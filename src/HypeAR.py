"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, List, Literal
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm

from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MinMetric
import pytorch_lightning as pl
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint
# from src.ema import EMA
from models.arch import tMLP, ProjectToTangent, Unbatch
# from manifolds import (
#     # Sphere,
#     # FlatTorus,
#     # Euclidean,
#     # ProductManifold,
#     # Mesh,
#     # SPD,
#     PoincareBall,
# )
# from src.manifolds.poincareball import PoincareBall as PoincareBall2
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
import manifolds
from manifolds.geodesic import geodesic
from solvers import projx_integrator_return_last, projx_integrator
from src.models.hyper_vae import HypFormer as HypVAE
from src.models.hyper_vae import HypAttention
from src.models.transormer_layers import DiTAttention
from src.models.PoincareTransformer import PoincareTransformerCausal
from src.manifolds.lorentz import Lorentz
from src.distribution.wrapped_normal import WrappedNormal, WrappedNormalLorentz, WrappedNormalPoincare
import pdb
import utils
import torch.nn as nn
import math
import time
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.product import ProductManifold
from geoopt import ManifoldParameter

class ManifoldARLitModule(pl.LightningModule):
    def __init__(self, cfg, sampling_metrics, glob_cfg):
        super().__init__()
        self.cfg = cfg
        self.glob_cfg = glob_cfg
        self.manifold_name = cfg.model.manifold
        self.manifold = eval(self.manifold_name)()

        # Model of the vector field.
        # self.model = HypAttention(cfg)
        self.euc_channels = glob_cfg.model.euc_channels
        self.hyp_channels = glob_cfg.model.hyp_channels
        self.nfe_steps = glob_cfg.flow_train.integrate.num_steps
        self.reconstruct_metrics = None
        if self.euc_channels ==0:
            # self.product_manifold = ProductManifold((self.manifold, self.hyp_channels))
            self.product_manifold = self.manifold
        if self.hyp_channels ==0:
            # self.product_manifold = ProductManifold((Euclidean(), self.euc_channels))
            self.product_manifold = Euclidean()
        if self.hyp_channels >0 and self.euc_channels >0:
            self.product_manifold = ProductManifold(
                (Euclidean(), self.euc_channels), (self.manifold, self.hyp_channels)
            )
        
        if self.glob_cfg.dataset.name in ['hyperbolic'] or self.euc_channels==0 :
            from .models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
            self.model = PoincareTransformerCausal(cfg, PoincareBall(), glob_cfg.model.latent_channels, glob_cfg.model.transformer_encoder.trans_num_layers, glob_cfg.model.transformer_encoder.trans_num_heads, glob_cfg.model.transformer_encoder.trans_dropout, 
                                                  glob_cfg.model.transformer_encoder.max_seq_len, glob_cfg.model.transformer_encoder.use_hyperbolic_attention, glob_cfg.model.transformer_encoder.attention_type, glob_cfg.model.transformer_encoder.attention_activation)
        else:
            self.model = DiTAttention(cfg, self.product_manifold)

        self.use_riemannian_optimizer = glob_cfg.loss.use_riemannian_optimizer
        if self.use_riemannian_optimizer:
            self.automatic_optimization = False
        # # Initialize learnable BOS token
        # self.bos_token = ManifoldParameter(
        #     self.product_manifold.random((1, 1, self.hyp_channels + self.euc_channels)),
        #     manifold=self.product_manifold
        # )
        
        # Optional: EOS prediction head
        # self.predict_eos = nn.Sequential(
        #     nn.Linear(self.hyp_channels + self.euc_channels, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)  # Output eos logits
        # )

        # how to use sampling_metrics?
        self.sampling_metrics = sampling_metrics
        self.name = glob_cfg.general.name + "-AR"
        self.val_counter = 0
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        self.val_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        self.test_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }

        # for logging best so far validation accuracy
        self.val_metrics_best = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        
        self.all_node_feats = []  # 收集所有有效的 node_feat
        self.all_edge_feats = []  # 收集所有有效的 edge_feat
        self.cnt = 0
        
        self.prior_count = torch.zeros(self.glob_cfg.loss.codebook_size).to(self.device)

    # add a function to load encoder+
    def load_VAE(self, VAE):
        self.VAE = VAE
        self.dataset_infos =  self.VAE.dataset_infos
        
    @torch.no_grad()
    def encode(self, batch):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # dense_data = dense_data.mask(node_mask)
        x, E = dense_data.X, dense_data.E
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.VAE.compute_extra_data(noisy_data)
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()
        # pdb.set_trace()
        # VQ VAE case
        if self.VAE.model.use_VQVAE:
            node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)

        else:
            # AE case
            if not self.VAE.model.use_VAE:
                node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)

            # VAE case  
            else:
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.VAE.model.encode(x, adj, E, node_mask)
                node_feat=None
                edge_feat=None
                if self.euc_channels>0:
                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    node_feat = euc_qz_x.rsample()
                if self.hyp_channels>0:
                    if not self.glob_cfg.model.use_poincare:
                        hyp_qz_x = WrappedNormalLorentz(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]), self.model.manifold_out)
                    else:
                        hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                    edge_feat = hyp_qz_x.rsample()
            
        return z, node_mask, node_feat, edge_feat
    
    
    @property
    def vecfield(self):
        return self.model

    @property
    def device(self):
        return self.model.parameters().__next__().device

    @torch.no_grad()
    def sample_decode(self, n_samples, x0=None, stage="valid"):
        start_time = time.time()
        x1, node_mask = self.sample(n_samples, x0)
        graph_list = self.VAE.generate_sample_from_z(x1, node_mask)
        end_time = time.time()
        print(f"Time taken for sample_decode: {end_time - start_time} seconds")
        # pdb.set_trace()
                # Only once after full pass
        current_path = os.getcwd()
        result_path = os.path.join(
            current_path,
            f'graphs/{self.name}/epoch{self.current_epoch}/'
        )
        self.VAE.visualization_tools.visualize(result_path, graph_list, n_samples)
        self.print("Visualization complete.")
        
        return graph_list

    @torch.no_grad()
    def sample(self, n_samples, x0=None):
        max_nodes = self.dataset_infos.n_nodes.shape[-1] # may be we need max_node = max_nodes - 1
        num_masked_tokens = torch.multinomial(self.dataset_infos.n_nodes, num_samples=n_samples, replacement=True)
        mask = torch.zeros((n_samples, max_nodes), device=self.device, dtype=torch.int64)
        for i, n in enumerate(num_masked_tokens):
            mask[i, :n] = 1
        mask = mask.bool()
        svq_temp = self.cfg.AR_sample.svq_temp
        if self.VAE.model.use_VQVAE:
            x1 = self.vq_autoregressive_generate(max_len=max_nodes, batch_size=n_samples, svq_temp=svq_temp, node_mask=mask)
        else:
            x1 = self.autoregressive_generate(max_len=max_nodes, batch_size=n_samples)
        return x1, mask

    @torch.no_grad()
    def autoregressive_generate(self, max_len, batch_size=1):
        bos = self.VAE.model.hyp_codebook.random_sample(batch_size)
        tokens = [bos]

        for _ in range(max_len-1):
            input_seq = torch.cat(tokens, dim=1)
            pred = self.model(input_seq)  # [B, T, D]
            next_token = pred[:, -1, :]
            tokens.append(next_token.unsqueeze(1))

        x1 = torch.cat(tokens, dim=1)
        return x1

    @torch.no_grad()
    def vq_autoregressive_generate(self, max_len, batch_size=1, svq_temp=1.0, node_mask=None):
        if self.cfg.AR_sample.use_prior_count:
            bos = self.VAE.model.hyp_codebook.random_sample(batch_size, prior_count=self.prior_count)
        else:
            bos = self.VAE.model.hyp_codebook.random_sample(batch_size)
        tokens = [bos.unsqueeze(1)]
        perplexity_total = 0
        for i in range(max_len-1):
            input_seq = torch.cat(tokens, dim=1)
            pred = self.model(input_seq)  # [B, T, D]
            next_token = pred[:, -1, :]
            mask = node_mask[:, i+1]
            embed_ind, hyp_quantize, perplexity = self.VAE.model.hyp_codebook.forward_hyp1(next_token, svq_temp=svq_temp, node_mask=mask)
            # print(f"predicted embed_ind: {embed_ind.detach().cpu()}")
            tokens.append(hyp_quantize.unsqueeze(1))
            perplexity_total += perplexity
        perplexity_total = perplexity_total / (max_len-1)
        self.log("perplexity_of_autoregressive_generation", perplexity_total, on_step=False, on_epoch=True, prog_bar=True)
        x1 = torch.cat(tokens, dim=1)
        return x1
    
    def loss_fn(self, batch: torch.Tensor):
        return self.autoregressive_loss(batch)

    def autoregressive_loss(self, batch):
        """
        x: [B, T, D] - sequence of points on manifold
        mask: [B, T] - optional padding mask
        """
        assert self.VAE.model.use_VQVAE
        if self.use_riemannian_optimizer and self.trainer.training:
            opt_euc, opt_hyp = self.optimizers()
            sched_euc, sched_hyp = self.lr_schedulers()
    
            self.log("lr_euc", sched_euc.get_last_lr()[0], prog_bar=True)
            self.log("lr_hyp", sched_hyp.get_last_lr()[0], prog_bar=False)

        _, node_mask, node_feat, edge_feat = self.encode(batch)
                
        if self.hyp_channels==0:
            x1 = node_feat
        if self.euc_channels==0:
            x1 = edge_feat
        if self.hyp_channels>0 and self.euc_channels>0:
            x1 = torch.concat((node_feat, edge_feat), dim=-1)

        # quantize x1
        with torch.no_grad():
            embed_ind, x1, perplexity = self.VAE.model.hyp_codebook.forward_hyp1(x1, svq_temp=None, node_mask=node_mask)
            self.log("perplexity_of_encoder_output", perplexity, on_step=True, on_epoch=True, prog_bar=True)
            # 把embed_ind[:, 0]加到prior_count中
            # 但是embed_ind还有batch维度
            self.prior_count[embed_ind[:, 0]] += 1

        # pdb.set_trace()
        # Expand BOS to match batch size
        B, T, D = x1.shape

        if self.cfg.AR_sample.use_discrete_codebook:
            # in indices space
            node_mask = node_mask[:, 1:]
            pred = self.model(embed_ind[:, :-1], node_mask)  # [B, T-1, Vocabulary size]
            embed_ind = embed_ind[:, 1:].view(-1)
            pred = pred.view(embed_ind.size(0), -1)
            node_mask = node_mask.view(-1)

            loss = F.cross_entropy(pred, embed_ind, reduction='none')
            if node_mask is not None:
                loss = (loss * node_mask).sum() / node_mask.sum()
            else:
                loss = loss.mean()



        # in latent space
        else:
            # bos = self.bos_token.expand(B, -1, -1)  # [B, 1, D]
            # x_input = torch.cat([bos, x1[:, :-1, :]], dim=1)  # [B, T, D]
            x_input = x1[:, :-1, :]
            node_mask = node_mask[:, 1:]
            pred = self.model(x_input, node_mask)  # [B, T-1, D]
            # In case of softmax loss
            logits = self.VAE.model.hyp_codebook.forward_hyp2(pred, svq_temp=1.0, node_mask=node_mask)
            # loss is the cross entropy of the embed_dist and the embed_ind
            # with mask node_mask
            embed_ind = embed_ind[:, 1:]
            embed_ind = embed_ind.reshape(-1)
            logits = logits.view(embed_ind.size(0), -1)   
            node_mask = node_mask.reshape(-1)
            loss = F.cross_entropy(logits, embed_ind, reduction='none')
            # flattent embed_ind
            # pdb.set_trace()
            if node_mask is not None:
                loss = (loss * node_mask).sum() / node_mask.sum()
            else:
                loss = loss.mean()
        
        # In case of L2 loss
        # else:
        #     x_target = x1  # [B, T, D]
        #     dist = self.product_manifold.dist(pred, x_target)  # [B, T-1]

        #     if node_mask is not None:
        #         loss = (dist * node_mask).sum() / node_mask.sum()
        #     else:
        #         loss = dist.mean()

        if self.use_riemannian_optimizer and self.trainer.training:
            opt_euc.zero_grad()
            opt_hyp.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer=opt_euc,
                # gradient_clip_val=self.glob_cfg.train.clip_grad,
                # gradient_clip_algorithm="norm"
                gradient_clip_val=self.cfg.train.manual_setting.gradient_clip_val,
                gradient_clip_algorithm=self.cfg.train.manual_setting.gradient_clip_algorithm

            )

            # utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)
            self.clip_gradients(
                optimizer=opt_hyp,
                gradient_clip_val=self.glob_cfg.train.clip_grad,
                gradient_clip_algorithm="norm"
            )
            opt_euc.step()
            opt_hyp.step()

        log_metric = loss.detach().item()
        log_metric_mean = loss.detach().item()
        if self.reconstruct_metrics is not None:
            for key, value in self.reconstruct_metrics.items():
                if self.glob_cfg.dataset.name == "qm9":
                    log_metric = log_metric * (2-torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device)))
                    log_metric_mean = log_metric_mean - torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))
                else:
                    log_metric = log_metric * torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))
                    log_metric_mean = log_metric_mean + torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))

        return {"loss": loss,
                "log_metric": log_metric,
                "log_metric_mean": log_metric_mean
            }


    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.autoregressive_loss(batch)

        if torch.isfinite(loss_dict['loss']):
            # log train metrics
            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)
                if isinstance(v, torch.Tensor):
                    self.train_metrics[k].update(v.cpu())
                else:
                    self.train_metrics[k].update(v)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss_dict['loss'].item()}.")
            return None

        return loss_dict


    def on_train_epoch_end(self):
        for train_metric in self.train_metrics.values():
            train_metric.reset()

        sched_euc, sched_hyp = self.lr_schedulers()
        sched_euc.step()
        sched_hyp.step()

    def shared_eval_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
        compute_loss: bool,
        compute_nll: bool,
    ) -> dict[str, torch.Tensor]:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        out = {}
        if compute_loss:
            loss_dict = self.loss_fn(batch)
            for k, v in loss_dict.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_epoch=True,
                    prog_bar=True,
                    # batch_size=batch.batch.max().item()+1,
                )
                if isinstance(v, torch.Tensor):
                    metrics[k].update(v.cpu())
                else:
                    metrics[k].update(v)
            out.update(loss_dict)

        return out
    
    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="val",
            compute_loss=True,
            compute_nll=self.cfg.val.compute_nll,
        )

    def compute_and_log_sampling_metrics(self, valid_result):
        dataset_name = self.glob_cfg.dataset.name
        base_statistics = {'sbm': [0.0008, 0.0332, 0.0255], 'planar': [0.0002, 0.0310, 0.0005], 'community': [0.02, 0.07, 0.01]}
        
        # Check if we have base statistics for this dataset
        if dataset_name in base_statistics:
            base_statistics = base_statistics[dataset_name]
            # Process datasets with base statistics for normalization
            degree_dist, clustering_dist, orbit_dist = valid_result['degree'], valid_result['clustering'], valid_result['orbit']
            degree_dist = degree_dist / base_statistics[0]
            clustering_dist = clustering_dist / base_statistics[1]
            orbit_dist = orbit_dist / base_statistics[2]
            print('degree dist: {:.3f}'.format(degree_dist))
            print('clustering dist: {:.3f}'.format(clustering_dist))
            print('orbit dist: {:.3f}'.format(orbit_dist))

            print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()
                                       
            valid_result["ratio_degree"] = degree_dist
            valid_result["ratio_clustering"] = clustering_dist
            valid_result["ratio_orbit_dist"] = orbit_dist
        else:
            # For datasets without base statistics, log the raw metrics
            print(f"No base statistics for {dataset_name}, using raw metrics")
            
            if 'degree' in valid_result:
                print('Raw degree dist: {:.3f}'.format(valid_result['degree']))
            if 'clustering' in valid_result:
                print('Raw clustering dist: {:.3f}'.format(valid_result['clustering']))
            if 'orbit' in valid_result:
                print('Raw orbit dist: {:.3f}'.format(valid_result['orbit']))
            
            if 'sampling/frac_unique' in valid_result:
                print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            if 'sampling/frac_unique_non_iso' in valid_result:
                print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            if 'sampling/frac_unic_non_iso_valid' in valid_result:
                print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()


        abstract_dataset = ['sbm', 'planar', 'community']
        
        if dataset_name in abstract_dataset:
            degree_dist, clustering_dist, orbit_dist = valid_result['degree'], valid_result['clustering'], valid_result['orbit']
            degree_dist = degree_dist / base_statistics[0]
            clustering_dist = clustering_dist / base_statistics[1]
            orbit_dist = orbit_dist / base_statistics[2]
            print('degree dist: {:.3f}'.format(degree_dist))
            print('clustering dist: {:.3f}'.format(clustering_dist))
            print('orbit dist: {:.3f}'.format(orbit_dist))

            print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()
                                       
            valid_result["ratio_degree"] = degree_dist
            valid_result["ratio_clustering"] = clustering_dist
            valid_result["ratio_orbit_dist"] = orbit_dist
        

        # 提取并打印指标
        for metric_name, value in valid_result.items():
            print(f'{metric_name}: {value:.3f}')
            # 使用 wandb 记录指标
            self.log(f'sampling_metrics/{metric_name}', value, on_epoch=True, prog_bar=True)


    def on_validation_epoch_start(self):
        self.prior_count = self.prior_count.to(self.device)
        if self.val_counter == 0:
            start = time.time()
            samples_left_to_generate = self.glob_cfg.general.samples_to_generate
            samples = []
            while samples_left_to_generate > 0:
                bs = self.glob_cfg.flow_train.batch_size.val
                to_generate = min(samples_left_to_generate, bs)
                samples.extend(self.sample_decode(n_samples=to_generate, stage="valid"))
                samples_left_to_generate -= to_generate
            
            self.sampling_metrics.reset()
            self.reconstruct_metrics = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank)
            self.sampling_metrics.reset()

            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank)
            self.sampling_metrics.reset()
    
    def on_validation_epoch_end(self):
        out = {}
        for key, val_metric in self.val_metrics.items():
            val_metric_value = (
                val_metric.compute()
            )  # get val accuracy from current epoch
            val_metric_best = self.val_metrics_best[key]
            val_metric_best.update(val_metric_value)
            self.log(
                f"val/best/{key}",
                val_metric_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )
            val_metric.reset()
            out[key] = val_metric_value
            
            
        self.val_counter += 1
        if self.val_counter % self.glob_cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.glob_cfg.general.samples_to_generate
            samples = []
            while samples_left_to_generate > 0:
                bs = self.glob_cfg.flow_train.batch_size.val
                to_generate = min(samples_left_to_generate, bs)
                samples.extend(self.sample_decode(n_samples=to_generate, stage="test"))
                samples_left_to_generate -= to_generate
            
            self.sampling_metrics.reset()
            self.reconstruct_metrics = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank)
            self.sampling_metrics.reset()

            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank)
            self.sampling_metrics.reset()
            # valid_result = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank)
            # self.compute_and_log_sampling_metrics(valid_result)
            # self.sampling_metrics.reset()
            
        return out
        
    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="test",
            compute_loss=self.cfg.test.get("compute_loss", False),
            compute_nll=self.cfg.test.get("compute_nll", False),
        )
        return None

    def on_test_epoch_end(self):
        for test_metric in self.test_metrics.values():
            test_metric.reset()
            
        samples_left_to_generate = self.glob_cfg.general.samples_to_generate
        samples = []
        while samples_left_to_generate > 0:
            bs = 64
            to_generate = min(samples_left_to_generate, bs)
            samples.extend(self.sample_decode(n_samples=to_generate, stage="test"))
            samples_left_to_generate -= to_generate

        self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank)
        self.sampling_metrics.reset()
        # valid_result = self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        # self.compute_and_log_sampling_metrics(valid_result)
        # self.sampling_metrics.reset()

    def configure_optimizers(self):
        if self.use_riemannian_optimizer:
            # 1. 参数拆分
            euc_params = [p for n, p in self.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]
            hyp_params = [p for n, p in self.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]

            # print hyp params name
            hyp_name = [n for n, p in self.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]
            print(hyp_name)
            # 2. 构建优化器
            optimizer_euc = hydra.utils.instantiate(
                self.cfg.optim.optimizer,
                params=euc_params,
                _convert_="partial",
            )
            optimizer_hyp = hydra.utils.instantiate(
                self.cfg.optim.optimizer_hyp,
                params=hyp_params,
                _convert_="partial",
            )

            # 3. 构建调度器（如果需要）
            schedulers = []
            optimizers = [{"optimizer": optimizer_euc}, {"optimizer": optimizer_hyp}]

            if self.cfg.optim.get("lr_scheduler", None) is not None:
                sched_euc = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer_euc,
                )
                sched_hyp = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer_hyp,
                )

                # 添加 scheduler（PyTorch Lightning 格式）
                schedulers = [
                    {
                        "optimizer": optimizer_euc,
                        "lr_scheduler": {
                            "scheduler": sched_euc,
                            "interval": self.cfg.optim.interval,
                        }
                    },
                    {
                        "optimizer": optimizer_hyp,
                        "lr_scheduler": {
                            "scheduler": sched_hyp,
                            "interval": self.cfg.optim.interval,
                        }
                    }
                ]
                return schedulers
            return optimizers
        else:
        
            optimizer = hydra.utils.instantiate(
                self.cfg.optim.optimizer,
                params=self.parameters(),
                _convert_="partial",
            )
            if self.cfg.optim.get("lr_scheduler", None) is not None:
                lr_scheduler = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer,
                )
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            "interval": "epoch",
                            "monitor": self.cfg.optim.monitor,
                            "frequency": self.cfg.optim.frequency,
                        },
                    }
                elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            "interval": self.cfg.optim.interval,
                        },
                    }
                else:
                    raise NotImplementedError("unsuported lr_scheduler")
            else:
                return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # if isinstance(self.model, EMA):
        #     self.model.update_ema()

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f'Gradient {name} {param.grad.abs().mean().item()}')
    #         else:
    #             print(f'Gradient {name} None')
    #     import pdb; pdb.set_trace()