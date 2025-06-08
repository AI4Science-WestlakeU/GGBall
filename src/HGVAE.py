import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from torch_geometric.utils import dense_to_sparse

from models.hyper_decoder import FermiDiracDecoder, SoftmaxDecoder
import manifolds
import models.hyper_encoder as encoders
import models.hyper_decoder as decoders
from src.models.hyper_vae import HypFormer as HypVAE
from torch_geometric.nn import GCNConv
from metrics.train_metrics import HGVAETrainLoss
import utils
from src.distribution.wrapped_normal import WrappedNormalLorentz, WrappedNormalPoincare, WrappedNormal
from src.manifolds.lorentz import Lorentz
# from src.manifolds.poincareball import PoincareBall
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
import math
from utils import PlaceHolder, check_on_manifold
import pdb
from debug import FullyConnectedResNet
from geoopt import ManifoldParameter
from geoopt.optim.radam import RiemannianAdam
import hydra

class HGVAE(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features, domain_features):
        super(HGVAE, self).__init__()
        
        self.cfg = cfg
        self.use_VAE = (cfg.loss.lambda_kl != 0)
        self.use_VQVAE = (cfg.loss.lambda_commitment_weight != 0)
        self.c = cfg.model.c
        if self.use_VAE:
            self.name = cfg.general.name + "-VAE-" + cfg.dataset.name
        elif self.use_VQVAE:
            self.name = cfg.general.name + "-VQVAE-" + cfg.dataset.name
        else:
            self.name = cfg.general.name + "-AE-" + cfg.dataset.name
        
        self.dataset_infos = dataset_infos
        self.manifold_name = cfg.model.manifold
        self.nodes_dist = dataset_infos.nodes_dist

        self.model = HypVAE(cfg)
        self.product_manifold = self.model.product_manifold
        self.euc_channels = self.model.euc_channels
        self.hyp_channels = self.model.hyp_channels
        
        self.shuffle_dense_graph = cfg.model.shuffle_dense_graph
        self.train_loss = HGVAETrainLoss(cfg, euc_channels=self.euc_channels, hyp_channels=self.hyp_channels, stage='train')
        self.val_loss = HGVAETrainLoss(cfg, euc_channels=self.euc_channels, hyp_channels=self.hyp_channels, stage='val')
        self.test_loss = HGVAETrainLoss(cfg, euc_channels=self.euc_channels, hyp_channels=self.hyp_channels, stage='test')
        self.train_metrics = train_metrics
        self.log_every_steps = cfg.general.log_every_steps
        self.sampling_metrics = sampling_metrics
        self.val_counter = 0
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.visualization_tools = visualization_tools
        self.reconstruct_metrics = None
        # self.btc_model = FullyConnectedResNet(self.model.manifold_out, 128, 128)
        self.use_riemannian_optimizer = cfg.loss.use_riemannian_optimizer
        if self.use_riemannian_optimizer:
            self.automatic_optimization = False
        self.prior_std = 0.5/math.sqrt(float(self.model.euc_channels + self.model.hyp_channels))


    @torch.no_grad()
    def reconstruct_step(self, batch):
        batch = batch.to(self.device)
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        # if self.shuffle_dense_graph:
        #     x, E, node_mask = utils.shuffle_dense_graph(x, E, node_mask)
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        # x = torch.cat((x, extra_data.X), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()
        if self.use_VQVAE:
            euc_feat, hyp_feat, z = self.model.encode(x, adj, E, node_mask)

            # pdb.set_trace() 
            # (Pdb) euc_feat.shape
            # torch.Size([32, 13, 128])
            # (Pdb) hyp_feat.shape
            # torch.Size([32, 13, 4])
            if self.euc_channels > 0:
                euc_quantize, euc_vq_ind, euc_vq_loss, euc_perplexity = self.model.codebook(euc_feat, codebook_type = "euc", node_mask=node_mask)
            else:
                euc_quantize = None
                euc_vq_loss = None
            if self.hyp_channels > 0:
                hyp_quantize, hyp_vq_ind, hyp_vq_loss, hyp_perplexity = self.model.codebook(hyp_feat, codebook_type = "hyp", node_mask=node_mask)
            else:
                hyp_quantize = None
                hyp_vq_loss = None
            
            if self.euc_channels > 0 and self.hyp_channels > 0:
                graph_feat = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            elif self.euc_channels > 0:
                graph_feat = euc_quantize
            elif self.hyp_channels > 0:
                graph_feat = hyp_quantize
            else:
                graph_feat = None
                
            euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
            
            reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
            reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)
            if euc2node_feat is not None:
                reconstruct_x += euc2node_feat
            if hyp2node_feat is not None:
                reconstruct_x += hyp2node_feat

            if euc2edge_feat is not None:
                reconstruct_E += euc2edge_feat
            if hyp2edge_feat is not None:
                reconstruct_E += hyp2edge_feat
        
        else:        
            if not self.use_VAE:
                node_feat, edge_feat, z = self.model.encode(x, adj, E, node_mask)
                if self.euc_channels == 0:
                    graph_feat = edge_feat
                elif self.hyp_channels == 0:
                    graph_feat = node_feat
                else:
                    graph_feat = torch.cat((node_feat, edge_feat), dim= -1)
                
                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
                pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
                qz_x_poincare = None
                pz_poincare = None
                if self.euc_channels == 0:
                    reconstruct_x = hyp2node_feat
                    reconstruct_E = hyp2edge_feat
                elif self.hyp_channels == 0:
                    reconstruct_x = euc2node_feat
                    reconstruct_E = euc2edge_feat
                else:
                    reconstruct_x = euc2node_feat + hyp2node_feat
                    reconstruct_E = euc2edge_feat + hyp2edge_feat
                
            else:
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.model.encode(x, adj, E, node_mask)

                # 让被mask的位置的值等于0
                if node_mask is not None:
                    mask_expanded = node_mask.unsqueeze(-1)
                    if self.euc_channels > 0:
                        euc_feat_mean = euc_feat_mean * mask_expanded
                        euc_feat_logvar = euc_feat_logvar * mask_expanded
                    if self.hyp_channels > 0:
                        hyp_feat_mean = hyp_feat_mean * mask_expanded
                        hyp_feat_logvar = hyp_feat_logvar * mask_expanded

                euc_qz_x = None
                euc_sample_z = None
                if self.euc_channels > 0:
                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    euc_sample_z = euc_qz_x.rsample()

                hyp_qz_x = None
                hyp_sample_z = None
                if self.hyp_channels > 0:
                    if not self.cfg.model.use_poincare:
                        if hyp_feat_mean is not None:
                            hyp_qz_x = WrappedNormalLorentz(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]), self.model.manifold_out)
                            hyp_sample_z = hyp_qz_x.rsample()
                        else:
                            hyp_qz_x = None
                            hyp_sample_z = None

                        hyp_pz = torch.distributions.Normal(
                            torch.zeros_like(hyp_feat_mean[..., 1:]),
                            torch.ones_like(hyp_feat_logvar[..., 1:]).mul(self.cfg.model.prior_std),
                        )
                        hyp_qz_x = torch.distributions.Normal(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]))
                    
                    else:
                        hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                        hyp_sample_z = hyp_qz_x.rsample()


                if euc_sample_z is not None and hyp_sample_z is not None:
                    graph_feat = torch.cat((euc_sample_z, hyp_sample_z), dim=-1)
                elif euc_sample_z is not None:
                    graph_feat = euc_sample_z
                elif hyp_sample_z is not None:
                    graph_feat = hyp_sample_z
                else:
                    graph_feat = None
                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                
                reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
                reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)

                if euc2node_feat is not None:
                    reconstruct_x += euc2node_feat 
                if hyp2node_feat is not None:
                    reconstruct_x += hyp2node_feat 
                if euc2edge_feat is not None:
                    reconstruct_E += euc2edge_feat 
                if hyp2edge_feat is not None:
                    reconstruct_E += hyp2edge_feat 
        
        sample_y = torch.zeros([1, 0]).float()

        # we are not using sample_adj right now
        sample = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask(node_mask=node_mask, collapse=True)
        X, E, y = sample.X, sample.E, sample.y
        diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0) 
        E[diag_mask.expand(E.shape[0], -1, -1)] = 0  # 扩展到 batch 维度并设置对角线为 0
        
        
        # n_nodes = node_mask.sum(-1)
        # batch_size = X.shape[0]
        # molecule_list = []
        # for i in range(batch_size):
        #     n = n_nodes[i]
        #     atom_types = X[i, :n].cpu()
        #     edge_types = E[i, :n, :n].cpu()
        #     molecule_list.append([atom_types, edge_types])
            
        # # Only once after full pass
        # current_path = os.getcwd()
        # result_path = os.path.join(
        #     current_path,
        #     f'graphs/{self.name}/reconstruct_epoch_{self.current_epoch}/'
        # )
        # self.visualization_tools.visualize(result_path, molecule_list, batch_size, "reconstruct_graph")
        # self.print("Visualization complete.")
        
        return X, E, node_mask.sum(-1)

    def interpolate(self, z1, z2, time_steps: int):
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        # z1 = z1.unsqueeze(0)
        # z2 = z2.unsqueeze(0)
        # z1的形状[dim, ] 
        two_pts = torch.stack((z1, z2), dim=1) # [1, 2, d]
        z = torch.zeros(time_steps+1, z1.size(0), z1.size(1)).to(self.device)
        for i in range(time_steps):
            # 这里需要使用双曲差值
            weight = torch.tensor([1 - i / time_steps, i / time_steps]).unsqueeze(0).expand(two_pts.size(0), 2).to(self.device) # [1, 2]
            mid_point = self.product_manifold.weighted_midpoint(
                xs = two_pts,
                weights = weight,
                reducedim = [1],
                keepdim = False
            )
            z[i, :, :] = mid_point
        z[time_steps, :, :] = z2
        return z

    @torch.no_grad()
    def test_interpolate(self, batch, steps: int = 10, batch_index: int = 0):
        # 设置随机种子以确保可重复性
        torch.manual_seed(34)
        torch.cuda.manual_seed(34)
        np.random.seed(34)
        
        # 随机选择两个点
        batch = batch.to(self.device)
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()

        node_feat, edge_feat, z = self.model.encode(x, adj, E, node_mask)
        # pdb.set_trace()
        n_nodes = node_mask.sum(-1)
        #  随机选择两个点，他们的n_nodes必须一样
        # random_start是在x.size(1)中随机选择一个index
        # random_end是在x.size(1)中随机选择一个index
        # 但是random_end不能和random_start一样，也不能和random_start的n_nodes不一样
        
        # 多做几个random start
        random_start_list = []
        for i in range(10):
            # 如何取消随机性在这里
            
            # pdb.set_trace()
            random_start = torch.randint(0, edge_feat.size(0), (1,))
            random_end = torch.randint(0, edge_feat.size(0), (1,))
            # random_start = torch.tensor(429)
            # random_end = torch.tensor(445)
            # pdb.set_trace
            while random_start == random_end or n_nodes[random_start] != n_nodes[random_end]:
                random_end = torch.randint(0, edge_feat.size(1), (1,))
                random_start = torch.randint(0, edge_feat.size(1), (1,))
            z1 = edge_feat[random_start, ...][0]
            z2 = edge_feat[random_end, ...][0]
            # z1 = edge_feat[429, ...]
            # z2 = edge_feat[445, ...]
            z = self.interpolate(z1, z2, steps)
            n_nodes_random = n_nodes[random_start]
            node_mask_random = node_mask[random_start, ...].expand(steps+1, -1)

            # 使用模型进行解码
            euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(z, adj, node_mask_random)
            reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
            reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)
            if euc2node_feat is not None:
                reconstruct_x += euc2node_feat
            if hyp2node_feat is not None:
                reconstruct_x += hyp2node_feat
            if euc2edge_feat is not None:
                reconstruct_E += euc2edge_feat
            if hyp2edge_feat is not None:
                reconstruct_E += hyp2edge_feat

            sample_y = torch.zeros([1, 0]).float()

            # we are not using sample_adj right now
            sample = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask_argmax(node_mask=node_mask_random, collapse=True)
            sample_soft = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask(node_mask=node_mask_random, collapse=True)
            X, E, y = sample.X, sample.E, sample.y
            X_soft, E_soft, y_soft = sample_soft.X, sample_soft.E, sample_soft.y
            diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0) 
            E[diag_mask.expand(E.shape[0], -1, -1)] = 0  # 扩展到 batch 维度并设置对角线为 0
            # top triangle mask E
            E = torch.triu(E, diagonal=1)
            # Make E symmetric
            E = E + E.transpose(1, 2)
            E_soft[diag_mask.expand(E_soft.shape[0], -1, -1)] = 0
            E_soft = torch.triu(E_soft, diagonal=1)
            # Make E_soft symmetric
            E_soft = E_soft + E_soft.transpose(1, 2)

            molecule_list = []
            
            # pdb.set_trace()
            for i in range(steps+1):
                n = n_nodes_random
                atom_types = X[i, :n].cpu()
                edge_types = E[i, :n, :n].cpu()
                molecule_list.append([atom_types, edge_types])
            

            result_path = os.path.join(
                '/home/crwang/code/graph-generation/HypeFlow/data/interpolate/',
                f'graphs/{self.name}/strips_ckpt2/interpolate_train_batch_index_{batch_index}_random_{random_start.item()}_{random_end.item()}_argmax/'
            )
            self.visualization_tools.visualize_interpolation_strip2(molecule_list, result_path, log_tag="interpolate_qm9")
            # pdb.set_trace()
            
            # # Only once after full pass
            # current_path = os.getcwd()
            # result_path = os.path.join(
            #     '/home/crwang/code/graph-generation/HypeFlow/data/interpolate/',
            #     f'graphs/{self.name}/interpolate_random_{random_start.item()}_{random_end.item()}_argmax/'
            # )
            # self.visualization_tools.visualize(result_path, molecule_list, steps+1, "interpolate_qm9")
            # self.print("Visualization complete.")
            
            # molecule_list_soft = []
            # for i in range(steps+1):
            #     n = n_nodes_random
            #     atom_types = X_soft[i, :n].cpu()
            #     edge_types = E_soft[i, :n, :n].cpu()
            #     molecule_list_soft.append([atom_types, edge_types])

            # result_path_soft = os.path.join(
            #     '/home/crwang/code/graph-generation/HypeFlow/data/interpolate/',
            #     f'graphs/{self.name}/interpolate_random_{random_start.item()}_{random_end.item()}_soft/'
            # )
            # self.visualization_tools.visualize(result_path_soft, molecule_list_soft, steps+1, "interpolate_qm9")
            # self.print("Visualization complete.")
            
        return X, E, node_mask.sum(-1)

            
            
            
            


    def training_step(self, batch, batch_idx):
        if self.use_riemannian_optimizer:
            opt_euc, opt_hyp = self.optimizers()
            sched_euc, sched_hyp = self.lr_schedulers()
            self.log("lr_euc", sched_euc.get_last_lr()[0], on_step=False, on_epoch=True, prog_bar=True)
            self.log("lr_hyp", sched_hyp.get_last_lr()[0], on_step=False, on_epoch=True, prog_bar=False)
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        if self.shuffle_dense_graph:
            x, E, node_mask = utils.shuffle_dense_graph(x, E, node_mask)
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        

        # 检查输入数据是否有NaN
        if torch.isnan(x).any() or torch.isnan(E).any():
            print("警告: 输入数据中存在NaN值")
            print(f"x中NaN的数量: {torch.isnan(x).sum().item()}")
            print(f"E中NaN的数量: {torch.isnan(E).sum().item()}")
            # 保存当前模型状态
            torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
            # 停止训练
            self.trainer.should_stop = True
            return {'loss': torch.tensor(0.0, device=self.device)}
        
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()
        
        # 检查处理后的数据是否有NaN
        if torch.isnan(x).any() or torch.isnan(E).any():
            print("警告: 处理后的数据中存在NaN值")
            print(f"x中NaN的数量: {torch.isnan(x).sum().item()}")
            print(f"E中NaN的数量: {torch.isnan(E).sum().item()}")
            torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
            self.trainer.should_stop = True
            return {'loss': torch.tensor(0.0, device=self.device)}
        
        # only euclidean
        # -> modification on model structure
        # -> modification on manifold / random normal
        
        
        # euclidean + poincare
        
        # euclidean + lorentz
        if self.use_VQVAE:
            euc_feat, hyp_feat, z = self.model.encode(x, adj, E, node_mask)
            if self.euc_channels > 0:
                euc_quantize, euc_vq_ind, euc_vq_loss, euc_perplexity = self.model.codebook(euc_feat, codebook_type = "euc", node_mask=node_mask)
            else:
                euc_quantize = None
                euc_vq_loss = None
            if self.hyp_channels > 0:
                hyp_quantize, hyp_vq_ind, hyp_vq_loss, hyp_perplexity = self.model.codebook(hyp_feat, codebook_type = "hyp", node_mask=node_mask)
                self.log("train/hyp_perplexity", hyp_perplexity, on_step=False, on_epoch=True)
                embed_norm = hyp_vq_loss['codebook_embed_norm'].detach()
                # wandb log
                self.log("train/hyp_embed_norm", embed_norm.mean().item(), on_step=False, on_epoch=True)
                self.log("train/hyp_embed_norm_min", embed_norm.min().item(), on_step=False, on_epoch=True)
                self.log("train/hyp_embed_norm_max", embed_norm.max().item(), on_step=False, on_epoch=True)
            else:
                hyp_quantize = None
                hyp_vq_loss = None
            
            if self.euc_channels > 0 and self.hyp_channels > 0:
                graph_feat = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            elif self.euc_channels > 0:
                graph_feat = euc_quantize
            elif self.hyp_channels > 0:
                graph_feat = hyp_quantize
            else:
                graph_feat = None
                
            # graph_feat = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
            
            data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
            pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
            
            euc_distribution = None # {'pz': euc_pz, 'qz_x': euc_qz_x}
            hyp_distribution = None # {'pz': hyp_pz, 'qz_x': hyp_qz_x}
            
            euc_feature = {'euc_feat':euc_feat, 'euc_vq_loss':euc_vq_loss} # {'mean':euc_feat_mean, 'var':euc_feat_logvar, 'sample':euc_sample_z}
            hyp_feature = {'hyp_feat':hyp_feat, 'hyp_vq_loss':hyp_vq_loss} # {'mean':hyp_feat_mean, 'var':hyp_feat_logvar, 'sample':hyp_sample_z}
                
            loss, to_log = self.train_loss.forward(data, pred, euc_distribution, hyp_distribution, euc_feature, hyp_feature, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=False)
            self.log_dict(to_log)
            
            # Initialize reconstruction with zero tensor (same shape as the expected output)
            reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
            reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)
            if euc2node_feat is not None:
                reconstruct_x += euc2node_feat * self.cfg.loss.lambda_euc2node
            if hyp2node_feat is not None:
                reconstruct_x += hyp2node_feat * self.cfg.loss.lambda_hyp2node

            if euc2edge_feat is not None:
                reconstruct_E += euc2edge_feat * self.cfg.loss.lambda_euc2edge
            if hyp2edge_feat is not None:
                reconstruct_E += hyp2edge_feat * self.cfg.loss.lambda_hyp2edge
                
            self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                            log=batch_idx % self.log_every_steps == 0)
           

        else:        
            if not self.use_VAE:
                node_feat, edge_feat, z = self.model.encode(x, adj, E, node_mask)
                if self.euc_channels == 0:
                    graph_feat = edge_feat
                elif self.hyp_channels == 0:
                    graph_feat = node_feat
                else:
                    graph_feat = torch.cat((node_feat, edge_feat), dim= -1)
                
                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
                pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
                qz_x_poincare = None
                pz_poincare = None
                sample_z_poincare = None
                loss, to_log = self.train_loss.forward(data, pred, pz_poincare, qz_x_poincare, node_feat, edge_feat, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=False)
                self.log_dict(to_log)
                if self.euc_channels == 0:
                    reconstruct_x = hyp2node_feat
                    reconstruct_E = hyp2edge_feat
                elif self.hyp_channels == 0:
                    reconstruct_x = euc2node_feat
                    reconstruct_E = euc2edge_feat
                else:
                    reconstruct_x = euc2node_feat * self.cfg.loss.lambda_euc2node + hyp2node_feat * self.cfg.loss.lambda_hyp2node
                    reconstruct_E = euc2edge_feat * self.cfg.loss.lambda_euc2edge + hyp2edge_feat * self.cfg.loss.lambda_hyp2edge
                self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                                log=batch_idx % self.log_every_steps == 0)
                
            else:
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.model.encode(x, adj, E, node_mask)
                
                # 检查编码器输出是否有NaN
                if self.euc_channels > 0:
                    if torch.isnan(euc_feat_mean).any() or torch.isnan(euc_feat_logvar).any():
                        print("警告: 编码器输出中存在NaN值")
                        print(f"euc_feat_mean中NaN的数量: {torch.isnan(euc_feat_mean).sum().item()}")
                        print(f"euc_feat_logvar中NaN的数量: {torch.isnan(euc_feat_logvar).sum().item()}")
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                if self.hyp_channels > 0:
                    if torch.isnan(hyp_feat_mean).any() or torch.isnan(hyp_feat_logvar).any():
                        print(f"hyp_feat_mean中NaN的数量: {torch.isnan(hyp_feat_mean).sum().item()}")
                        print(f"hyp_feat_logvar中NaN的数量: {torch.isnan(hyp_feat_logvar).sum().item()}")
                        torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                # 让被mask的位置的值等于0
                if node_mask is not None:
                    mask_expanded = node_mask.unsqueeze(-1)
                    if self.euc_channels > 0:
                        euc_feat_mean = euc_feat_mean * mask_expanded
                        euc_feat_logvar = euc_feat_logvar * mask_expanded
                    if self.hyp_channels > 0:
                        hyp_feat_mean = hyp_feat_mean * mask_expanded
                        hyp_feat_logvar = hyp_feat_logvar * mask_expanded
                

                # euclidean standard normal and hyperbolic standard normal         
                euc_pz = None
                euc_qz_x = None
                euc_sample_z = None
                if self.euc_channels > 0:
                    euc_pz = torch.distributions.Normal(
                        torch.zeros_like(euc_feat_mean),
                        torch.ones_like(euc_feat_logvar)
                    )

                
                # # 添加数值稳定处理（原有代码修改）
                # std = torch.exp(0.5 * euc_feat_logvar.clamp(min=-20, max=20))  # 限制logvar范围
                # std = std + 1e-8  # 防止零标准差

                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    euc_sample_z = euc_qz_x.rsample()

                hyp_pz = None
                hyp_qz_x = None
                hyp_sample_z = None
                if self.hyp_channels > 0:
                    if not self.cfg.model.use_poincare:
                        if hyp_feat_mean is not None:
                            hyp_pz = WrappedNormalLorentz(
                                torch.zeros_like(hyp_feat_mean[..., 1:]),
                                torch.ones_like(hyp_feat_logvar[..., 1:]).mul(self.cfg.model.prior_std),
                                self.model.manifold_out
                            )
                            hyp_qz_x = WrappedNormalLorentz(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]), self.model.manifold_out)
                            hyp_sample_z = hyp_qz_x.rsample()
                        else:
                            hyp_pz = None
                            hyp_qz_x = None
                            hyp_sample_z = None
                        # import pdb; pdb.set_trace()
                        
                        hyp_pz = torch.distributions.Normal(
                            torch.zeros_like(hyp_feat_mean[..., 1:]),
                            torch.ones_like(hyp_feat_logvar[..., 1:]).mul(self.cfg.model.prior_std),
                        )
                        hyp_qz_x = torch.distributions.Normal(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]))
                    
                    else:
                        if hyp_feat_mean is not None:
                            hyp_pz = WrappedNormalPoincare(
                                torch.zeros_like(hyp_feat_mean),
                                torch.ones_like(hyp_feat_logvar).mul(self.cfg.model.prior_std).mul(1/torch.tensor(self.cfg.model.hyp_channels, dtype=torch.float32, device=self.device)),
                                self.model.manifold
                            )
                            hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                            hyp_sample_z = hyp_qz_x.rsample()
                        else:
                            hyp_pz = None
                            hyp_qz_x = None
                            hyp_sample_z = None
                    

                # hyp_qz_x = torch.distributions.Normal(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]))
                if euc_sample_z is not None and hyp_sample_z is not None:
                    graph_feat = torch.cat((euc_sample_z, hyp_sample_z), dim=-1)
                elif euc_sample_z is not None:
                    graph_feat = euc_sample_z
                elif hyp_sample_z is not None:
                    graph_feat = hyp_sample_z
                else:
                    graph_feat = None  # 或者你可以根据需要设定一个默认值

                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                
                # 检查解码器输出是否有NaN
                if self.euc_channels > 0:
                    if torch.isnan(euc2node_feat).any() or torch.isnan(euc2edge_feat).any():
                        print("警告: 解码器输出中存在NaN值")
                        print(f"euc2node_feat中NaN的数量: {torch.isnan(euc2node_feat).sum().item()}")
                        print(f"euc2edge_feat中NaN的数量: {torch.isnan(euc2edge_feat).sum().item()}")
                        torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                if self.hyp_channels > 0:
                    if torch.isnan(hyp2node_feat).any() or torch.isnan(hyp2edge_feat).any():
                        print(f"hyp2node_feat中NaN的数量: {torch.isnan(hyp2node_feat).sum().item()}")
                        print(f"hyp2edge_feat中NaN的数量: {torch.isnan(hyp2edge_feat).sum().item()}")
                        torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
                pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
                
                euc_distribution = {'pz': euc_pz, 'qz_x': euc_qz_x}
                hyp_distribution = {'pz': hyp_pz, 'qz_x': hyp_qz_x}
                
                euc_feature = {'mean':euc_feat_mean, 'var':euc_feat_logvar, 'sample':euc_sample_z}
                hyp_feature = {'mean':hyp_feat_mean, 'var':hyp_feat_logvar, 'sample':hyp_sample_z}
                
                loss, to_log = self.train_loss.forward(data, pred, euc_distribution, hyp_distribution, euc_feature, hyp_feature, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=True)
                self.log_dict(to_log)
                
                # 检查损失是否有NaN
                if torch.isnan(loss):
                    print("警告: 损失计算中出现NaN值")
                    torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                    self.trainer.should_stop = True
                    return {'loss': torch.tensor(0.0, device=self.device)}
                
                # Initialize reconstruction with zero tensor (same shape as the expected output)
                reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
                reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)
                if euc2node_feat is not None:
                    reconstruct_x += euc2node_feat
                if hyp2node_feat is not None:
                    reconstruct_x += hyp2node_feat

                if euc2edge_feat is not None:
                    reconstruct_E += euc2edge_feat
                if hyp2edge_feat is not None:
                    reconstruct_E += hyp2edge_feat

                # 检查重建结果是否有NaN
                if torch.isnan(reconstruct_x).any() or torch.isnan(reconstruct_E).any():
                    print("警告: 重建结果中存在NaN值")
                    print(f"reconstruct_x中NaN的数量: {torch.isnan(reconstruct_x).sum().item()}")
                    print(f"reconstruct_E中NaN的数量: {torch.isnan(reconstruct_E).sum().item()}")
                    torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                    self.trainer.should_stop = True
                    return {'loss': torch.tensor(0.0, device=self.device)}
                
                self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                                log=batch_idx % self.log_every_steps == 0)
        

        
        if self.use_riemannian_optimizer:
            opt_euc.zero_grad()
            opt_hyp.zero_grad()

            # 反向传播
            self.manual_backward(loss)
            # pdb.set_trace()
            # from geoopt import ManifoldParameter
            # assert isinstance(self.model.hyp_codebook.codebook, ManifoldParameter)
            # >>> 这里插入监控梯度的部分 <<<
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad and isinstance(param, ManifoldParameter):
            #         if param.grad is not None:
            #             grad_norm = param.grad.norm()
            #             if "hyp_codebook" in name:
            #                 self.log(f'grad/{name}_norm', grad_norm, on_step=False, on_epoch=True, prog_bar=True)
            # clip
            # 分别优化
            self.clip_gradients(
                optimizer=opt_euc,
                gradient_clip_val=self.cfg.train.clip_grad,
                gradient_clip_algorithm="norm"
            )

            # utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)
            self.clip_gradients(
                optimizer=opt_hyp,
                gradient_clip_val=self.cfg.train.clip_grad,
                gradient_clip_algorithm="norm"
            )
            opt_euc.step()
            opt_hyp.step()

        # self.log("train_loss", loss)
                
        return {'loss':loss}
    
    
    def validation_step(self, batch, batch_idx):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        # if self.shuffle_dense_graph:
        #     x, E, node_mask = utils.shuffle_dense_graph(x, E, node_mask)
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        
        
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        # pdb.set_trace()
        
        # x = torch.cat((x, extra_data.X), dim=2).float()
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        # pdb.set_trace()
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()

        # import pdb; pdb.set_trace()
        # hyp_pz = WrappedNormalPoincare(
        #     torch.zeros(4,self.cfg.model.hyp_channels),
        #     torch.ones(4,self.cfg.model.hyp_channels).mul(self.cfg.model.prior_std),
        #     PoincareBall(dim=self.cfg.model.hyp_channels, c = 1/float(self.cfg.model.k_poin_out))
        # )
        # hyp_sample_z = hyp_pz.rsample()

        # hyp_pz2 = WrappedNormal(
        #     torch.zeros(4,self.cfg.model.hyp_channels),
        #     torch.ones(4,self.cfg.model.hyp_channels).mul(self.cfg.model.prior_std),
        #     PoincareBall(dim= self.cfg.model.hyp_channels,c = 1/float(self.cfg.model.k_poin_out))
        # )
        # hyp_sample_z2 = hyp_pz2.rsample()


        # hyp_pz2 = WrappedNormal(
        #     torch.zeros_like(self.cfg.model.hyp_channels),
        #     torch.ones_like(self.cfg.model.hyp_channels).mul(self.cfg.model.prior_std),
        #     PoincareBall(dim= self.cfg.model.hyp_channels,c = 1/float(self.cfg.model.k_poin_out))
        # )
        
        # VQ-VAE 无条件生成的核心修改点
        if self.use_VQVAE:
            euc_feat, hyp_feat, z = self.model.encode(x, adj, E, node_mask)

            if self.euc_channels > 0:
                euc_quantize, euc_vq_ind, euc_vq_loss, euc_perplexity = self.model.codebook(euc_feat, codebook_type = "euc", node_mask=node_mask)
            else:
                euc_quantize = None
                euc_vq_loss = None
            if self.hyp_channels > 0:
                hyp_quantize, hyp_vq_ind, hyp_vq_loss, hyp_perplexity = self.model.codebook(hyp_feat, codebook_type = "hyp", node_mask=node_mask)
            else:
                hyp_quantize = None
                hyp_vq_loss = None
            
            if self.euc_channels > 0 and self.hyp_channels > 0:
                graph_feat = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            elif self.euc_channels > 0:
                graph_feat = euc_quantize
            elif self.hyp_channels > 0:
                graph_feat = hyp_quantize
            else:
                graph_feat = None
                
            # graph_feat = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
            
            data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
            pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}

            euc_distribution = None # {'pz': euc_pz, 'qz_x': euc_qz_x}
            hyp_distribution = None # {'pz': hyp_pz, 'qz_x': hyp_qz_x}
            euc_feature = {'euc_feat':euc_feat, 'euc_vq_loss':euc_vq_loss} # {'mean':euc_feat_mean, 'var':euc_feat_logvar, 'sample':euc_sample_z}
            hyp_feature = {'hyp_feat':hyp_feat, 'hyp_vq_loss':hyp_vq_loss} # {'mean':hyp_feat_mean, 'var':hyp_feat_logvar, 'sample':hyp_sample_z}
 
            loss, to_log = self.val_loss.forward(data, pred, euc_distribution, hyp_distribution, euc_feature, hyp_feature, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=False, reconstruct_metrics=self.reconstruct_metrics)
            self.log_dict(to_log)
            
            # Initialize reconstruction with zero tensor (same shape as the expected output)
            reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
            reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)
            if euc2node_feat is not None:
                reconstruct_x += euc2node_feat
            if hyp2node_feat is not None:
                reconstruct_x += hyp2node_feat

            if euc2edge_feat is not None:
                reconstruct_E += euc2edge_feat
            if hyp2edge_feat is not None:
                reconstruct_E += hyp2edge_feat
                
            self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                            log=batch_idx % self.log_every_steps == 0,)
        
        else:
            if not self.model.use_VAE:
                node_feat, edge_feat, z = self.model.encode(x, adj, E, node_mask)
                if self.euc_channels == 0:
                    graph_feat = edge_feat
                elif self.hyp_channels == 0:
                    graph_feat = node_feat
                else:
                    graph_feat = torch.cat((node_feat, edge_feat), dim= -1)
            
                # reconstruct_x, reconstruct_E, reconstruct_adj = self.model.decode(graph_feat, adj)
                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                
                data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
                pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
                
                euc_distribution = None
                hyp_distribution = None
                sample_z_poincare = None
                loss, to_log = self.val_loss.forward(data, pred, euc_distribution, hyp_distribution, node_feat, edge_feat, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=False, reconstruct_metrics=self.reconstruct_metrics)
                self.log_dict(to_log)
                if self.euc_channels == 0:
                    reconstruct_x = hyp2node_feat
                    reconstruct_E = hyp2edge_feat
                elif self.hyp_channels == 0:
                    reconstruct_x = euc2node_feat
                    reconstruct_E = euc2edge_feat
                else:
                    reconstruct_x = euc2node_feat * self.cfg.loss.lambda_euc2node + hyp2node_feat * self.cfg.loss.lambda_hyp2node
                    reconstruct_E = euc2edge_feat * self.cfg.loss.lambda_euc2edge + hyp2edge_feat * self.cfg.loss.lambda_hyp2edge
                self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                                log=batch_idx % self.log_every_steps == 0,)
            
            
            else:
                
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.model.encode(x, adj, E, node_mask)
                
                # 检查编码器输出是否有NaN
                if self.euc_channels > 0:
                    if torch.isnan(euc_feat_mean).any() or torch.isnan(euc_feat_logvar).any():
                        print("警告: 编码器输出中存在NaN值")
                        print(f"euc_feat_mean中NaN的数量: {torch.isnan(euc_feat_mean).sum().item()}")
                        print(f"euc_feat_logvar中NaN的数量: {torch.isnan(euc_feat_logvar).sum().item()}")
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                if self.hyp_channels > 0:
                    if torch.isnan(hyp_feat_mean).any() or torch.isnan(hyp_feat_logvar).any():
                        print(f"hyp_feat_mean中NaN的数量: {torch.isnan(hyp_feat_mean).sum().item()}")
                        print(f"hyp_feat_logvar中NaN的数量: {torch.isnan(hyp_feat_logvar).sum().item()}")
                        torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                        self.trainer.should_stop = True
                        return {'loss': torch.tensor(0.0, device=self.device)}
                
                # 让被mask的位置的值等于0
                if node_mask is not None:
                    mask_expanded = node_mask.unsqueeze(-1)
                    if self.euc_channels > 0:
                        euc_feat_mean = euc_feat_mean * mask_expanded
                        euc_feat_logvar = euc_feat_logvar * mask_expanded
                    if self.hyp_channels > 0:
                        hyp_feat_mean = hyp_feat_mean * mask_expanded
                        hyp_feat_logvar = hyp_feat_logvar * mask_expanded
                
                

                euc_pz = None
                euc_qz_x = None
                euc_sample_z = None
                if self.euc_channels > 0:
                    euc_pz = torch.distributions.Normal(
                        torch.zeros_like(euc_feat_mean),
                        torch.ones_like(euc_feat_logvar)
                    )
                
                # # 在创建正态分布前添加数值检查
                # if torch.isnan(euc_feat_mean).any() or torch.isnan(euc_feat_logvar).any():
                #     raise ValueError("Encoder outputs contain NaN values")
                
                # # 添加数值稳定处理（原有代码修改）
                # std = torch.exp(0.5 * euc_feat_logvar.clamp(min=-20, max=20))  # 限制logvar范围
                # std = std + 1e-8  # 防止零标准差

                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    euc_sample_z = euc_qz_x.rsample()


                hyp_pz = None
                hyp_qz_x = None
                hyp_sample_z = None
                if self.hyp_channels > 0:
                    if not self.cfg.model.use_poincare:
                        if hyp_feat_mean is not None:
                            hyp_pz = WrappedNormalLorentz(
                                torch.zeros_like(hyp_feat_mean[..., 1:]),
                                torch.ones_like(hyp_feat_logvar[..., 1:]).mul(self.cfg.model.prior_std),
                                self.model.manifold_out
                            )
                            hyp_qz_x = WrappedNormalLorentz(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]), self.model.manifold_out)
                            hyp_sample_z = hyp_qz_x.rsample()
                        else:
                            hyp_pz = None
                            hyp_qz_x = None
                            hyp_sample_z = None
                        # import pdb; pdb.set_trace()
                        
                        hyp_pz = torch.distributions.Normal(
                            torch.zeros_like(hyp_feat_mean[..., 1:]),
                            torch.ones_like(hyp_feat_logvar[..., 1:]).mul(self.cfg.model.prior_std),
                        )
                        hyp_qz_x = torch.distributions.Normal(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]))
                    
                    else:
                        if hyp_feat_mean is not None:
                            # hyp_pz = WrappedNormalPoincare(
                            #     torch.zeros_like(hyp_feat_mean),
                            #     torch.ones_like(hyp_feat_logvar).mul(self.cfg.model.prior_std),
                            #     self.model.manifold
                            # )
                            hyp_pz = WrappedNormalPoincare(
                                torch.zeros_like(hyp_feat_mean),
                                torch.ones_like(hyp_feat_logvar).mul(self.cfg.model.prior_std).mul(1/torch.tensor(self.cfg.model.hyp_channels, dtype=torch.float32, device=self.device)),
                                self.model.manifold
                            )
                            hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                            hyp_sample_z = hyp_qz_x.rsample()
                        else:
                            hyp_pz = None
                            hyp_qz_x = None
                            hyp_sample_z = None
                    

                # hyp_qz_x = torch.distributions.Normal(hyp_feat_mean[..., 1:], torch.exp(0.5 * hyp_feat_logvar[..., 1:]))
                if euc_sample_z is not None and hyp_sample_z is not None:
                    graph_feat = torch.cat((euc_sample_z, hyp_sample_z), dim=-1)
                elif euc_sample_z is not None:
                    graph_feat = euc_sample_z
                elif hyp_sample_z is not None:
                    graph_feat = hyp_sample_z
                else:
                    graph_feat = None  # 或者你可以根据需要设定一个默认值

                euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, reconstruct_adj = self.model.decode(graph_feat, adj, node_mask)
                
                # # 检查解码器输出是否有NaN
                # if torch.isnan(euc2node_feat).any() or torch.isnan(euc2edge_feat).any():
                #     print("警告: 解码器输出中存在NaN值")
                #     print(f"euc2node_feat中NaN的数量: {torch.isnan(euc2node_feat).sum().item()}")
                #     print(f"euc2edge_feat中NaN的数量: {torch.isnan(euc2edge_feat).sum().item()}")
                #     if hyp2node_feat is not None:
                #         print(f"hyp2node_feat中NaN的数量: {torch.isnan(hyp2node_feat).sum().item()}")
                #         print(f"hyp2edge_feat中NaN的数量: {torch.isnan(hyp2edge_feat).sum().item()}")
                #     torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                #     self.trainer.should_stop = True
                #     return {'loss': torch.tensor(0.0, device=self.device)}
                
                data = {'node':node_labels, 'edge':edge_labels, 'adj':adj}
                pred = {'euc2node':euc2node_feat, 'hyp2node':hyp2node_feat, 'euc2edge':euc2edge_feat, 'hyp2edge':hyp2edge_feat , 'adj':reconstruct_adj}
                
                euc_distribution = {'pz': euc_pz, 'qz_x': euc_qz_x}
                hyp_distribution = {'pz': hyp_pz, 'qz_x': hyp_qz_x}
                
                euc_feature = {'mean':euc_feat_mean, 'var':euc_feat_logvar, 'sample':euc_sample_z}
                hyp_feature = {'mean':hyp_feat_mean, 'var':hyp_feat_logvar, 'sample':hyp_sample_z}
                
                loss, to_log = self.val_loss.forward(data, pred, euc_distribution, hyp_distribution, euc_feature, hyp_feature, node_mask, dataset_weight=self.dataset_infos, log=True, use_kl_loss=True,  reconstruct_metrics=self.reconstruct_metrics)
                self.log_dict(to_log)
                # 检查损失是否有NaN
                if torch.isnan(loss):
                    print("警告: 损失计算中出现NaN值")
                    torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                    self.trainer.should_stop = True
                    return {'loss': torch.tensor(0.0, device=self.device)}
                
                # Initialize reconstruction with zero tensor (same shape as the expected output)
                reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
                reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)

                # Add the corresponding features, only if they are not None
                if euc2node_feat is not None:
                    reconstruct_x += euc2node_feat * self.cfg.loss.lambda_euc2node
                if hyp2node_feat is not None:
                    reconstruct_x += hyp2node_feat * self.cfg.loss.lambda_hyp2node

                if euc2edge_feat is not None:
                    reconstruct_E += euc2edge_feat * self.cfg.loss.lambda_euc2edge
                if hyp2edge_feat is not None:
                    reconstruct_E += hyp2edge_feat * self.cfg.loss.lambda_hyp2edge
                
                # 检查重建结果是否有NaN
                if torch.isnan(reconstruct_x).any() or torch.isnan(reconstruct_E).any():
                    print("警告: 重建结果中存在NaN值")
                    print(f"reconstruct_x中NaN的数量: {torch.isnan(reconstruct_x).sum().item()}")
                    print(f"reconstruct_E中NaN的数量: {torch.isnan(reconstruct_E).sum().item()}")
                    torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                    self.trainer.should_stop = True
                    return {'loss': torch.tensor(0.0, device=self.device)}
                # pdb.set_trace()
                self.train_metrics(masked_pred_X=reconstruct_x, masked_pred_E=reconstruct_E, true_X=node_labels, true_E=edge_labels,
                                log=batch_idx % self.log_every_steps == 0, )
                # pdb.set_trace()
        return {'loss':loss}
    

    
    def test_step(self, batch, batch_idx):
        # dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # x, E = dense_data.X, dense_data.E
        # adj, edge_labels = utils.process_edge_attr(E, node_mask)
        # node_labels = x
        # B = x.size(0)
        # N = x.size(1)
        
        # test_manifold_lorentz = Lorentz(k=float(self.cfg.model.k_out))
        # ## attention: the fllowing lines is for PoincareBall, so we need to change it to PoincareBall
        # test_manifold = PoincareBall(dim= self.cfg.model.hidden_channels, c = 1/float(self.cfg.model.k_out))  # k = 1/c
        
        # _pz_mu = nn.Parameter(torch.zeros(B, N, self.cfg.model.hidden_channels), requires_grad=False)
        # _pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        
        # pz = WrappedNormal(_pz_mu.mul(1), F.softplus(_pz_logvar).div(math.log(2)).mul(self.cfg.model.prior_std), test_manifold)
        # sample_z = pz.rsample().to(x.device)
        # # sample_z = test_manifold.random_normal((x.size(0), x.size(1), self.cfg.model.out_channels + 1)).to(x.device)
        # sample_lorentz = test_manifold_lorentz.poincare_to_lorentz(sample_z)
        # samples_x, sample_E, sample_adj = self.model.decode(sample_z, None)
        
        
        return {'loss':0}
    
    @torch.no_grad()        
    def generate_sample_from_z(self, z, node_mask):
        batch_size = z.shape[0]
        n_nodes = node_mask.sum(dim=1)
        if self.use_VQVAE:
            if self.euc_channels > 0:
                euc_feat = z[..., :self.euc_channels]
                euc_quantize, _, _, _ = self.model.codebook(euc_feat, codebook_type = "euc", node_mask =node_mask)
            else:
                euc_quantize = None
            if self.hyp_channels > 0:
                hyp_feat = z[..., self.euc_channels:]
                hyp_quantize, _, _, _ = self.model.codebook(hyp_feat, codebook_type = "hyp", node_mask = node_mask)
            else:
                hyp_quantize = None

            if self.euc_channels > 0 and self.hyp_channels > 0:
                z = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            elif self.euc_channels > 0:
                z = euc_quantize
            elif self.hyp_channels > 0:
                z = hyp_quantize
            else:
                z = None

        euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, sample_adj = self.model.decode(z, None, node_mask)
        # Initialize reconstruction with zero tensor (same shape as the expected output)
        reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
        reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)

        # Add the corresponding features, only if they are not None
        if euc2node_feat is not None:
            reconstruct_x += euc2node_feat
        if hyp2node_feat is not None:
            reconstruct_x += hyp2node_feat
        if euc2edge_feat is not None:
            reconstruct_E += euc2edge_feat
        if hyp2edge_feat is not None:
            reconstruct_E += hyp2edge_feat
        sample_y = torch.zeros([1, 0]).float()
        # samples_x shape is [batch_size, n_nodes_max, n_class_nodes]
        # samples_E shape is [batch_size, n_nodes_max, n_nodes_max, n_class_edges])
        # sample_adj shape is [batch_size, n_nodes_max, n_nodes_max]
        
        sample = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask(node_mask=node_mask, collapse=True)
        sample_argmax = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask_argmax(node_mask=node_mask, collapse=True)
        X, E, y = sample.X, sample.E, sample.y
        X_argmax, E_argmax, y_argmax = sample_argmax.X, sample_argmax.E, sample_argmax.y
        diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0) 
        E[diag_mask.expand(E.shape[0], -1, -1)] = 0  # 扩展到 batch 维度并设置对角线为 0
        E_argmax[diag_mask.expand(E_argmax.shape[0], -1, -1)] = 0  # 扩展到 batch 维度并设置对角线为 0
        # E.fill_diagonal_(0) ## remove self loops

        # print("Examples of generated graphs:")
        # for i in range(min(5, X.shape[0])):
        #     print("E", E[i])
        #     print("X: ", X[i])

        # # 取E的上三角矩阵，变成对称矩阵
        # E = torch.triu(E, diagonal=1)
        # # 对称变成无向图
        # E = E + E.transpose(1, 2)

        # E_argmax = torch.triu(E_argmax, diagonal=1)
        # E_argmax = E_argmax + E_argmax.transpose(1, 2)

        molecule_list = []
        for i in range(batch_size):
            n = int(n_nodes[i])
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        molecule_list_argmax = []
        for i in range(batch_size):
            n = int(n_nodes[i])
            atom_types = X_argmax[i, :n].cpu()
            edge_types = E_argmax[i, :n, :n].cpu()
            molecule_list_argmax.append([atom_types, edge_types])

        return molecule_list, molecule_list_argmax

    
    
    def generate_sample_from_prior(self, num_nodes, adj=None):
        return
        
        
    @torch.no_grad()
    def sample_batch(self, batch_size: int, num_nodes=None, batch_id=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.nodes_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_nodes_max = torch.max(n_nodes).item()

        # Build the masks
        arange = torch.arange(n_nodes_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask = node_mask.float()
        
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        if self.use_VQVAE:
             # 从codebook中随机采样离散向量
            B, N = node_mask.shape[0], node_mask.shape[1]

            if self.euc_channels > 0:
                euc_indices = torch.randint(0, self.cfg.model.codebook_size, (B, N), device=self.device)
                euc_quantize = F.embedding(euc_indices, self.model.euc_codebook.codebook)
            if self.hyp_channels > 0:
                hyp_indices = torch.randint(0, self.cfg.model.codebook_size, (B, N), device=self.device)
                hyp_quantize = F.embedding(hyp_indices, self.model.hyp_codebook.codebook)         
             
            if self.euc_channels > 0 and self.hyp_channels > 0:
                z_0 = torch.cat((euc_quantize, hyp_quantize), dim=-1)
            elif self.euc_channels > 0:
                z_0 = euc_quantize
            elif self.hyp_channels > 0:
                z_0 = hyp_quantize
            else:
                z_0 = None
        

        else:
             z_0 = self.product_manifold.random((batch_size, n_nodes_max, self.model.euc_channels + self.model.hyp_channels), device=self.device, std=self.prior_std)
        # _pz_mu = nn.Parameter(torch.zeros(batch_size, n_nodes_max, self.dim - 1), requires_grad=False).to(self.device)
        # _pz_std = nn.Parameter(torch.zeros(1, 1), requires_grad=False).to(self.device)
        # pz_poincare = WrappedNormal(_pz_mu.mul(1), F.softplus(_pz_std).div(math.log(2)).mul(self.cfg.model.prior_std), self.model.manifold_out_poincare)

 
        # z_0 = self.from_p2l_correct(pz_poincare.rsample()) 
        
        # decode z_0
        euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, sample_adj = self.model.decode(z_0, None, node_mask)
        # Initialize reconstruction with zero tensor (same shape as the expected output)
        reconstruct_x = torch.zeros_like(euc2node_feat) if euc2node_feat is not None else torch.zeros_like(hyp2node_feat)
        reconstruct_E = torch.zeros_like(euc2edge_feat) if euc2edge_feat is not None else torch.zeros_like(hyp2edge_feat)

        # Add the corresponding features, only if they are not None
        if euc2node_feat is not None:
            reconstruct_x += euc2node_feat
        if hyp2node_feat is not None:
            reconstruct_x += hyp2node_feat

        if euc2edge_feat is not None:
            reconstruct_E += euc2edge_feat
        if hyp2edge_feat is not None:
            reconstruct_E += hyp2edge_feat
        sample_y = torch.zeros([1, 0]).float()
        # samples_x shape is [batch_size, n_nodes_max, n_class_nodes]
        # samples_E shape is [batch_size, n_nodes_max, n_nodes_max, n_class_edges])
        # sample_adj shape is [batch_size, n_nodes_max, n_nodes_max]
        
        # we are not using sample_adj right now
        sample = PlaceHolder(X=reconstruct_x, E=reconstruct_E, y=sample_y).mask(node_mask=node_mask, collapse=True)
        X, E, y = sample.X, sample.E, sample.y
        # TODO:
        # assert (E == torch.transpose(E, 1, 2)).all()
        diag_mask = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0) 
        E[diag_mask.expand(E.shape[0], -1, -1)] = 0  # 扩展到 batch 维度并设置对角线为 0
        
        # print("Examples of generated graphs:")
        # for i in range(min(5, X.shape[0])):
        #     print("E", E[i])
        #     print("X: ", X[i])

        # Prepare the chain for saving
        # Split the generated molecules
        keep_chain = 10
        chain_X_size = torch.Size((2, keep_chain, X.size(1)))
        chain_E_size = torch.Size((2, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = chain_X[torch.arange(chain_X.size(0) - 1, -1, -1)]
            chain_E = chain_E[torch.arange(chain_E.size(0) - 1, -1, -1)]

            # Repeat last frame to see final sample better
            # chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            # chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            # assert chain_X.size(0) == (number_chain_steps + 10)
            node_mask_keep_chain = node_mask[:keep_chain].to(torch.int32)
            e_mask1_keep_chain = node_mask_keep_chain.unsqueeze(2)             # bs, n, 1, 1
            e_mask2_keep_chain = node_mask_keep_chain.unsqueeze(1)             # bs, 1, n, 1

            chain_X[0] = torch.zeros_like(final_X_chain)
            chain_X[0][node_mask_keep_chain == 0] = -1
            chain_E[0] = torch.randint(0, 1, (final_E_chain.shape[0], final_E_chain.shape[1], final_E_chain.shape[2]), device=self.device)
            chain_E[0][(e_mask1_keep_chain * e_mask2_keep_chain).squeeze(-1) == 0] = -1

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='')
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, batch_size)
            self.print("Done.")



        return molecule_list

     
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Starting training with manifold:", self.manifold_name)
        # if self.local_rank == 0:
        #     utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        

    def on_train_epoch_end(self) -> None:
        sched_euc, sched_hyp = self.lr_schedulers()
        sched_euc.step()
        sched_hyp.step()

        to_log = self.train_loss.log_epoch_metrics()
        to_log["epoch"] = self.current_epoch
        self.log_dict(to_log, on_step=False, on_epoch=True)
        # 检查当前epoch的指标中是否有NaN
        for key, value in to_log.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                print(f"警告: 在epoch {self.current_epoch}的指标{key}中发现NaN值")
                # 保存当前模型状态
                torch.save(self.state_dict(), f'nan_detected_epoch_{self.current_epoch}.pt')
                # 停止训练
                self.trainer.should_stop = True
                return
        self.print(f"Epoch {self.current_epoch}: epoch_batch_loss: {to_log['train_epoch/batch_loss'] :.3f}"
                      f" -- epoch_euc2node_CE: {to_log['train_epoch/epoch_euc2node_CE'] :.3f}"
                      f" -- epoch_euc2edge_CE: {to_log['train_epoch/epoch_euc2edge_CE'] :.3f}"
                      f" -- epoch_hyp2node_CE: {to_log['train_epoch/epoch_hyp2node_CE'] :.3f}"
                      f" -- epoch_hyp2edge_CE: {to_log['train_epoch/epoch_hyp2edge_CE'] :.3f}"
                      f" -- epoch/epoch_consistency_node_loss: {to_log['train_epoch/epoch_consistency_node_loss'] :.3f}"
                      f" -- epoch/epoch_consistency_edge_loss: {to_log['train_epoch/epoch_consistency_edge_loss'] :.3f}"
                      f" -- epoch_kl_loss: {to_log['train_epoch/epoch_kl_loss'] :.3f}"
                      f" -- epoch_l2_loss: {to_log['train_epoch/epoch_l2_loss'] :.3f}"
                      f" -- epoch_node_acc: {to_log['train_epoch/epoch_node_acc']:.3f}"
                      f" -- epoch_node_f1: {to_log['train_epoch/epoch_node_f1']:.3f}"
                      f" -- epoch_edge_acc: {to_log['train_epoch/epoch_edge_acc']:.3f}"
                      f" -- epoch_edge_f1: {to_log['train_epoch/epoch_edge_f1']:.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        # if torch.cuda.is_available():
        #     print(torch.cuda.memory_summary())
    

    def on_validation_epoch_start(self):
        self.val_loss.reset()
        self.train_metrics.reset()
        # log epoch
        self.log("epoch", self.current_epoch, on_step=False, on_epoch=True)
        if self.val_counter == 0:
            print("Running reconstruction evaluation on validation set...")
            val_loader = self.trainer.datamodule.val_dataloader()
            recon_samples = []

            for batch_id, batch in enumerate(val_loader):
                recon_X, recon_E, n_nodes = self.reconstruct_step(batch)

                # === 3. 将 [X_hat, E_hat] 转为可评估结构 ===
                for i in range(recon_X.shape[0]):
                    n = n_nodes[i].detach().cpu()
                    atom_types = recon_X[i, :n].detach().cpu()
                    edge_types = recon_E[i, :n, :n].detach().cpu()
                    recon_samples.append([atom_types, edge_types])
           
            # Only once after full pass
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f'graphs/{self.name}/reconstruct_epoch_{self.current_epoch}/'
            )
            self.visualization_tools.visualize(result_path, recon_samples, min(len(recon_samples), 512), "reconstruct_graph")
            self.print("Visualization complete.")


            # === 4. 送入同一个评估模块 ===
            self.reconstruct_metrics = self.sampling_metrics(
                recon_samples,
                self.name + "_recon",
                self.current_epoch,
                val_counter=-1,
                test=False,
                local_rank=self.local_rank,
            )
            self.sampling_metrics.reset()
        
    def on_test_epoch_start(self):
        self.test_loss.reset()
        self.sampling_metrics.reset()
        # self.test_metrics.reset()
    
    def on_validation_epoch_end(self):
        to_log = self.val_loss.log_epoch_metrics()
        to_log["epoch"] = self.current_epoch
        self.log("val_epoch/batch_loss", to_log['val_epoch/batch_loss'], on_step=False, on_epoch=True)
        self.log("val_epoch/epoch_edge_acc", to_log['val_epoch/epoch_edge_acc'], on_step=False, on_epoch=True)
        self.log("val_epoch/log_metric", to_log['val_epoch/log_metric'], on_step=False, on_epoch=True)
        self.log_dict(to_log, on_step=False, on_epoch=True)

        self.print(f"--------------------------------------------------------------------------------------------------\n"
                    f"Validation - Epoch {self.current_epoch}: epoch_batch_loss: {to_log['val_epoch/batch_loss'] :.3f}"
                        f" -- epoch_euc2node_CE: {to_log['val_epoch/epoch_euc2node_CE'] :.3f}"
                        f" -- epoch_euc2edge_CE: {to_log['val_epoch/epoch_euc2edge_CE'] :.3f}"
                        f" -- epoch_hyp2node_CE: {to_log['val_epoch/epoch_hyp2node_CE'] :.3f}"
                        f" -- epoch_hyp2edge_CE: {to_log['val_epoch/epoch_hyp2edge_CE'] :.3f}"
                        f" -- epoch/epoch_consistency_node_loss: {to_log['val_epoch/epoch_consistency_node_loss'] :.3f}"
                      f" -- epoch/epoch_consistency_edge_loss: {to_log['val_epoch/epoch_consistency_edge_loss'] :.3f}"
                      f" -- epoch_kl_loss: {to_log['val_epoch/epoch_kl_loss'] :.3f}"
                      f" -- epoch_l2_loss: {to_log['val_epoch/epoch_l2_loss'] :.3f}"
                      f" -- epoch_node_acc: {to_log['val_epoch/epoch_node_acc']:.3f}"
                      f" -- epoch_node_f1: {to_log['val_epoch/epoch_node_f1']:.3f}"
                      f" -- epoch_edge_acc: {to_log['val_epoch/epoch_edge_acc']:.3f}"
                      f" -- epoch_edge_f1: {to_log['val_epoch/epoch_edge_f1']:.3f}"
                    f"--------------------------------------------------------------------------------------------------\n")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Validation - Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        
        
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            # 🆕 新增：重建验证集并比较分布
            print("Running reconstruction evaluation on validation set...")
            val_loader = self.trainer.datamodule.val_dataloader()
            recon_samples = []

            for batch_id, batch in enumerate(val_loader):
                recon_X, recon_E, n_nodes = self.reconstruct_step(batch)

                # === 3. 将 [X_hat, E_hat] 转为可评估结构 ===
                for i in range(recon_X.shape[0]):
                    n = n_nodes[i].detach().cpu()
                    atom_types = recon_X[i, :n].detach().cpu()
                    edge_types = recon_E[i, :n, :n].detach().cpu()
                    recon_samples.append([atom_types, edge_types])

                            # Only once after full pass
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f'graphs/{self.name}/reconstruct_epoch_{self.current_epoch}/'
            )
            self.visualization_tools.visualize(result_path, recon_samples, min(len(recon_samples), 512), "reconstruct_graph")
            self.print("Visualization complete.")

            # === 4. 送入同一个评估模块 ===
            self.reconstruct_metrics = self.sampling_metrics(
                recon_samples,
                self.name + "_recon",
                self.current_epoch,
                val_counter=-1,
                test=False,
                local_rank=self.local_rank,
            )
            print("Reconstruction eval done.\n")
            self.sampling_metrics.reset()

            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples = []
            id = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                samples.extend(self.sample_batch(batch_size=to_generate, num_nodes=None, batch_id=id))
                samples_left_to_generate -= to_generate
                id += to_generate
            
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank)
            print(f'Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

        self.print("Generating molecule visualizations...")
        molecule_list = []
        dataloader = self.trainer.datamodule.train_dataloader()
        
        if self.val_counter==1:
            batch = next(iter(dataloader))
            dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            X, E = dense_data.X, dense_data.E
            sample = PlaceHolder(X=X, E=E, y=None).mask_argmax(node_mask=node_mask, collapse=True)
            X, E = sample.X, sample.E
            # pdb.set_trace()
            n_nodes = node_mask.sum(-1)
            batch_size = X.shape[0]

            for i in range(batch_size):
                n = n_nodes[i]
                atom_types = X[i, :n].cpu()
                edge_types = E[i, :n, :n].cpu()
                molecule_list.append([atom_types, edge_types])
            
            # Only once after full pass
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f'graphs/{self.name}/train_data_first_batch/'
            )
            self.visualization_tools.visualize(result_path, molecule_list, batch_size)
            self.print("Visualization complete.")

    def on_test_epoch_end(self):
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples = []
        id = 0
        while samples_left_to_generate > 0:
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            samples.extend(self.sample_batch(batch_size=to_generate, num_nodes=None, batch_id=id))
            samples_left_to_generate -= to_generate
            id += to_generate

        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.sampling_metrics.reset()
        
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f'Gradient {name} {param.grad.abs().mean().item()}')
    #         else:
    #             print(f'Gradient {name} None')
    
    
    def configure_optimizers(self):
        # pdb.set_trace()
        if self.use_riemannian_optimizer:
            euc_params = [p for n, p in self.model.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]  # Euclidean parameters
            optimizer_euc = torch.optim.AdamW(euc_params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay, foreach=((self.euc_channels + self.hyp_channels)!=1))
            
            hyp_params = [p for n, p in self.model.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]  # Hyperbolic parameters
            # hyp_name = [n for n, p in self.model.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]  # Hyperbolic parameters
            optimizer_hyp = RiemannianAdam(hyp_params, lr=0.005, stabilize=10, weight_decay=self.cfg.train.weight_decay)
            
            schedulers = []
            optimizers = [{"optimizer": optimizer_euc}, {"optimizer": optimizer_hyp}]
            print(f"Using {len(euc_params)} Euclidean params and {len(hyp_params)} Hyperbolic params")
            if self.cfg.train.get("lr_scheduler", None) is not None:
                sched_euc = hydra.utils.instantiate(
                    self.cfg.train.lr_scheduler,
                    optimizer_euc,
                )
                sched_hyp = hydra.utils.instantiate(
                    self.cfg.train.lr_scheduler,
                    optimizer_hyp,
                )

                # 添加 scheduler（PyTorch Lightning 格式）
                schedulers = [
                    {
                        "optimizer": optimizer_euc,
                        "lr_scheduler": {
                            "scheduler": sched_euc,
                            "interval": self.cfg.train.interval,
                        }
                    },
                    {
                        "optimizer": optimizer_hyp,
                        "lr_scheduler": {
                            "scheduler": sched_hyp,
                            "interval": self.cfg.train.interval,
                        }
                    }
                ]
                return schedulers
            # optimizer = [optimizer_euc, optimizer_hyp]

            return optimizers
        
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay, foreach=((self.euc_channels + self.hyp_channels)!=1))
            scheduler = hydra.utils.instantiate(
                self.cfg.train.lr_scheduler,  # config 里写的是 lr_scheduler
                optimizer=optimizer
            )
            # return optimizer
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',      # 可选: 'step' 或 'epoch'
                    'frequency': 1,           # 每多少次调用一次 scheduler.step()
                },
                'gradient_clip_val': 1.0,  # 添加梯度裁剪
                'gradient_clip_algorithm': 'norm'
            }
        # return {
        #     'optimizer': self.optimizer,
        #     'gradient_clip_val': 1.0,  # 添加梯度裁剪
        #     'gradient_clip_algorithm': 'norm'
        # }
    
    
    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)


        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
    
    
        
        
        
        