import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, BCELossMetric, CrossEntropyMetric, \
    ProbabilityMetric, NLL, MSELossMetric, AccuracyMetric, F1ScoreMetric
from src.utils import has_analytic_kl
from src.distribution.wrapped_normal import WrappedNormalPoincare
from src.manifolds.lorentz import Lorentz
import pdb
from sklearn.metrics import f1_score
import numpy as np
import torch.nn.functional as F

class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log


class HGCNTrainLoss(nn.Module):
    def __init__(self, lambda_node, lambda_edge):
        super(HGCNTrainLoss, self).__init__()
        self.edge_existent_loss = BCELossMetric()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge

    def forward(self, data, pred, pos_scores, neg_scores, log: bool):
        pos_scores = torch.clamp(pos_scores, min=1e-7, max=1-1e-7)
        neg_scores = torch.clamp(neg_scores, min=1e-7, max=1-1e-7)
        
        edge_existent_loss = self.edge_existent_loss(pos_scores, torch.ones_like(pos_scores))
        edge_existent_loss += self.edge_existent_loss(neg_scores, torch.zeros_like(neg_scores))
        
        # 去除掩码
        # # 按最后一维检查是否有至少一个元素为 1
        # row_mask = torch.any(data['node'] == 1, dim=-1, keepdim=True)  # 保留最后一维，形状为 (batch_size, num_rows, 1)

        # # 广播 row_mask 到与 data['node'] 相同的形状
        # True_pos = row_mask.expand_as(data['node'])  # 形状变为 (batch_size, num_rows, num_features)

        # data['node'] = data['node'][True_pos]
        # pred['node'] = pred['node'][True_pos]
        
        # ## construct the true adjcency for the edge based on the true node values 
        # row_mask = torch.any(data['edge'] == 1, dim=-1)  # (B, N, 1)
        # row_mask = torch.any(row_mask, dim=-1, keepdim=True).unsqueeze(-1)  # (B, N, 1, 1)
  
        # True_mat = row_mask.expand_as(data['edge'])  # (B, N, N, D)
        
        # adj_masked = data['edge'][True_mat]  # 仅保留与有效节点相关的边
        # pred_adj_masked = pred['edge'][True_mat]
        
        node_dim = pred['node'].size(-1)
        edge_dim = pred['edge'].size(-1)
        
        node_loss = self.node_loss(pred['node'].reshape(-1, node_dim), data['node'].reshape(-1, node_dim))
        edge_loss = self.edge_loss(pred['edge'].reshape(-1, edge_dim), data['edge'].reshape(-1, edge_dim))

        loss = edge_existent_loss + self.lambda_node * node_loss + self.lambda_edge * edge_loss
        # node_loss = 0
        # edge_loss = 0
        # loss = edge_existent_loss + self.lambda_node * node_loss + self.lambda_edge * edge_loss
        
        if log:
            to_log = {'train_loss/batch_CE': loss.detach(),
                      'train_loss/edge_existent_CE': self.edge_existent_loss.compute(),
                      'train_loss/node_CE': self.node_loss.compute(),
                      'train_loss/edge_CE': self.edge_loss.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss
    
    
    def reset(self):
        for metric in (self.edge_existent_loss, self.node_loss, self.edge_loss):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_edge_existent_loss = self.edge_existent_loss.compute()
        epoch_node_loss = self.node_loss.compute()
        epoch_edge_loss = self.edge_loss.compute()

        to_log = {"train_epoch/epoch_edge_existent_bce": epoch_edge_existent_loss,
                  "train_epoch/epoch_node_ce": epoch_node_loss,
                  "train_epoch/epoch_edge_ce": epoch_edge_loss}
        if wandb.run:
            wandb.log(to_log)
        return to_log

def band_norm_loss(z, r_min=0.2, r_max=0.8, lam=1.0, eps=1e-9, mask=None):
    norm = torch.norm(z, dim=-1)  # batch 维
    loss_hi = torch.clamp(norm - r_max, min=0.0) ** 2
    loss_lo = torch.clamp(r_min - norm, min=0.0) ** 2
    # 取均值或求和均可
    return lam * ((loss_hi + loss_lo) * mask).sum() / mask.sum()

class HGVAETrainLoss(nn.Module):
    def __init__(self, cfg, euc_channels: int, hyp_channels: int, stage: str = "train"):
        super(HGVAETrainLoss, self).__init__()
        self.euc_channels = euc_channels
        self.hyp_channels = hyp_channels

        # self.edge_existent_loss = BCELossMetric()
        self.euc2node_loss = CrossEntropyMetric()
        self.hyp2node_loss = CrossEntropyMetric()
        self.euc2edge_loss = CrossEntropyMetric()
        self.hyp2edge_loss = CrossEntropyMetric()
        self.consistency_node_loss = CrossEntropyMetric()
        self.consistency_edge_loss = CrossEntropyMetric()
        self.node_acc = AccuracyMetric()
        self.node_f1 = F1ScoreMetric()
        self.edge_acc = AccuracyMetric()
        self.edge_f1 = F1ScoreMetric()
        # self.kl_loss = NLL()
        self.kl_loss = 0.0
        self.l2_loss = 0.0
        self.band_norm_loss = 0.0
        
        self.stage = stage
        self.cfg = cfg
        self.lambda_euc2node = cfg.loss.lambda_euc2node
        self.lambda_hyp2node = cfg.loss.lambda_hyp2node
        self.lambda_euc2edge = cfg.loss.lambda_euc2edge
        self.lambda_hyp2edge = cfg.loss.lambda_hyp2edge
        self.lambda_consistency_node = cfg.loss.lambda_consistency_node
        self.lambda_consistency_edge = cfg.loss.lambda_consistency_edge
        self.lambda_kl = cfg.loss.lambda_kl
        self.lambda_l2 = cfg.loss.lambda_l2
        self.lambda_band_norm = cfg.loss.lambda_band_norm

        self.stage = stage
        self.zs_euc_mean = None
        self.zs_euc_var = None
        self.zs_hyp_mean = None
        self.zs_hyp_var = None
        self.reconstruct_metrics = None
        
    def compute_and_log_statistics(self, zs_euc, zs_hyp):
        if self.euc_channels > 0:
            self.zs_euc_mean = torch.mean(zs_euc)
            self.zs_euc_var = torch.var(zs_euc)
        if self.hyp_channels > 0:
            self.zs_hyp_mean = torch.mean(zs_hyp)
            self.zs_hyp_var = torch.var(zs_hyp)

                
    def forward(self, data, pred, euc_dist, hyp_dist, zs_euc, zs_hyp, mask, dataset_weight, log: bool, use_kl_loss: bool = True, reconstruct_metrics=None):
        # edge_existent_loss = self.edge_existent_loss(pred['adj'], data['adj'])
        # 过滤掉节点个数为 0 的类别
        # pdb.set_trace()
        self.reconstruct_metrics = reconstruct_metrics
        non_zero_mask = dataset_weight.node_types > 0
        filtered_node_types = dataset_weight.node_types[non_zero_mask]
        # 计算权重
        weights = 1.0 / (filtered_node_types)
        weights = weights / weights.sum() * non_zero_mask.sum()  # 归一化
        # 恢复完整的权重向量
        full_weights = torch.zeros_like(dataset_weight.node_types, dtype=torch.float32)
        full_weights[non_zero_mask] = weights
        
        non_zero_mask_edge = dataset_weight.edge_types > 0
        filtered_edge_types = dataset_weight.edge_types[non_zero_mask_edge]
        # 计算权重
        weights_edge = 1.0 / (filtered_edge_types)
        weights_edge = weights_edge / weights_edge.sum() * non_zero_mask_edge.sum()  # 归一化
        # 恢复完整的权重向量
        full_weights_edge = torch.zeros_like(dataset_weight.edge_types, dtype=torch.float32)
        full_weights_edge[non_zero_mask_edge] = weights_edge 
        
        edge_mask = torch.einsum('bn, bm -> bnm', mask, mask)
        diag_indices = torch.arange(edge_mask.shape[-1], device=data['adj'].device)
        edge_mask[:, diag_indices, diag_indices] = 0
        # import pdb; pdb.set_trace()
        
        euc2node_loss = 0.0
        hyp2node_loss = 0.0
        euc2edge_loss = 0.0
        hyp2edge_loss = 0.0
        consistency_node_loss = 0.0
        consistency_edge_loss = 0.0
        # import pdb; pdb.set_trace()
        if self.euc_channels > 0:
            euc2node_loss = self.euc2node_loss(pred['euc2node'], data['node'], mask, weight=full_weights.to(torch.float32))
            euc2edge_loss = self.euc2edge_loss(pred['euc2edge'], data['edge'], edge_mask, weight=full_weights_edge.to(torch.float32))

        if self.hyp_channels > 0:
            hyp2node_loss = self.hyp2node_loss(pred['hyp2node'], data['node'], mask, weight=full_weights)
            hyp2edge_loss = self.hyp2edge_loss(pred['hyp2edge'], data['edge'], edge_mask, weight=full_weights_edge)

        if self.euc_channels > 0 and self.hyp_channels > 0:
            consistency_node_loss = self.consistency_node_loss(pred['euc2node'], pred['hyp2node'], mask)
            consistency_edge_loss = self.consistency_edge_loss(pred['euc2edge'], pred['hyp2edge'], edge_mask)
        
        # node_loss = self.node_loss(pred['node'], data['node'], mask)
        # edge_loss = self.edge_loss(pred['edge'], data['edge'], edge_mask)
        # node_loss = self.node_loss(pred['node'], data['node'], mask, weight=full_weights)
        # edge_loss = self.edge_loss(pred['edge'], data['edge'], edge_mask, weight=full_weights_edge)

        # pdb.set_trace()
        use_vq = self.cfg.loss.lambda_commitment_weight != 0
        if use_vq:
            # if self.stage == 'train':
            if self.euc_channels > 0 and self.hyp_channels > 0:
                self.l2_loss = (torch.norm(zs_euc['euc_feat'], p=2, dim=-1) * mask).sum() / mask.sum() + (torch.norm(zs_hyp['hyp_feat'], p=2, dim=-1) * mask).sum() / mask.sum()
            elif self.euc_channels > 0:
                self.l2_loss = (torch.norm(zs_euc['euc_feat'], p=2, dim=-1) * mask).sum() / mask.sum()
            elif self.hyp_channels > 0:
                self.l2_loss = (torch.norm(zs_hyp['hyp_feat'], p=2, dim=-1) * mask).sum() / mask.sum()
                self.band_norm_loss = band_norm_loss(zs_hyp['hyp_feat'], r_min=0.2, r_max=0.8, lam=1.0, mask=mask)

            loss = self.lambda_euc2node * euc2node_loss + self.lambda_hyp2node * hyp2node_loss \
                + self.lambda_euc2edge * euc2edge_loss + self.lambda_hyp2edge * hyp2edge_loss \
                + self.lambda_l2 * self.l2_loss + self.lambda_consistency_node * consistency_node_loss \
                + self.lambda_consistency_edge * consistency_edge_loss + self.lambda_band_norm * self.band_norm_loss
                
            if self.euc_channels > 0 and self.hyp_channels > 0:
                loss += zs_euc['euc_vq_loss']['loss'].mean() + zs_hyp['hyp_vq_loss']['loss'].mean()
            elif self.euc_channels > 0:
                loss += zs_euc['euc_vq_loss']['loss'].mean()
            elif self.hyp_channels > 0:
                loss += zs_hyp['hyp_vq_loss']['loss'].mean()
            # else:     
            #     loss = self.lambda_euc2node * euc2node_loss + self.lambda_hyp2node * hyp2node_loss \
            #         + self.lambda_euc2edge * euc2edge_loss + self.lambda_hyp2edge * hyp2edge_loss \
            #         + self.lambda_l2 * self.l2_loss + self.lambda_consistency_node * consistency_node_loss \
            #         + self.lambda_consistency_edge * consistency_edge_loss   
        else:
            if use_kl_loss:
                self.l2_loss = 0.0
                self.kl_loss = 0.0
                self.band_norm_loss = 0.0
                if self.euc_channels > 0:
                    euc_pz, euc_qz_x = euc_dist['pz'], euc_dist['qz_x']
                    euc_mean, euc_var = zs_euc['mean'], zs_euc['var']
                    euc_z = zs_euc['sample']
                    euc_kl_loss = torch.distributions.kl_divergence(euc_qz_x, euc_pz).sum(-1) if \
                        has_analytic_kl(type(euc_qz_x), type(euc_pz)) else \
                        euc_qz_x.log_prob(euc_z).sum(-1) - euc_pz.log_prob(euc_z).sum(-1)
                    self.l2_loss += (torch.norm(euc_mean, p=2, dim=-1)*mask).sum()/mask.sum() + (torch.norm(euc_var, p=2, dim=-1)*mask).sum()/mask.sum()
                    self.kl_loss += (euc_kl_loss*mask).sum()/mask.sum()
                if self.hyp_channels > 0:
                    hyp_pz, hyp_qz_x = hyp_dist['pz'], hyp_dist['qz_x']
                    hyp_mean, hyp_var = zs_hyp['mean'], zs_hyp['var']
                    hyp_z = zs_hyp['sample']
                    hyp_kl_loss = torch.distributions.kl_divergence(hyp_qz_x, hyp_pz).sum(-1) if \
                        has_analytic_kl(type(hyp_qz_x), type(hyp_pz)) else \
                        hyp_qz_x.log_prob(hyp_z).sum(-1) - hyp_pz.log_prob(hyp_z).sum(-1)
                    self.l2_loss += (torch.norm(hyp_mean, p=2, dim=-1)* mask).sum() / mask.sum() + (torch.norm(hyp_var, p=2, dim=-1)*mask).sum() / mask.sum()
                    self.kl_loss += (hyp_kl_loss*mask).sum()/mask.sum()
                    self.band_norm_loss += band_norm_loss(hyp_z, r_min=0.2, r_max=0.8, lam=1.0, mask=mask)
                # self.compute_and_log_statistics(euc_z, hyp_z)
                
                loss = self.lambda_euc2node * euc2node_loss + self.lambda_hyp2node * hyp2node_loss \
                    + self.lambda_euc2edge * euc2edge_loss + self.lambda_hyp2edge * hyp2edge_loss \
                    + self.lambda_l2 * self.l2_loss + self.lambda_consistency_node * consistency_node_loss \
                    + self.lambda_consistency_edge * consistency_edge_loss \
                    + self.lambda_kl * self.kl_loss + self.lambda_band_norm * self.band_norm_loss
                    
            else:
                if self.euc_channels > 0 and self.hyp_channels > 0:
                    self.l2_loss = (torch.norm(zs_euc, p=2, dim=-1) * mask).sum() / mask.sum() + (torch.norm(zs_hyp, p=2, dim=-1) * mask).sum() / mask.sum()
                elif self.euc_channels > 0:
                    self.l2_loss = (torch.norm(zs_euc, p=2, dim=-1) * mask).sum() / mask.sum()
                elif self.hyp_channels > 0:
                    self.l2_loss = (torch.norm(zs_hyp, p=2, dim=-1) * mask).sum() / mask.sum()
                    self.band_norm_loss = band_norm_loss(zs_hyp, r_min=0.2, r_max=0.8, lam=1.0, mask=mask)
                self.compute_and_log_statistics(zs_euc, zs_hyp)
                loss = self.lambda_euc2node * euc2node_loss + self.lambda_hyp2node * hyp2node_loss \
                    + self.lambda_euc2edge * euc2edge_loss + self.lambda_hyp2edge * hyp2edge_loss \
                    + self.lambda_l2 * self.l2_loss + self.lambda_consistency_node * consistency_node_loss \
                    + self.lambda_consistency_edge * consistency_edge_loss + self.lambda_band_norm * self.band_norm_loss
                
                self.kl_loss = 0.0
        
        with torch.no_grad():
            node_logits = 0.0
            if self.euc_channels == 0:
                node_logits += pred['hyp2node']
            elif self.hyp_channels == 0:
                node_logits += pred['euc2node']
            else:
                node_logits += pred['euc2node'] * self.lambda_euc2node + pred['hyp2node'] * self.lambda_hyp2node

            self.node_acc.update(node_logits, data['node'], mask)
            self.node_f1.update(node_logits, data['node'], mask)

            # 边预测融合
            edge_logits = 0.0
            if self.euc_channels == 0:
                edge_logits += pred['hyp2edge']
            elif self.hyp_channels == 0:
                edge_logits += pred['euc2edge']
            else:
                edge_logits += pred['euc2edge'] * self.lambda_euc2edge + pred['hyp2edge'] * self.lambda_hyp2edge

            self.edge_acc.update(edge_logits, data['edge'], edge_mask)
            self.edge_f1.update(edge_logits, data['edge'], edge_mask)

        # with torch.no_grad():
        #     self.node_acc.update(pred['euc2node']*self.lambda_euc2node+pred['hyp2node']*self.lambda_hyp2node, data['node'], mask)
        #     self.node_f1.update(pred['euc2node']*self.lambda_euc2node+pred['hyp2node']*self.lambda_hyp2node, data['node'], mask)
        #     self.edge_acc.update(pred['euc2edge']*self.lambda_euc2edge+pred['hyp2edge']*self.lambda_hyp2edge, data['edge'], edge_mask)
        #     self.edge_f1.update(pred['euc2edge']*self.lambda_euc2edge+pred['hyp2edge']*self.lambda_hyp2edge, data['edge'], edge_mask)
        
        euc2node_CE = 0.0
        euc2edge_CE = 0.0
        hyp2node_CE = 0.0
        hyp2edge_CE = 0.0
        consistency_node_loss = 0.0
        consistency_edge_loss = 0.0

        if self.euc_channels > 0:
            euc2node_CE = self.euc2node_loss.compute()
            euc2edge_CE = self.euc2edge_loss.compute()
        if self.hyp_channels > 0:
            hyp2node_CE = self.hyp2node_loss.compute()
            hyp2edge_CE = self.hyp2edge_loss.compute()
        if self.hyp_channels > 0 or self.euc_channels > 0:
            consistency_node_loss = self.consistency_node_loss.compute()
            consistency_edge_loss = self.consistency_edge_loss.compute()
        # pdb.set_trace()
        if log:
            to_log = {f'{self.stage}/batch_loss': loss.detach(),
                    f'{self.stage}/euc2node_CE': euc2node_CE,
                    f'{self.stage}/hyp2node_CE': euc2edge_CE,
                    f'{self.stage}/euc2edge_CE': hyp2node_CE,
                    f'{self.stage}/hyp2edge_CE': hyp2edge_CE,
                    f'{self.stage}/consistency_node_loss': consistency_node_loss,
                    f'{self.stage}/consistency_edge_loss': consistency_edge_loss,
                      f'{self.stage}/kl_loss': self.kl_loss,
                      f'{self.stage}/l2_loss': self.l2_loss,  # 记录 L2 loss
                      f'{self.stage}/band_norm_loss': self.band_norm_loss,
                      f'{self.stage}/node_acc': self.node_acc.compute(),
                      f'{self.stage}/node_f1': self.node_f1.compute(),
                      f'{self.stage}/edge_acc': self.edge_acc.compute(),
                      f'{self.stage}/edge_f1': self.edge_f1.compute(),
                      f'{self.stage}/zs_euc_mean': self.zs_euc_mean if self.zs_euc_mean is not None else 0.0,
                      f'{self.stage}/zs_euc_var': self.zs_euc_var if self.zs_euc_var is not None else 0.0,
                      f'{self.stage}/zs_hyp_mean': self.zs_hyp_mean if self.zs_hyp_mean is not None else 0.0,
                      f'{self.stage}/zs_hyp_var': self.zs_hyp_var if self.zs_hyp_var is not None else 0.0,
                    }

            if use_vq:
                to_log[f'{self.stage}/vq_loss_total'] = zs_hyp['hyp_vq_loss']['loss'].mean().item()
                to_log[f'{self.stage}/vq_loss'] = zs_hyp['hyp_vq_loss']['vq_loss'].mean().item() if isinstance(zs_hyp['hyp_vq_loss']['vq_loss'], torch.Tensor) else zs_hyp['hyp_vq_loss']['vq_loss']
                to_log[f'{self.stage}/commit_loss'] = zs_hyp['hyp_vq_loss']['commit_loss'].mean().item()
            # if wandb.run:
            #     wandb.log(to_log, commit=True)
    
        return loss, to_log
    
    def reset(self):
        for metric in (self.euc2node_loss, self.hyp2node_loss, self.euc2edge_loss, self.hyp2edge_loss, self.node_acc, self.node_f1, self.edge_acc, self.edge_f1, self.consistency_node_loss, self.consistency_edge_loss):
            metric.reset()
    
    def log_epoch_metrics(self):
        # epoch_edge_existent_loss = self.edge_existent_loss.compute()
        epoch_euc2node_loss = self.euc2node_loss.compute()
        epoch_hyp2node_loss = self.hyp2node_loss.compute()
        epoch_euc2edge_loss = self.euc2edge_loss.compute()
        epoch_hyp2edge_loss = self.hyp2edge_loss.compute()
        epoch_consistency_node_loss = self.consistency_node_loss.compute()
        epoch_consistency_edge_loss = self.consistency_edge_loss.compute()
        # write the same thing for node_acc
        epoch_node_acc = self.node_acc.compute()
        epoch_node_f1 = self.node_f1.compute()
        epoch_edge_acc = self.edge_acc.compute()
        epoch_edge_f1 = self.edge_f1.compute()
        
        loss = 0.0
        if self.euc_channels == 0:
            epoch_euc2node_loss = 0.0
            epoch_euc2edge_loss = 0.0
        if self.hyp_channels == 0:
            epoch_hyp2node_loss = 0.0
            epoch_hyp2edge_loss = 0.0
        if self.hyp_channels == 0 or self.euc_channels == 0:
            epoch_consistency_node_loss = 0.0
            epoch_consistency_edge_loss = 0.0
            loss += self.lambda_consistency_edge * epoch_consistency_edge_loss + self.lambda_consistency_node * epoch_consistency_node_loss
        loss += self.lambda_euc2node * epoch_euc2node_loss + self.lambda_euc2edge * epoch_euc2edge_loss + self.lambda_hyp2node * epoch_hyp2node_loss + self.lambda_hyp2edge * epoch_hyp2edge_loss + self.lambda_kl * self.kl_loss + self.lambda_l2 * self.l2_loss

        reconstruct_metric_sum = 0.0
        if self.reconstruct_metrics is not None:
            for key, value in self.reconstruct_metrics.items():
                reconstruct_metric_sum += value
        log_metric = epoch_edge_acc + epoch_node_acc + reconstruct_metric_sum - 0.1*loss.detach().item()

        to_log = {
            f"{self.stage}_epoch/batch_loss": loss,
            # f"{self.stage}_epoch/epoch_edge_existent_mse": epoch_edge_existent_loss,
            f"{self.stage}_epoch/epoch_euc2node_CE": epoch_euc2node_loss,
            f"{self.stage}_epoch/epoch_hyp2node_CE": epoch_hyp2node_loss,
            f"{self.stage}_epoch/epoch_euc2edge_CE": epoch_euc2edge_loss,
            f"{self.stage}_epoch/epoch_hyp2edge_CE": epoch_hyp2edge_loss,
            f"{self.stage}_epoch/epoch_consistency_node_loss": epoch_consistency_node_loss,
            f"{self.stage}_epoch/epoch_consistency_edge_loss": epoch_consistency_edge_loss,
            f"{self.stage}_epoch/epoch_kl_loss": self.kl_loss,
            f"{self.stage}_epoch/epoch_l2_loss": self.l2_loss,
            f"{self.stage}_epoch/epoch_band_norm_loss": self.band_norm_loss,
            f"{self.stage}_epoch/epoch_node_acc": epoch_node_acc,
            f"{self.stage}_epoch/epoch_node_f1": epoch_node_f1,
            f"{self.stage}_epoch/epoch_edge_acc": epoch_edge_acc,
            f"{self.stage}_epoch/epoch_edge_f1": epoch_edge_f1,
            f'{self.stage}_epoch/zs_euc_mean': self.zs_euc_mean if self.zs_euc_mean is not None else 0.0,
            f'{self.stage}_epoch/zs_euc_var': self.zs_euc_var if self.zs_euc_var is not None else 0.0,
            f'{self.stage}_epoch/zs_hyp_mean': self.zs_hyp_mean if self.zs_hyp_mean is not None else 0.0,
            f'{self.stage}_epoch/zs_hyp_var': self.zs_hyp_var if self.zs_hyp_var is not None else 0.0,
            f'{self.stage}_epoch/log_metric': log_metric,
        }
        
        print("Logging to WandB:", to_log)
        # if wandb.run:
        #     wandb.log(to_log, on_step=False, on_epoch=True,)
        return to_log
        
        


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



