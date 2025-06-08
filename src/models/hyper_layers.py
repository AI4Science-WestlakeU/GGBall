"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import pdb
from src.models.layers import DenseAtt
from geoopt import ManifoldParameter
from geoopt.optim.rsgd import RiemannianSGD
from geoopt.optim.radam import RiemannianAdam
from torch_scatter import scatter


from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.linear import PoincareLinear


def get_dim_act_curv(cfg):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not cfg.model.act:
        act = lambda x: x
    else:
        act = getattr(F, cfg.model.act)
    acts = [act] * (cfg.model.num_layers - 1)
    # pdb.set_trace()
    if cfg.model.manifold == 'Hyperboloid':
        dims = [cfg.model.feat_dim+1] + ([cfg.model.dim] * (cfg.model.num_layers - 1))
    else:
        dims = [cfg.model.feat_dim] + ([cfg.model.dim] * (cfg.model.num_layers - 1))

    dims += [cfg.model.dim]
    acts += [act]
    n_curvatures = cfg.model.num_layers

    if cfg.model.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([cfg.model.c]) for _ in range(n_curvatures)]
        if not cfg.general.gpus == -1:
            # curvatures = [curv.to(cfg.general.gpus) for curv in curvatures]
            curvatures = [curv.to('cuda:0') for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

# TODO: embedding of edge feature in hyperbolic space -> modify HygAgg (sparse matrix multiplication -> message + aggregation), Linear (edge feature update)
class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output

# class HyperbolicGraphConvolution(nn.Module):
#     """
#     Hyperbolic graph convolution layer with edge feature support.
#     """

#     def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
#         super(HyperbolicGraphConvolution, self).__init__()
#         self.manifold = manifold
#         self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
#         self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
#         self.hyp_act = HypAct(manifold, c_in, c_out, act)

#         # Linear transformation for edge features
#         self.edge_linear = nn.Linear(num_edge_class, out_features)  # Example: scalar edge features transformed to out_features
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input):
#         """
#         input: tuple (x, adj, edge_features)
#         x: Node features, shape (batch_size, num_nodes, in_features)
#         adj: Adjacency matrix, shape (batch_size, num_nodes, num_nodes)
#         edge_features: Edge features, shape (batch_size, num_nodes, num_nodes)
#         """
#         x, adj, edge_features = input

#         # Step 1: Node feature transformation using Hyperbolic Linear
#         h = self.linear(x)  # Shape: (batch_size, num_nodes, out_features)

#         # Step 2: Edge feature transformation
#         edge_features = edge_features.unsqueeze(-1)  # Shape: (batch_size, num_nodes, num_nodes, 1)
#         edge_features = self.edge_linear(edge_features)  # Transform edge features
#         edge_features = self.dropout(edge_features)  # Shape: (batch_size, num_nodes, num_nodes, out_features)

#         # Step 3: Aggregation with edge features
#         h = self.agg(h, adj, edge_features)  # Aggregated features with edge weights

#         # Step 4: Hyperbolic activation
#         h = self.hyp_act(h)  # Shape: (batch_size, num_nodes, out_features)

#         # Return updated node features and adjacency matrix
#         return h, adj, edge_features


class LorentzHypLinear(nn.Module):
    """
    Parameters:
        manifold (manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, manifold_out=None):
        super().__init__()
        # self.in_features = in_features + 1  # + 1 for time dimension
        self.in_features = in_features
        self.out_features = out_features - 1 # - 1 for time dimension
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out
        
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        '''
        x: [batch_size, num_nodes, in_features]
        '''

        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)

        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
 
        return x


# class LorenzHypLinear(nn.Module):
#     """
#     Parameters:
#         manifold (manifold): The manifold to use for the linear transformation.
#         in_features (int): The size of each input sample.
#         out_features (int): The size of each output sample.
#         bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
#         dropout (float, optional): The dropout probability. Default is 0.1.
#     """

#     def __init__(self, manifold, in_features, out_features, bias=True, manifold_out=None):
#         super().__init__()
#         self.in_features = in_features + 1  # + 1 for time dimension
#         self.out_features = out_features
#         self.bias = bias
#         self.manifold = manifold
#         self.manifold_out = manifold_out
        
#         # use euclidean linear layer in hyperbolic space ?
#         # self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
#         # fix the initialization of linear layer
#         self.linear = _HypLinear(self.manifold, self.in_features, self.out_features, use_bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
#         init.constant_(self.linear.bias, 0)

#     def forward(self, x, x_manifold='hyp'):
#         if x_manifold != 'hyp':
#             x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
#             x = self.manifold.expmap0(x)
#         x_space = self.linear(x)

#         x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
#         x = torch.cat([x_time, x_space], dim=-1)

#         # Adjust for a different manifold if specified
#         if self.manifold_out is not None:
#             x = x * (self.manifold_out.k / self.manifold.k).sqrt()
#         return x
 

# class _HypLinear(nn.Module):
#     def __init__(self, manifold, in_features, out_features, use_bias, dropout=0.1):
#         super(_HypLinear, self).__init__()
#         self.manifold = manifold
#         self.in_features = in_features
#         self.out_features = out_features
#         self.c = 1.0 / manifold.k
#         self.dropout = dropout
#         self.use_bias = use_bias
#         self.bias = nn.Parameter(torch.Tensor(out_features))
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
#     def forward(self, x):
#         drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
#         mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
#         res = self.manifold.proj(mv, self.c)
#         if self.use_bias:
#             bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
#             hyp_bias = self.manifold.expmap0(bias, self.c)
#             hyp_bias = self.manifold.proj(hyp_bias, self.c)
#             res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
#             res = self.manifold.proj(res, self.c)
#         return res

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, c={}'.format(
#             self.in_features, self.out_features, self.c
#         )
    

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            # support_t = torch.spmm(adj, x_tangent)  ## 原本的只能处理二维的，HGCN文章输入的没有batch维度
            # GCN 的邻接矩阵聚合操作
            # pdb.set_trace()
            # if isinstance(adj, torch.sparse.Tensor):
            #     # 如果邻接矩阵是稀疏矩阵，逐批处理
            #     support_t = []
            #     for i in range(adj.size(0)):
            #         support_t.append(torch.spmm(adj[i], x_tangent[i]))
            #     support_t = torch.stack(support_t, dim=0)
            # else:
                # 如果邻接矩阵是稠密矩阵，直接批量操作
            #TODO: consider masked edges/nodes
            support_t = torch.bmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )



class HypLayerNorm(nn.Module):
    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features - 1 # - 1 for time dimension
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypNormalization(nn.Module):
    def __init__(self, manifold, manifold_out=None):
        super(HypNormalization, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypActivation(nn.Module):
    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        '''
        x: [batch_size, num_nodes, in_features]
        '''
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        # Adjust for a different manifold if specified
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypDropout(nn.Module):
    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)

            # Adjust for a different manifold if specified
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class PoincareDropout(nn.Module):
    def __init__(self, manifold, dropout, manifold_out=None):
        super(PoincareDropout, self).__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        if training:
            x = self.dropout(x)
        return x



class HypCLS(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        cls_emb = self.manifold.random_normal((self.out_channels, self.in_channels + 1), mean=0, std=1. / math.sqrt(self.in_channels + 1))
        self.cls = ManifoldParameter(cls_emb, self.manifold, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def forward(self, x, x_manifold='hyp', return_type='neg_dist'):
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1))  # project to Lorentz

        dist = -2 * self.manifold.k - 2 * self.manifold.cinner(x, self.cls) + self.bias

        # dist = self.manifold.cdist(x, self.cls) + self.bias
        # dist = dist.clamp(min=1e-6)

        if return_type == 'neg_dist':
            return - dist
        elif return_type == 'prob':
            return 10 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return - 10*torch.log(1.0 + dist)
        else:
            raise NotImplementedError


class Optimizer(object):
    def __init__(self, model, args):
        # Extract optimizer types and parameters from arguments
        euc_optimizer_type = getattr(args, 'euc_optimizer_type', args.optimizer_type)  # Euclidean optimizer type
        hyp_optimizer_type = getattr(args, 'hyp_optimizer_type', args.hyp_optimizer_type)  # Hyperbolic optimizer type
        euc_lr = getattr(args, 'euc_lr', args.lr)  # Euclidean learning rate
        hyp_lr = getattr(args, 'hyp_lr', args.hyp_lr)  # Hyperbolic learning rate
        euc_weight_decay = getattr(args, 'euc_weight_decay', args.weight_decay)  # Euclidean weight decay
        hyp_weight_decay = getattr(args, 'hyp_weight_decay', args.hyp_weight_decay)  # Hyperbolic weight decay

        # Separate parameters for Euclidean and Hyperbolic parts of the model
        euc_params = [p for n, p in model.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]  # Euclidean parameters
        hyp_params = [p for n, p in model.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]  # Hyperbolic parameters

        # Print the number of Euclidean and Hyperbolic parameters
        # print(f">> Number of Euclidean parameters: {sum(p.numel() for p in euc_params)}")
        # print(f">> Number of Hyperbolic parameters: {sum(p.numel() for p in hyp_params)}")
        # Initialize Euclidean optimizer

        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        else:
            raise NotImplementedError("Unsupported Euclidean optimizer type")

        # Initialize Hyperbolic optimizer if there are Hyperbolic parameters
        if hyp_params:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=50, weight_decay=hyp_weight_decay)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=50, weight_decay=hyp_weight_decay)
            else:
                raise NotImplementedError("Unsupported Hyperbolic optimizer type")

            # Store both optimizers
            self.optimizer = [optimizer_euc, optimizer_hyp]
        else:
            # Store only Euclidean optimizer if there are no Hyperbolic parameters
            self.optimizer = [optimizer_euc]

    def step(self):
        # Perform optimization step for each optimizer
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        # Reset gradients to zero for each optimizer
        for optimizer in self.optimizer:
            optimizer.zero_grad()
            



class LorentzLinear(nn.Module):
    # Lorentz Hyperbolic Graph Neural Layer
    def __init__(self, manifold, in_features, out_features, c, drop_out, use_bias):
        super(LorentzLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features-1))   # -1 when use mine mat-vec multiply
        self.weight = nn.Parameter(torch.Tensor(out_features - 1, in_features))  # -1, 0 when use mine mat-vec multiply
        self.reset_parameters()

    def report_weight(self):
        print(self.weight)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)
        # print('reset lorentz linear layer')

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.drop_out, training=self.training)
        mv = self.manifold.matvec_regular(drop_weight, x, self.bias, self.c, self.use_bias)
        return mv

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class PoincareMPNN(Module):
    """
    Poincare aggregtation and combine layer
    """
    def __init__(self, manifold, in_features, out_features):
        super(PoincareMPNN, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.norm1 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.euc_edge_linear = nn.Linear(3*in_features, out_features, bias=True)
        self.euc_node_linear = nn.Linear(2*in_features, out_features, bias=True)
        self.node_linear = PoincareLinear(manifold, in_features, out_features, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1, 2*out_features, bias=True)
        )
        self.adaLN_modulation_node = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 3*in_features, bias=True)
        )

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
    
    def hyper_distance(self, x, y):
        try:
            # Compute hyperbolic distance with safeguards
            hyper_distance = self.manifold.cdist(x, y)
            
            # Check for NaN/Inf values
            if torch.isnan(hyper_distance).any() or torch.isinf(hyper_distance).any():
                print("Warning: NaN/Inf detected in PoincareMPNN hyper_distance")
                # Replace problematic values with a reasonable default (1.0)
                hyper_distance = torch.where(
                    torch.isnan(hyper_distance) | torch.isinf(hyper_distance),
                    torch.ones_like(hyper_distance),
                    hyper_distance
                )
            
            # Apply clamping to ensure reasonable distance values
            hyper_distance = hyper_distance.clamp(min=1e-6, max=100.0)
            return hyper_distance
            
        except Exception as e:
            print(f"Error in PoincareMPNN hyper_distance: {e}")
            # Return a reasonable fallback distance
            return torch.ones_like(x[..., 0:1]).expand_as(y[..., 0:1])

    def edge_model(self, node_features, edge_features, edge_index_b, edge_index):
        # Extract features for connected nodes with numerical checks
        hi, hj = node_features[edge_index_b, edge_index[:, 0], :], node_features[edge_index_b, edge_index[:, 1], :]
        
        # Compute hyperbolic distance with numerical stability
        hypdis_ij = self.hyper_distance(node_features, node_features)[edge_index_b, edge_index[:, 0], edge_index[:, 1]]
        
        # Apply modulation safely with error checking
        shift_msa, scale_msa = self.adaLN_modulation(hypdis_ij.unsqueeze(-1).clamp(min=1e-6, max=100.0)).chunk(2, dim=1)

        # Safe concatenation of node features with edge features
        edge_features = self.manifold.logmap0(edge_features)
        hi = self.manifold.logmap0(hi)
        hj = self.manifold.logmap0(hj)
        edge_inputs = torch.cat([hi, hj, edge_features], dim=-1)
        
        # Apply normalization and modulation with checks
        normalized_edge_inputs = self.norm1(self.euc_edge_linear(edge_inputs))
        if torch.isnan(normalized_edge_inputs).any() or torch.isinf(normalized_edge_inputs).any():
            print("Warning: NaN/Inf detected after normalization in edge_model")
            normalized_edge_inputs = torch.zeros_like(normalized_edge_inputs)
        
        # Apply modulation with numerical stability
        edge_features_updated = self.modulate(normalized_edge_inputs, shift_msa, scale_msa)
        
        # Final check for NaN/Inf values
        if torch.isnan(edge_features_updated).any() or torch.isinf(edge_features_updated).any():
            print("Warning: NaN/Inf detected in final edge_features_updated")
            edge_features_updated = torch.where(
                torch.isnan(edge_features_updated) | torch.isinf(edge_features_updated),
                torch.zeros_like(edge_features_updated),
                edge_features_updated
            )
        
        return edge_features_updated
               
    def node_model(self, node_features, edge_features_updated, edge_index, adaLN_node):
        node_features = self.manifold.logmap0(node_features)
        aggregated_messages = scatter(edge_features_updated, edge_index[:, 0], dim=0, reduce='mean', dim_size=node_features.size(0))
        # if mask is not None:
        #     aggregated_messages = aggregated_messages * mask.unsqueeze(-1)
        #     node_features = node_features * mask.unsqueeze(-1)
        # node_inputs = torch.cat([node_features, aggregated_messages], dim=-1) # this step will out of hyperbolic space
        if adaLN_node:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation_node(aggregated_messages).chunk(3, dim=1)
            node_output = node_features + gate_msa * self.modulate(self.norm2(self.euc_node_linear(torch.cat([node_features, aggregated_messages], dim=-1))), shift_msa, scale_msa)
            node_output = self.manifold.expmap0(node_output)
            node_output = self.node_linear(node_output)
        else:
            node_inputs = torch.cat([node_features, aggregated_messages], dim=-1)
            node_output = node_inputs + self.node_linear(node_inputs)
        return node_output


    def forward(self, x, adj, e, adaLN_node = True):
        """
        Perform aggregation and update for all nodes in the graph.

        Args:
            x: Tensor of shape [B, N, D], node features.
            adj: Tensor of shape [B, N, N], adjacency matrix.
            e: Tensor of shape [B, N, N, D], edge features.

        Returns:
            Updated node features of shape [B, N, D].
        """
        B, N, D = x.shape
        # x = self.manifold.logmap0(x)
        # Convert adjacency matrix to edge_index
        edge_index = adj.nonzero(as_tuple=False)  # Shape [E, 3]
        edge_index_b = edge_index[:, 0]  # Batch indices
        edge_index_nodes = edge_index[:, 1:]  # Node indices
        
        # Extract edge features
        edge_features = e[edge_index_b, edge_index_nodes[:, 0], edge_index_nodes[:, 1], :]

        # Apply edge model
        edge_features_updated = self.edge_model(x, edge_features, edge_index_b, edge_index_nodes)
        # Apply node model
        x_flat = x.view(-1, D)
        h_next_flat = self.node_model(x_flat, edge_features_updated, edge_index_nodes, adaLN_node = adaLN_node)

        # Reshape back to [B, N, D]
        h_next = h_next_flat.view(B, N, D)

        e[edge_index_b, edge_index_nodes[:, 0], edge_index_nodes[:, 1], :] = self.manifold.expmap0(edge_features_updated)

        return h_next, e


class LorentzMPNN(Module):
    """
    lorentz aggregtation and combine layer
    """
    def __init__(self, manifold, manifold_out, in_features, out_features):
        super(LorentzMPNN, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.norm1 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        # self.message_layer = LorentzHypLinear(manifold, in_features*3 - 2, out_features, bias=True)
        # self.update_layer = LorentzHypLinear(manifold, in_features*2 - 1, out_features, bias=True)
        # self.edge_linear = LorentzHypLinear(manifold, 3*in_features, out_features, bias=True)
        self.euc_edge_linear = nn.Linear(3*in_features, out_features, bias=True)
        self.euc_node_linear = nn.Linear(2*in_features, out_features, bias=True)
        self.node_linear = LorentzHypLinear(manifold, in_features, out_features, bias=True, manifold_out=manifold_out)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1, 2*out_features, bias=True)
        )
        self.adaLN_modulation_node = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 3*in_features, bias=True)
        )
        
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
        
    def hyper_distance(self, x, y):
        try:
            # Compute hyperbolic distance with safeguards
            hyper_distance = self.manifold.cdist(x, y)
            
            # Check for NaN/Inf values
            if torch.isnan(hyper_distance).any() or torch.isinf(hyper_distance).any():
                print("Warning: NaN/Inf detected in LorentzMPNN hyper_distance")
                # Replace problematic values with a reasonable default (1.0)
                hyper_distance = torch.where(
                    torch.isnan(hyper_distance) | torch.isinf(hyper_distance),
                    torch.ones_like(hyper_distance),
                    hyper_distance
                )
            
            # Apply clamping to ensure reasonable distance values
            hyper_distance = hyper_distance.clamp(min=1e-6, max=100.0)
            return hyper_distance
            
        except Exception as e:
            print(f"Error in LorentzMPNN hyper_distance: {e}")
            # Return a reasonable fallback distance
            return torch.ones_like(x[..., 0:1]).expand_as(y[..., 0:1])

    def edge_model(self, node_features, edge_features, edge_index_b, edge_index):        
        try:
            # Extract features for connected nodes with numerical checks
            hi, hj = node_features[edge_index_b, edge_index[:, 0], :], node_features[edge_index_b, edge_index[:, 1], :]
            
            # Compute hyperbolic distance with numerical stability
            hypdis_ij = self.hyper_distance(node_features, node_features)[edge_index_b, edge_index[:, 0], edge_index[:, 1]]
            
            # Apply modulation safely with error checking
            try:
                shift_msa, scale_msa = self.adaLN_modulation(hypdis_ij.unsqueeze(-1).clamp(min=1e-6, max=100.0)).chunk(2, dim=1)
                
                # Check modulation parameters for issues
                if torch.isnan(shift_msa).any() or torch.isinf(shift_msa).any():
                    print("Warning: NaN/Inf detected in shift_msa")
                    shift_msa = torch.zeros_like(shift_msa)
                    
                if torch.isnan(scale_msa).any() or torch.isinf(scale_msa).any():
                    print("Warning: NaN/Inf detected in scale_msa")
                    scale_msa = torch.zeros_like(scale_msa)
            except Exception as inner_e:
                print(f"Error in adaLN_modulation: {inner_e}")
                # Create default modulation parameters
                shift_msa = torch.zeros_like(edge_features)
                scale_msa = torch.zeros_like(edge_features)
            
            # Safe concatenation of node features with edge features
            edge_inputs = torch.cat([hi, hj, edge_features], dim=-1)
            
            # Apply normalization and modulation with checks
            normalized_edge_inputs = self.norm1(self.euc_edge_linear(edge_inputs))
            if torch.isnan(normalized_edge_inputs).any() or torch.isinf(normalized_edge_inputs).any():
                print("Warning: NaN/Inf detected after normalization in edge_model")
                normalized_edge_inputs = torch.zeros_like(normalized_edge_inputs)
            
            # Apply modulation with numerical stability
            edge_features_updated = self.modulate(normalized_edge_inputs, shift_msa, scale_msa)
            
            # Final check for NaN/Inf values
            if torch.isnan(edge_features_updated).any() or torch.isinf(edge_features_updated).any():
                print("Warning: NaN/Inf detected in final edge_features_updated")
                edge_features_updated = torch.where(
                    torch.isnan(edge_features_updated) | torch.isinf(edge_features_updated),
                    torch.zeros_like(edge_features_updated),
                    edge_features_updated
                )
            
            return edge_features_updated
            
        except Exception as e:
            print(f"Error in LorentzMPNN edge_model: {e}")
            # Return a zero tensor as fallback
            return torch.zeros_like(edge_features)


    def node_model(self, node_features, edge_features_updated, edge_index, adaLN_node):
        aggregated_messages = scatter(edge_features_updated, edge_index[:, 0], dim=0, reduce='mean', dim_size=node_features.size(0))
        # if mask is not None:
        #     aggregated_messages = aggregated_messages * mask.unsqueeze(-1)
        #     node_features = node_features * mask.unsqueeze(-1)
        # node_inputs = torch.cat([node_features, aggregated_messages], dim=-1) # this step will out of hyperbolic space
        # node_inputs = self.manifold.mid_point(torch.stack((node_features, aggregated_messages), dim=1))
        if adaLN_node:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation_node(aggregated_messages).chunk(3, dim=1)
            node_output = node_features + gate_msa * self.modulate(self.norm2(self.euc_node_linear(torch.cat([node_features, aggregated_messages], dim=-1))), shift_msa, scale_msa)
            node_output = self.node_linear(node_output)
        else:
            node_inputs = torch.cat([node_features, aggregated_messages], dim=-1)
            node_output = node_inputs + self.node_linear(node_inputs)
        return node_output


    def forward(self, x, adj, e, adaLN_node = True):
        """
        Perform aggregation and update for all nodes in the graph.

        Args:
            x: Tensor of shape [B, N, D], node features.
            adj: Tensor of shape [B, N, N], adjacency matrix.
            e: Tensor of shape [B, N, N, D], edge features.

        Returns:
            Updated node features of shape [B, N, D].
        """
        B, N, D = x.shape
        x = self.manifold.logmap0(x)
        # Convert adjacency matrix to edge_index
        edge_index = adj.nonzero(as_tuple=False)  # Shape [E, 3]
        edge_index_b = edge_index[:, 0]  # Batch indices
        edge_index_nodes = edge_index[:, 1:]  # Node indices (source, target)

        # Extract edge features
        edge_features = e[edge_index_b, edge_index_nodes[:, 0], edge_index_nodes[:, 1], :]

        # Apply edge model
        edge_features_updated = self.edge_model(x, edge_features, edge_index_b, edge_index_nodes)

        # Apply node model
        x_flat = x.view(-1, D)
        h_next_flat = self.node_model(x_flat, edge_features_updated, edge_index_nodes, adaLN_node = adaLN_node)

        # Reshape back to [B, N, D]
        h_next = h_next_flat.view(B, N, D)

        return h_next

    
        
    # def forward(self, x, adj, e):
    #     """
    #     Perform aggregation and update for all nodes in the graph.

    #     Args:
    #         x: Tensor of shape [B, N, D], node features.
    #         adj: Tensor of shape [B, N, N], adjacency matrix.
    #         e: Tensor of shape [B, N, N, D], edge features.

    #     Returns:
    #         Updated node features of shape [B, N, D].
    #     """
    #     B, N, D = x.shape

    #     # Initialize aggregated message tensor
    #     m_next = torch.zeros_like(x)

    #     # Iterate over all nodes to aggregate messages
    #     for v in range(N):
    #         # Get neighbors for node v based on adjacency matrix
    #         neighbors = adj[:, v, :]  # Shape [B, N]
    #         # Create boolean mask for neighbors
    #         neighbor_mask = neighbors.bool()  # Shape [B, N]

    #         # Extract neighbor features
    #         neighbor_indices = torch.nonzero(neighbor_mask, as_tuple=True)  # Get indices for neighbors
    #         batch_indices, node_indices = neighbor_indices[0], neighbor_indices[1]

    #         neighbor_features = torch.zeros(B, N, D, device=x.device)  # Shape [B, N, D]
    #         neighbor_features[batch_indices, node_indices] = x[batch_indices, node_indices]  # Populate features

    #         # Extract edge features (from BNND tensor)
    #         edge_features = torch.zeros(B, N, D, device=e.device)  # Shape [B, N, D]
    #         edge_features[batch_indices, node_indices] = e[batch_indices, v, node_indices, :]  # Populate features
            
    #         # Concatenate node, neighbor, and edge features
    #         combined_features = torch.cat([
    #             x[:, v, :].unsqueeze(1).expand(-1, neighbor_features.size(1), -1),
    #             neighbor_features,
    #             edge_features
    #         ], dim=-1)  # Shape [B, num_neighbors, 3 * D]

    #         # Apply message passing layer
    #         messages = self.message_layer(combined_features)  # Shape [B, num_neighbors, D]

    #         # Aggregate messages (sum over neighbors)
    #         m_next[:, v, :] = messages.sum(dim=1)  # Shape [B, D]

    #     # Combine messages and update node features
    #     updated_features = torch.cat([x, m_next], dim=-1)  # Shape [B, N, 2 * D]
    #     h_next = self.update_layer(updated_features)  # Shape [B, N, D]

    #     return h_next
        
    
class LorentzGraphNeuralNetwork(nn.Module):
    def __init__(self, manifold_in, manifold_out, in_features, out_features, in_edge_feature, act = F.relu):
        super(LorentzGraphNeuralNetwork, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.agg = LorentzMPNN(manifold_in, manifold_out, out_features, out_features)
        self.lorentz_act = HypActivation(manifold_out, act)

    def forward(self, input):
        '''
        input: tuple (x, adj, e)
        x: Node features, shape (batch_size, num_nodes, in_features)
        adj: Adjacency matrix, shape (batch_size, num_nodes, num_nodes)
        e: Edge features, shape (batch_size, num_nodes, num_nodes, in_edge_feature)
        '''
        x, adj, e = input
        h = self.agg.forward(x, adj, e)
        h = self.lorentz_act.forward(h)
        output = h, adj, e
        return output


class PoincareGraphNeuralNetwork(nn.Module):
    def __init__(self, manifold, in_features, out_features, in_edge_feature, act = F.relu, use_layernorm=False):
        super(PoincareGraphNeuralNetwork, self).__init__()
        self.manifold = manifold
        self.agg = PoincareMPNN(manifold, out_features, out_features)
        self.act = act
        # Add a normalization layer for numerical stability
        if use_layernorm:
            self.norm = PoincareLayerNorm(manifold, out_features)
        
    
    def forward(self, input):
        x, adj, e = input
        
        # Check for NaN/Inf values in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected in input to PoincareGraphNeuralNetwork")
            x = torch.where(
                torch.isnan(x) | torch.isinf(x),
                torch.zeros_like(x),
                x
            )
            # Ensure the points remain on the manifold
            if hasattr(self.manifold, 'proj'):
                x = self.manifold.proj(x)
            
        # Apply MPNN aggregation with error handling
        h, e = self.agg.forward(x, adj, e)
        
        # Check for NaN/Inf values after aggregation
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("Warning: NaN/Inf detected after PoincareMPNN aggregation")
            h = torch.where(
                torch.isnan(h) | torch.isinf(h),
                x.clone(),  # Fall back to input if output is problematic
                h
            )
        
        # Apply normalization for stability if needed
        if hasattr(self, 'norm'):
            h = self.norm(h)
            
        output = h, adj, e
        return output


class PoincareLayerNorm(nn.Module):
    def __init__(self, manifold, in_features):
        super(PoincareLayerNorm, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True, eps=1e-6)
        self.norm.weight.data.fill_(0.1)
        self.norm.bias.data.fill_(0)

    def forward(self, x):
        x = self.manifold.logmap0(x)
        x = self.norm(x)
        x = self.manifold.expmap0(x)
        return x


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        device = b.device
        a = torch.sparse_coo_tensor(indices, values, shape, device=device)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class LorentzGraphDecoder(nn.Module):
    # Lorentzian graph neural network decoder
    def __init__(self, manifold, in_feature, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGraphDecoder, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.out_features = out_features + 1 # original output equal to num_classes
        self.in_features = in_feature
        self.linear = LorentzLinear(manifold, in_feature-1, self.out_features, c_in, drop_out, False)
        self.agg = LorentzAgg(manifold, c_in, use_att, self.out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        init.constant_(self.bias, 0)

    def forward(self, input):
        x, adj = input
        # print('=====x', x.shape, self.in_features)
        h = self.linear.forward(x) ## problem is h1+
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)
        b = self.manifold.ptransp0(h, self.bias, self.c_in)
        b = self.manifold.exp_map_x(h, b, self.c_in)
        poincare_h = self.manifold.lorentz2poincare(h, self.c_in)
        output = poincare_h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()