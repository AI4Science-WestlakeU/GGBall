import torch
import torch.nn as nn

import manifolds
from src.models.layers import GraphAttentionLayer, GraphConvolution, Linear
import pdb


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

        
    def forward(self, dist, r=None, t=None):
        # pdb.set_trace()
        if r is not None:
            self.r = r
        if t is not None:
            self.t = t
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class SoftmaxDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(SoftmaxDecoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.nn = nn.Linear(1, 1)

    def forward(self, dist):
        probs = self.sigmoid(self.nn(dist[...,None]).squeeze(-1))
        return probs
    
    
class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, cfg):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, cfg.model.manifold)()
        self.input_dim = cfg.model.dim
        self.output_dim = cfg.model.node_classes
        self.bias = cfg.model.bias
        self.cls = Linear(self.input_dim, self.output_dim, cfg.train.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )
        

class LPDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean edge classification models.
    """

    def __init__(self, c, cfg):
        super(LPDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, cfg.model.manifold)()
        self.input_dim = cfg.model.dim
        self.output_dim = cfg.model.edge_classes
        self.bias = cfg.model.bias
        # self.linear = nn.Linear(self.input_dim, self.input_dim).to(x.device)
        self.relu = nn.ReLU()
        self.dropout = cfg.train.dropout
        self.cls = Linear(self.input_dim, self.output_dim, cfg.train.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        B = x.size(0)
        N = x.size(1)
        # 延迟初始化 self.linear，动态调整输出维度
        # if self.linear is None or self.linear.out_features != N * self.input_dim:
        #     self.linear = nn.Linear(self.input_dim, N * self.input_dim).to(x.device)
        
        # TODO: we can concat node feature as edge feature, and predict edge label        
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        h = torch.einsum("bnd, bmd -> bnmd", h, h)
        h = self.relu(h)
        h = h.view(B, N, N, self.input_dim)
        return super(LPDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'FD': FermiDiracDecoder,
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}