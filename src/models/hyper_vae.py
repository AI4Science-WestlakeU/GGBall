import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.models.hyper_layers import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypCLS, LorentzHypLinear, PoincareLinear, PoincareLayerNorm, PoincareDropout
from src.models.hyper_encoder import HGCN, LGCN, PoincareGCN
from src.models.hyper_decoder import FermiDiracDecoder
from src.manifolds.lorentz import Lorentz
# from src.manifolds.poincareball import PoincareBall
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
from utils import check_on_manifold
from src.models.transormer_layers import DiTBlock, TimestepEmbedder
# from geoopt.manifolds.euclidean import Euclidean
from src.manifolds.euclidean import Euclidean2 as Euclidean
from geoopt.manifolds.product import ProductManifold
from src.models.vq import VectorQuantize
from src.models.PoincareTransformer import PoincareTransformer
import wandb
import geoopt
class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, cfg=None):
        """
        Initializes a TransConvLayer instance.

        Args:
            manifold: The manifold to use for the layer.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            num_heads: The number of attention heads.
            use_weight: Whether to use weights for the attention mechanism. Defaults to True.
            args: Additional arguments for the layer, including attention_type, power_k, and trans_heads_concat.

        Returns:
            None
        """
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = cfg.model.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(LorentzHypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(LorentzHypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(LorentzHypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)], requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels - 1, out_channels - 1, bias=True)  # -1 for time dimension
        self.power_k = cfg.model.power_k
        self.trans_heads_concat = cfg.model.trans_heads_concat

        if self.trans_heads_concat:
            self.final_linear = nn.Linear(out_channels * self.num_heads, out_channels, bias=True)

    def full_attention(self, qs, ks, vs, output_attn=False):
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)

        # negative squared distance (less than 0)
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]

        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def get_pad_x(self, x, mask, pad_index = 0):
        '''
        x: [B, N, H, D]
        mask: [B, N]
        return: [B, N, H, D]
        '''
        B, N, H, D = x.shape
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, H, D)
        pad_x = torch.zeros_like(x)
        pad_x[mask] = x[mask]
        return pad_x


    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, mask = None, output_attn=False):    
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]

        phi_qs = (F.relu(qs) + 1e-6) / self.norm_scale.abs()  # [B, N, H, D]
        phi_ks = (F.relu(ks) + 1e-6) / self.norm_scale.abs()  # [B, N, H, D]
        # v = (F.relu(v) + 1e-6) / self.norm_scale.abs()

        phi_qs = self.fp(phi_qs, p=self.power_k)  # [B, N, H, D]
        phi_ks = self.fp(phi_ks, p=self.power_k)  # [B, N, H, D]
        
        # Step 0: Pad phi_ks
        if mask is not None:
            phi_qs = self.get_pad_x(phi_qs, mask)
            phi_ks = self.get_pad_x(phi_ks, mask)
            v = self.get_pad_x(v, mask)

        # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
        k_transpose_v = torch.einsum('bnhm,bnhd->bhmd', phi_ks, v)  # [B, H, D, D] 

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('bnhm,bhmd->bnhd', phi_qs, k_transpose_v)  # [B, N, H, D]

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        denominator = torch.einsum('bnhd,bhd->bnh', phi_qs, torch.einsum('bnhd->bhd', phi_ks))  # [B, N, H]
        denominator = denominator.unsqueeze(-1)  # [B, N, H, D] for broadcasting

        # Step 4: Normalize the numerator with the denominator
        attn_output = numerator / (denominator + 1e-6)  # [B, N, H, D]

        # Map vs through v_map_mlp and ensure it is the correct shape
        vss = self.v_map_mlp(v)  # [B, N, H, D]
        if mask is not None:
            vss = self.get_pad_x(vss, mask)
        attn_output = attn_output + vss  # preserve its rank, [B, N, H, D]

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(attn_output.shape[0], -1, self.num_heads * self.out_channels))
        else:
            attn_output = attn_output.mean(dim=2)

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)


        if output_attn:
            # Calculate attention weights
            attention = torch.einsum('bnhd,bmhd->bnmh', phi_qs, phi_ks)  # [B, N, M, H]
            attention = attention / (denominator + 1e-6)  # Normalize

            # Average attention across heads if needed
            attention = attention.mean(dim=-1)  # [N, M]
            return attn_output, attention
        else:
            return attn_output

    def forward(self, query_input, source_input, mask=None, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input, x_manifold='hyp'))
            k_list.append(self.Wk[i](source_input, x_manifold='hyp'))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input, x_manifold='hyp'))
            else:
                v_list.append(source_input)
                
        ## fixed code

        query = torch.stack(q_list, dim=2)  # [B, N, H, D]
        key = torch.stack(k_list, dim=2)  # [B, N, H, D]
        value = torch.stack(v_list, dim=2)  # [B, N, H, D]
        
        ## add code, current is B, H, N, D， change to N, H, D
        # query = torch.reshape(query, (-1, query.shape[1], query.shape[3]))
        # key = torch.reshape(key, (-1, key.shape[1], key.shape[3]))
        # value = torch.reshape(value, (-1, value.shape[1], value.shape[3]))
        ## calculate the attention between all nodes in all batches, not only in one graph


        if output_attn:
            if self.attention_type == 'linear_focused':
                if mask is not None:
                    attention_output, attn = self.linear_focus_attention(
                        query, key, value, mask, output_attn)  # [B, N, H, D]
                else:
                    attention_output, attn = self.linear_focus_attention(
                        query, key, value, output_attn)
            elif self.attention_type == 'full':
                if mask is not None:
                    attention_output, attn = self.full_attention(
                        query, key, value, mask, output_attn)
                else:
                    attention_output, attn = self.full_attention(
                        query, key, value, output_attn)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                if mask is not None:
                    attention_output = self.linear_focus_attention(
                        query, key, value, mask)  # [B, N, H, D]
                else:
                    attention_output = self.linear_focus_attention(
                        query, key, value)
            elif self.attention_type == 'full':
                if mask is not None:
                    attention_output = self.full_attention(
                        query, key, value, mask)
                else:
                    attention_output = self.full_attention(
                        query, key, value)  # [N, H, D]

        final_output = attention_output

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, euc_channels, cfg):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.num_layers = cfg.model.trans_num_layers
        self.num_heads = cfg.model.trans_num_heads
        self.dropout_rate = cfg.model.trans_dropout
        self.use_bn = cfg.model.trans_use_bn
        self.residual = cfg.model.trans_use_residual
        self.use_act = cfg.model.trans_use_act
        self.use_weight = cfg.model.trans_use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(LorentzHypLinear(self.manifold_in, self.in_channels, self.euc_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.euc_channels))

        self.add_pos_enc = cfg.model.add_positional_encoding
        self.positional_encoding = LorentzHypLinear(self.manifold_in, self.in_channels, self.euc_channels,
                                             self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device = self.manifold_in.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.euc_channels, self.euc_channels,
                               num_heads=self.num_heads,
                               use_weight=self.use_weight, cfg=cfg))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.euc_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)
        self.mlp = LorentzMLP(self.manifold_hidden, self.euc_channels, layers=[self.euc_channels, self.euc_channels])

        self.fcs.append(LorentzHypLinear(self.manifold_hidden, self.euc_channels, self.euc_channels, self.manifold_out))

    def forward(self, x_input, mask=None):
        layer_ = []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='hyp')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='hyp')
            self.epsilon = self.epsilon.to(x.device)
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=2))  # fix as second dim

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            # b = x.shape[0]
            if mask is not None:
                x = conv(x, x, mask)
            else:
                x = conv(x, x)
            # add code, change to N, H, D
            # x = torch.reshape(x, (b, -1, x.shape[1]))
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=2))
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            # if self.dropout_rate > 0:
            #     x = self.dropout(x, training=self.training)
            x = self.mlp(x, self.activation)
            
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x_input):
        layer_, attentions = [], []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='hyp')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='hyp')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class TransConv2(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, euc_channels, cfg):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.num_layers = cfg.model.trans_num_layers
        self.num_heads = cfg.model.trans_num_heads
        self.dropout_rate = cfg.model.trans_dropout
        self.use_bn = cfg.model.trans_use_bn
        self.residual = cfg.model.trans_use_residual
        self.use_act = cfg.model.trans_use_act
        self.use_weight = cfg.model.trans_use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(LorentzHypLinear(self.manifold_in, self.in_channels, self.euc_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.euc_channels))

        self.add_pos_enc = cfg.model.add_positional_encoding
        self.positional_encoding = LorentzHypLinear(self.manifold_in, self.in_channels, self.euc_channels,
                                             self.manifold_hidden)

        self.add_time_enc = cfg.model.add_time_encoding
        if self.add_time_enc:
            self.time_encoding = TimestepEmbedder(self.in_channels-1)
            self.time_linear = LorentzHypLinear(self.manifold_in, self.in_channels, self.euc_channels,
                                                    self.manifold_hidden)
        # self.epsilon = torch.tensor([1.0], device=self.manifold_in.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.euc_channels, self.euc_channels,
                               num_heads=self.num_heads,
                               use_weight=self.use_weight, cfg=cfg))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.euc_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)
        self.mlp = LorentzMLP(self.manifold_hidden, self.euc_channels, layers=[self.euc_channels, self.euc_channels])

        self.fcs.append(LorentzHypLinear(self.manifold_hidden, self.euc_channels, self.euc_channels, self.manifold_out))

    def forward(self, x_input, t=None, mask=None):
        layer_ = []
        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='hyp')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='hyp')
            x = self.manifold_in.mid_point(torch.stack((x, x_pos), dim=2))  # fix as second dim

        if self.add_time_enc:
            t_emb = self.time_encoding(t.view(-1))
            t_emb = self.time_linear(t_emb, x_manifold='euc')
            t_emb = t_emb.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)
            x = self.manifold_in.mid_point(torch.stack((x, t_emb), dim=2))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            if self.add_time_enc:
                t_emb = self.time_encoding(t.view(-1))
                t_emb = self.time_linear(t_emb, x_manifold='euc')
                t_emb = t_emb.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)
                x = self.manifold_in.mid_point(torch.stack((x, t_emb), dim=2))
                # b = x.shape[0]
            if mask is not None:
                x = conv(x, x, mask)
            else:
                x = conv(x, x)
            # add code, change to N, H, D
            # x = torch.reshape(x, (b, -1, x.shape[1]))
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=2))
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            # if self.dropout_rate > 0:
            #     x = self.dropout(x, training=self.training)
            x = self.mlp(x, self.activation)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x_input):
        layer_, attentions = [], []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='euc')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_in.mid_point(torch.stack((x, self.epsilon * x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_in.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]



class LorentzMLP(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, manifold, input, layers= [16, 16], normalize = False, dropout_rate = 0):
        """

        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(LorentzMLP, self).__init__()
        self.manifold = manifold
        self.layers = torch.nn.ModuleList([LorentzHypLinear(self.manifold, input, layers[0], bias=True)])
        for i in range(len(layers)-1):
            self.layers.append(LorentzHypLinear(self.manifold, layers[i], layers[i+1], bias=True))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([HypLayerNorm(manifold, c, manifold) for c in [input]+layers])
        self.dropout = HypDropout(manifold, dropout_rate, manifold)
        # self.reset_parameters()

    def forward(self, in_tensor, activation, applyActOnTheLastLyr=False):
        h = in_tensor
        for i in range(len(self.layers)):
            residual = h  # Save the input for the residual connection
            if self.norm_layers!=None:
                if len(h.shape)==3:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(h.shape[0], -1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            
            if residual.shape == h.shape:
                h = h + residual
            
            if i != (len(self.layers)-1) or applyActOnTheLastLyr:
                h = activation(h)
        return h
    


class PoincareMLP(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, manifold, input, layers= [16, 16], dropout_rate = 0, use_resnet = False, use_layernorm = False):
        """

        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(PoincareMLP, self).__init__()
        self.manifold = manifold
        self.layers = torch.nn.ModuleList([PoincareLinear(self.manifold, input, layers[0])])
        for i in range(len(layers)-1):
            self.layers.append(PoincareLinear(self.manifold, layers[i], layers[i+1]))

        self.norm_layers = None
        if use_layernorm:
            self.norm_layers =  torch.nn.ModuleList([PoincareLayerNorm(manifold, c) for c in [input]+layers])
        self.dropout = PoincareDropout(manifold, dropout_rate)
        self.use_resnet = use_resnet
        # self.reset_parameters()

    def forward(self, in_tensor, applyActOnTheLastLyr=False):
        h = in_tensor
        for i in range(len(self.layers)):
            residual = h  # Save the input for the residual connection
            if self.norm_layers!=None:
                if len(h.shape)==3:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(h.shape[0], -1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            
            if self.use_resnet and residual.shape == h.shape:
                h = self.manifold.mobius_add(residual, h)
            
            # if i != (len(self.layers)-1) or applyActOnTheLastLyr:
            #     h = activation(h)
        return h


class AdjDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        self.mlp = LorentzMLP(self.manifold, input=self.euc_channels, layers=[self.euc_channels, self.out_channels], normalize=True, dropout_rate=0)
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        # self.fermi_dirac = FermiDiracDecoder(self.cfg.model.r, self.cfg.model.t)
    
    def hyper_distance(self, x, y):
        return self.manifold.cdist(x, y)
        
    def forward(self, z):
        node_feat = self.mlp(z, activation=F.relu) # [B, N, D]
        # calculate the hybolic distance between the node feature
        adj = self.hyper_distance(node_feat, node_feat)
        adj = self.linear(adj.unsqueeze(-1)).squeeze(-1)
        # adj = self.fermi_dirac(adj)
        adj = self.sigmoid(adj)
        # adj digonal should be 0
        # adj = adj - torch.eye(adj.size(1), device=adj.device)
        return adj

class EucEncoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        self.node_mlp = nn.Sequential(
            nn.Linear(self.euc_channels, 4*self.euc_channels),
            nn.ReLU(),
            nn.Linear(4*self.euc_channels, self.out_channels)
        )
        self.node_mlp2 = PoincareMLP(self.manifold, input=self.euc_channels, layers=[self.euc_channels, self.euc_channels], dropout_rate=0.1, use_resnet=False, use_layernorm=False)
        
    def forward(self, z):
        z = self.node_mlp2(z)
        z = self.manifold.logmap0(z)
        node_feat = self.node_mlp(z)
        return node_feat
    
class HypEncoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        if isinstance(manifold, Lorentz):
            self.act = HypActivation(self.manifold, activation=F.relu)
            self.edge_mlp = LorentzMLP(self.manifold, input=self.euc_channels, layers=[self.euc_channels, self.out_channels], normalize=True, dropout_rate=0.1)
        else:
            self.edge_mlp = PoincareMLP(self.manifold, input=self.euc_channels, layers=[self.euc_channels, self.out_channels], dropout_rate=0.1, use_resnet=use_resnet, use_layernorm=use_layernorm)
    
    def forward(self, z):
        # pdb.set_trace()
        if isinstance(self.manifold, Lorentz):
            edge_feat = self.edge_mlp(z, activation=self.act)
        else:
            edge_feat = self.edge_mlp(z)
        return edge_feat


class Euc2NodeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        # self.mlp = LorentzMLP(self.manifold, input=self.in_channels, layers=[self.euc_channels, self.euc_channels], normalize=True, dropout_rate=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(self.euc_channels, self.out_channels)  ## after mlp, hypmlp will not change the output channel now, the output includes the time dimension to predict the res
        
    def forward(self, z):
        node_feat = self.mlp(z)
        node_feat = self.linear(node_feat)
        return node_feat
    
class Hyp2NodeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        # self.mlp = LorentzMLP(self.manifold, input=self.in_channels, layers=[self.euc_channels, self.euc_channels], normalize=True, dropout_rate=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(self.euc_channels, self.out_channels)  ## after mlp, hypmlp will not change the output channel now, the output includes the time dimension to predict the res
        self.edge_mlp = PoincareMLP(self.manifold, input=self.in_channels, layers=[self.in_channels, self.in_channels], dropout_rate=0.1, use_resnet=use_resnet, use_layernorm=use_layernorm)
    def forward(self, z):
        z = self.edge_mlp(z)
        z = self.manifold.logmap0(z)
        node_feat = self.mlp(z)
        node_feat = self.linear(node_feat)
        return node_feat
    
    
    
class Euc2EdgeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, method="pairwise_interaction", init_thresholds=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        # self.mlp = LorentzMLP(self.manifold, input= 2 * self.euc_channels, layers=[self.euc_channels, self.out_channels+1], normalize=True, dropout_rate=0)

        self.decoder = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
        )      

        self.prediction_head3 = nn.Linear(self.euc_channels, self.euc_channels)
        self.prediction_head2 = nn.Linear(self.euc_channels, self.out_channels)

        self.method = method
        if self.method == "pairwise_interaction":
            self.interaction_mlp = nn.Sequential(
                nn.Linear(2 * self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )
            # Edge classification layer
            self.edge_classifier = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )              
   
        elif self.method == "pairwise_distance":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )              
            
            
        elif self.method == "inner_product":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )      

        elif self.method == "fermi_dirac":                     
            self.r = nn.Parameter(torch.tensor([2.0], requires_grad=True))
            self.t = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.fermi_dirac = FermiDiracDecoder(self.r, self.t)

        elif self.method == "logmap":
            self.prediction_head = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )


    def hyper_distance(self, x, y):
        return self.manifold.cdist(x, y)
    
    def forward(self, z):
        # z is not on manifold
        # check_on_manifold(self.manifold, z, "node input to decoder")
        z = self.decoder(z)
        
        # check_on_manifold(self.manifold, z, "node in decoder after linear layer")
        # z is not on manifold
        # z = self.manifold.logmap0(z)
        
        # distance = distance.unsqueeze(-1)
        # distance = self.prediction_head(distance)
        
        # edge_feat = self.fermi_dirac(distance)
        
        if self.method == "pairwise_interaction":
            B, N, D1 = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            z_j = z.unsqueeze(1).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            pair_features = torch.cat([z_i, z_j], dim=-1)  # Shape: (B, N, N, 2 * D1)
            
            h = self.interaction_mlp(pair_features)
            edge_feat = self.edge_classifier(h)
            return edge_feat
            
        
        elif self.method == "fermi_dirac": 
            distance = self.hyper_distance(z, z)
            edge_feat = self.fermi_dirac(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat
        
        elif self.method == "inner_product":
            distance = self.manifold.inner(z,z)
            edge_feat = self.prediction_head(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat
        
        elif self.method == "pairwise_distance":
            distance = self.hyper_distance(z, z)
            edge_feat = self.prediction_head(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat
        
        elif self.method == "logmap":
            B, N, D = z.shape
            # 创建点对索引
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
            
            # 计算每对点之间的logmap
            logmap_vectors = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]
            
            # 使用预测头处理logmap向量
            edge_feat = self.prediction_head(logmap_vectors)
            return edge_feat
        # distance = self.hyper_distance(z, z)
        
        # thresholds = torch.sort(self.thresholds)[0]
        # distance = distance.unsqueeze(-1)
        # # 初始化 logits，形状 [B, N, N, num_classes]
        # B, N, _, _ = distance.shape
        # logits = torch.zeros(B, N, N, self.out_channels, device=distance.device)
        # thresholds = thresholds.view(1, 1, 1, -1)  # 调整形状以适配 broadcast
        # weights = distance - thresholds  # 广播减法，避免梯度丢失
        # # weights = torch.cat([distance - thresholds[i] for i in range(self.out_channels + 1)], dim=-1)
        # logits = F.softmax(weights, dim=-1)
        
        # # edge_feat = self.prediction_head(distance.unsqueeze(-1)).squeeze(-1)
        # return logits

class Hyp2EdgeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, method="pairwise_interaction", init_thresholds=None, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        # self.mlp = LorentzMLP(self.manifold, input= 2 * self.euc_channels, layers=[self.euc_channels, self.out_channels+1], normalize=True, dropout_rate=0)

        if isinstance(self.manifold, Lorentz):
            self.decoder = LorentzMLP(self.manifold, input=self.in_channels, layers=[self.euc_channels, self.euc_channels], normalize=True, dropout_rate=0)
            self.act = HypActivation(self.manifold, activation=F.relu)      
        else:
            self.decoder = PoincareMLP(self.manifold, input=self.in_channels, layers=[self.euc_channels, self.euc_channels], dropout_rate=0, use_resnet=use_resnet, use_layernorm=use_layernorm)    

        self.method = method
        if self.method == "pairwise_interaction":
            self.interaction_mlp = nn.Sequential(
                nn.Linear(2 * self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )
            # Edge classification layer
            self.edge_classifier = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )              
  
        elif self.method == "pairwise_distance":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )              
            
            
        elif self.method == "inner_product":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )              
            
        elif self.method == "logmap" or self.method=="logmap_symmetry":
            self.prediction_head = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )
        

        elif self.method == "fermi_dirac":                     
            self.r = nn.Parameter(torch.tensor([2.0], requires_grad=True))
            self.t = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.fermi_dirac = FermiDiracDecoder(self.r, self.t)

        elif self.method == "tangent_mlp":
            self.tan_mlp = nn.Sequential(
                nn.Linear(3*self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )
        
        elif self.method == "strong_mlp":
            self.strong_mlp = nn.Sequential(
                nn.Linear(3*self.euc_channels + 2, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )
        elif self.method == "transh":
            self.R_mat = nn.Parameter(torch.eye(self.euc_channels).repeat(self.R,1,1)) # 旋转
            self.t_vec = nn.Parameter(torch.zeros(self.R, self.euc_channels))          # 位移
        elif self.method == "gyroplane":
            self.n_vec = nn.Parameter(torch.randn(self.out_channels, self.euc_channels))
            self.bias  = nn.Parameter(torch.zeros(self.out_channels))
        elif self.method == "multirel_distance":
            self.rel_emb = geoopt.ManifoldParameter(
            manifold.random_normal((self.R, self.euc_channels)),
            manifold=self.manifold)
            # global scale / bias for 1.1
            self.alpha = nn.Parameter(torch.tensor(10.0))
            self.beta  = nn.Parameter(torch.tensor(0.0))

    def hyper_distance(self, x, y):
        return self.manifold.cdist(x, y)
    
    # ---------- 1.2  多关系：Möbius 位移 + 距离 Softmax --------------------
    def forward_multirel_distance(self, z):
        """K-class edge type；learn K relation embeddings r_k"""
        z = self.decoder(z)
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        # Möbius 迁移后求距
        rel = self.rel_emb.view(1, 1, 1, self.R, D)         # [1,1,1,R,D]
        z_i_r = self.manifold.mobius_add(z_i.unsqueeze(3), rel)  # [B,N,N,R,D]
        dist = self.manifold.dist(z_i_r, z_j.unsqueeze(3))       # [B,N,N,R]
        score = -self.alpha * dist                              # R-way logit
        return score                                            # CrossEntropyLoss

    # ---------- 2  Gyroplane（超平面）--------------------------------------
    def forward_gyroplane(self, z):
        """超平面 Logistic / Softmax"""
        z = self.decoder(z)
        B, N, D = z.shape
        # build pair diff in tangent
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        # logmap0 差分
        v = self.manifold.logmap0(self.manifold.mobius_add(
            self.manifold.mobius_neg(z_i), z_j))   # [B,N,N,D]

        # 线性超平面： <v, n_k> + b_k
        logits = torch.einsum('b m n d , k d -> b m n k', v, self.n_vec) + self.bias
        return logits                              # binary: k=1;  multi-class: k=R


    # ---------- 3  Tangent-space MLP ---------------------------------------
    def forward_tangent_mlp(self, z):
        """logmap0 → Euclidean MLP"""
        z = self.decoder(z)
        v = self.manifold.logmap0(z)               # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        logmap_vectors1 = self.manifold.logmap(z_i, z_j)
        B, N, D = v.shape
        v_i = v.unsqueeze(2).expand(B, N, N, D)
        v_j = v.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([v_i, v_j, logmap_vectors1], dim=-1)   # [B,N,N,3D]
        out = self.tan_mlp(pair)                   # 自定义 MLP → [B,N,N,out_channels]
        return out
    
    def forward_strong_mlp(self, z):
        """logmap0 → Euclidean MLP"""
        z = self.decoder(z)
        v = self.manifold.logmap0(z)               # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        logmap_vectors1 = self.manifold.logmap(z_i, z_j)
        ## calculate distance
        dist = self.manifold.dist(z_i, z_j).unsqueeze(-1) # [B,N,N,1]
        B, N, D = v.shape
        v_i = v.unsqueeze(2).expand(B, N, N, D)
        v_j = v.unsqueeze(1).expand(B, N, N, D)
        ## calculate angle between vi and vj,
        v_i_unit = F.normalize(v_i, dim=-1, eps=1e-7)
        v_j_unit = F.normalize(v_j, dim=-1, eps=1e-7)
        cos_angle = (v_i_unit * v_j_unit).sum(-1, keepdim=True)\
                 .clamp(-1.0 + 1e-6, 1.0 - 1e-6)        # [B,N,N,1]

        pair = torch.cat([v_i, v_j, logmap_vectors1, dist, cos_angle], dim=-1)   # [B,N,N,3D + 2]
        out = self.strong_mlp(pair)                   # 自定义 MLP → [B,N,N,out_channels]
        return out


    # ---------- 4  TransH / RotH in 𝓗 -------------------------------------
    def forward_transh(self, z):
        """关系平移 / 旋转后再距"""
        z = self.decoder(z)                        # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        # 将每条关系的旋转矩阵 R_k (或位移向量 t_k) 储存为 nn.Parameter
        z_i_r = self.trans_rotate(z_i, self.R_mat)     # 可选旋转
        z_i_r = self.manifold.mobius_add(z_i_r, self.t_vec)  # 再平移
        dist = -self.manifold.dist(z_i_r, z_j)         # [B,N,N,R]
        # pdb.set_trace()
        return dist                                    # CrossEntropyLoss

    def forward(self, z):
        # z is not on manifold
        # check_on_manifold(self.manifold, z, "node input to decoder")
        if isinstance(self.manifold, Lorentz):
            z = self.decoder(z, activation=self.act)
        else:
            z = self.decoder(z)
        # check_on_manifold(self.manifold, z, "node in decoder after linear layer")
        # z is not on manifold
        # z = self.manifold.logmap0(z)
        
        # distance = distance.unsqueeze(-1)
        # distance = self.prediction_head(distance)
        
        # edge_feat = self.fermi_dirac(distance)
        
        if self.method == "pairwise_interaction":
            z = self.manifold.logmap0(z)
            B, N, D1 = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            z_j = z.unsqueeze(1).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            pair_features = torch.cat([z_i, z_j], dim=-1)  # Shape: (B, N, N, 2 * D1)
            
            h = self.interaction_mlp(pair_features)
            edge_feat = self.edge_classifier(h)
            return edge_feat
            
        # elif self.method == "manifold_difference":
        #     B, N, D1 = z.shape
        #     z_i = z.unsqueeze(2).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
        #     z_j = z.unsqueeze(1).expand(B, N, N, D1)  # Shape: (B, N,
        #     # calculate manifold.logmap(z_i, z_j)
        #     # linear classifier
        #     h = self.interaction_mlp(pair_features)
        #     edge_feat = self.edge_classifier(h)
        #     return edge_feat
        
        elif self.method == "fermi_dirac": 
            distance = self.hyper_distance(z, z)
            edge_feat = self.fermi_dirac(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat
        
        elif self.method == "inner_product":
            distance = self.manifold.inner(z,z)
            edge_feat = self.prediction_head(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat
        
        elif self.method == "pairwise_distance":
            distance = -self.hyper_distance(z, z)
            edge_feat = self.prediction_head(distance.unsqueeze(-1)).squeeze(-1)
            edge_feat = torch.sigmoid(edge_feat)
            return edge_feat

        elif self.method == "logmap":
            B, N, D = z.shape
            # 创建点对索引
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
            
            # 计算每对点之间的logmap
            logmap_vectors = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]
            
            # 使用预测头处理logmap向量
            edge_feat = self.prediction_head(logmap_vectors)
            return edge_feat
        
        elif self.method=="logmap_symmetry":
            B, N, D = z.shape
            # 创建点对索引
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
            
            # 计算每对点之间的logmap
            logmap_vectors1 = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]
            logmap_vectors2 = self.manifold.logmap(z_j, z_i)  # [B, N, N, D]

            # 使用预测头处理logmap向量
            edge_feat = self.prediction_head(logmap_vectors1) + self.prediction_head(logmap_vectors2)
            return edge_feat
        
        elif self.method == 'pairwise_distance_logit':
            logits = self.forward_pairwise_distance_logit(z)


        elif self.method == 'multirel_distance':
            logits = self.forward_multirel_distance(z)      # [B,N,N,R]


        elif self.method == 'gyroplane':
            logits = self.forward_gyroplane(z)


        elif self.method == 'tangent_mlp':
            logits = self.forward_tangent_mlp(z)


        elif self.method == 'transh':
            logits = self.forward_transh(z)

        elif self.method == 'strong_mlp':
            logits = self.forward_strong_mlp(z)

        return logits
class HypAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.manifold_in = Lorentz(k=float(cfg.model.k_out))
        self.manifold_hidden = Lorentz(k=float(cfg.model.k_out))
        self.manifold_out = Lorentz(k=float(cfg.model.k_out))

        self.in_channels = cfg.model.in_channels
        self.euc_channels = cfg.model.euc_channels
        self.trans_conv = TransConv2(self.manifold_in, self.manifold_hidden, self.manifold_out, self.in_channels,
                                    self.euc_channels, cfg)
        
        
        self.lorentz_to_decode = torch.nn.Linear(self.euc_channels, self.euc_channels)
        self.final_decode = torch.nn.Linear(self.euc_channels, self.euc_channels)
        
        # self.time_embedder = nn.Linear(1, self.euc_channels)
    def forward(self, t, x, mask=None):
        #TODO: add topological information
        z = self.trans_conv(x, t, mask=mask)
        
        # z = self.lorentz_to_decode(z)
        # todo: add stronger transformer, because now is in tangent space
        # z = self.manifold_out.proju(x, z)
        # print(z.shape)
        # print("z min/max", z.min(), z.max())
        # import pdb; pdb.set_trace()
        
        # vector_field = self.manifold_out.logmap(x, z)
        # import pdb; pdb.set_trace()
        z = self.lorentz_to_decode(z)
        # z = torch.arcsinh(z)
        z = self.final_decode(z)
        z = torch.tanh(z)
        vector_field = self.manifold_out.proju(x, z)
        # import pdb; pdb.set_trace()
        
        #TODO: add DiT like structure, to encode time embedding t
        return vector_field




class HypFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # TODO: 这里的manifold out, manifold in应该是复合空间，而不是单独的双曲空间
        self.use_VAE = (cfg.loss.lambda_kl != 0)
        self.use_VQVAE = (cfg.loss.lambda_commitment_weight != 0)
        self.in_channels = cfg.model.latent_channels
        self.euc_channels = cfg.model.euc_channels
        self.hyp_channels = cfg.model.hyp_channels
        self.latent_channels = cfg.model.latent_channels
               
        
        # 只是LGCN用的
        self.manifold_in = Lorentz(k=float(cfg.model.k_in))
        self.manifold_hidden = Lorentz(k=float(cfg.model.k_hidden))
        if cfg.model.use_poincare:
            self.manifold_out =  PoincareBall(c = 1/float(self.cfg.model.k_poin_out))  # poincare ball上曲率暂时不变
        else:
            self.manifold_out = Lorentz(k=float(cfg.model.k_out))

        
        # manifold out是productmanifold
        if self.cfg.model.use_poincare:
            self.manifold = PoincareBall(c = 1/float(self.cfg.model.k_poin_out))
        else:
            self.manifold = Lorentz(k=float(cfg.model.k_out))
        # self.product_manifold = ProductManifold((Euclidean(), self.euc_channels), (self.manifold, self.hyp_channels))
        self.product_manifold = self.manifold
        # use double layer MLP instead of one single linear
        # pdb.set_trace()
        self.extra_feature_mlp = nn.Sequential(
            nn.Linear(self.cfg.model.lgcn_in_channels, self.latent_channels),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels),
            nn.LayerNorm(self.latent_channels)
        )
        self.extra_edge_feature_mlp = nn.Sequential(
            nn.Linear(self.cfg.model.lgcn_in_edge_channels, self.latent_channels),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels),
            nn.LayerNorm(self.latent_channels)
        )
        
        if cfg.model.use_poincare:
            self.hyper_graph_conv = PoincareGCN(cfg.model.k_poin_out, cfg)
        else:
            self.hyper_graph_conv = LGCN(cfg.model.k_in, cfg)
        # 如果只有欧式空间，那么就用GCN encoder

        # 这里选择使用transformer与否，还是只有GCN
        if cfg.model.transformer_encoder.Hypformer_use:
            # self.trans_conv_linear = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, self.in_channels,
            #                             self.latent_channels, cfg)
            self.trans_conv_linear = PoincareTransformer(cfg, self.manifold, self.in_channels, cfg.model.transformer_encoder.trans_num_layers, cfg.model.transformer_encoder.trans_num_heads, cfg.model.transformer_encoder.trans_dropout, 
                                                        cfg.model.transformer_encoder.max_seq_len, cfg.model.transformer_encoder.use_hyperbolic_attention, cfg.model.transformer_encoder.attention_type, cfg.model.transformer_encoder.attention_activation,
                                                        use_pe=cfg.model.transformer_encoder.use_pe)
        else:
            if cfg.model.use_poincare:
                self.trans_conv_linear = PoincareLinear(self.manifold, self.in_channels, self.latent_channels)
                self.layernorm = PoincareLayerNorm(self.manifold_out, self.latent_channels)
            else:
                self.trans_conv_linear = LorentzHypLinear(self.manifold_out, self.in_channels, self.latent_channels)
                self.layernorm = HypLayerNorm(self.manifold_out, self.latent_channels)

        if self.euc_channels==0 and cfg.model.transformer_decoder.Hypformer_use:
            self.trans_decoder = PoincareTransformer(cfg, self.manifold, self.hyp_channels, cfg.model.transformer_decoder.trans_num_layers, cfg.model.transformer_decoder.trans_num_heads, cfg.model.transformer_decoder.trans_dropout, 
                                                        cfg.model.transformer_decoder.max_seq_len, cfg.model.transformer_decoder.use_hyperbolic_attention, cfg.model.transformer_decoder.attention_type, cfg.model.transformer_decoder.attention_activation,
                                                        use_pe=cfg.model.transformer_decoder.use_pe)

        # if self.decoder_type == 'euc':
        #     self.decode_trans = nn.Linear(self.euc_channels, self.out_channels)
        #     self.decode_graph = nn.Linear(self.euc_channels, self.out_channels)
        # elif self.decoder_type == 'hyp':
        #     self.decode_graph = LorentzHypLinear(self.manifold_out, self.euc_channels, self.euc_channels)
        #     self.decode_trans = HypCLS(self.manifold_out, self.euc_channels, self.out_channels)
        # else:
        #     raise NotImplementedError
        
        # self.adj_decoder = AdjDecoder(self.manifold_out, self.euc_channels, self.euc_channels, self.out_channels)
        
        if self.use_VQVAE:
            if self.euc_channels > 0:
                self.euc_encoder = EucEncoder(self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                self.euc_codebook = VectorQuantize(self.euc_channels, self.cfg.model.codebook_size)
            if self.hyp_channels > 0:
                self.hyp_encoder = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels, self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                self.hyp_codebook = VectorQuantize(self.hyp_channels, self.cfg.model.codebook_size, vq_loss_weight=self.cfg.loss.lambda_vq_loss_weight, 
                commitment_weight=self.cfg.loss.lambda_commitment_weight, orthogonal_reg_weight=self.cfg.loss.lambda_orthogonal_reg_weight, 
                use_hyperbolic=cfg.model.use_poincare,manifold=self.manifold, kmeans_init=cfg.loss.use_kmeans, sample_codebook_temp=cfg.model.sample_codebook_temp)
        else:
            if self.use_VAE:
                if self.euc_channels > 0:
                    self.euc_encoder_mean = EucEncoder(self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                    self.euc_encoder_var = EucEncoder(self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                if self.hyp_channels > 0:
                    self.hyp_encoder_mean = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels, self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                    self.hyp_encoder_var = EucEncoder(self.manifold_out, self.latent_channels, self.latent_channels, self.hyp_channels, )
            else:
                if self.euc_channels > 0:
                    self.euc_encoder = EucEncoder(self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                if self.hyp_channels > 0:
                    self.hyp_encoder = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels, self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)


        self.decode_method = cfg.model.decode_method
        # decoder部分 - 交叉decode
        if self.euc_channels > 0:   
            self.euc2node_decoder = Euc2NodeDecoder(self.manifold_out,  self.euc_channels, self.euc_channels, cfg.model.node_classes, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
            self.euc2edge_decoder = Euc2EdgeDecoder(Euclidean(),  self.euc_channels, self.euc_channels, cfg.model.edge_classes, method=self.decode_method, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
        if self.hyp_channels > 0:
            self.hyp2node_decoder = Hyp2NodeDecoder(self.manifold_out,  self.hyp_channels, self.hyp_channels, cfg.model.node_classes, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)       
            self.hyp2edge_decoder = Hyp2EdgeDecoder(self.manifold_out,  self.hyp_channels, self.hyp_channels,cfg.model.edge_classes, method=self.decode_method, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
        

    def forward(self, dataset):
     
        raise NotImplementedError


    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def encode(self, x, adj, e, mask):
        # pdb.set_trace()
        x_dim = x.shape[-1]
        # check_on_manifold(self.manifold_in, x, "node input to hypformer")
        # pdb.set_trace()
        # add extra feature MLP layer, and extra edge feature MLP layer
        # pdb.set_trace()
        x = self.extra_feature_mlp(x)
        e = self.extra_edge_feature_mlp(e)


        x = self.hyper_graph_conv.encode(x, adj, e, x_manifold='euc')
        x_norm = torch.norm(x, p=2, dim=-1)
        # print seperator
        print(f"--------------------------------")
        # can we wandb log this?
        # name it "after GNN x_norm norm min"
        epoch = wandb.run.summary.get("epoch", 0)
        to_log = {
            "epoch": epoch,
            "train/after_GNN_x_norm_min": x_norm.min().item(),
            "train/after_GNN_x_norm_max": x_norm.max().item(),
            "train/after_GNN_x_norm_mean": x_norm.mean().item()
        }
        wandb.log(to_log)
        print(f"After GNN x_norm norm min: {x_norm.min().item():.4f}, After GNN norm max: {x_norm.max().item():.4f}, After GNN  norm mean: {x_norm.mean().item():.4f}")
        # Print parameters in hyper_graph_conv
        # print("Parameters in hyper_graph_conv:")
        # for name, param in self.hyper_graph_conv.named_parameters():
        #     print(f"  {name}: value={param.mean()}")
        # TODO: 添加判断，是否使用transformer层
        # check_on_manifold(self.manifold_in, x, "node after GCN")
        # z = self.trans_conv(x, mask)
        # pdb.set_trace()
        # print x's abs mean
        # print("x's abs mean: ", torch.abs(x).mean())
        if self.cfg.model.transformer_encoder.Hypformer_use:
            z = self.trans_conv_linear(x, mask)
            z_norm = torch.norm(z, p=2, dim=-1)
            print(f"After Transformer x_norm norm min: {z_norm.min().item():.4f}, After Transformer norm max: {z_norm.max().item():.4f}, After Transformer  norm mean: {z_norm.mean().item():.4f}")
            # can we wandb log this?
            to_log.update({
                "train/after_Transformer_x_norm_min": z_norm.min().item(),
                "train/after_Transformer_x_norm_max": z_norm.max().item(),
                "train/after_Transformer_x_norm_mean": z_norm.mean().item()
            })
        else:
            z = self.trans_conv_linear(x)
        #print("z's abs mean: ", torch.abs(z).mean())
        # check_on_manifold(self.manifold_hidden, z, "node after Transformer")
        if self.use_VQVAE:            
            if self.euc_channels > 0:
                node_feat = self.euc_encoder(z)
            else:
                node_feat = None
            if self.hyp_channels > 0:
                edge_feat = self.hyp_encoder(z)
                # edge_feat = z
                edge_feat_norm = torch.norm(edge_feat, p=2, dim=-1)
                print(f"After HypEncoder edge_feat_norm norm min: {edge_feat_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_norm.mean().item():.4f}")
                to_log.update({
                    "train/after_HypEncoder_edge_feat_norm_min": edge_feat_norm.min().item(),
                    "train/after_HypEncoder_edge_feat_norm_max": edge_feat_norm.max().item(),
                    "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_norm.mean().item()
                })
            else:
                edge_feat = None
            wandb.log(to_log)
            return node_feat, edge_feat, z
        
        else:
            if not self.cfg.model.transformer_encoder.Hypformer_use:
                # # print z's abs mean
                z = self.layernorm(z)
            if not self.use_VAE:
                if self.euc_channels > 0:
                    node_feat = self.euc_encoder(z)
                else:
                    node_feat = None
                if self.hyp_channels > 0:
                    edge_feat = self.hyp_encoder(z)
                else:
                    edge_feat = None
                edge_feat_norm = torch.norm(edge_feat, p=2, dim=-1)
                print(f"After HypEncoder edge_feat_norm norm min: {edge_feat_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_norm.mean().item():.4f}")
                to_log.update({
                    "train/after_HypEncoder_edge_feat_norm_min": edge_feat_norm.min().item(),
                    "train/after_HypEncoder_edge_feat_norm_max": edge_feat_norm.max().item(),
                    "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_norm.mean().item()
                })
                wandb.log(to_log)
                return node_feat, edge_feat, z
            
            else:
                if self.euc_channels > 0:
                    node_feat_mean = self.euc_encoder_mean(z)
                    # should layernorm inside the euclidean space
                    node_feat_var = self.euc_encoder_var(z)
                else:
                    node_feat_mean = None
                    node_feat_var = None
                if self.hyp_channels > 0:
                    edge_feat_mean = self.hyp_encoder_mean(z)
                    edge_feat_var = self.hyp_encoder_var(z)
                    edge_feat_mean_norm = torch.norm(edge_feat_mean, p=2, dim=-1)
                    print(f"After HypEncoder edge_feat_norm norm min: {edge_feat_mean_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_mean_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_mean_norm.mean().item():.4f}")
                    to_log.update({
                        "train/after_HypEncoder_edge_feat_norm_min": edge_feat_mean_norm.min().item(),
                        "train/after_HypEncoder_edge_feat_norm_max": edge_feat_mean_norm.max().item(),
                        "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_mean_norm.mean().item()
                    })
                    edge_feat_var_norm = torch.norm(edge_feat_var, p=2, dim=-1)
                    print(f"After HypEncoder edge_feat_var_norm norm min: {edge_feat_var_norm.min().item():.4f}, After HypEncoder edge_feat_var_norm norm max: {edge_feat_var_norm.max().item():.4f}, After HypEncoder edge_feat_var_norm norm mean: {edge_feat_var_norm.mean().item():.4f}")
                    to_log.update({
                        "train/after_HypEncoder_edge_feat_var_norm_min": edge_feat_var_norm.min().item(),
                        "train/after_HypEncoder_edge_feat_var_norm_max": edge_feat_var_norm.max().item(),
                        "train/after_HypEncoder_edge_feat_var_norm_mean": edge_feat_var_norm.mean().item()
                    })
                else:
                    edge_feat_mean = None
                    edge_feat_var = None
                wandb.log(to_log)
                return node_feat_mean, node_feat_var, edge_feat_mean, edge_feat_var, z
            

    def codebook(self, feat, codebook_type = "euc", node_mask=None):
        if codebook_type == "euc" and self.euc_channels > 0:
            quantize, vq_ind, vq_loss, perplexity = self.euc_codebook(feat, node_mask)
        elif codebook_type == "hyp" and self.hyp_channels > 0:
            quantize, vq_ind, vq_loss, perplexity = self.hyp_codebook(feat, node_mask)
        return quantize, vq_ind, vq_loss, perplexity
    

    def reparameterize(self, mean, log_std):
        try:
            # Add numerical stability checks for mean
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                print("Warning: NaN or Inf detected in mean during reparameterization")
                mean = torch.where(
                    torch.isnan(mean) | torch.isinf(mean),
                    torch.zeros_like(mean),
                    mean
                )
                
            # Add numerical stability checks for log_std
            if torch.isnan(log_std).any() or torch.isinf(log_std).any():
                print("Warning: NaN or Inf detected in log_std during reparameterization")
                log_std = torch.where(
                    torch.isnan(log_std) | torch.isinf(log_std),
                    torch.ones_like(log_std) * -5,  # Small but valid log std
                    log_std
                )
                
            # Clamp log_std to avoid extreme values
            log_std = torch.clamp(log_std, min=-20, max=2)
            
            # Standard normal sampling with numerical stability
            std = torch.exp(0.5 * log_std)
            
            # Handle potential infs or nans in std
            if torch.isnan(std).any() or torch.isinf(std).any():
                print("Warning: NaN or Inf detected in std after exp")
                std = torch.where(
                    torch.isnan(std) | torch.isinf(std),
                    torch.ones_like(std) * 0.01,  # Small but valid std
                    std
                )
                
            # Sample from standard normal and scale
            eps = torch.randn_like(std)
            z = mean + eps * std
            
            # Final safety check on output
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("Warning: NaN or Inf detected in reparameterization output")
                z = torch.where(
                    torch.isnan(z) | torch.isinf(z),
                    mean.detach(),  # Fall back to mean if sampling produces NaN/Inf
                    z
                )
                
            return z
            
        except Exception as e:
            print(f"Error in reparameterize: {e}")
            # In case of unexpected error, return mean as fallback
            return mean.detach()
    

    def decode(self, z, adj, mask=None):
        if self.euc_channels==0 and self.cfg.model.transformer_decoder.Hypformer_use:
            z = self.trans_decoder(z, mask)
        if self.euc_channels > 0:
            z_euc = z[..., :self.euc_channels]
            euc2node_feat = self.euc2node_decoder(z_euc)
            euc2edge_feat = self.euc2edge_decoder(z_euc)
        else:
            z_euc = None
            euc2node_feat = None
            euc2edge_feat = None

        if self.hyp_channels > 0:
            z_hyp = z[..., self.euc_channels:]
            hyp2node_feat = self.hyp2node_decoder(z_hyp)
            hyp2edge_feat = self.hyp2edge_decoder(z_hyp)
        else:
            z_hyp = None
            hyp2node_feat = None
            hyp2edge_feat = None
        
        # pdb.set_trace()
        return euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, adj