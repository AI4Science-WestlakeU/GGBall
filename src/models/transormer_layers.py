import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.manifolds.lorentz import Lorentz

# DiT transformer
class DiTAttention(nn.Module):
    def __init__(self, cfg, product_manifold):
        super().__init__()
        self.manifold_in = Lorentz(k=float(cfg.model.k_out))
        self.manifold_hidden = Lorentz(k=float(cfg.model.k_out))
        self.manifold_out = Lorentz(k=float(cfg.model.k_out))
        self.product_manifold = product_manifold
        hidden_size = cfg.model.hidden_channels + cfg.model.edge_dim
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.trans_conv = nn.ModuleList([
            DiTBlock(cfg, self.product_manifold) for _ in range(cfg.model.trans_num_layers)
        ])
        
        self.final_decode = torch.nn.Linear(hidden_size, hidden_size)
        
        # self.time_embedder = nn.Linear(1, self.hidden_channels)
    def forward(self, t, x, mask=None):
        z = x + self.positional_encoding(x)
        t = self.t_embedder(t)
        for block in self.trans_conv:
            z = block(z, t, mask)                  
        # z = self.trans_conv(x, t, mask=mask)
        # z = self.final_decode(x)
        # vector_field = self.product_manifold.proju(x, z)
        z = self.product_manifold.projx(z)
        # vector_field = self.product_manifold.logmap(x, z)
        vector_field = z
        if mask is not None:
            mask = mask.unsqueeze(-1).bool()
            vector_field = vector_field.masked_fill(~mask, 0)
        # vector_field = self.product_manifold.logmap(x, z)
        return vector_field
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
    
      
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    
    
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

 
class BertSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_prob):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: None,
    ):
        mixed_query_layer = self.query(hidden_states)


        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # our attention mask is 1 for positions we want to attend and 0 for ones we want to ignore
            # let the mask have -10000.0 where we want to ignore
            attention_mask = torch.einsum('bi,bj->bij', attention_mask, attention_mask).unsqueeze(1)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -10000.0)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer
    
    

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, cfg, manifold):
        super().__init__()
        hidden_size = cfg.model.hidden_channels + cfg.model.edge_dim
        num_heads = cfg.model.trans_num_heads
        dropout = cfg.model.trans_dropout
        mlp_ratio= 4.0
        self.manifold = manifold
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BertSelfAttention(num_attention_heads=num_heads, hidden_size=hidden_size, dropout_prob=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = self.manifold.projx(x)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.manifold.projx(x)
        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        if t.ndim == 0:
            args = t[None].float() * freqs[None]
        else:
            args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb