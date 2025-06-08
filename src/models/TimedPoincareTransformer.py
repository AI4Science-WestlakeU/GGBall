import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from scipy.special import beta
import pdb
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.linear import PoincareLinear
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.learned_positional_embedding import PoincareLearnedPositionalEmbedding
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.multinomial_logistic_regression import UnidirectionalPoincareMLR
import time



class PoincareSplitConcat(nn.Module):
    """Handles Poincaré β-split and β-concatenation operations"""
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        
    def split(self, x: torch.Tensor, split_sizes: List[int]) -> List[torch.Tensor]:
        """
        Poincaré β-split operation
        Args:
            x: Input tensor in Poincaré ball
            split_sizes: List of sizes for each split
        Returns:
            List of split tensors in Poincaré ball
        """
        # Map to tangent space
        x_tangent = self.manifold.logmap0(x)
        
        # Split in tangent space
        splits = torch.split(x_tangent, split_sizes, dim=-1)
        
        # Compute beta coefficients
        n = x.size(-1)
        beta_n = beta(n/2, 1/2)
        
        # Project each split back to Poincaré ball with proper scaling
        results = []
        for split in splits:
            ni = split.size(-1)
            beta_ni = beta(ni/2, 1/2)
            # Scale and project back
            scaled = split * (beta_ni / beta_n)
            results.append(self.manifold.expmap0(scaled))
            
        return results
    
    def concat(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Poincaré β-concatenation operation
        Args:
            tensors: List of tensors in Poincaré ball
        Returns:
            Concatenated tensor in Poincaré ball
        """
        # Map each tensor to tangent space
        tangents = [self.manifold.logmap0(x) for x in tensors]
        
        # Compute total dimension
        n = sum(x.size(-1) for x in tensors)
        beta_n = beta(n/2, 1/2)
        
        # Scale each tangent vector
        scaled_tangents = []
        for x in tangents:
            ni = x.size(-1)
            beta_ni = beta(ni/2, 1/2)
            scaled_tangents.append(x * (beta_n / beta_ni))
        
        # Concatenate in tangent space
        concat_tangent = torch.cat(scaled_tangents, dim=-1)
        
        # Project back to Poincaré ball
        return self.manifold.expmap0(concat_tangent)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations in Poincaré ball.
    Adapted from DiT implementation.
    """
    def __init__(self, hidden_size, manifold, frequency_embedding_size=256):
        super().__init__()
        self.manifold = manifold
        # First map to Euclidean space with MLP
        self.euclidean_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Then project to Poincaré ball
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
        t_emb = self.euclidean_mlp(t_freq)
        # Map to Poincaré ball
        # t_norm = torch.norm(t_emb, dim=-1, keepdim=True).clamp(min=1e-8)
        # t_normalized = t_emb / t_norm
        # # Scale norm to be within ball
        # factor = torch.rand_like(t_norm) * 0.9  # Scale to ensure it's inside the ball
        # t_poincare = t_normalized * factor
        # return self.manifold.projx(t_poincare)  # Ensure it's on the manifold

        return t_emb


def modulate(x, shift, scale):
    """Apply modulation to layer norm output (AdaLN)"""
    return x * (1 + scale) + shift


class TimedPoincareTransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, manifold, use_hyperbolic_attention=False, 
                 attention_type='distance', attention_activation='exp'):
        super(TimedPoincareTransformerLayer, self).__init__()
        self.manifold = manifold
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        
        # Self-attention
        self.self_attn = TimedPoincareMultiheadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            manifold=manifold,
            use_hyperbolic_attention=use_hyperbolic_attention,
            attention_type=attention_type,
            attention_activation=attention_activation
        )
        
        # Feed-forward network
        self.ffn = PoincareFeedForward(
            model_dim=model_dim,
            ffn_dim=model_dim * 4,
            dropout=dropout,
            manifold=manifold
        )
        
        # Layer norms (without affine parameters, as they'll be modulated by time)
        self.attn_layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.ffn_layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            # PoincareLinear(manifold, model_dim, 6 * model_dim, bias=True)
            nn.Linear(model_dim, 6 * model_dim, bias=True)
        )
        
    def forward(self, x, t_emb, mask=None):
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        # TODO：不同位置的x的shift_msa应该是不同的？
        # Self-attention block with AdaLN
        residual = x
        # Map to tangent space for layer norm
        x_tangent = self.manifold.logmap0(x)
        x_norm = self.attn_layer_norm(x_tangent)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        x_norm_poincare = self.manifold.expmap0(x_modulated)
        
        # Apply attention
        attn_output = self.self_attn(x_norm_poincare, mask=mask)
        # attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        # Apply gated residual connection
        attn_output = self.manifold.logmap0(attn_output)
        attn_output = gate_msa * attn_output

        x = self.manifold.mobius_add(residual, self.manifold.expmap0(attn_output))
        # x = self.manifold.expmap0(x)
        # x = self.manifold.projx(x)
        
        # Feed-forward block with AdaLN
        residual = x
        
        # Map to tangent space for layer norm
        x_tangent = self.manifold.logmap0(x)
        x_norm = self.ffn_layer_norm(x_tangent)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)
        x_norm_poincare = self.manifold.expmap0(x_modulated)
        
        # Apply feed-forward
        ffn_output = self.ffn(x_norm_poincare)
        # ffn_output = F.dropout(ffn_output, p=self.dropout, training=self.training)
        
        # Apply gated residual connection
        ffn_output = self.manifold.logmap0(ffn_output)
        ffn_output = gate_mlp * ffn_output
        ffn_output = self.manifold.expmap0(ffn_output)
        x = self.manifold.mobius_add(residual, ffn_output)
        # x = self.manifold.expmap0(x)
        # x = self.manifold.projx(x)

        return x


class TimedPoincareMultiheadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, manifold, use_hyperbolic_attention=False,
                 attention_type='distance', attention_activation='exp'):
        super(TimedPoincareMultiheadAttention, self).__init__()
        self.manifold = manifold
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        
        # Query, key, value projections
        self.q_proj = PoincareLinear(manifold, model_dim, model_dim, bias=True)
        self.k_proj = PoincareLinear(manifold, model_dim, model_dim, bias=True)
        self.v_proj = PoincareLinear(manifold, model_dim, model_dim, bias=True)
        
        # Output projection
        self.out_proj = PoincareLinear(manifold, model_dim, model_dim, bias=True)
        
        # Split and concat operations
        self.split_concat = PoincareSplitConcat(manifold)
        
        # Hyperbolic attention parameters
        if use_hyperbolic_attention:
            self.tau = nn.Parameter(torch.ones(1))  # inverse temperature
            self.gamma = nn.Parameter(torch.zeros(1))  # bias parameter
        
    def forward(self, x, mask=None):
        # import pdb; pdb.set_trace()
        batch_size, seq_len, _ = x.size()
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split into heads using Poincaré β-split
        split_size = [self.head_dim] * self.num_heads
        q_heads = self.split_concat.split(q, split_size)
        k_heads = self.split_concat.split(k, split_size)
        v_heads = self.split_concat.split(v, split_size)
        
        # Process each head
        attn_outputs = []

        # Stack instead of appending in a loop
        q_stacked = torch.stack(q_heads, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
        k_stacked = torch.stack(k_heads, dim=1)
        v_stacked = torch.stack(v_heads, dim=1)

        if self.use_hyperbolic_attention:
            # Compute hyperbolic attention scores
            attn_weights = self._hyperbolic_attention_weights(q_stacked, k_stacked)
        else:
            # Compute regular attention scores
            attn_weights = self._regular_attention_weights(q_stacked, k_stacked)
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)

        if self.attention_activation == 'exp':
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attention_activation == 'sigmoid':
            attn_weights = torch.sigmoid(attn_weights)
        elif self.attention_activation == 'identity':
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        
        # Apply attention weights to values
        if self.use_hyperbolic_attention:
            attn_output = self._apply_hyperbolic_attention(attn_weights, v_stacked)
        else:
            attn_output = self._apply_regular_attention(attn_weights, v_stacked)

        output = self.split_concat.concat(attn_output)

        output = self.out_proj(output)

        return output
    
    def _hyperbolic_attention_weights(self, q, k):
        """Compute attention weights using hyperbolic distance or inner product"""
        if self.attention_type == 'distance':
            # Compute negative hyperbolic distance (closer = higher score)
            # distances = -self.manifold.dist(q.unsqueeze(2), k.unsqueeze(1))
            distances = -self.manifold.dist(q.unsqueeze(3), k.unsqueeze(2))
            return self.tau * distances - self.gamma
        else:  # inner_product
            # Map to tangent space and compute inner product
            q_tangent = self.manifold.logmap0(q)
            k_tangent = self.manifold.logmap0(k)
            return torch.matmul(q_tangent, k_tangent.transpose(-2, -1))
    
    def _regular_attention_weights(self, q, k):
        """Compute regular attention scores"""
        # Map to tangent space for regular attention
        q_tangent = self.manifold.logmap0(q)
        k_tangent = self.manifold.logmap0(k)
        return torch.matmul(q_tangent, k_tangent.transpose(-2, -1)) / (self.head_dim ** 0.5)
    
    def _apply_hyperbolic_attention(self, attn_weights, v):
        """Apply attention weights using Möbius gyromidpoint"""
        batch_size, num_heads, seq_len, _ = v.size()
        # # attention [b, m, m]
        # # v [b, m, d]
        # # Apply Möbius gyromidpoint for each position in each batch
        # import pdb; pdb.set_trace()
        # # For gyromidpoint calculation, we still need to loop, but we can process all heads at once
        # results = []
        # for h in range(num_heads):
        #     head_results = []
        #     for b in range(batch_size):
        #         # Process each sequence position for this batch and head
        #         batch_output = []
        #         for i in range(seq_len):
        #             weights = attn_weights[b, h, i]  # [seq_len]
        #             weighted_sum = self.manifold.weighted_midpoint(v[b, h], weights) # [N, D] * [N] -> [D]
        #             batch_output.append(weighted_sum)
        #         head_results.append(torch.stack(batch_output)) # [N, D]
        #     head_result = torch.stack(head_results)  # [batch_size, seq_len, head_dim]
        #     results.append(head_result)
        
        # output_1 = torch.stack(results)

        head_results = self.manifold.weighted_midpoint_bmm(
            xs=v,                 # [batch_size, num_heads, seq_len, head_dim]
            weights=attn_weights, # [batch_size, num_heads, seq_len, seq_len]
            lincomb=False,
            project=True
        )  # [batch_size, num_heads, seq_len, head_dim]

        #
        result_list = list(torch.unbind(head_results, dim=1))

        return result_list # -> [b, m, d]
    
    def _apply_regular_attention(self, attn_weights, v):
        """Apply regular attention weights"""
        # For regular attention, we compute in tangent space and then map back
        v_tangent = self.manifold.logmap0(v)
        attention_output = torch.bmm(attn_weights, v_tangent)
        return self.manifold.expmap0(attention_output)


class PoincareFeedForward(nn.Module):
    def __init__(self, model_dim, ffn_dim, dropout, manifold):
        super(PoincareFeedForward, self).__init__()
        self.manifold = manifold
        
        # Two-layer feed-forward network
        self.linear1 = PoincareLinear(manifold, model_dim, ffn_dim, bias=True)
        self.linear2 = PoincareLinear(manifold, ffn_dim, model_dim, bias=True)
        self.dropout = dropout
        
    def forward(self, x):
        # First linear layer with ReLU-like activation
        x = self.linear1(x)
        
        # Apply hyperbolic ReLU (approximated by mapping to tangent space, applying ReLU, and mapping back)
        # x_tangent = self.manifold.logmap0(x)
        # x_tangent_relu = F.relu(x_tangent)
        # x = self.manifold.expmap0(x_tangent_relu)
        
        # x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x


class TimedPoincareTransformer(nn.Module):
    def __init__(self, cfg, manifold, in_channels, num_layers, num_heads, dropout, max_seq_len, use_hyperbolic_attention, attention_type, attention_activation):
        super(TimedPoincareTransformer, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.model_dim = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.model_dim // self.num_heads
        self.max_seq_len = max_seq_len
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type  # 'distance' or 'inner_product'
        self.attention_activation = attention_activation  # 'exp', 'sigmoid', or 'identity'

        # Time embedding
        self.t_embedder = TimestepEmbedder(self.model_dim, manifold)
        
        # Positional embedding
        self.pos_embedding = PoincareLearnedPositionalEmbedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=self.model_dim,
            padding_idx=0,
            ball=self.manifold
            # requires_grad=True
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TimedPoincareTransformerLayer(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                manifold=self.manifold,
                use_hyperbolic_attention=self.use_hyperbolic_attention,
                attention_type=self.attention_type,
                attention_activation=self.attention_activation
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = PoincareLinear(
            manifold=self.manifold,
            in_dim=self.model_dim,
            out_dim=self.model_dim,
            bias=True
        )
        
    def forward(self, t, x, mask=None):
        """
        Args:
            t: Time tensor of shape [batch_size]
            x: Input tensor of shape [batch_size, seq_len, model_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, model_dim]
        """
        # Get time embedding

        t_emb = self.t_embedder(t)
        
        # Get positions
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(positions)
        
        # Mobius addition of input and positional embeddings
        x = self.manifold.mobius_add(x, pos_emb)
        
        # Apply transformer layers with time embedding
        for layer in self.layers:
            x = layer(x, t_emb, mask)
        
        # Apply output projection
        x = self.output_projection(x)
        
        return x 