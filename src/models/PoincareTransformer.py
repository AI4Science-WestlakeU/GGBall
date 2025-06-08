## poincare linear adapted from hyperbolic neural network++
## activation: no need 

## LayNorm: logmap - ln - expmap
## Add: x mobius_add f(x)
# attention: adapted from hyperbolic neural network++
## initialization

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from scipy.special import beta
import pdb
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.linear import PoincareLinear
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.learned_positional_embedding import PoincareLearnedPositionalEmbedding
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.modules.multinomial_logistic_regression import UnidirectionalPoincareMLR


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

class PoincareTransformerCausal(nn.Module):
    def __init__(self, cfg, manifold, in_channels, num_layers, num_heads, dropout, max_seq_len, use_hyperbolic_attention, attention_type, attention_activation, use_pe=True):
        super(PoincareTransformerCausal, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.model_dim = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.model_dim // self.num_heads
        self.max_seq_len = max_seq_len
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        self.use_pe = use_pe

        if self.use_pe:
            # Positional embedding
            self.pos_embedding = PoincareLearnedPositionalEmbedding(
                num_embeddings=self.max_seq_len,
                embedding_dim=self.model_dim,
                padding_idx=0,
                ball=self.manifold  
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PoincareTransformerLayer(
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
    
    def generate_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).bool()  # [1, T, T]
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, model_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, model_dim]
        """
        # Get positions
        # pdb.set_trace()
        if self.use_pe:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            # Add positional embeddings
            pos_emb = self.pos_embedding(positions)
            # Mobius addition of input and positional embeddings
            x = self.manifold.mobius_add(x, pos_emb)
        

        B, N, D = x.shape
        # Generate autoregressive (causal) mask: [1, T, T]
        causal_mask = self.generate_causal_mask(N, x.device)  # [1, N, N]

        # Combine with padding mask (if any)
        if mask is not None:  # mask: [B, T]
            # padding mask: [B, 1, T]
            pad_mask = mask.unsqueeze(1)
            # causal mask: [1, N, N]
            # final mask: [B, T, T]
            combined_mask = pad_mask & causal_mask
        else:
            combined_mask = causal_mask.expand(B, -1, -1)  # [B, T, T]


        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, combined_mask)
        
        # Apply output projection
        x = self.output_projection(x)
        
        return x
    

class PoincareTransformer(nn.Module):
    def __init__(self, cfg, manifold, in_channels, num_layers, num_heads, dropout, max_seq_len, use_hyperbolic_attention, attention_type, attention_activation, use_pe=True):
        super(PoincareTransformer, self).__init__()
        self.cfg = cfg
        self.manifold = manifold
        self.model_dim = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.model_dim // self.num_heads
        self.max_seq_len = max_seq_len
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        self.use_pe = use_pe

        if self.use_pe:
            # Positional embedding
            self.pos_embedding = PoincareLearnedPositionalEmbedding(
                num_embeddings=self.max_seq_len,
            embedding_dim=self.model_dim,
            padding_idx=0,
            ball=self.manifold
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PoincareTransformerLayer(
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
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, model_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, model_dim]
        """
        # Get positions
        if self.use_pe:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            # Add positional embeddings
            pos_emb = self.pos_embedding(positions)
            # Mobius addition of input and positional embeddings
            x = self.manifold.mobius_add(x, pos_emb)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply output projection
        x = self.output_projection(x)
        
        return x


class PoincareTransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, manifold, use_hyperbolic_attention=False, 
                 attention_type='distance', attention_activation='exp'):
        super(PoincareTransformerLayer, self).__init__()
        self.manifold = manifold
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        
        # Self-attention
        self.self_attn = PoincareMultiheadAttention(
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
        
        # Layer norms
        self.attn_layer_norm = PoincareLayerNorm(model_dim, manifold)
        self.ffn_layer_norm = PoincareLayerNorm(model_dim, manifold)
        
    def forward(self, x, mask=None):
        # Self-attention block
        residual = x
        x = self.attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.manifold.mobius_add(residual, x)
        
        # Feed-forward block
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.manifold.mobius_add(residual, x)
        
        return x


class PoincareMultiheadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, manifold, use_hyperbolic_attention=False,
                 attention_type='distance', attention_activation='exp'):
        super(PoincareMultiheadAttention, self).__init__()
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
        # pdb.set_trace()
        if mask is not None:
            if mask.dim() == 2:
                # padding mask: [B, T] -> [B, 1, 1, T]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                # mask = mask.expand(-1, self.num_heads, attn_weights.size(-2), -1)  # [B, H, T, T]
            elif mask.dim() == 3:
                # causal mask or combined mask: [B, T, T] -> [B, 1, T, T]
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
                mask = mask.expand(-1, self.num_heads, -1, -1)  # [B, H, T, T]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

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
            
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output
    
    def _hyperbolic_attention_weights(self, q, k):
        """Compute attention weights using hyperbolic distance or inner product"""
        if self.attention_type == 'distance':
            # Compute negative hyperbolic distance (closer = higher score)
            distances = -self.manifold.dist(q.unsqueeze(3), k.unsqueeze(2))
            return self.tau * distances - self.gamma
        else:  # inner_product
            # Map to tangent space and compute inner product
            q_tangent = self.manifold.logmap0(q)
            k_tangent = self.manifold.logmap0(k)
            return torch.matmul(q_tangent, k_tangent.transpose(-2, -1))
    
    def _regular_attention_weights(self, q, k):
        """Compute regular attention scores"""
        return torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
    
    def _apply_hyperbolic_attention(self, attn_weights, v):
        """Apply attention weights using Möbius gyromidpoint"""
        head_results = self.manifold.weighted_midpoint_bmm(
            xs=v,                 # [batch_size, num_heads, seq_len, head_dim]
            weights=attn_weights, # [batch_size, num_heads, seq_len, seq_len]
            lincomb=False,
            project=True
        )  # [batch_size, num_heads, seq_len, head_dim]

        result_list = list(torch.unbind(head_results, dim=1))
        return result_list # -> [b, m, d]
    
    def _apply_regular_attention(self, attn_weights, v):
        """Apply regular attention weights"""
        return torch.matmul(attn_weights, v)


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
        x_tangent = self.manifold.logmap0(x)
        x_tangent_relu = F.relu(x_tangent)
        x = self.manifold.expmap0(x_tangent_relu)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x


class PoincareLayerNorm(nn.Module):
    def __init__(self, normalized_shape, manifold):
        super(PoincareLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.ln.weight.data.fill_(0.1)
        self.ln.bias.data.fill_(0)
        self.manifold = manifold


    def forward(self, x):
        # Map to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        
        # Apply Euclidean layer norm
        x_tangent_normalized = self.ln(x_tangent)
        
        # Map back to hyperbolic space
        x_normalized = self.manifold.expmap0(x_tangent_normalized)
        
        return x_normalized