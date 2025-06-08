import math
from typing import List, Optional

import torch
import torch.nn as nn
from scipy.special import beta
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt.manifolds.stereographic import PoincareBall

from geoopt_plusplus.manifolds.stereographic.math import (
    _mobius_add,
    _mobius_scalar_mul,
    _project,
    weighted_midpoint
)

from ..modules.multinomial_logistic_regression import unidirectional_poincare_mlr


class PoincareLinear(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, bias=True, out_split=1, gain=1.):
        super(PoincareLinear, self).__init__()
        gain = 1. ###
        self.ball = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_split = out_split
        weight = torch.empty(in_dim, out_dim).normal_( 
            mean=0, std=(2 * self.in_dim * self.out_dim / out_split) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.out_dim / out_split / 2, 1 / 2)
        self.beta_n = beta(self.out_dim / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        x = poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c,
            # out_split=self.out_split)
            out_split=1)
        if self.out_split > 1:
            size = x.size()
            x = self.ball.logmap0(x).contiguous().view(*size[:-1], self.out_split, size[-1] // self.out_split)
            x = self.ball.expmap0(x * self.beta_ni / self.beta_n)
        return x

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, out_split={}, bias={}'.format(
            self.in_dim, self.out_dim, self.out_split, self.bias.requires_grad
        )


class PoincareConcatLinear(nn.Module):
    def __init__(self, in_stacks, in_dim, out_dim, bias=True, ball=None, gain=1.):
        super().__init__()
        gain = 1. ###
        self.ball = ball
        self.in_stacks = in_stacks
        self.in_dim = in_dim
        self.out_dim = out_dim
        weight = torch.empty(in_stacks * in_dim, out_dim).normal_( 
            mean=0, std=1. / (2 * self.in_dim * in_stacks * self.out_dim) ** 0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.in_dim / 2, 1 / 2)
        self.beta_n = beta(self.in_dim * self.in_stacks / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        size = x.size()
        x = self.ball.logmap0(x).contiguous().view(*size[:-2], self.in_stacks * self.in_dim)
        x = self.ball.expmap0(x * self.beta_n / self.beta_ni)
        return poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c)
    
    def extra_repr(self):
        return (f'in_stacks={self.in_stacks},'
        f' in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias.requires_grad}')



# @torch.jit.script
def poincare_linear(x, weight_g, weight_v, bias, c, out_split : int = 1):
    # Ensure weight_v is properly normalized
    weight_v_norm = weight_v.norm(dim=0).clamp_min(1e-15)
    weight_v_normalized = weight_v / weight_v_norm
    
    # Apply numerical stability to parameters
    rc = c.sqrt().clamp_min(1e-15)
    
    # Apply MLR with numerical safeguards
    try:
        x = unidirectional_poincare_mlr(x, weight_g, weight_v_normalized, bias, c)
        
        # Check for NaN/Inf values after MLR
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected after unidirectional_poincare_mlr")
            # Apply safer fallback computation or reset problematic values
            x = torch.where(
                torch.isnan(x) | torch.isinf(x),
                torch.zeros_like(x),
                x
            )
        
        # Apply sinh transformation with numerical stability
        x_scaled = (rc * x).clamp(-20, 20)  # Prevent extreme values
        x = x_scaled.sinh() / rc
        
        # Check again for NaN/Inf values
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected after sinh transformation")
            x = torch.where(
                torch.isnan(x) | torch.isinf(x),
                torch.zeros_like(x),
                x
            )
        
        if out_split > 1:
            size = x.size()
            x = x.view(*size[:-1], out_split, size[-1] // out_split)
        
        # Apply projection with numerical stability
        x_norm_squared = x.pow(2).sum(dim=-1, keepdim=True).clamp(0, 1e15)
        denominator = (1 + (1 + c * x_norm_squared).sqrt()).clamp_min(1e-15)
        result = x / denominator
        
        # Final safety check before returning
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf detected in final result")
            result = torch.where(
                torch.isnan(result) | torch.isinf(result),
                torch.zeros_like(result),
                result
            )
        
        return result
        
    except Exception as e:
        print(f"Error in poincare_linear: {e}")
        # Return a safe fallback value - a zero tensor of the expected shape
        if out_split > 1:
            size = x.size()
            return torch.zeros(*size[:-1], out_split, size[-1] // out_split, device=x.device, dtype=x.dtype)
        else:
            return torch.zeros_like(x)


def main():
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 初始化参数
    manifold = PoincareBall(c=1.0)  # 创建双曲空间，曲率c=1.0
    in_dim = 4      # 输入维度
    out_dim = 3     # 输出维度
    batch_size = 2  # 批次大小
    
    # 创建模型
    model = PoincareLinear(
        manifold=manifold,
        in_dim=in_dim,
        out_dim=out_dim,
        bias=True
    )
    
    # 创建输入数据
    x = torch.randn(batch_size, 1, in_dim)  # 创建随机输入
    x = manifold.expmap0(x)  # 将输入映射到双曲空间
    
    print("=== PoincareLinear 测试 ===")
    print(f"模型结构:\n{model}")
    print(f"\n输入形状: {x.shape}")
    print(f"输入数据:\n{x}")
    
    # 前向传播
    output = model(x)
    
    print(f"\n输出形状: {output.shape}")
    print(f"输出数据:\n{output}")
    
    # 验证输出是否在双曲空间上
    is_on_manifold = manifold.check_point_on_manifold(output)
    print(f"\n输出是否在双曲空间上: {is_on_manifold}")
    
    # 打印一些模型参数信息
    print("\n模型参数信息:")
    print(f"weight_v 形状: {model.weight_v.shape}")
    print(f"weight_g 形状: {model.weight_g.shape}")
    print(f"bias 形状: {model.bias.shape}")

if __name__ == "__main__":
    main()