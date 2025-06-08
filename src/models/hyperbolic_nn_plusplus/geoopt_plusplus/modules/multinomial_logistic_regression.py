import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter, ManifoldTensor
from geoopt_plusplus.manifolds.stereographic.math import arsinh, artanh
from geoopt_plusplus.utils import *


class UnidirectionalPoincareMLR(nn.Module):
    __constants__ = ['feat_dim', 'num_outcome']

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super(UnidirectionalPoincareMLR, self).__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome
        weight = torch.empty(feat_dim, num_outcome).normal_( 
            mean=0, std=(self.feat_dim) ** -0.5 / self.ball.c.data.sqrt())
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return unidirectional_poincare_mlr(
            x, self.weight_g, self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), self.bias, self.ball.c)
    
    def extra_repr(self):
        return 'feat_dim={}, num_outcome={}, bias={}'.format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )

class WeightTiedUnidirectionalPoincareMLR(nn.Module):
    __constants__ = ['feat_dim', 'num_outcome']

    def __init__(self, feat_dim, num_outcome, bias=True, ball=None):
        super().__init__()
        self.ball = ball
        self.feat_dim = feat_dim
        self.num_outcome = num_outcome
        self.bias = nn.Parameter(torch.empty(num_outcome), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x, weight):
        weight = weight.t()
        weight_g = weight.norm(dim=0)
        return unidirectional_poincare_mlr(
            x, weight_g, weight / weight_g.clamp_min(1e-15), self.bias, self.ball.c)
    
    def extra_repr(self):
        return 'feat_dim={}, num_outcome={}, bias={}'.format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )


# @torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters - add numerical stability
    rc = c.sqrt().clamp_min(1e-15)
    drcr = 2. * rc * r
    
    # Safeguard against excessive values in sinh/cosh computation
    drcr_clamped = drcr.clamp(-20, 20)  # prevent extreme sinh/cosh values
    
    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True).clamp(0, 1e15)
    
    # Compute matmul with numerical stability check
    matmul_result = torch.matmul(rcx, z_unit)
    if torch.isnan(matmul_result).any() or torch.isinf(matmul_result).any():
        print("Warning: NaN/Inf detected in matmul result in MLR")
        matmul_result = torch.where(
            torch.isnan(matmul_result) | torch.isinf(matmul_result),
            torch.zeros_like(matmul_result),
            matmul_result
        )
    
    # Safe computation of cosh and sinh
    drcr_cosh = drcr_clamped.cosh()
    drcr_sinh = drcr_clamped.sinh()
    
    # Compute argument to arsinh with safeguards
    numerator = (2. * matmul_result * drcr_cosh - (1. + cx2) * drcr_sinh)
    
    # Ensure the denominator is non-zero
    denominator = torch.clamp_min(1. - cx2, 1e-15)
    
    arsinh_arg = numerator / denominator
    
    # Clamp the argument to arsinh to prevent extreme values
    arsinh_arg_clamped = arsinh_arg.clamp(-1e5, 1e5)
    
    # Compute final result with numerical stability
    result = 2 * z_norm / rc * arsinh(arsinh_arg_clamped)
    
    # Final safety check
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("Warning: NaN/Inf detected in MLR result")
        result = torch.where(
            torch.isnan(result) | torch.isinf(result),
            torch.zeros_like(result),
            result
        )
    
    return result

