"""Euclidean manifold."""

from .base import Manifold
from geoopt.manifolds.euclidean import Euclidean as geoopt_Euclidean
import torch

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
    


class Euclidean2(geoopt_Euclidean):
    def __init__(self):
        super(Euclidean2, self).__init__()
        self.name = 'Euclidean2'

    def sqdist(self, p1, p2, c):
        return (p1 - p2).pow(2).sum(dim=-1)
    
    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算两个张量之间的欧式距离矩阵
        
        Args:
            x: 形状为 [B, N, D] 的张量
            y: 形状为 [B, N, D] 的张量
            
        Returns:
            形状为 [B, N, N] 的距离矩阵
        """
        # 确保输入张量的形状正确
        assert x.dim() == 3 and y.dim() == 3, "输入张量必须是3维的"
        assert x.shape[0] == y.shape[0], "批次大小必须相同"
        assert x.shape[2] == y.shape[2], "特征维度必须相同"
        
        # 计算欧式距离
        # 1. 计算 x^2
        x_squared = torch.sum(x * x, dim=2, keepdim=True)  # [B, N, 1]
        
        # 2. 计算 y^2
        y_squared = torch.sum(y * y, dim=2, keepdim=True)  # [B, N, 1]
        
        # 3. 计算 -2xy
        xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, N]
        
        # 4. 组合得到距离矩阵: x^2 + y^2 - 2xy
        distance_matrix = x_squared + y_squared.transpose(1, 2) - 2 * xy
        
        # 5. 确保距离非负（由于浮点数计算可能会有小的负值）
        distance_matrix = torch.clamp(distance_matrix, min=0.0)
        
        # 6. 开平方得到最终距离
        distance_matrix = torch.sqrt(distance_matrix)
        
        return distance_matrix