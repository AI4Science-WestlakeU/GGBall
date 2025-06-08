from .geoopt_plusplus.modules.linear import PoincareLinear, PoincareConcatLinear
from .geoopt_plusplus.manifolds.stereographic.math import (
    _mobius_add,
    _mobius_scalar_mul,
    _project,
    weighted_midpoint
)

__all__ = [
    'PoincareLinear',
    'PoincareConcatLinear',
    '_mobius_add',
    '_mobius_scalar_mul',
    '_project',
    'weighted_midpoint',
]

# 版本信息
__version__ = '0.1.0'