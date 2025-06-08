"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import torch
import torch.nn as nn

# import arch.model.diffeq_layers as diffeq_layers
# from arch.model.actfn import Sine, Softplus


# ACTFNS = {
#     "swish": diffeq_layers.TimeDependentSwish,
#     "sine": Sine,
#     "srelu": Softplus,
# }

# here, replace tMLP with tHGCN or other models
# def tMLP(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish", fourier=None):
#     assert num_layers > 1, "No weak linear nets here"
#     d_out = d_in if d_out is None else d_out
#     actfn = ACTFNS[actfn]
#     if fourier:
#         layers = [
#             diffeq_layers.diffeq_wrapper(
#                 PositionalEncoding(n_fourier_features=fourier)
#             ),
#             diffeq_layers.ConcatLinear_v2(d_in * fourier * 2, d_model),
#         ]
#     else:
#         layers = [diffeq_layers.ConcatLinear_v2(d_in, d_model)]

#     for _ in range(num_layers - 2):
#         layers.append(actfn(d_model))
#         layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_model))
#     layers.append(actfn(d_model))
#     layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_out))
#     return diffeq_layers.SequentialDiffEq(*layers)


# If you want Swish, you can define a small helper:
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def tMLP(
    d_in: int, 
    d_out: int = None, 
    d_model: int = 256, 
    num_layers: int = 6, 
    activation: str = "swish",
    fourier: int = None
):
    """
    A simple Multi-Layer Perceptron that uses PyTorch's built-in nn.Linear and nn.ReLU/Swish.

    :param d_in: dimension of the input
    :param d_out: dimension of the output (defaults to d_in if not specified)
    :param d_model: intermediate hidden dimension
    :param num_layers: number of Linear layers
    :param activation: activation function ("swish" or "relu")
    :param fourier: optional int to define whether to do some fourier-based input expansion
    """
    assert num_layers > 1, "Must have at least 2 layers."
    d_out = d_in if d_out is None else d_out

    # Choose the activation
    if activation.lower() == "swish":
        act_layer = Swish()
    elif activation.lower() == "relu":
        act_layer = nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = []

    # Optional Fourier expansion of inputs (just an example; you can customize)
    if fourier is not None:
        # For demonstration, let's pretend we do some positional encoding here
        # That might expand your input dimension from d_in -> d_in * fourier_factor
        fourier_dim = d_in * fourier * 2  # roughly mimicking the original code
        layers.append(nn.Linear(fourier_dim, d_model))
    else:
        layers.append(nn.Linear(d_in, d_model))

    # Intermediate layers
    for _ in range(num_layers - 2):
        layers.append(act_layer)
        layers.append(nn.Linear(d_model, d_model))

    # Final layer(s)
    layers.append(act_layer)
    layers.append(nn.Linear(d_model, d_out))

    return nn.Sequential(*layers)

class PositionalEncoding(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x):
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)


class Unbatch(nn.Module):
    def __init__(self, vecfield):
        super().__init__()
        self.vecfield = vecfield

    def forward(self, t, x):
        has_batch = x.ndim > 1
        if not has_batch:
            x = x.reshape(1, -1)
            t = t.reshape(-1)
        v = self.vecfield(t, x)
        if not has_batch:
            v = v[0]
        return v


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield, manifold, metric_normalize):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold
        self.metric_normalize = metric_normalize

    def forward(self, t, x):
        x = self.manifold.projx(x)
        v = self.vecfield(t, x)
        v = self.manifold.proju(x, v)

        if self.metric_normalize and hasattr(self.manifold, "metric_normalized"):
            v = self.manifold.metric_normalized(x, v)

        return v


if __name__ == "__main__":
    # print(diffeq_layers.ConcatLinear_v2(3, 64))

    import torch

    model = tMLP(d_in=3, d_model=64, num_layers=3)
    t = torch.randn(2, 1)
    x = torch.randn(2, 3)

    print(model(t, x))