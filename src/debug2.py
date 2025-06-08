import torch
from src.manifolds.lorentz import Lorentz
from src.manifolds.geodesic import geodesic
from src.manifolds.poincareball import PoincareBall
from torch.func import vjp, jvp, vmap, jacrev
# manifold = Lorentz()
import pdb

dim = 128
manifold = PoincareBall(dim=dim)
x1 = torch.randn([dim]).double() * 10
x0 = torch.randn([dim]).double()
x1 = manifold.projx(x1)
x0 = manifold.projx(x0)

path = geodesic(manifold, x0, x1)
x1_pred = path(torch.Tensor([1.0]).to(x1))


print(f"in {dim} dim")
print(f"x0 abs mean: {x0.abs().mean()}")
print(f"x1_pred abs mean: {x1_pred.abs().mean()}")
print(f"x1 abs mean: {x1.abs().mean()}")
print(f"diff between x0 and x1, with euclidean L2 norm {torch.norm(x1 - x0)}")
print(f"diff between x0 and x1 on th manifold is {manifold.dist(x1, x0)}")
print(f"diff between ground truth x1 and pred x1, with euclidean L2 norm {torch.norm(x1_pred - x1)}")
print(f"diff between ground truth x1 and pred x1, on the manifold {manifold.dist(x1_pred, x1)}")



x1 = x1.to("cuda")
x0 = x0.to("cuda")
manifold.to("cuda")

def cond_u(x0, x1, t):
    path = geodesic(manifold, x0, x1)
    with torch.no_grad():
        x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
    return x_t, u_t


t_list = torch.linspace(0, 1, 1001).to(x1)
x_t_list = []
u_t_list = []
for i in range(1001):
    x_t, u_t = cond_u(x0, x1, t_list[i].unsqueeze(-1))
    x_t.squeeze_(0)
    u_t.squeeze_(0)
    x_t_list.append(x_t)
    u_t_list.append(u_t)
    
from src.solvers import debug_projx_integrator_return_last, debug_projx_integrator_return_last_2
x1_sample = debug_projx_integrator_return_last(
        manifold,
        u_t_list,
        x0,
        t=torch.linspace(0, 1, 1001).to(x1),
        method="euler",
        projx=True,
        local_coords=True,
        pbar=True,
        node_mask = None,
    )

print(f"diff between xt_list_last and x1_sample,is {manifold.dist(x1_sample, x_t_list[-1])}, norm is {torch.norm(x1_sample - x_t_list[-1])}")
print(f"diff between x1_sample and ground truth x1 is {manifold.dist(x1, x1_sample)}, norm is {torch.norm(x1 - x1_sample)}")
print(f"diff between ground truth x1 and xt_list_last, is {manifold.dist(x1, x_t_list[-1])}, norm is {torch.norm(x1 - x_t_list[-1])}")

pdb.set_trace()