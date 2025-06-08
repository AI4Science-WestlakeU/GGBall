"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
from tqdm import tqdm
import pdb

@torch.no_grad()
def projx_integrator(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False, node_mask=None
):
    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
        "euler_retraction": euler_retraction_step,
        "euler_normalize_step": euler_normalize_step,
    }[method]

    xts = [x0]
    vts = []

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    xt = x0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt, node_mask)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None, node_mask=node_mask
        )
        if projx:
            xt = manifold.projx(xt)
        vts.append(vt)
        xts.append(xt)
    vts.append(odefunc(t1, xt))
    return torch.stack(xts), torch.stack(vts)


@torch.no_grad()
def projx_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False, node_mask=None
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
        "euler_retraction": euler_retraction_step,
        "euler_normalize_step": euler_normalize_step,
        "x0_pred": x0_pred_step, 
        "vt_prediction": vt_prediction_step,
        "x1_prediction": x1_prediction_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        #x1[9,7:10, -10:]
        vt = odefunc(t0.unsqueeze(0).unsqueeze(0), xt, node_mask)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None, node_mask=node_mask
        )
        # import pdb; pdb.set_trace()
        if projx:
            xt = manifold.projx(xt)
    return xt


@torch.no_grad()
def debug_projx_integrator_return_last(
    manifold, u, x0, t, method="euler", projx=True, local_coords=False, pbar=False, node_mask=None
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
        "euler_retraction": euler_retraction_step,
        "euler_normalize_step": euler_normalize_step
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    i = 0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = u[i]
        xt = step_fn(
            None, xt, vt, t0, dt, manifold=manifold if local_coords else None, node_mask=node_mask
        )
        if projx:
            xt = manifold.projx(xt)
        i += 1
    return xt


@torch.no_grad()
def debug_projx_integrator_return_last_2(
    manifold, x1, x0, t, method="euler", projx=True, local_coords=False, pbar=False, node_mask=None
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    i = 0
    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = manifold.logmap(xt, x1)
        xt = step_fn(
            None, xt, vt, t0, dt, manifold=manifold if local_coords else None, node_mask=node_mask
        )
        if projx:
            xt = manifold.projx(xt)
        i += 1
    return xt



def x0_pred_step(odefunc, xt, x1, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        vt = manifold.logmap(xt, x1)
        return manifold.expmap(xt, dt * vt / (1-t0))
    else:
        vt = x1 - xt
        return xt + dt * vt
    
    
def euler_step(odefunc, xt, vt, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt
    
def euler_retraction_step(odefunc, xt, vt, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        xt = xt + dt * vt
        xt = manifold.projx(xt)
        return xt
    
        # return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


def euler_normalize_step(odefunc, xt, vt, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        _norm = torch.norm(vt, dim=-1, keepdim=True)
        xt = xt + dt * vt / _norm
        xt = manifold.projx(xt)
        return xt
    
        # return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt

def midpoint_step(odefunc, xt, vt, t0, dt, manifold=None, node_mask=None):
    
    half_dt = 0.5 * dt
    if manifold is not None:
        x_mid = xt + half_dt * vt
        v_mid = odefunc(t0 + half_dt, x_mid, mask=node_mask)
        v_mid = manifold.transp(x_mid, xt, v_mid)
        return manifold.expmap(xt, dt * v_mid)
    else:
        x_mid = xt + half_dt * vt
        return xt + dt * odefunc(t0 + half_dt, x_mid, mask=node_mask)


def rk4_step(odefunc, xt, vt, t0, dt, manifold=None, node_mask=None):
    k1 = vt
    if manifold is not None:
        raise NotImplementedError
    else:
        k2 = odefunc(t0 + dt / 3, xt + dt * k1 / 3, mask=node_mask)
        k3 = odefunc(t0 + dt * 2 / 3, xt + dt * (k2 - k1 / 3), mask=node_mask)
        k4 = odefunc(t0 + dt, xt + dt * (k1 - k2 + k3), mask=node_mask)
        return xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

def vt_prediction_step(odefunc, xt, x1, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        vt = manifold.logmap(xt, x1)
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt

def x1_prediction_step(odefunc, xt, x1, t0, dt, manifold=None, node_mask=None):
    if manifold is not None:
        vt = manifold.logmap(xt, x1)
        return manifold.expmap(xt, dt * vt / (1-t0))
    else:
        vt = x1 - xt
        return xt + dt * vt
