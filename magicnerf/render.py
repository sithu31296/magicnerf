import torch
import math
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


EPS = 1e-5

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat([torch.ones([accumulated_transmittance.shape[0], 1], device=alphas.device), accumulated_transmittance[:, :-1]], dim=-1)
    # return torch.cumprod(alphas + 1e-10, 1)

def render_rays(model, rays_o, rays_d, near=0, far=0.5, n_samples=192):
    """Render rays by computing the output of model applied on rays
    Args:
        model               : NeRF model
        rays_o (n_rays, 3)  : ray origins
        rays_d (n_rays, 3)  : ray directions
        near                : near bound
        far                 : far bound
        n_samples           : number of samples per ray
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]
    t = torch.linspace(near, far, n_samples, device=device).expand(n_rays, n_samples)    # (n_rays, n_samples)

    # perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.               # (n_rays, n_samples-1) interval mid points
    # get intervals between samples
    lower = torch.cat([t[:, :1], mid], dim=-1)  
    upper = torch.cat([mid, t[:, -1:]], dim=-1)

    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(n_rays, 1)], dim=-1)

    # compute 3D points along each ray
    xyz = rays_o.unsqueeze(1) + t.unsqueeze(2) * rays_d.unsqueeze(1)   # (n_rays, n_samples, 3)
    # expand the ray_directions tensor to match the shape of x
    dir = rays_d.expand(n_samples, n_rays, 3).transpose(0, 1)
    
    xyz, dir = xyz.reshape(-1, 3), dir.reshape(-1, 3)
    rgb, sigma = model(xyz, dir)
    rgb = rgb.view(n_rays, n_samples, 3)
    sigma = sigma.view(n_rays, n_samples)

    alpha = 1 - torch.exp(-sigma * delta)  
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # compute the pixel values as a weighted sum of colors along each ray
    c = (weights * rgb).sum(dim=1)
    # depth = (weights * t).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)    # regularization for white background

    return c + 1 - weight_sum.unsqueeze(-1)
