import torch
import math
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


EPS = 1e-5

def compute_accumulated_transmittance(alphas):
    ones = torch.ones([alphas.shape[0], 1], device=alphas.device)
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat([ones, accumulated_transmittance[:, :-1]], dim=-1)


def sample_pdf(bins, weights, n_samples, det=False):
    # get pdf
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # take uniform samples
    if det:
        u = torch.linspace(0., 1., n_samples).expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    
    # invert cdf
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def sample_along_rays(near, far, n_samples, n_rays, device):
    """Sample points along rays (Stratified sampling)
    Args:
        near        : near bound
        far         : far bound
        n_samples   : number of sampled points for each ray
    """
    # uniform samples along rays
    t = torch.linspace(near, far, n_samples, device=device).expand(n_rays, n_samples)    # (n_rays, n_samples)
    # perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.               # (n_rays, n_samples-1) interval mid points
    # get intervals between samples
    lower = torch.cat([t[:, :1], mid], dim=-1)  
    upper = torch.cat([mid, t[:, -1:]], dim=-1)
    rand = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * rand
    return t


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
    t = sample_along_rays(near, far, n_samples, n_rays, device)
    one_e_10 = torch.tensor([1e10], device=device).expand(n_rays, 1)
    delta = torch.cat([t[:, 1:] - t[:, :-1], one_e_10], dim=-1)
    # delta = delta * rays_d[..., None, :].norm(p=2, dim=-1)    # to convert to real world distance units

    # compute 3D points along each ray
    xyz = rays_o.unsqueeze(1) + t.unsqueeze(2) * rays_d.unsqueeze(1)   # (n_rays, n_samples, 3)
    # expand the ray_directions tensor to match the shape of x
    dir = rays_d.expand(n_samples, n_rays, 3).transpose(0, 1)
    
    xyz, dir = xyz.reshape(-1, 3), dir.reshape(-1, 3)
    rgb, sigma = model(xyz, dir)
    rgb = rgb.view(n_rays, n_samples, 3)
    sigma = sigma.view(n_rays, n_samples)

    alpha = 1 - torch.exp(-sigma * delta)  
    # ray termination probability
    weights = compute_accumulated_transmittance(1 - alpha + 1e-10).unsqueeze(2) * alpha.unsqueeze(2)

    # compute the pixel values as a weighted sum of colors along each ray
    c = (weights * rgb).sum(dim=1)
    # depth = (weights * t).sum(dim=1)

    # regularization for white background
    acc_map = weights.sum(-1).sum(-1)
    c = c + (1 - acc_map.unsqueeze(-1))
    return c
