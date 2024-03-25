import torch
import math
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat([torch.ones([accumulated_transmittance.shape[0], 1], device=alphas.device), accumulated_transmittance[:, :-1]], dim=-1)


def render_rays(model, ray_origins, ray_directions, hn=0, hf=0.5, bins=192):
    device = ray_origins.device

    t = torch.linspace(hn, hf, bins, device=device).expand(ray_origins.shape[0], bins)

    # perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat([t[:, :1], mid], dim=-1)
    upper = torch.cat([mid, t[:, -1:]], dim=-1)
    
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u # [bs, bins]
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)], dim=-1)

    # compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [bs, bins, 3]

    # expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)   # [bs, bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)    # regularization for white background

    return c + 1 - weight_sum.unsqueeze(-1)
