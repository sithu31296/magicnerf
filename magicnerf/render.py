import torch
import math
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


EPS = 1e-5


# hierarchical sampling (S5.2)
def sample_pdf(bins, weights, n_importance, det=False):
    """Sample n_importance from bins with distribution defined by weights.
    Args:
        bins (n_rays, n_samples + 1)    :   where n_samples is the number of coarse samples per ray - 2
        weights (n_rays, n_samples)
        n_importance                    :   the number of samples to draw from the distribution
        det                             :   deterministic or not
    Returns:
        samples: the sampled samples
    """
    n_rays, n_samples = weights.shape
    weights = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # (n_rays, n_samples)
    # cumulative distribution function
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (n_rays, n_samples+1) padded to 0-1 inclusive

    # take uniform samples
    if det:
        u = torch.linspace(0., 1., n_importance, device=bins.device)
        u = u.expand(n_rays, n_importance)
    else:
        u = torch.rand(n_rays, n_importance, device=bins.device)
    
    # invert cdf
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, n_samples)

    inds_sampled = torch.stack([below, above], -1).view(n_rays, 2*n_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(n_rays, n_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(n_rays, n_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled anyway, therefore any value for it is fine (set to 1 here)
    denom[denom < EPS] = 1
   
    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples




# def render_rays(model, rays, n_samples=64, chunk=1024*32):
#     """Render rays by computing the output of model applied on rays
#     Args:
#         model   : NeRF model
#         rays (n_rays, 3+3+2)    : ray origins, ray directions, near and far depth bounds
#         n_samples               : number of coarse samples per ray
#         use_disp                : whether to sample in disparity space (inverse depth)
#         perturb                 : factor perturb the sampling position on the ray (for coarse model only)
#         noise_std               : factor to perturb the model's prediction of sigma
#         n_importance            : number of fine samples per ray
#         chunk                   : chunk size in batched inference
#         white_back              : whether the background is white (dataset dependent)
#         test_time               : whether it is test (inference only) or not. If True, it will not do inference on coarse rgb to save time
#     Returns:
#         result                  : dict containing final rgb and depth maps for coarse and fine models
#     """
#     device = rays.device
#     n_rays = rays.shape[0]
#     rays_o, rays_d = rays[:, :3], rays[:, 3:6]
#     near, far = rays[:, 6:7], rays[:, 7:8]
#     # sample depth points
#     z_steps = torch.linspace(0, 1, n_samples, device=device)    # (n_samples)

#     # use linear_sampling in depth space
#     z_vals = near * (1 - z_steps) + far * z_steps
#     z_vals = z_vals.expand(n_rays, n_samples)

#     # perturb sampling depths 
#     z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])         # (n_rays, n_samples-1) interval mid points
#     # get intervals between samples
#     upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
#     lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

#     perturb_rand = torch.rand(z_vals.shape, device=device)
#     z_vals = lower + (upper - lower) * perturb_rand

#     # compute 3d points along each ray
#     xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)   # (n_rays, n_samples, 3)
#     dir = rays_d.expand(n_rays, n_samples, 3)                               # (n_rays, n_samples, 3)

#     xyz, dir = xyz.view(-1, 3), dir.view(-1, 3)
#     inputs = torch.cat([xyz, dir], dim=-1)

#     B = inputs.shape[0]
#     out_chunks = []
#     for i in range(0, B, chunk):
#         inputs = inputs[i:i+chunk]
#         out_chunks.append(model(inputs))
    
#     out = torch.cat(out_chunks, 0)
#     rgbsigma = out.view(n_rays, n_samples, 4)
#     rgb = rgbsigma[:, :3]                                                   # (n_rays, n_samples, 3)
#     sigma = rgbsigma[:, 3]                                                  # (n_rays, n_samples)





def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat([torch.ones([accumulated_transmittance.shape[0], 1], device=alphas.device), accumulated_transmittance[:, :-1]], dim=-1)


def render_rays(model, rays_o, rays_d, near=0, far=0.5, bins=192):
    device = rays_o.device
    t = torch.linspace(near, far, bins, device=device).expand(rays_o.shape[0], bins)    # (batch, bins)

    # perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat([t[:, :1], mid], dim=-1)
    upper = torch.cat([mid, t[:, -1:]], dim=-1)
    
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u # [bs, bins]
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(rays_o.shape[0], 1)], dim=-1)

    # compute the 3D points along each ray
    x = rays_o.unsqueeze(1) + t.unsqueeze(2) * rays_d.unsqueeze(1)   # [bs, bins, 3]

    # expand the ray_directions tensor to match the shape of x
    rays_d = rays_d.expand(bins, rays_d.shape[0], 3).transpose(0, 1)

    colors, sigma = model(x.reshape(-1, 3), rays_d.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)   # [bs, bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)    # regularization for white background

    return c + 1 - weight_sum.unsqueeze(-1)
