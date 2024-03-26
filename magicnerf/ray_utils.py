import torch
import numpy as np
import matplotlib.pyplot as plt



def get_ray_directions(H, W, K):
    """Get ray directions for all pixels in camera coordinate
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/standard-coordinate-systems
    
    Args:
        H, W, K : image height, image width, camera intrinsics
    Returns:
        dirs: (H, W, 3), the direction of the rays in camera coordinate
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i, j = i.T, j.T 
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)    # (H, W, 3)
    return dirs


def get_rays(dirs, c2w):
    """Get ray origin and normalized directions in world coordiante for all pixels in one image
    Args:
        dirs    : (H, W, 3) precomputed ray directions in camera coordinate
        c2w     : (4, 4)    transformation matrix from camera coordinate to world coordinate
    Returns:
        rays_o  : (H*W, 3)  the origin of the rays in world coordinate
        rays_d  : (H*W, 3)  the normalized direction of the rays in world coordinate
    """
    # rotate ray directions from camera coordinate to world coordinate
    rays_d = dirs @ c2w[:3, :3].T    # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # the origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o.view(-1, 3), rays_d.view(-1, 3)


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Transform rays from world coordinates to Normalized Device Coordiantes (NDC)
    NDC: a space such that the canvas is a cube with side [-1, 1] in each axis.

    For detailed derivation, please see:
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    
    In practice, use NDC if and only if the scene is unbounded (has a large depth)
    See https://github.com/bmild/nerf/issues/18

    Args:
        near (n_rays) or float  :   the depths of the near plane
        rays_o (n_rays, 3)      :   the origin of the rays in world coordinate
        rays_d (n_rays, 3)      :   the direction of the rays in world coordinate
    Returns:
        rays_o (n_rays, 3)      :   the origin of the rays in NDC
        rays_d (n_rays, 3)      :   the direction of the rays in NDC
    """
    # shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]   # t_n = -(n + o_z) / d_z
    rays_o = rays_o + t[..., None] * rays_d         # o_n = o + t_n * d

    # projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]     # o_x = -f/(W/2) * (o_x / o_z)
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]     # o_y = -f/(H/2) * (o_y / o_z)
    o2 = 1. + 2. * near / rays_o[..., 2]                                # o_z = 1 + (2 * n / o_z)

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]) # d_x = -f/(W/2) * (d_x/d_z - o_x/o_z)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]) # d_y = -f/(H/2) * (d_y/d_z - o_y/o_z)
    d2 = -2. * near / rays_o[..., 2]                                                                    # d_z = -2 * n / o_z

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)
    return rays_o, rays_d




if __name__ == '__main__':
    H = W = 800
    focal = 1111
    K = torch.tensor([
        [focal, 0, 0],
        [0, focal, 0],
        [0, 0, 1]
    ])
    dirs = get_ray_directions(H, W, K)
    print(dirs.shape)

    c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    rays_o, rays_d = get_rays(dirs, c2w)
    print(rays_o.shape, rays_d.shape)

    rays_o, rays_d = get_ndc_rays(H, W, focal, 0.8, rays_o, rays_d)
    print(rays_o.shape, rays_d.shape)

    
    dirs2 = dirs.view(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    plt.show()