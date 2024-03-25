import torch
import matplotlib.pyplot as plt
from kornia import create_meshgrid



def get_ray_directions(image_height, image_width, focal_length):
    """Get ray directions for all pixels in camera coordinate
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/standard-coordinate-systems
    
    Returns:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(image_height, image_width, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = torch.stack([(i - image_width/2)/focal_length, -(j-image_height/2)/focal_length, -torch.ones_like(i)], dim=-1) # (H, W, 3)
    return directions, i, j


def get_rays(directions, c2w):
    """Get ray origin and normalized directions in world coordinate for all pixels in one image
    Args:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Returns:
        rays_o: (H*W, 3) the origin of the rays in world coordinate
        rays_d: (H*W, 3) the normalized direction of the rays in world coordinate
    """
    # rotate any directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # the origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    return rays_o.view(-1, 3), rays_d.view(-1, 3)


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Transform rays from world coordinate to NDC
    NDC: space such that the canvs is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    
    In practice, use NDC if and only if the scene is unbounded (has a large depth)
    See https://github.com/bmild/nerf/issues/18

    Args:
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Returns:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d



if __name__ == '__main__':
    H = W = 800
    focal = 1111
    dirs = get_ray_directions(H, W, focal)
    
    dirs2 = dirs.view(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    plt.show()