import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
from magicnerf.ray_utils import *



trans_t = lambda t: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
], dtype=torch.float32)

rot_phi = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)


rot_theta = lambda theta: torch.tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.tensor([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32) @ c2w
    return c2w


class Blender(Dataset):
    """A dataset of synthetically rendered images 
    Stats:
        * 8 scenes
        * 100 training images
        * 100 validation images
        * 200 test images
        * 800x800 size
    Directory:
        SCENE_NAME
            |_ train
                |_ r_*.png
            |_ val
                |_ r_*.png
            |_ test
                |_ r_*.png
                |_ r_*_depth_0000.png
                |_ r_*_normal_0000.png
            transforms_train.json
            transforms_val.json
            transforms_test.json

    Transform json detials:
        * camera_angle_x: FOV in x dimension
        * frames: list of dicts that contain the camera transform matrices for each image
    """
    def __init__(self, root, scene='lego', split='train', size=(400, 400)) -> None:
        super().__init__()
        self.root = Path(root)
        splits = ['train', 'val', 'test']
        assert split in splits
        self.split = split
        self.scene = scene
        self.size = size
        self.transforms = T.ToTensor()

        # bounds, common for all scenes in this dataset
        self.near = 2
        self.far = 6
        self.bounds = np.array([self.near, self.far])
        self.rgbs, self.rays, = self.read_data()

    def read_image(self, img_path: str) -> Image:
        img = Image.open(img_path)
        img = img.resize(self.size, Image.Resampling.LANCZOS)
        img = self.transforms(img)  # (4, 800, 800) # RGBA
        img = img.view(4, -1).T     # (800*800, 4)  
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
        return img

    def read_data(self):
        with open(self.root / self.scene / f"transforms_{self.split}.json") as f:
            meta = json.load(f)        # keys [camera_angle_x, frames]

        w, h = self.size
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])    # original focal length when W=800
        focal *= w / 800                                            # modify focal length to match size
        K = torch.tensor([
            [focal, 0, w//2],
            [0, focal, h//2],
            [0, 0, 1]
        ])

        # ray directions for all pixels, same for all images (same H, W, K)
        dirs = get_ray_directions(h, w, K)  # (H, W, 3)

        rgbs, rays = [], []
        for frame in meta['frames']:
            img = self.read_image(self.root / self.scene / f"{frame['file_path']}.png")     # (640000, 3)                                    
            c2w = torch.from_numpy(np.array(frame['transform_matrix']) ).float()            # (4, 4)
            rays_o, rays_d = get_rays(dirs, c2w)                                            # (640000, 3), (640000, 3)   
            rgbs.append(img)
            rays.append(torch.cat([rays_o, rays_d], dim=-1))
        
        rgbs = torch.cat(rgbs, 0)    # (100*640000, 3)
        rays = torch.cat(rays, 0)    # (100*640000, 6)
        return rgbs, rays
 

    def create_render_poses(self):
        render_poses = []
        for angle in np.linspace(-180, 180, 40+1)[:-1]:
            render_poses.append(pose_spherical(angle, -30.0, 4.0))
        return torch.stack(render_poses, dim=0)
        
    def __len__(self):
        return len(self.rgbs)
    
    def __getitem__(self, index):
        return self.rgbs[index], self.rays[index]



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Blender("/data4/sithu/datasets/", "lego", "test", (800, 800))
    dataloader = DataLoader(dataset, 100)
    rgbs, rays = next(iter(dataloader))
    print(rays.shape)