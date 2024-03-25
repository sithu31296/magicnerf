import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
from ray_utils import *



class BlenderDataset(Dataset):
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
    def __init__(self, root, scene_name='lego', split='train', size=(800, 800)) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.scene_name = scene_name
        assert size[0] == size[1], "Image size should be equal"
        self.size = size

        self.transforms = T.ToTensor()

        self.read_meta()

        self.white_back = True

    def read_meta(self):
        with open(self.root / self.scene_name / f"transforms_{self.split}.json") as f:
            self.meta = json.load(f)

        w, h = self.size
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length when W=800
        self.focal *= self.size[0] / 800                                    # modify focal length to match size
        print(self.focal)

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal)  # (H, W, 3)

        # create buffer of all rays and rgb data
        if self.split == 'train':
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []

            for frame in self.meta['frames']:
                image_path = self.root / self.scene_name / f"{frame['file_path']}.png"
                img = Image.open(image_path)
                img = img.resize(self.size, Image.Resampling.LANCZOS)
                img = self.transforms(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.image_paths.append(image_path)
                self.all_rgbs.append(img)

                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.from_numpy(pose).float()
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.poses.append(pose)
                self.all_rays.append(torch.cat([rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1])], dim=1)) # (h*w, 8)
            
            self.all_rays = torch.cat(self.all_rays, dim=0)
            self.all_rgbs = torch.cat(self.all_rgbs, dim=0)

        
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'val':
            return 8    # only valide 8 images (to support <= 8 gpus)
        return len(self.meta['frames'])
    
    def __getitem__(self, index):
        if self.split == 'train':
            sample = {
                "rays": self.all_rays[index],
                "rgbs": self.all_rgbs[index]
            }
        else:   # create data for each image separately
            c2w = torch.from_numpy(self.poses[index]).float()
            image_path = self.image_paths[index]
            img = Image.open(image_path)
            img = img.resize(self.size, Image.Resampling.LANCZOS)
            img = self.transforms(img)
            valid_mask = (img[-1] > 0).flatten()    # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)     # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1])], dim=1)    # (H*W, 8)

            sample = {
                "rays": rays,
                "rgbs": img,
                "c2w": c2w,
                "valid_mask": valid_mask
            }
        return sample



if __name__ == '__main__':
    dataset = BlenderDataset("/home/sithu/datasets/nerf_synthetic", "lego", "train")
    sample = next(iter(dataset))
    print(sample['rays'].shape)
    print(sample['rgbs'].shape)