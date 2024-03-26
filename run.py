import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from magicnerf.model import MLP
from magicnerf.datasets import Blender
from magicnerf.render import render_rays



def main():
    batch_size = 6144    # 100 views
    lr  = 5.e-4
    epochs = 16
    weight_decay = 1e-4

    half_res = True
    n_samples = 64
    n_importance = 64
    use_viewdirs = True
    white_bg = True
    redner_test = True
    n_rand = 1024

    bins = 192
    chunk_size = 10
    H, W = 400, 400
    root = "./data/nerf_synthetic"
    device = torch.device('cuda:0')

    # trainset = Blender(root, "lego", 'train', (H, W))
    # testset = Blender(root, "lego", "test", (H, W))

    trainset = torch.from_numpy(np.load("/home/sithu/datasets/llff/training_data.pkl", allow_pickle=True))
    testset = torch.from_numpy(np.load("/home/sithu/datasets/llff/testing_data.pkl", allow_pickle=True))

    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    # near, far = trainset.near, trainset.far
    near, far = 2, 6

    model = MLP().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    losses = []
    for epoch in range(epochs):
        for batch in train_loader:
            rays_o = batch[:, :3].to(device)
            rays_d = batch[:, 3:6].to(device)
            rgbs = batch[:, 6:].to(device)
            # rgbs = rgbs.view(-1, 3)
            # rays = rays.view(-1, 6)
            # rays_o = rays[:, :3].to(device)
            # rays_d = rays[:, 3:6].to(device)
            # rgbs = rgbs.to(device)
            
            rendered_rgbs = render_rays(model, rays_o, rays_d, near, far, bins)
            loss = ((rgbs - rendered_rgbs) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            losses.append(loss.item())
        print(f"Epochs: {epoch+1}/{epochs}\tLoss: {sum(losses) / len(losses):.2f}")
        scheduler.step()

        for index in range(10):
            batch = testset[index]
            # rgbs = rgbs.view(-1, 3)
            # rays = rays.view(-1, 6)
            # rgbs = rgbs.to(device)
            # rays_o = rays[:, :3].to(device)
            # rays_d = rays[:, 3:6].to(device)
            rays_o = batch[:, :3].to(device)
            rays_d = batch[:, 3:6].to(device)
            rgbs = batch[:, 6:].to(device)

            data = []   # list of rendered images
            # iterate over chunks
            for i in range(int(np.ceil(H / chunk_size))):
                # get chunk of rays
                rays_o_ = rays_o[i*W*chunk_size : (i+1)*W*chunk_size]
                rays_d_ = rays_d[i*W*chunk_size : (i+1)*W*chunk_size]
                with torch.inference_mode():
                    rendered_rgbs = render_rays(model, rays_o_, rays_d_, near, far, bins)
                    data.append(rendered_rgbs)

            img = torch.cat(data).cpu().numpy().reshape(H, W, 3)

            plt.figure()
            plt.imshow(img)
            plt.savefig(f"novel_views/img_{index}.png", bbox_inches='tight')
            plt.close()
        

if __name__ == '__main__':
    main()