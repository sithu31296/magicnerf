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


img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mse_sum = lambda x, y: torch.sum((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



def main():
    lr  = 5.e-4
    epochs = 16
    weight_decay = 1e-4

    bins = 64
    chunk_size = 10
    H, W = 400, 400
    root = "/data4/sithu/datasets/"
    device = torch.device('cuda:7')
    batch_size = H*W    

    trainset = Blender(root, "lego", 'train', (H, W))
    testset = Blender(root, "lego", "test", (H, W))

    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    test_loader = DataLoader(testset, H*W, shuffle=False)
    near, far = trainset.near, trainset.far

    model = MLP().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    for epoch in range(epochs):
        losses = []
        for rgbs, rays in train_loader:
            rgbs = rgbs.to(device)
            rays_o = rays[:, :3].to(device)
            rays_d = rays[:, 3:6].to(device)
            
            rendered_rgbs = render_rays(model, rays_o, rays_d, near, far, bins)
            loss = img2mse_sum(rgbs, rendered_rgbs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            losses.append(loss.item())
        print(f"Epochs: {epoch+1}/{epochs}\tLoss: {sum(losses) / len(losses):.2f}")
        scheduler.step()

        losses = []
        for index, (rgbs, rays) in enumerate(test_loader):
            if index == 10:
                break
            rgbs = rgbs.to(device)
            rays_o = rays[:, :3].to(device)
            rays_d = rays[:, 3:6].to(device)

            rendered_rgbs = []   # list of rendered images
            # iterate over chunks
            for i in range(int(np.ceil(H / chunk_size))):
                # get chunk of rays
                rays_o_ = rays_o[i*W*chunk_size : (i+1)*W*chunk_size]
                rays_d_ = rays_d[i*W*chunk_size : (i+1)*W*chunk_size]
                with torch.inference_mode():
                    rendered_rgbs.append(render_rays(model, rays_o_, rays_d_, near, far, bins))
                    
            rendered_rgbs = torch.clamp(torch.cat(rendered_rgbs), 0, 1)
            losses.append(img2mse(rgbs, rendered_rgbs))

            plt.figure()
            plt.imshow(rendered_rgbs.cpu().numpy().reshape(H, W, 3))
            plt.savefig(f"novel_views/img_{index}.png", bbox_inches='tight')
            plt.close()
        
        psnr = mse2psnr(torch.mean(torch.tensor(losses))).item()
        print(f"PSNR: {psnr:.2f}")
        

if __name__ == '__main__':
    main()