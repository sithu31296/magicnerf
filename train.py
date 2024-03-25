import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from model import NerfModel
from dataset import load_dataset
from render import render_rays


@torch.inference_mode()
def test(model, hn, hf, dataset, device, chunk_size=10, img_index=0, bins=192, H=400, W=400):
    """
    hn: near plance distance
    hf: far plane distance
    dataset: dataset to render
    chunk_size: chunk size for memory efficiency
    img_index: image index to render
    bins: number of bins for density estimation
    H: image height
    W: image width
    """
    start = img_index * H * W
    end = (img_index + 1) * H * W
    ray_origins = dataset[start:end, :3]
    ray_directions = dataset[start:end, 3:6]

    data = []   # list of generated pixel values
    for i in range(int(np.ceil(H / chunk_size))):    # iterate over chunks
        # get chunk of rays
        start = i * W * chunk_size
        end = (i+1) * W * chunk_size
        ray_origins_ = ray_origins[start:end].to(device)
        ray_directions_ = ray_directions[start:end].to(device)

        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn, hf, bins)
        data.append(regenerated_px_values)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f"novel_views/img_{img_index}.png", bbox_inches='tight')
    plt.close()



def train(model, optimizer, scheduler, dataloader, device, hn=0, hf=1, bins=192):
    losses = []
    for batch in tqdm(dataloader):
        ray_origins = batch[:, :3].to(device)
        ray_directions = batch[:, 3:6].to(device)
        gt_px_values = batch[:, 6:].to(device)

        regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, bins=bins)

        loss = ((gt_px_values - regenerated_px_values) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    scheduler.step()



def main():
    batch_size = 2048
    d_pos = 10
    d_dir = 4
    dim = 256
    lr  = 5.e-4
    epochs = 16

    hn = 2
    hf = 6
    bins = 192
    H = 400
    W = 400
    chunk_size = 10

    device = torch.device('cuda')

    train_set = load_dataset("data/training_data.pkl")
    test_set = load_dataset("data/testing_data.pkl")

    train_loader = DataLoader(train_set, batch_size, True)

    model = NerfModel(d_pos, d_dir, dim)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train(model, optimizer, scheduler, train_loader, device, hn, hf, bins)

        for img_index in range(10):
            test(model, hn, hf, test_set, device, chunk_size, img_index, bins, H, W)

if __name__ == '__main__':
    main()