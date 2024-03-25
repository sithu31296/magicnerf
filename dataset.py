import torch
import numpy as np
from torch.utils.data import DataLoader



def load_dataset(path):
    return torch.from_numpy(np.load(path, allow_pickle=True))




if __name__ == '__main__':
    dataset = load_dataset("data/training_data.pkl")

    data = next(iter(dataset))
    print(data.shape)

    ray_origins = data[:3]
    ray_directions = data[3:6]
    pixels = data[6:]

    
    print(ray_directions)
    print(ray_origins)
    print(pixels)