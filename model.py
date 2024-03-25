import torch 
from torch import nn, Tensor
from torch.nn import functional as F


class NerfModel(nn.Module):
    def __init__(self, d_pos=10, d_dir=4) -> None:
        super().__init__()
        dim = 128
        self.block1 = nn.Sequential(
            nn.Linear(d_pos * 6 + 3, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU()
        )

        # density estimation
        self.block2 = nn.Sequential(
            nn.Linear(d_pos * 6 + dim + 3, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim + 1)
        )

        # color estimation
        self.block3 = nn.Sequential(
            nn.Linear(d_dir * 6 + dim + 3, dim//2), nn.ReLU(),
            nn.Linear(dim//2, 3), nn.Sigmoid()
        )
        self.d_pos = d_pos
        self.d_dir = d_dir


    @staticmethod
    def positional_encoding(x: Tensor, dim: int):
        out = [x]
        for j in range(dim):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
    
    def forward(self, pos: Tensor, dir: Tensor):
        emb_pos = self.positional_encoding(pos, self.d_pos)
        emb_dir = self.positional_encoding(dir, self.d_dir)

        h = self.block1(emb_pos)
        tmp = self.block2(torch.cat([h, emb_pos], dim=1))

        # density estimation (use relu to ensure that the output volume density is non-negative)
        sigma = F.relu(tmp[:, -1])

        # color estimation
        c = self.block3(torch.cat([tmp[:, :-1], emb_dir], dim=1))
        return c, sigma
    


if __name__ == '__main__':
    d_pos = 10
    d_dir = 4
    bs = 2
    model = NerfModel(d_pos, d_dir)
    x = torch.randn(bs, 3)
    d = torch.randn(bs, 3)

    color, density = model(x, d)
    print(color.shape, density.shape)