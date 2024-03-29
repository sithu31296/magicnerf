import torch 
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F



class Embedding(nn.Module):
    def __init__(self, n_freqs) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, n_freqs-1, n_freqs)

    def forward(self, x: Tensor) -> Tensor:
        """Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Args:
            x: (B, self.in_channels)
        Returns:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class MLP(nn.Module):
    """Reasonable MLP 

    dim = 32
    three hidden layers
    sigmoid function to regularize the sigma to [0, 1] (since extracting scene geometry from the raw density requires carful tuning of the density thresold
    and leads to artifacts due to the ambiguity present in the density field). 
    in this case, a fixed level set of 0.5 could be used to extract the mesh
    """
    def __init__(self) -> None:
        super().__init__()
        # parameters
        dim = 32    
        d_pos = 3
        d_dir = 3
        pos_freq = 10
        dir_freq = 4
        self.skip = 2   # (8, 4), (6, 4), (4, 2)

        self.pos_embed = Embedding(pos_freq)
        self.dir_embed = Embedding(dir_freq)

        self.pos_encoder = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                input_dim = d_pos * pos_freq * 2 + 3
            elif i == self.skip:
                input_dim = dim + (d_pos * pos_freq * 2 + 3)
            else:
                input_dim = dim
            self.pos_encoder.append(nn.Linear(input_dim, dim))

        self.sigma = nn.Linear(dim, 1)
        self.rgb_encoder = nn.Sequential(
            nn.Linear(dim + (d_dir * dir_freq * 2 + 3), dim//2), 
            nn.ReLU(),
            nn.Linear(dim//2, 3), 
            nn.Sigmoid(),
        )
    
    def forward(self, pos: Tensor, dir: Tensor):
        pos, dir = self.pos_embed(pos), self.dir_embed(dir)
        residual = pos.clone()
        for i, layer in enumerate(self.pos_encoder):
            if i == self.skip:
                pos = torch.cat([pos, residual], dim=-1)
            pos = F.relu(layer(pos))
            
        sigma = F.relu(self.sigma(pos))
        rgb = self.rgb_encoder(torch.cat([pos, dir], dim=-1))
        return rgb, sigma
    

if __name__ == '__main__':
    d_pos = 3
    d_dir = 3
    bs = 100
    model = MLP()
    x = torch.randn(bs, d_pos+d_dir)
    # pos, dir = torch.split(x, [3, 3], dim=-1)

    y = model(x)
    print(y.shape)