import torch 
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



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
    def __init__(self) -> None:
        super().__init__()
        # parameters
        dim = 128
        d_pos = 3
        d_dir = 3
        pos_freq = 10
        dir_freq = 4
        self.skip = 4   # skip connection at fifth layer

        self.pos_embed = Embedding(pos_freq)
        self.dir_embed = Embedding(dir_freq)

        self.pos_encoder = nn.ModuleList([])
        for i in range(8):
            if i == 0:
                input_dim = d_pos * pos_freq * 2 + 3
            elif i == self.skip:
                input_dim = (d_pos * pos_freq * 2 + 3) + dim
            else:
                input_dim = dim
            self.pos_encoder.append(nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU()))

        self.feat_encoder = nn.Linear(dim, dim)
        self.sigma = nn.Sequential(nn.Linear(dim, 1), nn.ReLU())
        self.dir_encoder = nn.Sequential(nn.Linear(dim + (d_dir * dir_freq * 2 + 3), dim//2), nn.ReLU())
        self.rgb = nn.Sequential(nn.Linear(dim//2, 3), nn.Sigmoid())

    @staticmethod
    def positional_encoding(x: Tensor, dim: int):
        out = [x]
        for j in range(dim):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
    
    def forward(self, pos: Tensor, dir: Tensor):
        # pos, dir = torch.split(x, [3, 3], dim=-1)
        pos, dir = self.pos_embed(pos), self.dir_embed(dir)
        residual = pos.clone()

        for i, layer in enumerate(self.pos_encoder):
            if i == self.skip:
                pos = torch.cat([pos, residual], dim=-1)
            pos = layer(pos)
            
        sigma = self.sigma(pos)
        feat = self.feat_encoder(pos)
        feat = self.dir_encoder(torch.cat([feat, dir], dim=-1))
        rgb = self.rgb(feat)
        # return torch.cat([rgb, sigma], dim=-1)
        return rgb, sigma
    

if __name__ == '__main__':
    d_pos = 3
    d_dir = 3
    bs = 100
    model = MLP()
    x = torch.randn(bs, d_pos+d_dir)

    y = model(x)
    print(y.shape)