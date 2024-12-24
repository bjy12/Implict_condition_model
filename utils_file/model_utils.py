import numpy as np
import torch
import torch.nn as nn


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)





def default(x, d):
    return d if x is None else x





def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas

def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]

class Embedder:
    def __init__(self, num_freqs):
        self.num_freqs = num_freqs

        embed_fns = []
        d = 1 # input_dims
        out_dim = 0

        # include input
        embed_fns.append(lambda x: x)
        out_dim += d

        freq_bands = 2. ** torch.linspace(0., num_freqs-1, num_freqs)

        for freq in freq_bands:
            embed_fns.append(lambda x: torch.sin(x * freq))
            out_dim += d
            embed_fns.append(lambda x: torch.cos(x * freq))
            out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)