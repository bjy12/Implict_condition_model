import torch
import torch.nn as nn

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim  ## self.kwargs["input_dims"] + self.kwargs["input_dims"] * 2 * self.kwargs["num_freqs"]

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class GlobalFeatureFilter(nn.Module):
    def __init__(self, input_ch , output_ch):
        super(GlobalFeatureFilter, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        if self.input_ch != self.output_ch:
            # 可以添加更复杂的变换
            self.transform = nn.Sequential(
                nn.Linear(self.input_ch, self.input_ch * 2),
                nn.ReLU(),
                nn.Linear(self.input_ch * 2, self.output_ch)
            )
    
    def forward(self, x):
        # x: (B, H, W, D, C)
        if self.input_ch != self.output_ch:
            B, H, W, D, C = x.shape
            x = x.reshape(B * H * W * D, C)
            x = self.transform(x)
            x = x.reshape(B, H, W, D, -1)
        
        return {'outputs': x}