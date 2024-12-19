import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor

from model.pcc.network import ContextCluster_Denoised




class DenoisedModel(ModelMixin , ConfigMixin):
    @register_to_config
    def __init__(self,
                 model_type: str = 'pcc', 
                 layers=[1, 1, 1, 1],
                 norm_layer='GroupNorm',
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 downsamples=[True, True, True, True],
                 proposal_w=[2, 2, 2, 2],
                 proposal_h=[2, 2, 2, 2],
                 proposal_d=[2, 2, 2, 2],
                 fold_w=[1, 1, 1, 1],
                 fold_h=[1, 1, 1, 1],
                 fold_d=[1, 1, 1, 1],
                 heads=[4, 4, 8, 8],
                 head_dim=[24, 24, 24, 24],
                 down_patch_size=3,
                 down_pad=1,
                 with_coord=False,
                 time_embed_dims = None,
                 sample_size = 64, 
                 in_channels = 260,
                 out_channels = 1, 
                 ):
        super().__init__()
        self.sample_size = sample_size


        self.model_type = model_type

        if self.model_type == 'pcc':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = ContextCluster_Denoised(
                            layers=layers, embed_dims=embed_dims, norm_layer=norm_layer,
                            mlp_ratios=mlp_ratios, downsamples=downsamples,
                            down_patch_size=down_patch_size, down_pad=down_pad,
                            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                            heads=heads, head_dim=head_dim,with_coord=with_coord , time_embed_dims=time_embed_dims,
                            in_channels = in_channels , out_channels = out_channels) 
        else:
            raise NotImplementedError   
    def forward(self , inputs: Tensor , t: Tensor) -> Tensor:
        with self.autocast_context:

            return self.model(inputs , t)