import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from torch import Tensor

from model.pcc.network import ContextCluster_Denoised
from model.SongUNet.networks import SongUNet
from typing import Any, Dict, List, Optional





class DenoisedModel(ModelMixin , ConfigMixin):
    @register_to_config
    def __init__(self,
                 model_type: str = 'pcc', 
                 pcc_config: Optional[dict] = None,
                 unet_config: Optional[dict] = None, 
                 ):
        super().__init__()

        self.model_type = model_type

        if self.model_type == 'pcc':
            if pcc_config is None:
                raise ValueError("PCC config is required when model_type is 'pcc'")
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = ContextCluster_Denoised(
                **pcc_config
            )
            # self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            # self.model = ContextCluster_Denoised(
            #                 layers=layers, embed_dims=embed_dims, norm_layer=norm_layer,
            #                 mlp_ratios=mlp_ratios, downsamples=downsamples,
            #                 down_patch_size=down_patch_size, down_pad=down_pad,
            #                 proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            #                 fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            #                 heads=heads, head_dim=head_dim,with_coord=with_coord , time_embed_dims=time_embed_dims,
            #                 in_channels = in_channels , out_channels = out_channels) 
        elif self.model_type == 'unet':
            if unet_config is None:
                raise ValueError("UNet config is required when model_type is 'unet'")
            self.model = SongUNet(
                **unet_config
            )
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}")             
    def forward(self , inputs: Tensor , t: Tensor) -> Tensor:
        if self.model_type == 'pcc':
            with self.autocast_context:
                return self.model(inputs , t)
        elif self.model_type == 'unet':
            return self.model(inputs , t , class_labels=None) 