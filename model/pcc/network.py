# -*- coding = utf-8 -*-
import random

import torch
import torch.nn as nn
from timm.models.layers import to_3tuple
from torch.autograd import Variable

#from model.context_cluster3D import ContextCluster, basic_blocks
from model.pcc.pcc_net import ContextCluster_Denoised 
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=5, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# generator for PCC-GAN
class PCC_Net(nn.Module):
    def __init__(self,
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
                 in_channels = 4 ,
                 out_channels = 1 , 
                 ):
        super(PCC_Net, self).__init__()
        self.sample_size = sample_size 
        self.in_channels = 4
        self.out_channels = 1 

        # generator for PCC-GAN
        self.CoCs = ContextCluster_Denoised(
            layers=layers, embed_dims=embed_dims, norm_layer=norm_layer,
            mlp_ratios=mlp_ratios, downsamples=downsamples,
            down_patch_size=down_patch_size, down_pad=down_pad,
            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim,with_coord=with_coord , time_embed_dims=time_embed_dims
        )


    def forward(self, x , time_step):
        EPET = self.CoCs(x , time_step)
        return EPET
