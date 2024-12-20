import inspect
from typing import Optional

import os
import sys
import random
import torch
import torch.nn.functional as F
import pdb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from torch import Tensor

from model.denoised_model import DenoisedModel
from model.project_condition_model import ProjectionImplictConditionModel
from utils_file.model_utils import get_custom_betas

from einops import rearrange

class Points_WiseImplict_ConditionDiffusionModel(ProjectionImplictConditionModel):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        #pcc_net models
        denoised_model_config: dict = {},
        **kwargs
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        #! denoise model  
        self.noised_pred_model = DenoisedModel(**denoised_model_config)

    def forward_train(self, 
                      coords_idensity: Optional[Tensor] ,  #  ( b , h ,  w ,  d , 3 + 1)  3 is x y z 1 is idensity
                      points_proj: Optional[Tensor] , # (b ,  , n_points , d )
                      xray_projs: Optional[Tensor] ,  # (b , 2 , 1 , u ,v )
                    return_intermediate_steps: bool = False
        ):
        B  =   coords_idensity.shape[0]
        coords = coords_idensity[:,:,:,:,:3]
        # x_0 is idensity  only add noise to idensity 
        # because we want learn identisy ditribution 
        x_0 = coords_idensity[:,:,:,:,3:] 
        #pdb.set_trace()
        # Sample random noise from idensity 
        noise = torch.randn_like(x_0)
        
        # Sample random timesteps for each idensity
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), 
            device=self.device, dtype=torch.long) 
        
        # Add noise to idensity 
        x_t =  self.scheduler.add_noise(x_0, noise, timestep)
        #pdb.set_trace()
        # Conditioning
        x_t_input = self.get_input_with_conditioning(x_t, xray_projs , points_proj , coords)
        #pdb.set_trace()
        # Forward predict nosie 
        noise_pred = self.noised_pred_model(x_t_input, timestep)
        # Check
        noise_pred = rearrange(noise_pred," b c h w d -> b h w d c")
        #pdb.set_trace()
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
        #pdb.set_trace()
        loss = F.mse_loss(noise_pred , noise)

        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)
        
        return loss
    
    def process_batch(self, batch: dict):
        angles = batch['angles']
        proj = batch['projs']
        points = batch['points']
        idensity = batch['points_gt']
        points_proj = batch['points_proj']    
        #pdb.set_trace()
        #points = rearrange(points , ' b n c -> b c n')
        coords_idensity = torch.cat([points , idensity]  , dim=-1)

        return coords_idensity , proj , points_proj




    def forward(self, batch: dict , mode = 'train', **kwargs):
        coords_idensity , proj , points_proj  = self.process_batch(batch)
        coords_idensity, proj, points_proj = self.process_batch(batch)
        # print("Feature stats:", {
        #     "coords_range": (coords_idensity.min(), coords_idensity.max()),
        #     "proj_range": (proj.min(), proj.max()),
        #     "points_proj_range": (points_proj.min(), points_proj.max())
        # })


        return self.forward_train(coords_idensity , points_proj , proj)


         
