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
from tqdm import tqdm

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
    
    @torch.no_grad()
    def forward_sample(self, 
                       coords_idensity: Optional[Tensor] ,  #  ( b , h ,  w ,  d , 3 + 1)  3 is x y z 1 is idensity
                       points_proj: Optional[Tensor] , # (b ,  , n_points , d )
                       xray_projs: Optional[Tensor] ,  # (b , 2 , 1 , u ,v )
                       scheduler: Optional[str] = 'ddpm' ,
                       num_inference_steps: Optional[int] = 1000,
                       eta: Optional[float] = 0.0,  # for DDIM
                       return_sample_every_n_steps: int =  100 ,
                       disable_tqdm: bool = False,
    ):  
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        device = self.device
        #pdb.set_trace()
        b , h , w , d , _ = coords_idensity.shape
        coords = coords_idensity[...,:3]
        idensity = coords_idensity[...,3:4]
        c = 1  # only predict idensity
        x_t = torch.randn(b , h , w , d ,c , device=device)

        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}
        
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)
        for i , t in enumerate(progress_bar):
            x_t_input = self.get_input_with_conditioning(x_t ,  xray_projs ,  points_proj , coords )
            #pdb.set_trace()
            noise_pred = self.noised_pred_model(x_t_input,  t.reshape(1).expand(b))
            #pdb.set_trace()
            x_t = rearrange(x_t , ' b h w d c -> b c h w d ')
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
            x_t = rearrange(x_t , ' b c h w d -> b h w d c')
            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        idensity = idensity.cpu().numpy()
        if return_all_outputs:
            pdb.set_trace()
            all_outputs = torch.stack(all_outputs, dim=1)
            all_outputs = (all_outputs / 2 + 0.5 ).clamp(0,1)
            all_outputs.cpu().numpy()
            return all_outputs , idensity
        else:
            x_t = (x_t / 2 + 0.5 ).clamp(0,1)
            sample_result = x_t.cpu().numpy()
            return sample_result , idensity


    def forward(self, batch: dict , mode = 'train', **kwargs):
        coords_idensity , proj , points_proj  = self.process_batch(batch)
        if mode == 'train':
            return self.forward_train(coords_idensity , points_proj , proj)
        elif mode == 'sample':
            sample_params = {
                'num_inference_steps': kwargs.get('num_inference_steps', 1000),
                'eta': kwargs.get('eta', 0.0),
                'return_sample_every_n_steps': kwargs.get('return_sample_every_n_steps', 100),
                'disable_tqdm': kwargs.get('disable_tqdm', False),
                'scheduler': kwargs.get('scheduler', 'ddpm')
            }            
            return self.forward_sample(coords_idensity , points_proj , proj  , **sample_params)
        # print("Feature stats:", {
        #     "coords_range": (coords_idensity.min(), coords_idensity.max()),
        #     "proj_range": (proj.min(), proj.max()),
        #     "points_proj_range": (points_proj.min(), points_proj.max())
        # })


        return 


         
