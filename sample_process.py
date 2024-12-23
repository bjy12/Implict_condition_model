import os
import torch
from tqdm import tqdm
from config.train_cfg_pcc import ProjectConfig
from typing import Any, Iterable, List, Optional
from accelerate import Accelerator
from pathlib import Path
from dataset.data_utils import sitk_save

import pdb

@torch.no_grad
def sample(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = 'sample',
):
    pdb.set_trace()
    model.eval()
    progress_bar = tqdm(dataloader , disable=(not accelerator.is_main_process))

    output_dir: Path = Path(output_dir)
    pred_dir = os.path.join(output_dir , 'pred')
    gt_dir = os.path.join(output_dir, 'gt')
    os.makedirs(pred_dir , exist_ok=True)
    os.makedirs(gt_dir , exist_ok=True)
    #? Evaluate way to do  PSNR SSIM  qualitity of reconstruction CT 
    #pdb.set_trace()
    loss_dict = {

    }

    # sample_way  
    for batch_idx , batch in enumerate(progress_bar):
        # only patch wise sample 
        pdb.set_trace()
        name = batch['name']
        pred_idensity , gt_idensity = model(batch , mode='sample' ,
                                            num_inference_steps=cfg.run.num_inference_steps , 
                                            return_sample_every_n_steps= cfg.run.return_sample_every_n_steps)
        save_batch_image(pred_idensity,save_path=pred_dir, name=name, batch_idx=batch_idx , ts=cfg.run.num_inference_steps)
        save_batch_image(gt_idensity , save_path=gt_dir, name=name , batch_idx=batch_idx ,  ts=cfg.run.num_inference_steps)
        

def save_batch_image(images , save_path , name , batch_idx ,ts ):
    N = images.shape[0]
    #pdb.set_trace()
    for i in range(N):
        # Update progress bar
        print(f"Saving image {i + 1}/{N}...")
        pdb.set_trace()
        np_image = images[i,:,:,:,0]
        nii_save_path = os.path.join(save_path ,f"inf_ts_{ts}_b_idx_{batch_idx}_{i}_sample_.nii.gz")
        sitk_save(nii_save_path , np_image , uint8 = True)




