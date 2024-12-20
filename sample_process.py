import torch
from tqdm import tqdm
from config.train_cfg_pcc import ProjectConfig
from typing import Any, Iterable, List, Optional
from accelerate import Accelerator
from pathlib import Path

@torch.no_grad
def sample(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = 'sample',
):
    
    model.eval()
    progress_bar = tqdm(dataloader , disable=(not accelerator.is_main_process))

    output_dir: Path = Path(output_dir)

    #? Evaluate way to do  PSNR SSIM  qualitity of reconstruction CT 

    loss_dict = {

    }
    # sample_way  
    for batch_idx , batch in enumerate(progress_bar):
        # only patch wise sample 
        model(batch , mode='sample' ,   )



