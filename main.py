import datetime
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional

import hydra
import torch
#import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

from utils_file import training_utils
from config.train_cfg_pcc import ProjectConfig
from model import get_model, Points_WiseImplict_ConditionDiffusionModel
from dataset import get_dataset
try:
    import lovely_tensors
    lovely_tensors.monkey_patch()
except ImportError:
    pass  # lovely tensors is not necessary but it really is lovely, I do recommend it

import warnings
warnings.filterwarnings("ignore") # torch.meshgrid

torch.multiprocessing.set_sharing_strategy('file_system')
import pdb
#auto make dirs config
@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):

    #pdb.set_trace()
    #logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    # Accelerator
    print(f'Current working directory: {os.getcwd()}')
    
    logging_dir = os.path.join(os.getcwd(),'logger')
    os.makedirs(logging_dir , exist_ok=True)
    #pdb.set_trace()
    accelerator_project_config = ProjectConfiguration(
         project_dir=os.getcwd(), logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
        cpu = cfg.run.cpu,
        mixed_precision=cfg.run.mixed_precision,
        log_with=cfg.logging.logger,
        project_config=accelerator_project_config
    )
    if cfg.logging.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training.")

    #pdb.set_trace()
    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    training_utils.set_seed(cfg.run.seed)
    # Mdoel 
    #pdb.set_trace()
    model = get_model(cfg)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    #pdb.set_trace()

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')
    #pdb.set_trace()
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    #Dataset
    #pdb.set_trace()
    dataloader_train = get_dataset(cfg)
    
    # Compute total training batch size
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    # Setup.
    #! to do add val dataset 
    model, optimizer, scheduler, dataloader_train = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train)

    #pdb.set_trace()
    if accelerator.is_main_process:
        accelerator.init_trackers('demo')
    model: Points_WiseImplict_ConditionDiffusionModel
    optimizer: torch.optim.Optimizer

    #pdb.set_trace()
    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataloader_train.dataset):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Batch size per device = {cfg.dataloader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
    print(f'    Max training steps = {cfg.run.max_steps}')
    print(f'    Training state = {train_state}')

    while True:
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, 
            header=log_header)
        
        for i , batch in enumerate(progress_bar):
            #pdb.set_trace()
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            #model.train()
            #pdb.set_trace()
            # Gradient accumulation
            with accelerator.accumulate(model):
                #pdb.set_trace()
                # Forward
                loss = model(batch, mode='train')

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # grad_norm_unclipped = training_utils.compute_grad_norm(model.parameters())  # useless w/ mixed prec
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1

                # Exit if loss was NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            # Gradient accumulation
            if accelerator.sync_gradients:

                # Logging
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    # 'grad_norm_unclipped': grad_norm_unclipped,  # useless w/ mixed prec
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                #pdb.set_trace()
                if (cfg.logging.logger_opt and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                    accelerator.log(log_dict, step=train_state.step)
            
                # Update EMA
                if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                    model_ema.update(model.parameters())
                
                # Save a checkpoint
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'model_ema': model_ema.state_dict() if model_ema else {},
                        'cfg': cfg
                    }
                    checkpoint_path = 'checkpoint-latest.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')
                
                #? Visulaize
                # End training after the desired number of steps/epochs
                if train_state.step >= cfg.run.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    
                    accelerator.end_training()
                    time.sleep(5)
                    return
                

        train_state.epoch += 1
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)




    


if __name__ == '__main__':
    main()