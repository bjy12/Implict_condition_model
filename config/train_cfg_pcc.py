import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir

@dataclass
class BaseTrainingConfig:
    # Dir
    logging_dir: str
    output_dir: str

    # Logger and checkpoint
    logger: str = 'tensorboard'
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 20
    valid_epochs: int = 100
    valid_batch_size: int = 1
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None

    # Diffuion Models
    model_config: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100

    #pcc_net models
    layers: list = field(default_factory=lambda: [2, 2, 2, 2])  # 使用 default_factory
    embed_dims: list = field(default_factory=lambda: [64, 128, 256, 512])  # 64, 128, 256, 512
    mlp_ratios: list = field(default_factory=lambda: [8, 8, 4, 4])
    heads: list = field(default_factory=lambda: [4, 4, 8, 8])
    head_dim: list = field(default_factory=lambda: [24, 24, 24, 24])
    with_coord: bool = True
    time_embed_dims: list = field(default_factory=lambda: [16])
    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 1
 
    # Training
    seed: int = None
    num_epochs: int = 200
    train_batch_size: int = 2
    dataloader_num_workers: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    eval_vis: bool = False
    # Dataset
    dataset_name: str = None
    image_root: str = '/root/share/pcc_gan_demo/coords_with_idensity/train/coords'
    coord_root: str = None
    files_list_path: str = None


    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # EMA
    use_ema: bool = False
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4

    # Hub
    push_to_hub: bool = False
    hub_model_id: str = ''


@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './outputs/${run.name}/${now:%Y-%m-%d--%H-%M-%S}'


@dataclass
class RunConfig:

    name: str = 'implict_diff'
    job: str = 'train'
    mixed_precision: str = 'no'
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = False
    vis_before_training: bool = False
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 10_000
    checkpoint_freq: int = 10
    val_freq: int = 1_000
    vis_freq: int = 1_000
    log_step_freq: int = 1
    print_step_freq: int = 100

    # Inference config
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    num_sample_batches: Optional[int] = None
    sample_from_ema: bool = False 
    sample_save_evolutions: bool = True  # temporarily set by default

    # Training config
    freeze_feature_model: bool = True

    # Coloring training config
    coloring_training_noise_std: float = 0.0
    coloring_sample_dir: Optional[str] = None


@dataclass
class ImageEncoderConfig:
    n_channels: int = 1
    out_classes:  int =128
    bilinear:  bool = False


@dataclass
class PointCloudProjectionModelConfig:
        
    # Feature extraction arguments
    use_local_features: bool = True
    use_global_features: bool = True

    #points_wise_type 
    encoder_type: str = 'view_mixer'
    num_views: int = 2 


    # Image encoder parameters (in a dict to match __init__)
    image_encoder: Dict = field(default_factory=lambda: {
        'n_channels': 1,
        'n_classes': 128,
        'bilinear': False
    })
    
    #global feature process 

    gl_f_input_c: int = 319
    gl_f_output_c: int = 128

    # 
    merge_mode: str = 'concat'


@dataclass
class PointCloudDiffusionModelConfig(PointCloudProjectionModelConfig):
    # Diffusion arguments
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom'

    # Denoised model parameters (in a dict to match __init__)
    denoised_model_config: Dict = field(default_factory=lambda: {
        'model_type': 'pcc',
        'layers': [2, 2, 2, 2],
        'norm_layer': 'GroupNorm',
        'embed_dims': [64, 128, 256, 512],
        'mlp_ratios': [8, 8, 4, 4],
        'downsamples': [True, True, True, True],
        'proposal_w': [2, 2, 2, 2],
        'proposal_h': [2, 2, 2, 2],
        'proposal_d': [2, 2, 2, 2],
        'fold_w': [1, 1, 1, 1],
        'fold_h': [1, 1, 1, 1],
        'fold_d': [1, 1, 1, 1],
        'heads': [4, 4, 8, 8],
        'head_dim': [24, 24, 24, 24],
        'down_patch_size': 3,
        'down_pad': 1,
        'with_coord': True,
        'time_embed_dims': [16],
        'sample_size': 64,
        'in_channels': 260,
        'out_channels': 1
    })


@dataclass
class DatasetConfig:
    type: str

@dataclass
class XrayPointsDataset(DatasetConfig):
    type: str = 'XrayPoints'
    root: str = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/'
    files_list: str = 'F:/Code_Space/Implict_condition_model/dataset/files_list/pelvic_coord_train_16.txt'
    
    geo_config_path: str = 'F:/Code_Space/Implict_condition_model/config/geo_config/config_block_64.yaml'
    #sample_points setting
    blocks_size : int = 64 
    sample_points_type: str = 'overlap_block'

    #project setting 
    n_views: int = 2 


@dataclass
class DataloaderConfig:
    batch_size: int = 1 
    num_workers: int = 1 
        


@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 1e-3
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=1e-6,
    ))


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))

@dataclass
class LoggingConfig:
    logger: str = 'tensorboard'
    logger_opt: bool = True
    wandb_project: str = 'mhcdiff'

@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())

@dataclass
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    decay: float = 0.999
    update_every: int = 20

@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps="${run.max_steps}",
    ))
@dataclass
class CheckpointConfig:
    resume: Optional[str] = None
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True
@dataclass
class ProjectConfig:
    run: RunConfig
    logging: LoggingConfig #* logging option 
    optimizer: OptimizerConfig #* optimier option
    model: PointCloudDiffusionModelConfig #* model config
    ema: ExponentialMovingAverageConfig #* ema model use option
    scheduler: SchedulerConfig      #* learning rate scheduler
    dataset: DatasetConfig
    dataloader: DataloaderConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {'logging': 'default'},
        {'model': 'diffrec'},
        {'optimizer': 'adam'},
        {'scheduler': 'cosine'},
        {'ema': 'default'},
        {'dataset': 'XrayPoints'},
        {'dataloader':'default'},
        {'checkpoint': 'default'},
    ])



cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='diffrec', node=PointCloudDiffusionModelConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='dataset' ,  name='XrayPoints', node=XrayPointsDataset)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='config', node=ProjectConfig)
