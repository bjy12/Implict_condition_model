import torch
from config.train_cfg_pcc import XrayPointsDataset ,  DataloaderConfig , ProjectConfig
from .dataset import XrayPointsCTDataset


PATH_DICT = {
            'image': 'images/{}.nii.gz',
            'projs': 'projections/{}.pickle',
            'projs_vis': 'projections_vis/{}.png',
            'blocks_vals': 'blocks/{}_block-{}.npy',
            'blocks_coords': 'blocks/blocks_coords.npy'
            }



def get_dataset(cfg: ProjectConfig):
    dataset_cfg : XrayPointsDataset = cfg.dataset
    dataloader_cfg : DataloaderConfig = cfg.dataloader


    train_dataset = XrayPointsCTDataset(dataset_cfg , PATH_DICT , cfg.run.job )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers)
    

    return train_data_loader




