import torch
from config.train_cfg_pcc import XrayPointsDataset ,  DataloaderConfig , ProjectConfig ,SliceDatasetConfig
from .dataset import XrayPointsCTDataset
from .dataset_v1 import XrayPointsCTDatasetV2
from .slice_dataset import SliceDataset
import pdb
PATH_DICT = {
            'image': 'images/{}.nii.gz',
            'projs': 'projections/{}.pickle',
            'projs_vis': 'projections_vis/{}.png',
            'blocks_vals': 'blocks/{}_block-{}.npy',
            'blocks_coords': 'blocks/blocks_coords.npy'
            }



def get_dataset(cfg: ProjectConfig):
    #dataset_cfg : XrayPointsDataset = cfg.dataset
    dataset_cfg : SliceDatasetConfig = cfg.dataset
    dataloader_cfg : DataloaderConfig = cfg.dataloader

    # train_dataset = XrayPointsCTDataset(dataset_cfg , PATH_DICT , 'train' )
    # test_dataset = XrayPointsCTDataset(dataset_cfg , PATH_DICT , 'test')

          
    train_dataset = SliceDataset(dataset_cfg , 'train')
    test_dataset  = SliceDataset(dataset_cfg , 'test')
    #pdb.set_trace()
    #train_dataset = XrayPointsCTDatasetV2(dataset_cfg , PATH_DICT , 'train' )
    #test_dataset = XrayPointsCTDatasetV2(dataset_cfg , PATH_DICT , 'test')



    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers)
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers)
    

    return train_data_loader , test_data_loader




