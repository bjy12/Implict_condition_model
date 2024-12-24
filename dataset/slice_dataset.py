import os
import json
import yaml
import scipy
import pickle
import numpy as np
import pdb
from torch.utils.data import Dataset

from dataset.data_utils import sitk_load , get_filesname_from_txt
from dataset.geometry_v1 import Geometry
from config.train_cfg_pcc import XrayPointsDataset , SliceDatasetConfig 
import random
import glob
from copy import deepcopy





class SliceDataset(Dataset):
    def __init__(self,
                 cfg: SliceDatasetConfig , 
                 mode = 'train'):
        super().__init__()

        self.data_root = cfg.root
        if mode == 'train':
            files_list = cfg.train_files_list
        elif mode == 'test':
            files_list = cfg.test_files_list
        name_list = get_filesname_from_txt(files_list)
        random.shuffle(name_list)
        self.path_list = []
        for name in name_list:
            pattern = f"{name}_*_*.npz"
            search_path = os.path.join(self.data_root, pattern)
            matched_files = glob.glob(search_path)
            self.path_list.extend(matched_files)
        random.shuffle(self.path_list)
        #pdb.set_trace()
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        path = self.path_list[index]
        npz = np.load(path)
        #pdb.set_trace()
        value  = npz['intensity']
        value  = value.astype(np.float32) / 255.
        value  = value[None,...]
        coords = npz['coordinates']
        coords = (coords - 0.5) * 2 

        ret_dict ={

            'value' : value,
            'coords' : coords,
        }

        return ret_dict


