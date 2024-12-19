import sys 
import numpy as np
import yaml 
import os 
import pickle
import pdb
import torch
import torch.utils.data as data
import SimpleITK as sitk
from copy import deepcopy
from dataset.data_utils import get_filesname_from_txt,sitk_load
from dataset.geometry import Geometry
from config.train_cfg_pcc import XrayPointsDataset

import random
from dataclasses import dataclass
import pdb
class XrayPointsCTDataset(data.Dataset):
    def __init__(self,  cfg: XrayPointsDataset , path_dict , job='train'):
        super().__init__()
        # path setting 
        #pdb.set_trace()
        self.root_dir = cfg.root
        files_list = cfg.files_list 
        self.files_name_list = get_filesname_from_txt(files_list)
        random.shuffle(self.files_name_list)
        #pdb.set_trace()
        # _path_dict setting 
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path
        #pdb.set_trace()
        # geo setting  for projection points to 2d from 3d 
        with open(cfg.geo_config_path , 'r') as f:
            self.config_geo = yaml.safe_load(f)
        #pdb.set_trace()
        self.geo = Geometry(self.config_geo['projector'])       
        #coords setting
        self.blocks = np.load(self._path_dict['blocks_coords'])
        #view_setting 
        self.n_view = cfg.n_views
        #sample poitns setting
        self.sample_points_type = cfg.sample_points_type
        self.blocks_size = cfg.blocks_size
        self.npoints = (self.blocks_size ** 3)
    def __len__(self):
        return len(self.files_name_list)


    def load_path_dict(self, path_dict):
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path  

    def load_path_dict(self, path_dict):
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path         
    

    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.root_dir, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32

        return image
    
    def load_block(self, name , b_idx):
        #pdb.set_trace()
        path = self._path_dict['blocks_vals'].format(name, b_idx)
        block = np.load(path) # uint8
        return block    
    
    def sample_projections(self, name , n_view=None):
        #* view index 1 is AP 
        #* view index 0 is LV
        #* if n_view is 1 then will use AP view 
        with open(os.path.join(self.root_dir, self._path_dict['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']
            angles = data['angles'] 
            projs_max = data['projs_max']
        #pdb.set_trace()
        if n_view is None:
            n_view = self.n_view
        views = np.linspace(0,len(projs) , n_view , endpoint=False).astype(int)
        #pdb.set_trace()
        projs = projs[views].astype(np.float32) / 255.0

        # norm to [-1 , 1 ]
        projs = (projs * 2. ) - 1. 

        projs = projs[:,None , ...]
        angles = angles[views]
        #* not sure if this is correct
        #projs = projs * projs_max / 0.2

        return projs , angles

    def project_points(self,points,angles):
        
        points_proj = []
        for a in angles:
            p  = self.geo.project(points,a)
            points_proj.append(p)
            
        points_proj = np.stack(points_proj, axis=0).astype(np.float32) # [M,N,2]
        
        return points_proj
    
    def sample_points(self, points, values):
        
        choice = np.random.choice(len(points), size=self.npoints, replace=False)
        #pdb.set_trace()
        points = points[choice]
        values = values[choice]
        values = values.astype(np.float32) / 255.
        # norm
        values = (values * 2 ) - 1

        return points , values      
    
    def get_blocks_random(self , name):
        b_idx = np.random.randint(len(self.blocks))
        
        block_values = self.load_block(name , b_idx )

        block_coords = self.blocks[b_idx]

        return block_values , block_coords
    
    def get_blocks_overlap(self , name):
        b_idx = np.random.randint(len(self.blocks))
 
        block_values = self.load_block(name , b_idx )
        block_values = block_values.astype(np.float32) / 255.
        
        # norm
        block_values = (block_values * 2 ) - 1
        coords = self.blocks[b_idx]
        #pdb.set_trace()
        temp = np.concatenate([coords , block_values] , axis=3 )
        temp = temp.reshape(-1,4)
        points = temp[:,:3]
        #pdb.set_trace()


        return coords , points , block_values

    def __getitem__(self, index):
        #pdb.set_trace()
        name = self.files_name_list[index]
        #get xray images 
        projs , angles = self.sample_projections(name)
        #pdb.set_trace()
        if self.sample_points_type == 'block_random':
            block_values , block_coords = self.get_blocks_random(name)
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt[None, :]
            points_proj = self.project_points(points,angles)

        elif self.sample_points_type == 'overlap_block':
            points ,coords , points_gt = self.get_blocks_overlap(name)
            points_proj = self.project_points(coords, angles)
            points =  (points - 0.5 ) * 2 
            #pdb.set_trace()

        
        #pdb.set_trace()

        ret_dict = { 
            'name': name,
            'angles': angles,           # [M,]
            'projs': projs,             # [M, 1,  u,  v], projections
            'points': points,           # [N, H , W , D ,3], center xyz of volumes ~[-1, 1]
            'points_gt': points_gt,     # [N, H , W , D ,1] (or [W', H', D'] only when is_train is False)
            'points_proj': points_proj, # [M, N , 2 ]
        }

        return ret_dict


        
# main 
if __name__ == '__main__':

    @dataclass
    class DatasetConfig:
        type: str

    @dataclass
    class XrayPointsDataset(DatasetConfig):
        type: str = 'XrayPoints'
        root: str = 'F:/Data_Space/Pelvic1K/cnetrilize_blocks_64/'
        files_list: str = './dataset/files_list/pelvic_coord_train_16.txt'
        geo_config_path: str = './config/geo_config/config_block_64.yaml'
        blocks_size : int = 64 

    PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy'}

    dataset = XrayPointsCTDataset(XrayPointsCTDataset ,PATH_DICT , 'train' )

    sample = dataset[0]


