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
from config.train_cfg_pcc import XrayPointsDataset
import random
from copy import deepcopy


class XrayPointsCTDatasetV2(Dataset):
    def __init__(
            self,
            cfg: XrayPointsDataset ,
            path_dict , 
            mode = 'train'                      
        ):
        super().__init__()
        
        self.data_root = cfg.root
        if mode == 'train':
            files_list = cfg.train_files_list
        elif mode == 'test':
            files_list = cfg.test_files_list
        name_list = get_filesname_from_txt(files_list)
        random.shuffle(name_list)

        # load dataset info
        # load dataset config
        with open(cfg.geo_config_path, 'r') as f:
            dst_cfg = yaml.safe_load(f)
            out_res = np.array(dst_cfg['dataset']['resolution'])
            self.geo = Geometry(dst_cfg['projector'])
        self.geo_cfg = dst_cfg
        self.block_size = dst_cfg['dataset']['block_size'][0]
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.data_root, self._path_dict[key])
            self._path_dict[key] = path

    
        # prepare points
        if mode == 'train':
            # load blocks' coordinates [train only]
            self.blocks = np.load(self._path_dict['blocks_coords'])
        else:
            # prepare sampling points
            points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]] # TODO: CT resolution
            points = points.astype(np.float32)
            points = points.reshape(3, -1)
            points = points.transpose(1, 0) # N, 3
            self.points = points / (out_res - 1)
        
        # other parameters
        self.name_list = name_list
        self.is_train = (mode == 'train')
        self.num_views = cfg.n_views
        #self.random_views = random_views
        #self.view_offset = view_offset

        # for acceleration when testing
        self.points_proj = None

    def __len__(self):
        return len(self.name_list)
    
    def sample_projections(self, name, n_view=None):
        # -- load projections
        with open(os.path.join(self.data_root, self._path_dict['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        if n_view is None:
            n_view = self.num_views

        # -- sample projections
        views = np.linspace(0, len(projs), n_view, endpoint=False).astype(int) # endpoint=False as the random_views is True during training, i.e., enabling view offsets.
        #offset = np.random.randint(len(projs) - views[-1]) if self.random_views else self.view_offset
        #views += offset

        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]

        # -- de-normalization
        projs = projs * projs_max / 0.2

        return projs, angles
    
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join( self.data_root, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32
        # if self.out_res_scale < 1.:
        #     image = scipy.ndimage.zoom(image, self.out_res_scale, order=3, prefilter=False)
        return image
    
    def load_block(self, name, b_idx):
        path = self._path_dict['blocks_vals'].format(name, b_idx)
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values):
        # values [block]: uint8
        #choice = np.random.choice(len(points), size=self.npoint, replace=False)
        points = points
        values = values
        values = values.astype(np.float32) / 255.
        return points, values

    def project_points(self, points, angles):
        points_proj = []
        for a in angles:
            p = self.geo.project(points, a)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0) # [M, N, 2]
        return points_proj

    def __getitem__(self, index):
        name = self.name_list[index]
        pdb.set_trace()
        # -- load projections
        projs, angles = self.sample_projections(name)

        # -- load sampling points
        if not self.is_train:
            points = self.points
            points_gt = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # [N, 3]
            pdb.set_trace()
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt[None, :]

        # -- project points
        if self.is_train or self.points_proj is None:
            points_proj = self.project_points(points, angles) # given the same geo cfg
            self.points_proj = points_proj
        else:
            points_proj = self.points_proj
        #pdb.set_trace()
        points = points.transpose(1,0)  # fisrt transpose   3 , N  and then reshape (3 , block_size , block_size , block_siez)  for keep order 
        points = points.reshape(3, self.block_size ,self.block_size ,self.block_size)
        points_gt = points_gt.reshape(1,self.block_size , self.block_size , self.block_size)
        pdb.set_trace()

        # -- collect data
        ret_dict = {
            # M: the number of views
            # N: the number of sampled points
            'dst_name': self.dst_name,
            'name': name,
            'angles': angles,           # [M,]
            'projs': projs,             # [M, 1, W, H], projections
            'points': points,           # [3 , H ,W, D], center xyz of volumes ~[0, 1]
            'points_gt': points_gt,     # [1, H , W ,D] (or [W', H', D'] only when is_train is False)
            'points_proj': points_proj, # [M, N, 2]
        }
        return ret_dict
