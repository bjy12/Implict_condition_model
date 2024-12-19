from dataclasses import dataclass 
from dataset.dataset import XrayPointsCTDataset

# main 
if __name__ == '__main__':

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
    PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy'}
    config = XrayPointsDataset()
    dataset = XrayPointsCTDataset(config ,PATH_DICT , 'train' )

    sample = dataset[0]