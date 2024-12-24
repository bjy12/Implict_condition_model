import numpy as np
import os
import sys
import yaml
import pdb
import SimpleITK as sitk
import glob
from tqdm import tqdm
def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    #pdb.set_trace()
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing


def get_filesname_from_txt(txt_file_path):
    files = []
    with open(txt_file_path, 'r') as f:
        file_name = f.readlines()
        for file in file_name:
            file_name = file.strip()
            #file_path = os.path.join(base_dir, file_name)
            files.append(file_name)   
    return files


class CT_Coord_Slicer:
    def __init__(self , ct_root , name_list_file , ct_res ,value_range):
        self.ct_root = ct_root
        self.name_list = get_filesname_from_txt(name_list_file)
        self.ct_res = ct_res
        self._value_range =  value_range
        range_ct = np.arange(ct_res)
        self.coords = np.stack(np.meshgrid(range_ct, range_ct, range_ct, indexing='ij'), axis=0)
        #pdb.set_trace()
        ct_path_list = []
        for name in self.name_list:
            path = os.path.join(self.ct_root , f"{name}.nii.gz")
            ct_path_list.append(path)
        self.ct_path_list = ct_path_list
    def get_all_slices_with_meta(self , save_dir):
        os.makedirs(save_dir , exist_ok=True)
        axes_names = ['axial', 'sagittal', 'coronal']
        for ct_idx, ct_path in enumerate(tqdm(self.ct_path_list, desc="Processing CTs")):
            #pdb.set_trace()
            ct_name = ct_path.split('\\')[-1].split('.')[0]
            image , spacing = sitk_load(ct_path)
            #pdb.set_trace()
            image = self.normalize(image)
            for aixs , axis_name in enumerate(axes_names):

                n_slices = image.shape[0]

                for slice_idx in range(n_slices):
                    if aixs == 0:
                        slice_image = image[slice_idx, :, :]
                        slice_coords =  self.coords[:,slice_idx,:,:]
                    elif aixs == 1:
                        slice_image = image[:,slice_idx, :]
                        slice_coords =  self.coords[:,:,slice_idx,:]
                    else:
                        slice_image = image[:,:, slice_idx]
                        slice_coords =  self.coords[:,:,:,slice_idx]

                    #pdb.set_trace()
                    slice_image = (slice_image * 255).astype(np.uint8)
                    slice_coords = slice_coords / (self.ct_res - 1 )
                    slice_coords = slice_coords.astype(np.float32)
                    save_file = os.path.join(save_dir , f"{ct_name}_{axis_name}_{slice_idx}")
                    np.savez(
                        save_file,
                        intensity=slice_image,
                        coordinates=slice_coords,
                        slice_idx=slice_idx,
                        ct_name=ct_name,
                        axis=axis_name
                    )

    def normalize(self, image) :
        min_value, max_value = self._value_range
        image = image
        image = np.clip(image, a_min=min_value, a_max=max_value)
        image = (image - min_value) / (max_value - min_value)
        #pdb.set_trace()
        return image       



if __name__ == "__main__":
    # ct_root = 'F:/Data_Space/Pelvic1K/centrolize_images'
    # name_list_file = 'F:/Code_Space/Implict_condition_model/dataset/files_list/pelvic_coord_train_16.txt'
    # ct_res = 256
    # ct_slicer = CT_Coord_Slicer(ct_root , name_list_file , ct_res , value_range=[0, 255])
    # #pdb.set_trace()
    save_root = 'F:/Data_Space/Pelvic1K/slice_aixs_dataset'
    pattern = "0001_*_*.npz"
    search_path = os.path.join(save_root, pattern)
    matched_files = glob.glob(search_path)
    pdb.set_trace()
    # ct_slicer.get_all_slices_with_meta(save_root)

    # demo_slic_paht = 'F:/Code_Space/Implict_condition_model/slice_aixs/0047_axial_0.npz'
    # slice = np.load(demo_slic_paht)
    # pdb.set_trace()
    

