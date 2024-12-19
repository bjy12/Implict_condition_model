import numpy as np 
#from dataset.data_utils import sitk_load
import pdb
import SimpleITK as sitk


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


if __name__ == '__main__':
    nii_path = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/images/0001.nii.gz'

    sitk_img  , _ =  sitk_load(nii_path , uint8=True)

    coords_path = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/blocks/blocks_coords.npy'
    coords = np.load(coords_path)
    coord_100 = coords[100]
    coord_100 = coord_100.reshape(-1,3)
    selec_p = np.int32(coord_100 * 255 )
    pdb.set_trace()
    coords_value_path = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/blocks/0001_block-100.npy'
    pdb.set_trace()
    value = np.load(coords_value_path)
    value = value.reshape(-1,1)
    pdb.set_trace()
    p_v_img = sitk_img[selec_p[:,0] , selec_p[:,1] , selec_p[:,2]]
    # 
    value = value[: , 0]
    pdb.set_trace()
    are_equal = np.array_equal(p_v_img, value)
    print("p_v_img 和 selec_p_v 是否完全相等:", are_equal)



