import numpy as np
import os
import sys
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from copy import deepcopy
import SimpleITK as sitk
import scipy
import yaml
from projector import Projector , visualize_projections
from saver import Saver , PATH_DICT
from tqdm import tqdm
import pdb



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



def sitk_save(path, image, spacing=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)


def _load_raw(path):
    #pdb.set_trace()
    #name = path.split('/')[-1].split('.')[0].split('_')[2]
    name = path.split('\\')[-1].split('.')[0]

    image, spacing = sitk_load(path)
    #data = {}
    return {
        'name': name,
        'image': image,
        'spacing': spacing
    }

def _resample(data , config ):
    # resample data (spacing)
    data['image'] = scipy.ndimage.zoom( 
        # numerical offsets depending on the value range 
        # NOTE: normalization first or later will (slightly) affect this
        data['image'], 
        data['spacing'] / config["spacing"], 
        order=3, 
        prefilter=False
    )
    data['spacing'] = deepcopy(np.array(config["spacing"]))
    return data

def _crop_pad(data , crop_size , value_range):
    # crop or add padding (resolution)
    processed = []
    original = []
    shape = data['image'].shape
    for i in range(3):
        if shape[i] >= crop_size[i]:
            # center crop
            processed.append({
                'left': 0,
                'right': crop_size[i]
            })
            offset = (shape[i] - crop_size[i]) // 2
            original.append({
                'left': offset,
                'right': offset + crop_size[i]
            })
        else:
            # padding
            offset = (crop_size[i] - shape[i]) // 2
            processed.append({
                'left': offset,
                'right': offset + shape[i]
            })
            original.append({
                'left': 0,
                'right': shape[i]
            })


    def slice_array(a, index_a, b, index_b):
        a[
            index_a[0]['left']:index_a[0]['right'],
            index_a[1]['left']:index_a[1]['right'],
            index_a[2]['left']:index_a[2]['right']
        ] = b[
            index_b[0]['left']:index_b[0]['right'],
            index_b[1]['left']:index_b[1]['right'],
            index_b[2]['left']:index_b[2]['right']
        ]
        return a
    
    # NOTE: 'mask' is used for evaluation with masked region, not used currently
    data['mask'] = slice_array( 
        np.zeros(crop_size),
        processed,
        np.ones_like(data['image']),
        original
    )
    data['image'] = slice_array(
        np.full(crop_size, fill_value=value_range[0], dtype=np.float32),
        processed,
        data['image'],
        original
    )
    return data

def _normalize_zoom_(data , value_range,normalize_type):
    min_value, max_value = value_range 
    image = data 
    image = np.clip(image, a_min=min_value, a_max=max_value)
    #pdb.set_trace()
    image_max = image.max()
    image_min = image.min()
    if normalize_type == "zero2one":
        image = (image - image_min) / (image_max - image_min)
    else:
    # normalize data to -1 1 
        image = (image - image_min) / (image_max - image_min)
        image = image * 2 - 1 
    
    return  image
def _generate_overlap_blocks(data ,ct_resolution , block_size , stride):
    block_size = np.array(block_size).astype(int)
    ct_resolution = np.array(ct_resolution).astype(int)
    ct_res = ct_resolution[0]
    nx, ny, nz = block_size
    ct = data['image']
    #pdb.set_trace()
    cuts_h = max(1, (ct_res - nx) // stride + 1)
    cuts_w = max(1, (ct_res - ny) // stride + 1)
    cuts_d = max(1, (ct_res - nz) // stride + 1)

    coords_h = np.arange(ct_res) 
    coords_w = np.arange(ct_res)
    coords_d = np.arange(ct_res)
    coords = np.stack(np.meshgrid(coords_h, coords_w, coords_d, indexing='ij'), axis=-1)
    #pdb.set_trace()
    total_cuts = cuts_h * cuts_w * cuts_d

    blocks_list = []
    coords_list = []
    value_list = []
    with tqdm(total=total_cuts) as pbar:
        for i in range(cuts_h):
            start_h = i * stride
            #pdb.set_trace()
            end_h = min(start_h + nx, ct_res)
            
            for j in range(cuts_w):
                start_w = j * stride
                end_w = min(start_w + ny, ct_res)
                
                for k in range(cuts_d):
                    start_d = k * stride
                    end_d = min(start_d + nz, ct_res)
                    #block_idx = i * cuts_w * cuts_d + j * cuts_d + k
                    coords_block = coords[start_h:end_h, start_w:end_w, start_d:end_d]    
                    #pdb.set_trace()
                    value_block = ct[start_h:end_h, start_w:end_w, start_d:end_d]
                    value_block = value_block[:,:,:,None]
                    value_list.append(value_block)
                    blocks_list.append(coords_block)
                    coords_list.append(coords_block.reshape(-1,3))
                    
                    pbar.update(1)   

    blocks_list = np.stack(blocks_list, axis=0) # [N, *, 3]
    #pdb.set_trace()
    blocks_list = blocks_list / (ct_resolution - 1) # coords starts from 0
    blocks_list = blocks_list.astype(np.float32)
    #pdb.set_trace()
    _block_info = {
        'coords': blocks_list,  #  M , H W D 3  
        'list': coords_list   #   a list [ N , 3 ] 
    }
    #pdb.set_trace()
    return _block_info , value_list

     

     
def _generate_blocks(_block_size , ct_resolution):
    #pdb.set_trace()
    _block_size = np.array(_block_size).astype(int)
    ct_resolution = np.array(ct_resolution).astype(int)
    nx, ny, nz = _block_size

    assert (ct_resolution % _block_size).sum() == 0, \
        f'resolution {ct_resolution} is not divisible by block_size {_block_size}'
    offsets = (ct_resolution / _block_size).astype(int)

    base = np.mgrid[:nx, :ny, :nz] # [3, nx, ny, nz]
    base = base.reshape(3, -1).transpose(1, 0) # [*, 3]
    base = base * offsets
    
    block_list = []
    for x in range(offsets[0]):
        for y in range(offsets[1]):
            for z in range(offsets[2]):
                block = base + np.array([x, y, z])
                block_list.append(block)
    #pdb.set_trace()
    blocks_coords = np.stack(block_list, axis=0) # [N, *, 3]
    #pdb.set_trace()
    blocks_coords = blocks_coords / (ct_resolution - 1) # coords starts from 0
    blocks_coords = blocks_coords.astype(np.float32)
    #pdb.set_trace()
    _block_info = {
        'coords': blocks_coords,
        'list': block_list
    }
    return _block_info
#* 随机切分block
def _convert_blocks(data , block_size , ct_res):
    block_info = _generate_blocks(block_size , ct_res)
    #pdb.set_trace()
    blocks_vals = [
        data['image'][b[:, 0], b[:, 1], b[:, 2]]
        for b in block_info['list']
    ]
    data['blocks_vals'] = blocks_vals
    data['blocks_coords'] = block_info['coords']
    return data , block_info['list']


def _convert_overlap_blocks(data , block_size , ct_res , stride = 32):
    block_info , values_blocks= _generate_overlap_blocks( data,ct_res ,  block_size , stride)
    #pdb.set_trace()
    # blocks_vals = [
    #     data['image'][b[:, 0], b[:, 1], b[:, 2]]
    #     for b in block_info['list']
    # ]
    data['blocks_vals'] = values_blocks
    #pdb.set_trace()
    data['blocks_coords'] = block_info['coords']
    return data 



#* 不随机切分block 
def _convert_blocks_v2(data , ct_res):
    ct = data['image']
    h , w ,d  = ct.shape
    range_ = np.arange(0, h , step=1)
    grid = np.meshgrid(range_ , range_ , range_)
    coords = np.stack(grid,axis=0)
    coord_index = coords.reshape(3,-1).transpose(1,0)
    #pdb.set_trace()
    coords_norm = coord_index / (ct_res[0] - 1)
    coords_norm = coords_norm.astype(np.float32)

    coords_vals = ct[coord_index[:,0] ,coord_index[:,1] ,coord_index[:,2]]
    data['blocks_vals'] = coords_vals
    data['blocks_coords'] = coords_norm

    #pdb.set_trace()

    return data 

#* 从图像中心切割出一个固定大小的图像
def _convert_blocks_v3(data  , crop_size):
    ct = data['image']
    h , w ,d  = ct.shape
    range_ = np.arange(0, crop_size , step=1)
    grid = np.meshgrid(range_ , range_ , range_)
    coords = np.stack(grid,axis=0)
    coord_index = coords.reshape(3,-1).transpose(1,0)
    #* 从图像中心切割出一个固定大小的图像

    #pdb.set_trace()
    coords_norm = coord_index / (crop_size - 1)
    coords_norm = coords_norm.astype(np.float32)

    # get the mid 
    mid_index = h // 2 
    start_index = int(mid_index-(crop_size/2))
    end_index = int(mid_index+(crop_size/2))
    coords_vals = ct[start_index: end_index ,start_index : end_index,start_index :end_index]
    data['blocks_vals'] = coords_vals
    data['blocks_coords'] = coords_norm
    data['image'] = coords_vals

    #pdb.set_trace()

    return data 



def _process(data , config , is_scale ):
    # data->resample->crop->normalize
    #pdb.set_trace()
    #* 正常处理
    if is_scale:
        resampled_data = _resample(data , config)
        data = _crop_pad(resampled_data , config['resolution'] , config['value_range'])
    #pdb.set_trace()
    img = data["image"]
    normalize_img = _normalize_zoom_(img , config['value_range'] , normalize_type="zero2one") 
    #pdb.set_trace()
    data['image'] = normalize_img
    return data

    
def process_dataset_demo( nii_path ,config , block_type='random'  , is_scale=True):

    ct_process_config = config['dataset']
    projector_config = config['projector']
    #pdb.set_trace()
    tigre_projector = Projector(config=projector_config)
    #pdb.set_trace()
    data = _load_raw(nii_path)
    
    #pdb.set_trace()
    # data =  sitk_load(nii_path)
    data = _process(data , ct_process_config , is_scale)
    #pdb.set_trace()
    if block_type == 'random':
        data  , block_list = _convert_blocks(data , ct_process_config['block_size'] , ct_process_config['zoom_size'])
    if block_type == 'full':
        data   = _convert_blocks_v2(data , ct_process_config['zoom_size'])
    if block_type == 'overlap':
        data  =  _convert_overlap_blocks(data ,ct_process_config['block_size'] , ct_process_config['zoom_size']  )
    #pdb.set_trace()
    projs = tigre_projector(data['image'])
    #pdb.set_trace()
    data.update(projs)
    
    # proj = projs['projs']
    # angles = projs['angles']
    #visualize_projections("./projs_img.png" , proj , angles)
    #visualize_projections("./projs_transpose.png" , proj[:, ::-1, :] , angles)

    return data



def visulize_proj_and_points_(projs , data , block_list):
    image = data['image'] # x y z 
    #image = image.transpose(2,1,0) # z y x 
    blocks_coord = data['blocks_coords']
    blocks_vals = data['blocks_vals']
    proj_image = projs['projs']
    angles = projs['angles']
    pdb.set_trace()


    #可视化3个方向的 slice 的切片
    axis_1 = np.linspace(0,127,128)
    axis_2 = np.linspace(0,127,128)

    mid_points = np.array(63)

    #slice_d_1_x , slice_d_1_y , slice_d_1_z = np.meshgrid(mid_points, axis_1 , axis_2 , indexing='ij')
    slice_d_2_x , slice_d_2_y , slice_d_2_z = np.meshgrid(axis_1, mid_points , axis_2 , indexing='ij')
    #slice_d_3_x , slice_d_3_y , slice_d_3_z = np.meshgrid(axis_1 , axis_2 , mid_points , indexing='ij')
    #pdb.set_trace()

    #slice_d1 = np.stack([slice_d_1_x,slice_d_1_y,slice_d_1_z] , axis=-1)
    slice_d2 = np.stack([slice_d_2_x,slice_d_2_y,slice_d_2_z] , axis=-1)
    #slice_d3 = np.stack([slice_d_3_x,slice_d_3_y,slice_d_3_z] , axis=-1)
    #pdb.set_trace()
    
    #slice_d1 =  slice_d1.reshape(-1,3).astype(int)
    slice_d2 =  slice_d2.reshape(-1,3).astype(int)
    #slice_d3 =  slice_d3.reshape(-1,3).astype(int)

    #slice_d1_v = image[slice_d1[:,0] , slice_d1[:,1],slice_d1[:,2]]
    pdb.set_trace()
    x_ = slice_d2[:,0]
    y_ = slice_d2[:,1]
    z_ = slice_d2[::-1,2]


    pdb.set_trace()
    slice_d2_v = image[x_ ,y_,z_]
    #slice_d3_v = image[slice_d3[:,0] , slice_d3[:,1],slice_d3[:,2]]
    #slice_d1_v = slice_d1_v.reshape(128,128)
    slice_d2_v = slice_d2_v.reshape(128,128).transpose(1,0)
    #slice_d3_v = slice_d3_v.reshape(128,128)
    
    
    # 创建一个包含三个子图的图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制 slice_d1_v
    # axes[0].imshow(slice_d1_v, cmap='gray')
    # axes[0].set_xlabel('X')
    # axes[0].set_ylabel('Y')
    # axes[0].set_title('Slice D1')

    # 绘制 slice_d2_v
    axes[1].imshow(slice_d2_v, cmap='gray')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Slice D2')

    # 绘制 slice_d3_v
    # axes[2].imshow(slice_d3_v, cmap='gray')
    # axes[2].set_xlabel('X')
    # axes[2].set_ylabel('Y')
    # axes[2].set_title('Slice D3')

    #调整子图之间的间距
    plt.tight_layout()

    #显示图像
    plt.show()
    #pdb.set_trace()
    # select random blocks list 
    indx = np.random.choice(int(len(slice_d2)), 1, replace=False)
    sp_points = slice_d2[indx[0]]
    pdb.set_trace()
     
    sp_points = sp_points / 127 
    sp_points = sp_points.astype(np.float32)

    slice_d2 = slice_d2 / 127 
    slice_d2 = slice_d2.astype(np.float32)

    sp_points = np.expand_dims(sp_points , axis=0)
    pdb.set_trace()
    sp_points[: , :2] -= 0.5
    sp_points[: , 2 ] = 0.5 - sp_points[:,2]
    sp_points *= 128.0 * 1.0

    slice_d2[: , :2] -= 0.5
    slice_d2[:,2] = 0.5 - slice_d2[:,2]
    slice_d2 *= 128.0 * 1.0
    pdb.set_trace()


  
    angle = -1 * angles[1]

    rot_M = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [            0,              0, 1]
    ])


    sp_points = sp_points @ rot_M.T
    slice_d2 = slice_d2 @ rot_M.T


    #select_block = select_block @ rot_M.T

    # points_coord = points_coord @ rot_M.T
    # slice_coord = slice_coord @ rot_M.T

    d2 = 1200
    d1 = 800

    coeff = (d2) / (d1 - sp_points[:, 0]) # N,
    d_sp_points = sp_points[:, [2, 1]] * coeff[:, None] # [N, 2] float

    # coeff = (d2) / (d1 - points_coord[:, 0]) # N,
    # d_points = points_coord[:, [2, 1]] * coeff[:, None] # [N, 2] float


    coeff_slice = (d2) / (d1 - slice_d2[:,0])
    pdb.set_trace()
    d_slice_d2 = slice_d2[:,[2,1]] * coeff_slice[:,None]
    pdb.set_trace()




    d_points_x = np.array([-400.0])
    d_points_x = np.expand_dims(d_points_x,axis=0)
    d_sp_points_xyz = np.concatenate([d_points_x , d_sp_points] , axis=1)
     
    d_points_x = d_points_x.repeat(16384,1)
    d_points_x = d_points_x.transpose(1,0)
    d_slice_proj = np.concatenate([d_points_x , d_slice_d2], axis=1)

    pdb.set_trace()
    #s_d_points_xyz = np.concatenate([d_points_x ,d_points ] , axis=1)   
    
    
    #pdb.set_trace()
    # d_points_x ,  points  ,  coord_center  , source_points 

    source_points = np.array([ 800.0 , 0.0,  0.0])
    coord_center  = np.array([ 0.0,0.0,0.0])

    
    # #pdb.set_trace()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # # 绘制点
    # pdb.set_trace()
    #ax.scatter(points_coord[:, 0], points_coord[:, 2], points_coord[:, 1], label='points')
    ax.scatter(sp_points[:, 0], sp_points[:, 2], sp_points[:, 1], label='sp_points')
    ax.scatter(slice_d2[:,0] , slice_d2[:,2] , slice_d2[:,1] , c= slice_d2_v , alpha = 0.1 , s = 1)
    ax.scatter(d_sp_points_xyz[:, 0], d_sp_points_xyz[:, 1], d_sp_points_xyz[:, 2], label='d_sp_points_xyz' )
    pdb.set_trace()
    ax.scatter(d_slice_proj[:,0] ,d_slice_proj[:,1] ,d_slice_proj[:,2] , c =proj_image[1]  ,alpha = 0.1 , s = 1 )
    ax.scatter(source_points[0], source_points[1], source_points[2], label='source_points')
    ax.scatter(coord_center[0], coord_center[1], coord_center[2], label='coord_center')
    # # 添加图例
    ax.legend()
    # # 显示图形
    plt.show() 

if __name__ == '__main__':
    #nii_path = 'F:/Data_Space/Pelvic1K/final_clip_img/dataset6_CLINIC_0001_data_post.nii'
    config_path = './data_process/data_process_cfg/config_block_64.yaml'
    with open(config_path , 'r') as f :
        config = yaml.safe_load(f)
    nii_root = 'F:/Data_Space/Pelvic1K/centrolize_images'
    root_dir = 'F:/Data_Space/Pelvic1K/cnetrilize_overlap_blocks_64/'
    blocks_type = 'overlap'
    save_blocks = True
    is_scale = False
    saver = Saver(root_dir=root_dir , path_dict = PATH_DICT , is_all_block=save_blocks)
    file_list = os.listdir(nii_root)
    for file_name in tqdm(file_list , desc='save coord blocks'):
        nii_file_path  = os.path.join(nii_root,file_name)
        #pdb.set_trace()
        data = process_dataset_demo(nii_file_path , config, block_type=blocks_type , is_scale=is_scale)
        #pdb.set_trace()
        saver.save(data)


        #pdb.set_trace()



  
    




