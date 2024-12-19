import os
import numpy as np
import sys
import scipy
import SimpleITK as sitk
import pdb


def compute_centroid(image):
    """计算图像的质心"""
    mask = image != 0  # 创建掩膜，选择非零像素
    return scipy.ndimage.center_of_mass(mask)  # 返回质心坐标 (z, y, x)

def centralize(image):
    """将图像中的非零对象转移到图像中心"""
    centroid = compute_centroid(image)  # 获取图像的质心坐标
    
    # 创建一个掩膜，选择非零像素
    mask = image != 0
    
    # 使用闭运算填充空洞
    filled_mask = scipy.ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
    #pdb.set_trace()
    # 计算平移量，使质心对齐到图像中心 (128, 128, 128)
    shift = np.array(image.shape) // 2 - np.array(centroid)  # 质心移动到中心
    
    # 使用 ndimage.shift 对图像进行平移
    centered_image = np.zeros_like(image)
    # 对非零区域应用平移
    for x, y, z in zip(*np.where(filled_mask)):
        new_x ,new_y, new_z = (int(x + shift[0]), int(y + shift[1]), int(z + shift[2]))
        #pdb.set_trace()
        # 保证新坐标在图像范围内
        new_x ,new_y, new_z = np.clip([new_x, new_y, new_z], 0, np.array(image.shape) - 1)
        centered_image[new_x , new_y, new_z] = image[x, y, z]
    
    return centered_image


def centrolize_process(image_root , save_root):
    os.makedirs(save_root,exist_ok=True)
    for file in os.listdir(path):
        #pdb.set_trace()
        file_path = os.path.join(image_root , file)
        image_np ,_ = sitk_load(file_path , uint8=True)
        centrialized_image = centralize(image_np)
        after_save = os.path.join(save_root,f"{file.split('.')[0]}.nii.gz")
        sitk_save(after_save , centrialized_image ,uint8=True)




#mian 
if __name__ == '__main__':
    path = 'F:/Data_Space/Pelvic1K/all_blocks_proj_256/images'
    center_path = 'F:/Data_Space/Pelvic1K/centrolize_images/'


    centrolize_process(image_root=path , save_root=center_path)

