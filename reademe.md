#Implict condition diffusion Model for CT recontruction from 2 Xray Image 

1. configsetting in config files 

2. training  use   python ./main.py
3. version_0  2024.12.19  使用投影模型将2D图像特征投影到3D空间,并且获取全局向量和coordinate embedding 向量,  获取该向量作为 condition 送入到diffusion model中. 这这个diffusion model 内的 预测噪声网络是Image as Points 2D 版本修改为3D 。    

