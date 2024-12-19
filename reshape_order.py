import torch
from einops import rearrange
import pdb
# 假设输入张量的形状为 (B, C, h, w, d)
B, C, h, w, d = 2, 1, 2, 2, 2  # 举例
points_proj = torch.randn(B, C, h, w, d)
pdb.set_trace()
# 使用 einops 将 (B, C, h, w, d) 展平为 (B, C, h*w*d)
points_proj_flat = rearrange(points_proj, 'b c h w d -> b c (h w d)')

# 恢复维度为 (B, C, h, w, d)
points_proj_reshaped = rearrange(points_proj_flat, 'b c (h w d) -> b c h w d' , h=h ,w=w, d=d)

# 验证是否一致
print("Original shape:", points_proj.shape)
print("Flat shape:", points_proj_flat.shape)
print("Reshaped back shape:", points_proj_reshaped.shape)
pdb.set_trace()
print(torch.allclose(points_proj, points_proj_reshaped))  # 应该是 True