import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(999)
np.random.seed(666)

# 数据加载
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]

# 数据分割
n_train = 100
test_img, test_pose = images[101], poses[101]
images = images[:n_train]
poses = poses[:n_train]

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians=5000):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # 可优化参数
        self.xyz = nn.Parameter(torch.randn((num_gaussians, 3)) * 4)
        self.scale = nn.Parameter(torch.randn((num_gaussians, 3)) * 0.1)
        self.rot = nn.Parameter(torch.randn((num_gaussians, 4)) * 0.1)
        self.color = nn.Parameter(torch.sigmoid(torch.rand(num_gaussians, 3)))
        self.opacity = nn.Parameter(torch.sigmoid(torch.randn((num_gaussians, 1))))
        
        # 协方差矩阵激活函数
        self.scale_activation = torch.exp
        self.rot_activation = F.normalize

    def build_covariance3D(self):
        # 构建3D协方差矩阵
        scale = self.scale_activation(self.scale)
        rot = self.rot_activation(self.rot)
        
        # 创建旋转矩阵
        qr, qi, qj, qk = rot.unbind(-1)
        R = torch.stack([
            1 - 2*(qj**2 + qk**2),   2*(qi*qj - qk*qr),   2*(qi*qk + qj*qr),
            2*(qi*qj + qk*qr),   1 - 2*(qi**2 + qk**2),   2*(qj*qk - qi*qr),
            2*(qi*qk - qj*qr),   2*(qj*qk + qi*qr),   1 - 2*(qi**2 + qj**2)
        ], dim=-1).reshape(-1, 3, 3)
        
        # 创建缩放矩阵
        S = torch.diag_embed(scale)
        
        # 协方差矩阵 Σ = R S S^T R^T
        return R @ S @ S.transpose(1,2) @ R.transpose(1,2)

#从相机坐标系做投影变化到ndc坐标系
def project_gaussians(means3D, cov3D, viewmat, focal, width, height):
    focal = float(focal)  # 转换 focal 为 float 类型
    # 坐标变换到相机空间
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    means_cam = (means3D - T) @ R
    
    # 投影到图像平面
    x, y, z = means_cam.unbind(-1)
    x_proj = focal * x / z + width / 2
    y_proj = focal * y / z + height / 2
    
    # 计算2D协方差矩阵
    W = viewmat[:3, :3]
    cov3D_cam = W @ cov3D @ W.t()
    # 创建雅克比矩阵，形状应为 (N, 2, 3)
    J = torch.zeros((len(means3D), 2, 3), device=device)
    J[:, 0, 0] = focal / z
    J[:, 0, 2] = -focal * x / (z**2)
    J[:, 1, 1] = focal / z
    J[:, 1, 2] = -focal * y / (z**2)
    
    cov2D = J @ cov3D_cam @ J.transpose(1, 2)
    
    return x_proj, y_proj, cov2D, z
def compute_alpha(means2D, cov2D, opacity, pixel_coords, radius=3):
    dx = pixel_coords[..., 0].unsqueeze(1) - means2D[..., 0].unsqueeze(0)
    dy = pixel_coords[..., 1].unsqueeze(1) - means2D[..., 1].unsqueeze(0)
    
    # 计算马氏距离
    cov_inv = torch.linalg.inv(cov2D + 1e-6*torch.eye(2, device=device))
    dist = dx**2 * cov_inv[..., 0, 0] + 2*dx*dy*cov_inv[..., 0, 1] + dy**2 * cov_inv[..., 1, 1]
    
    # 高斯衰减
    opacity = opacity.squeeze(-1).unsqueeze(0)  # 调整为 (1, num_gaussians)
    alpha = opacity * torch.exp(-0.5 * dist)
    alpha[dist > radius**2] = 0
    
    return alpha

def render(gaussians:GaussianModel, pose, focal, H, W):
    # 准备数据
    viewmat = torch.inverse(torch.tensor(pose, device=device))
    means3D = gaussians.xyz
    cov3D = gaussians.build_covariance3D()
    
    # 投影高斯
    x_proj, y_proj, cov2D, depth = project_gaussians(means3D, cov3D, viewmat, focal, W, H)
    
    # 生成像素网格
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    pixel_coords = torch.stack([x, y], dim=-1).float()
    
    # 计算每个像素的alpha值
    alpha = compute_alpha(torch.stack([x_proj, y_proj], dim=-1), cov2D, 
                        gaussians.opacity, pixel_coords.reshape(-1, 2))
    
    # 颜色混合
    color = gaussians.color.unsqueeze(0) * alpha.unsqueeze(-1)  # [H*W, N, 3]
    
    # 按深度排序，高斯深度 depth 为 (N, )，需复制到每个像素上
    sorted_idx = torch.argsort(depth, descending=True)  # shape: (N,)
    # 将 sorted_idx 扩展到 batch 维度：(H*W, N)
    sorted_idx = sorted_idx.unsqueeze(0).expand(color.shape[0], -1)
    
    # 对 color 和 alpha 进行排序
    color = torch.gather(color, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))
    alpha = torch.gather(alpha, 1, sorted_idx)
    
    # Alpha合成
    weights = alpha * torch.cumprod(1 - alpha + 1e-6, dim=1)
    rgb = (weights.unsqueeze(-1) * color).sum(dim=1)
    
    return rgb.reshape(H, W, 3)

# 训练参数
num_gaussians = 5000
lr = 0.001
epochs = 1
batch_size = 1024

# 初始化模型
model = GaussianModel(num_gaussians).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    with tqdm(total=n_train, desc=f" Epoch {epoch+1}", ncols=100) as p_bar:
        for img_idx in range(n_train):
            # 获取当前视角数据
            pose = poses[img_idx]
            target = torch.tensor(images[img_idx], device=device)
            
            # 渲染图像
            rendered = render(model, pose, focal, H, W)
            
            # 计算损失
            loss = mse_loss(rendered, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            p_bar.update(1)
            p_bar.set_postfix_str(f"Loss: {loss.item():.4f}")
    # for img_idx in tqdm(range(n_train), desc=f"Epoch {epoch+1}"):
    #     # 获取当前视角数据
    #     pose = poses[img_idx]
    #     target = torch.tensor(images[img_idx], device=device)
        
    #     # 渲染图像
    #     rendered = render(model, pose, focal, H, W)
        
    #     # 计算损失
    #     loss = mse_loss(rendered, target)
        
    #     # 反向传播
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    #     total_loss += loss.item()
    
    # 验证和保存
    with torch.no_grad():
        test_pose = torch.tensor(test_pose, device=device)
        test_img = torch.tensor(test_img, device=device)
        rendered_test = render(model, test_pose, focal, H, W)
        test_loss = mse_loss(rendered_test, test_img)
        psnr = -10 * torch.log10(test_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/n_train:.4f} | Test PSNR: {psnr:.2f}")
        
        # 保存结果
        savedir = "gaussian_models"
        if (epoch+1) % 1 == 0:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            # 将rendered_test的值clamp到0-1之间
            rendered_test = torch.clamp(rendered_test, 0, 1)
            plt.imsave(f'{savedir}/gaussian_{epoch+1}.png', rendered_test.cpu().numpy())
            # plt.imsave(f'gaussian_{epoch+1}.png', rendered_test.cpu().numpy())