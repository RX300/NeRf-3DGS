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
    def __init__(self, num_gaussians=5000,near = 0.1,far = 100.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        # 可优化参数
        # 将xyz在[near,far]范围内随机生成
        self.xyz = nn.Parameter(torch.rand((num_gaussians, 3)))
        #self.xyz = nn.Parameter(torch.rand((num_gaussians, 3)) * (far - near) + near)
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

def getProjectionMatrix(znear, zfar, focal, H, W):
    import math
    fovX = 2 * np.arctan(W / (2 * focal))
    fovY = 2 * np.arctan(H / (2 * focal))
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros((4, 4), device=device)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

#将ndc坐标系转换为像素坐标系
def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

# 3D 高斯协方差投影
def compute_cov2d(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    """
    Args:
        mean: Tensor of shape (N_gaussians, 3)，表示 N 个高速球的 3D 均值点坐标。
        focal_x, focal_y: 标量，焦距沿 x,y 方向的值。
        tan_fovx, tan_fovy: 标量，x 和 y 方向视场角正切值。
        cov3D: Tensor，可以是 (N_gaussians, 3, 3) 或 (3, 3)，表示完整的 3×3 协方差矩阵。
        viewmatrix: Tensor of shape (4, 4)，按照 GLM 中列主序排列。
    Returns:
        Tensor of shape (N_gaussians, 2, 2)，每个点对应的 2D 协方差矩阵，
        计算过程为： cov = T^T · Vrk · T，其中 T = W * J，
        W 从 viewmatrix 中提取，J 是修正后的雅可比矩阵。
    """
    # 确保 mean 有 batch 维度，形状 (N_gaussians, 3)
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    N = mean.shape[0]
    device = mean.device

    # 将 mean 扩展为齐次坐标 (N_gaussians, 4)
    ones = torch.ones((N, 1), device=device, dtype=mean.dtype)
    mean_h = torch.cat([mean, ones], dim=1)  # (NN_gaussians, 4)

    # 转换到相机坐标 (N, 3)
    t = torch.matmul(mean_h, viewmatrix.t())[:, :3]

    # 对 t[:,0] 和 t[:,1] 进行 clamp 限制
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[:, 0] / t[:, 2]
    tytz = t[:, 1] / t[:, 2]
    t_modified_x = torch.clamp(txtz, -limx, limx) * t[:, 2]
    t_modified_y = torch.clamp(tytz, -limy, limy) * t[:, 2]
    t = torch.stack([t_modified_x, t_modified_y, t[:, 2]], dim=1)  # (N, 3)

    # 构造雅可比矩阵 J (N_gaussians, 3, 3)，第三行均为 0
    J = torch.zeros((N, 3, 3), device=device, dtype=mean.dtype)
    J[:, 0, 0] = focal_x / t[:, 2]
    J[:, 0, 2] = - (focal_x * t[:, 0]) / (t[:, 2]**2)
    J[:, 1, 1] = focal_y / t[:, 2]
    J[:, 1, 2] = - (focal_y * t[:, 1]) / (t[:, 2]**2)

    # 从 viewmatrix 中提取 3x3 矩阵 W（GLM 排列方式）
    view_flat = viewmatrix.flatten()
    W = torch.stack([
        torch.stack([view_flat[0], view_flat[4], view_flat[8]]),
        torch.stack([view_flat[1], view_flat[5], view_flat[9]]),
        torch.stack([view_flat[2], view_flat[6], view_flat[10]])
    ], dim=0)  # (3, 3)
    W_expanded = W.unsqueeze(0).expand(N, -1, -1)  # (N_gaussians, 3, 3)

    # 计算 T = W * J
    T_mat = torch.bmm(W_expanded, J)  # (N_gaussians, 3, 3)

    # 处理 cov3D：如果 cov3D 已经是完整矩阵，则形状应该为 (N_gaussians, 3, 3) 或 (3, 3)
    if cov3D.ndim == 2 and cov3D.shape == (3, 3):
        Vrk = cov3D.unsqueeze(0).expand(N, -1, -1)
    elif cov3D.ndim == 3:
        Vrk = cov3D
    else:
        raise ValueError("cov3D 的形状必须为 (N,3,3) 或 (3,3)！")

    # 计算 2D 协方差矩阵： cov = T^T · Vrk · T
    cov2d = torch.bmm(T_mat.transpose(1, 2), torch.bmm(Vrk, T_mat))  # (N_gaussians, 3, 3)
    return cov2d[:, :2, :2] 

#从世界坐标系->相机坐标系做投影变化->ndc坐标系，对于高斯的均值，还需要做一次视口变化
def project_gaussians(means3D, cov3D, viewmat, focal, width, height):
    focal = float(focal)
    # 坐标变换到相机空间
    means_vec4 = torch.cat([means3D, torch.ones_like(means3D[..., :1])], dim=-1)
    means_cam = torch.matmul(means_vec4, viewmat.t())
    X_cam, Y_cam, Z_cam = means_cam[..., 0], means_cam[..., 1], means_cam[..., 2]
    
    # 正确应用焦距到投影坐标
    x = (X_cam * focal) / Z_cam
    y = (Y_cam * focal) / Z_cam
    
    # 生成投影矩阵并变换到裁剪空间
    proj_matrix = getProjectionMatrix(2, 4, focal, H, W).to(device)
    means_proj = torch.matmul(means_cam, proj_matrix.t())
    # 透视除法前的 w 值 + 0.000001f
    means_w = means_proj[..., 3:] + 0.000001
    means_proj = means_proj[..., :3] /means_w  # 透视除法
    
    # 视口变换到像素坐标
    x_pix = ndc2Pix(means_proj[..., 0], width)
    y_pix = ndc2Pix(means_proj[..., 1], height)
    
    # 修正雅可比矩阵计算
    # J = torch.zeros((len(means3D), 2, 3), device=device)
    # inv_Z = 1.0 / Z_cam
    # J[:, 0, 0] = focal * inv_Z
    # J[:, 0, 2] = - (x * inv_Z)
    # J[:, 1, 1] = focal * inv_Z
    # J[:, 1, 2] = - (y * inv_Z)
    
    # cov3D_cam = viewmat[:3, :3] @ cov3D @ viewmat[:3, :3].t()
    # cov2D = J @ cov3D_cam @ J.transpose(1, 2)
    # cov2D[..., 0, 0] += 0.3
    # cov2D[..., 1, 1] += 0.3
    # print(cov2D.shape)
    torch_tan_fovx = torch.tensor(np.tan(0.5), device=device)
    torch_tan_fovy = torch.tensor(np.tan(0.5), device=device)
    cov2D = compute_cov2d(means3D, focal, focal, torch_tan_fovx, torch_tan_fovy, cov3D, viewmat)
     # 返回 x, y, cov2D, depth
    return x_pix, y_pix, cov2D, means_proj[..., 2]  

def compute_alpha(means2D, cov2D, opacity, pixel_coords, radius=3):
    # 添加一个小的方差，防止奇异矩阵
    h_var = 0.3
    det_cov = cov2D[..., 0, 0] * cov2D[..., 1, 1] - cov2D[..., 0, 1] * cov2D[..., 0, 1]
    cov2D[:, 0, 0] += h_var
    cov2D[:, 1, 1] += h_var
    det_cov_plus_h_cov = cov2D[..., 0, 0] * cov2D[..., 1, 1] - cov2D[..., 0, 1] * cov2D[..., 0, 1]
    h_convolution_scaling = 1.0
    det = det_cov_plus_h_cov
    det_inv = 1.0 / det
    cov2D_inv = torch.zeros_like(cov2D)
    cov2D_inv[..., 0, 0] = cov2D[..., 1, 1] * det_inv
    cov2D_inv[..., 1, 1] = cov2D[..., 0, 0] * det_inv
    cov2D_inv[..., 0, 1] = -cov2D[..., 0, 1] * det_inv
    cov2D_inv[..., 1, 0] = cov2D_inv[..., 0, 1]
    opacity = opacity*h_convolution_scaling
    # 计算 alpha
    dx = pixel_coords[..., 0].unsqueeze(1) - means2D[..., 0].unsqueeze(0)
    dy = pixel_coords[..., 1].unsqueeze(1) - means2D[..., 1].unsqueeze(0)
    cov_inv = torch.linalg.inv(cov2D + 1e-6*torch.eye(2, device=device))
    power = -0.5 *(cov_inv[..., 0, 0]*dx*dx+cov_inv[..., 1, 1]*dy*dy)-cov_inv[..., 0, 1]*dx*dy
    alpha = torch.minimum(torch.tensor(0.99, device=device), opacity.reshape(1, -1) * torch.exp(power))
    # # 计算马氏距离
    # cov_inv = torch.linalg.inv(cov2D + 1e-6*torch.eye(2, device=device))
    # dist = dx**2 * cov_inv[..., 0, 0] + 2*dx*dy*cov_inv[..., 0, 1] + dy**2 * cov_inv[..., 1, 1]
    # # 高斯衰减
    # opacity = opacity.squeeze(-1).unsqueeze(0)  # 调整为 (1, num_gaussians)
    # alpha = opacity * torch.exp(-0.5 * dist)
    # alpha[dist > radius**2] = 0
    
    return alpha

def render(gaussians:GaussianModel, pose, focal, H, W):
    # 准备数据
    viewmat = torch.inverse(torch.tensor(pose, device=device))
    means3D = gaussians.xyz
    cov3D = gaussians.build_covariance3D()
    
    # 投影高斯
    x_proj, y_proj, cov2D, depth = project_gaussians(means3D, cov3D, viewmat, focal, W, H)
    
    # 生成像素网格
    #y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    y,x= torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
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
    #color => [H*W, N_gaussian, 3] 
    # alpha => [H*W, N_gaussian]

    # Alpha合成
    threshold = 0.01  # 可以根据需要调整阈值
    # 如果 alpha 小于阈值，则将其视为0，从而使得 1 - alpha = 1，起到“跳过”的作用
    alpha_filtered = torch.where(alpha < threshold, torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype), alpha)
    one_minus_alpha_filtered = 1 - alpha_filtered
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha_filtered * torch.cumprod(torch.cat([ones, one_minus_alpha_filtered], dim=-1), dim=-1)[..., :-1]
    # weights =>[H*W, N_gaussian]
    print(weights.shape)
    rgb = (weights.unsqueeze(-1) * color).sum(dim=1)
    # ones = torch.ones(alpha[..., :1].shape, device=device)
    # weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]
    # rgb = (weights.unsqueeze(-1) * color).sum(dim=1)
    
    return rgb.reshape(H, W, 3)

# 训练参数
num_gaussians = 5000
lr = 0.001
epochs = 300

# 初始化模型
model = GaussianModel(num_gaussians,near=2.0,far=4.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 改用SGD优化器
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
        if (epoch+1) % 10 == 0:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            # 将rendered_test的值clamp到0-1之间
            rendered_test = torch.clamp(rendered_test, 0, 1)
            plt.imsave(f'{savedir}/gaussian_{epoch+1}.png', rendered_test.cpu().numpy())