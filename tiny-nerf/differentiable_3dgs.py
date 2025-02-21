import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda")
torch.manual_seed(999)
np.random.seed(666)

# 加载数据，位姿，焦距
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print("images.shape:", images.shape)
print("poses.shape:", poses.shape)
print("focal:", focal)
n_train = 100  # 使用前n_train张图片用于训练

test_img, test_pose = images[101], poses[101]
images = images[:n_train]
poses = poses[:n_train]

# 球谐基函数
def spherical_harmonics_basis(direction, sh_degree=2):
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    basis = []
    basis.append(0.5 * np.sqrt(1.0 / np.pi) * torch.ones_like(x))  # l=0
    if sh_degree >= 1:
        basis.append(np.sqrt(3.0/(4*np.pi)) * y)  # l=1
        basis.append(np.sqrt(3.0/(4*np.pi)) * z)  # l=1
        basis.append(np.sqrt(3.0/(4*np.pi)) * x)  # l=1
    if sh_degree >= 2:
        basis.append(0.5 * np.sqrt(15.0/(4*np.pi)) * x * y)  # l=2
        basis.append(0.5 * np.sqrt(15.0/(4*np.pi)) * y * z)  # l=2
        basis.append(0.5 * np.sqrt(5.0/(16*np.pi)) * (3*z**2 - 1))  # l=2
        basis.append(0.5 * np.sqrt(15.0/(4*np.pi)) * x * z)  # l=2
        basis.append(0.5 * np.sqrt(15.0/(16*np.pi)) * (x**2 - y**2))  # l=2
    return torch.stack(basis, dim=-1)

# 3D高斯模型
class GaussSplatModel(nn.Module):
    def __init__(self, voxel_res=64, sh_degree=2):
        super().__init__()
        self.voxel_res = voxel_res
        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree + 1)**2
        self.n_features = 10 + 3 * self.sh_dim  # 3D位置 + 3D缩放 + 四元数 + 不透明度 + 球谐系数

        # 初始化高斯点的参数
        self.positions = nn.Parameter(torch.zeros([voxel_res, voxel_res, voxel_res, 3]))  # 高斯位置 (x, y, z)
        self.scales = nn.Parameter(torch.ones([voxel_res, voxel_res, voxel_res, 3]))  # 高斯缩放 (sx, sy, sz)
        self.rotations = nn.Parameter(torch.zeros([voxel_res, voxel_res, voxel_res, 4]))  # 四元数 (qw, qx, qy, qz)
        self.opacity = nn.Parameter(torch.ones([voxel_res, voxel_res, voxel_res, 1]))  # 不透明度
        self.sh_coeff = nn.Parameter(torch.zeros([voxel_res, voxel_res, voxel_res, self.sh_dim]))  # 球谐系数

    def forward(self, points, n_samples=64):
        """
        points: 输入的点坐标 [N_pts, 3]
        返回: RGB颜色和sigma（密度） [N_pts, 3] 和 [N_pts, 1]
        """
        N_pts = points.shape[0]

        # 归一化坐标到[-1, 1]（假设场景在[-4, 4]^3范围）
        pts_norm = points / 4.0
        
        # 获取每个高斯的参数：位置，缩放，旋转，不透明度，球谐系数
        pos = self.positions.permute(3, 0, 1, 2)[None]  # [1, 3, R, G, B]
        scale = self.scales.permute(3, 0, 1, 2)[None]  # [1, 3, R, G, B]
        rot = self.rotations.permute(3, 0, 1, 2)[None]  # [1, 4, R, G, B]
        opacity = self.opacity.permute(3, 0, 1, 2)[None]  # [1, 1, R, G, B]
        sh = self.sh_coeff.permute(3, 0, 1, 2)[None]  # [1, sh_dim, R, G, B]

        # 计算每个点到高斯的距离，使用缩放矩阵
        diff = pts_norm[..., None, :] - pos  # [N_pts, 3, R, G, B]
        scale_inv = 1.0 / (scale + 1e-6)  # 防止除以0
        diff_scaled = diff * scale_inv  # 对点的坐标进行缩放

        # 计算高斯的密度（使用标准的3D高斯公式）
        density = torch.exp(-0.5 * torch.sum(diff_scaled**2, dim=-1))  # 高斯密度

        # 计算旋转矩阵应用到点云的方向
        # 四元数旋转，假设每个高斯都有一个方向
        rot_matrix = self.quaternion_to_matrix(rot)
        rotated_diff = torch.matmul(rot_matrix, diff_scaled)  # [N_pts, 3, R, G, B]
        
        # 计算RGB颜色
        rays_d_norm = rotated_diff / torch.norm(rotated_diff, dim=-1, keepdim=True)  # 方向单位化
        basis = spherical_harmonics_basis(rays_d_norm, self.sh_degree)  # 计算球谐基函数
        rgb = torch.sigmoid(torch.sum(sh * basis, dim=-1))  # [N_pts, 3]
        
        # 计算密度（不透明度）
        sigma = F.softplus(opacity)  # [N_pts]

        return rgb, sigma.unsqueeze(-1)  # 返回RGB和sigma

    def quaternion_to_matrix(self, quat):
        """将四元数转换为旋转矩阵"""
        q0, q1, q2, q3 = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        rot_matrix = torch.stack([
            1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2,
            2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1,
            2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2
        ], dim=-1).view(-1, 3, 3)  # [N_pts, 3, 3]
        return rot_matrix

# 渲染函数
def get_rgb_w_gauss_splat(plenoxelvoxels:GaussSplatModel, pts, rays_d, z_vals, device, noise_std=.0, use_view=False):
    pts_flat = torch.reshape(pts, [-1, 3])
    rgb, sigma = plenoxelvoxels(pts_flat)  # 调用模型得到RGB和sigma

    batchsize = pts.shape[0]
    uniform_N = pts.shape[1]
    rgb = rgb.reshape([batchsize, uniform_N, 3])
    sigma = sigma.reshape([batchsize, uniform_N])

    delta = z_vals[..., 1:] - z_vals[..., :-1]
    INF = torch.ones(delta[..., :1].shape, device=device).fill_(1e10)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    if noise_std > 0.:
        sigma += torch.randn(sigma.size(), device=device) * noise_std

    alpha = 1. - torch.exp(-sigma * delta)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

    return rgb, weights

def render_rays_gauss(net_coarse, net_fine, rays, bound, N_samples, device, noise_std=.0, use_view=False, use_hierarchy=False):
    rays_o, rays_d = rays
    bs = rays_o.shape[0]
    near, far = bound
    uniform_N, important_N = N_samples
    z_vals = uniform_sample_point(near, far, uniform_N, device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    # Run network
    if important_N is not None and use_hierarchy:
        with torch.no_grad():
            rgb_coarse, weights_coarse = get_rgb_w_gauss(net_coarse, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)    

        z_vals_batch = z_vals.unsqueeze(0).expand([bs, uniform_N])
        pts_combined, z_vals = hierarchical_sampling(rays_o, rays_d, weights_coarse, z_vals_batch, num_fine=important_N)

        rgb, weights = get_rgb_w_gauss(net_fine, pts_combined, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)
    else:
        rgb, weights = get_rgb_w_gauss(net_coarse, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map

# 初始化高斯模型
gaussVoxel_coarse = GaussModel(voxel_res=64).to(device)
gaussVoxel_fine = GaussModel(voxel_res=128).to(device)
optimizer = torch.optim.Adam([{'params': gaussVoxel_coarse.parameters(), 'lr': 5e-4},
                              {'params': gaussVoxel_fine.parameters(), 'lr': 5e-4}])
mse = torch.nn.MSELoss()

# 训练循环
print("Start training!")
for e in range(epoch):
    rays = rays[torch.randperm(N), :]
    train_iter = iter(torch.split(rays, Batch_size, dim=0))

    avg_loss = 0
    with tqdm(total=iterations, desc=f"Epoch {e+1}", ncols=150, dynamic_ncols=True) as p_bar:
        for i in range(iterations):
            train_rays = next(train_iter)
            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            
            rgb_coarse, _, _ = render_rays_gauss(gaussVoxel_coarse, gaussVoxel_fine, rays_od, bound=bound, N_samples=N_samples, device=device)
            rgb_fine, _, _ = render_rays_gauss(gaussVoxel_coarse, gaussVoxel_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_hierarchy=True)
            
            loss_coarse = mse(rgb_coarse, target_rgb)
            loss_fine = mse(rgb_fine, target_rgb)
            loss = loss_coarse + loss_fine
            avg_loss += loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({
                'loss': '{0:1.5f}'.format(loss.item()),
                'loss_coarse': '{0:1.5f}'.format(loss_coarse.item()),
                'loss_fine': '{0:1.5f}'.format(loss_fine.item())
            })
            p_bar.update(1)
    print(f"Epoch {e+1} avgLoss: {avg_loss/iterations}")
    
    # 进行验证
    with torch.no_grad():
        rgb_list = []
        pixel_num = test_rays_o.shape[0]
        val_iterations = pixel_num // Batch_size + int(pixel_num % Batch_size != 0)

        with tqdm(total=val_iterations, desc=f"Validation Epoch {e+1}", ncols=100) as val_bar:
            for i in range(val_iterations):
                start = i * Batch_size
                end = min(start + Batch_size, pixel_num)

                batch_rays_o = test_rays_o[start:end]
                batch_rays_d = test_rays_d[start:end]
                rays_od = (batch_rays_o, batch_rays_d)

                rgb_coarse, _, __ = render_rays_gauss(gaussVoxel_coarse, gaussVoxel_fine, rays_od, bound=bound, N_samples=N_samples, device=device)
                rgb_fine, _, __ = render_rays_gauss(gaussVoxel_coarse, gaussVoxel_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_hierarchy=True)
                rgb_list.append(rgb_fine)

                val_bar.update(1)

        rgb_pred = torch.cat(rgb_list, dim=0)
        mse_loss = mse(rgb_pred, test_rgb).cpu()
        loss = mse_loss 
        print(f"Validation Loss after Epoch {e+1}: {loss.item():1.5f}")
        
        psnr = -10. * torch.log(loss) / torch.log(torch.tensor([10.]))
        print(f"PSNR after Epoch {e+1}: {psnr.item()}")
        
        rgb_pred = rgb_pred.reshape(H, W, 3).cpu().detach().numpy()
        save_path = f'GaussVoxel_results'
        if (e+1) % 20 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(save_path+f'/{e+1}.png', rgb_pred)
            torch.save(gaussVoxel_fine.state_dict(), save_path+f'/model_{e+1}.pth')
            with open(save_path+'/psnr_loss.txt', 'a') as f:
                f.write(f"Epoch {e+1} PSNR: {psnr.item()} Loss: {loss.item()}\n")