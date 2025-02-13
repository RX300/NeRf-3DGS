import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import HTML
from base64 import b64encode
import imageio
from NeRFDataset import NeRFDatasetUnified

class NeRFParams:
    def __init__(self):
        self.data_path = '' # 数据集路径
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 训练设备
        self.only_render = False # 是否仅渲染
        self.nerf_coarse_path = '' # 粗网络模型路径
        self.nerf_fine_path = '' # 细网络模型路径
        self.epochs = 100
        self.batch_size = 1024
        self.bound = (2., 6.)
        self.N_samples = (64, 128)
        self.use_view = True
        self.use_hierarchy = True
        self.H:float = 256
        self.W:float = 256
        self.focal:float = 1.0
        self.n_train = 100 # 使用的训练图像数量
        self.n_val = 6 # 使用的验证图像数量

def load_nerf_data(data_path):
    data = np.load(data_path)
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    return images, poses, focal, H, W

# 采样光线
def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5 + 0.5) / f, -(j - H * 0.5 + 0.5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    # rays_d => (H, W, 3)
    # c2w[:3,-1]提取了最后一列的前三行元素，也就是相机的位置
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    # rays_o => (H, W, 3)
    return rays_o, rays_d

class NeRFDataset(Dataset):
    def __init__(self, npz_file,start_idx=0,n_train=100,transform=None,device='cpu'):
        """
        初始化 NeRF 数据集
        :param npz_file: 数据集文件路径，例如 'tiny_nerf_data.npz'
        :param n_train: 使用的训练图像数量
        :param transform: 可选的图像预处理
        """
        self.transform = transform
        self.data = np.load(npz_file)
        self.images = self.data['images']         # [N_images, H, W, 3]
        self.poses = self.data['poses']           # [N_images, 4, 4]
        self.focal = self.data['focal']           # 标量
        self.H, self.W = self.images.shape[1:3]
        self.focal = self.focal
        self.device = device
        self.start_idx = start_idx
        self.n_train = n_train
        self.rays_num=self.H*self.W*self.n_train
        # 仅使用前 n_train 张图像
        self.all_rays = []
        for i in range(self.start_idx,self.start_idx + self.n_train):
            pose = self.poses[i]
            img = self.images[i]
            # 如果设置了 transform，则预处理图像（注意：预处理后需保证仍为 numpy 数组）
            if self.transform is not None:
                img = self.transform(img)
            # 计算当前图像对应的光线信息
            rays_o, rays_d = sample_rays_np(self.H, self.W, self.focal, pose)
            # 将所有信息铺平，每个像素对应一条射线
            rays_o = rays_o.reshape(-1, 3)            # [H*W, 3]
            rays_d = rays_d.reshape(-1, 3)            # [H*W, 3]
            rgb = img.reshape(-1, 3)                  # [H*W, 3]
            # 拼接成一个 shape 为 [H*W, 9] 的数组（前3是原点，中间3是方向，后3是颜色）
            rays = np.concatenate([rays_o, rays_d, rgb], axis=1)
            self.all_rays.append(rays)
        self.all_rays = np.concatenate(self.all_rays, axis=0)
        self.all_rays = torch.from_numpy(self.all_rays).float().to(device)

    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        """
        返回单个采样数据，包括：光线原点、光线方向、RGB颜色
        """
        ray = self.all_rays[idx]
        return ray[:3], ray[3:6], ray[6:9]
    
# 定义NeRF模型
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=24, skip=4, use_view_dirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skip = skip
        self.use_view_dirs = use_view_dirs

        # 定义全连接层
        self.net = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D-1):
            if i == skip:
                self.net.append(nn.Linear(W + input_ch, W))
            else:
                self.net.append(nn.Linear(W, W))

        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        if use_view_dirs:
            self.proj = nn.Linear(W + input_ch_views, W // 2)
        else:
            self.proj = nn.Linear(W, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_pts, input_views=None):
        h = input_pts.clone()
        for i, layer in enumerate(self.net):
            h = F.relu(layer(h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        alpha = F.relu(self.alpha_linear(h))
        feature = self.feature_linear(h)

        if self.use_view_dirs and input_views is not None:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, alpha

def uniform_sample_point(tn, tf, N_samples, device): # 归一化 采样的点，可以理解为把光线的原点移到near平面
    k = torch.rand([N_samples], device=device) / float(N_samples)
    # 在 0-1 之间均匀采样 N_samples+1 个点，第一个点是0，最后一个点是1
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value

def positional_encoding_pts(x, L=10):
    """
    对点坐标进行位置编码，包含原始坐标和正弦余弦编码(注意这里把原始坐标也包括了)。
    输入:
        x: 张量，形状为 [batch_size, 3]
        L: 编码级数
    输出:
        编码后的张量，形状为 [batch_size, 3 + 2 * L * 3] = [batch_size, 63]
    """
    pe = [x]
    for i in range(L):
        pe += [torch.sin(2**i * x), torch.cos(2**i * x)]
    return torch.cat(pe, -1)

def positional_encoding_views(x, L=4):
    """
    对视角方向进行位置编码，仅包含正弦和余弦编码。
    输入:
        x: 张量，形状为 [batch_size, 3]
        L: 编码级数
    输出:
        编码后的张量，形状为 [batch_size, 2 * L * 3] = [batch_size, 24]
    """
    pe = []
    for i in range(L):
        pe += [torch.sin(2**i * x), torch.cos(2**i * x)]
    return torch.cat(pe, -1)

def hierarchical_sampling(rays_o, rays_d, weights, z_vals_coarse, num_fine=128):
    """
    分层采样流程
    :param rays_o: 光线原点 [batch_size, 3]
    :param rays_d: 光线方向（已归一化） [batch_size, 3]
    :param weights: 权重 [batch_size, num_coarse]
    :param z_vals_coarse: 粗采样点的深度 [batch_size, num_coarse]
    :param near/far: 采样范围 [batch_size, 1]
    :return: 合并后的采样点 [batch_size, num_coarse+num_fine, 3]
    """
    batch_size = rays_o.shape[0]

    # ---------------------- 细采样阶段 ----------------------
    # 生成概率密度函数(PDF)
    weights = weights + 1e-5  # 防止除零
    pdf = weights / torch.sum(weights, -1, keepdims=True)  # [batch_size, num_coarse]

    # 逆变换采样（分箱）
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[...,:-1]], -1)  # [batch_size, num_coarse]

    # 生成均匀分布的样本
    u = torch.rand(batch_size, num_fine, device=rays_o.device)  # [batch_size, num_fine]

    # 查找索引（利用分箱策略）
    idx = torch.searchsorted(cdf, u, right=True)  # [batch_size, num_fine]
    lower = torch.max(torch.zeros_like(idx), idx - 1)
    upper = torch.min(torch.full_like(idx, cdf.shape[-1]-1), idx)
    idx_g = torch.stack([lower, upper], -1)  # [batch_size, num_fine, 2]

    # 线性插值
    cdf_expand = cdf.unsqueeze(1).expand(-1, num_fine, -1)  # [batch_size, num_fine, num_coarse]
    z_vals_expand = z_vals_coarse.unsqueeze(1).expand(-1, num_fine, -1)  # [batch_size, num_fine, num_coarse]
    cdf_g = torch.gather(cdf_expand, dim = 2, index = idx_g)  # [batch_size, num_fine, 2]
    z_vals_coarse_g = torch.gather(z_vals_expand, dim = 2, index = idx_g)  # [batch_size, num_fine, 2]

    denom = cdf_g[..., 1] - cdf_g[..., 0] #denom是当前子区间的累积分布函数（CDF）值的差值
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = z_vals_coarse_g[..., 0] + t * (z_vals_coarse_g[..., 1] - z_vals_coarse_g[..., 0])

    # 合并粗采样和细采样的点
    z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_vals_fine], -1), -1) # [batch_size, num_coarse+num_fine]
    pts_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_combined.unsqueeze(-1)

    return pts_fine, z_vals_combined

def get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=.0, use_view=False):
    """
    计算点的颜色和权重
    :param net: NeRF模型
    :param pts: 采样点 [batch_size, num of sample points in one ray, 3]
    :param rays_d: 光线方向 [batch_size, 3]
    :param z_vals: 深度值 [batch_size, num of sample points in one ray]
    :param noise_std: 噪声标准差
    :param use_view: 是否使用视角方向
    """
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_d => tensor(Batch_Size, 3)
    # z_vals => tensor(Batch_Size, uniform_N)
    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    # pts_flat => tensor(Batch_Size*uniform_N, 3)
    pts_flat = positional_encoding_pts(pts_flat, L=10)
    # pts_flat => tensor(Batch_Size*uniform_N, 3*2*10)
    dir_flat = None
    if use_view:
        dir_flat = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), p=2, dim=-1)
        dir_flat = positional_encoding_views(dir_flat, L=4)
        # dir_flat => tensor(Batch_Size*uniform_N, 3*2*4)

    rgb, sigma = net(pts_flat, dir_flat)
    # rgb => tensor(Batch_Size*uniform_N, 3)
    # sigma => tensor(Batch_Size*uniform_N)
    batchsize=pts.shape[0]
    uniform_N=pts.shape[1]
    rgb = rgb.reshape([batchsize, uniform_N, 3])
    sigma = sigma.reshape([batchsize, uniform_N])
    # rgb => tensor(Batch_Size, uniform_N, 3)
    # sigma => tensor(Batch_Size, uniform_N)

    # get the interval
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    # delta => tensor(uniform_N-1)
    INF = torch.ones(delta[..., :1].shape, device=device).fill_(1e10)
    # INF => tensor(1)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)
    # delta => tensor(Batch_Size, uniform_N)

    # add noise to sigma
    if noise_std > 0.:
        sigma += torch.randn(sigma.size(), device=device) * noise_std

    # get weights
    alpha = 1. - torch.exp(-sigma * delta)
    # alpha => tensor(Batch_Size, uniform_N)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]
    # weights => tensor(Batch_Size, uniform_N)
    return rgb, weights

# 渲染光线
def render_rays_coarseTofine(net_coarse,net_fine, rays, bound, N_samples, device, noise_std=.0, use_view=False,use_hierarchy=False):
    '''
    Render rays from coarse to fine
    :param net_coarse: Coarse network
    :param net_fine: Fine network
    :param rays: Ray origin and direction,是一个list[ray_o,ray_d],包含两个tensor,分别是ray_o[Batch_Size, 3]和ray_d[Batch_Size, 3]
    :param bound: Near and far bound,是一个tuple(near, far)
    :param N_samples: Number of samples,是一个tuple(uniform_N, important_N)
    '''
    rays_o, rays_d = rays
    bs = rays_o.shape[0]
    near, far = bound
    uniform_N, important_N = N_samples
    z_vals = uniform_sample_point(near, far, uniform_N, device)
    # z_vals => tensor(uniform_N)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    # rays_o[..., None, :].shape => torch.Size([Batch_Size, 1, 3]),None是为了增加一个维度
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_o, rays_d => tensor(Batch_Size, 3)

    # Run network
    if important_N is not None and use_hierarchy:
        with torch.no_grad():
            rgb_coarse, weights_coarse = get_rgb_w(net_coarse, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)    
            # samples = sample_pdf_point(z_vals_mid, weights[..., 1:-1], important_N, device)
            z_vals2 = z_vals.unsqueeze(0).expand([bs, uniform_N])
            pts_combined, z_vals = hierarchical_sampling(rays_o, rays_d, weights_coarse, z_vals2, num_fine=important_N)

        rgb, weights = get_rgb_w(net_fine, pts_combined, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)   
    else:
        #打印所有参数是否需要梯度
        rgb, weights = get_rgb_w(net_coarse, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)

    # rgb => tensor(Batch_Size, uniform_N, 3)
    # weights => tensor(Batch_Size, uniform_N)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    # rgb_map => tensor(Batch_Size, 3)
    # depth_map => tensor(Batch_Size)
    # acc_map => tensor(Batch_Size)
    return rgb_map, depth_map, acc_map

# 训练NeRF
def train_nerf(trainDataloader:DataLoader, validationDataloader:DataLoader, params:NeRFParams, device, epochs=100, save_path='tiny_nerf.pth'):
    net_coarse = NeRF(input_ch=63,use_view_dirs=params.use_view).to(device)
    torch.manual_seed(0)#这里必须再刷新一次种子，否则第二个模型的梯度不会更新
    net_fine = NeRF(input_ch=63,use_view_dirs=params.use_view).to(device)
    optimizer = torch.optim.Adam(
        [
        {'params': net_coarse.parameters(), 'lr': 5e-4},
        {'params': net_fine.parameters(), 'lr': 5e-4}
        ]
    )
    mse = nn.MSELoss()
    
    iterations = len(trainDataloader)
    device = params.device
    for epoch in range(epochs):
        net_coarse.train()
        net_fine.train()
        epoch_loss = 0.0
        with tqdm(total=iterations, desc=f"Epoch {epoch+1}", ncols=150, dynamic_ncols=True) as p_bar:
            for batch in trainDataloader:
                batch = [b.to(device) for b in batch]
                batch_rays_o, batch_rays_d, batch_target_rgb = batch
                optimizer.zero_grad()
                # 渲染光线
                rgb_coarse, _, _ = render_rays_coarseTofine(net_coarse,net_fine, (batch_rays_o, batch_rays_d), bound=params.bound, N_samples=params.N_samples, device=device, use_view=params.use_view,use_hierarchy=params.use_hierarchy)
                rgb_fine, depth_pred, acc_pred = render_rays_coarseTofine(net_coarse,net_fine, (batch_rays_o, batch_rays_d), bound=params.bound, N_samples=params.N_samples, device=device, use_view=params.use_view,use_hierarchy=params.use_hierarchy)
                # 计算损失
                loss_coarse = mse(rgb_coarse, batch_target_rgb)
                loss_fine = mse(rgb_fine, batch_target_rgb)
                loss = loss_coarse + loss_fine
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # 更新进度条
                p_bar.set_postfix({
                'loss': '{0:1.5f}'.format(loss.item()),
                'loss_coarse': '{0:1.5f}'.format(loss_coarse.item()),
                'loss_fine': '{0:1.5f}'.format(loss_fine.item())
                })
                p_bar.update(1)
        avg_loss = epoch_loss / len(trainDataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        # validation
        n_val = params.n_val
        net_coarse.eval()
        net_fine.eval()
        with torch.no_grad():
            rgb_list = []  # 存储所有批次的RGB结果
            avg_loss = 0.0
            with tqdm(total=len(validationDataloader), desc="Validation", ncols=150, dynamic_ncols=True) as p_bar:
                for batch in validationDataloader:
                    batch_rays_o, batch_rays_d, batch_target_rgb = batch
                    rgb, _, _ = render_rays_coarseTofine(net_coarse,net_fine, (batch_rays_o, batch_rays_d), bound=params.bound, N_samples=params.N_samples, device=device, use_view=params.use_view,use_hierarchy=params.use_hierarchy)
                    loss = mse(rgb, batch_target_rgb)
                    avg_loss += loss.item()
                    rgb_list.append(rgb)
                    p_bar.update(1)
      
                avg_loss /= len(validationDataloader)
                avglosstensor = torch.tensor(avg_loss,dtype=torch.float32)
                print(f"Validation Loss: {avg_loss:.4f}")
                #计算PSNR
                avg_psnr = -10. * torch.log(avglosstensor) / torch.log(torch.tensor([10.]))
                print(f"Validation PSNR: {avg_psnr.item():.4f}")

                if((epoch+1) % 1 == 0):
                    # 保存验证集的渲染结果，将rgb_list保存为n_val个图片
                    rgb_pred = torch.cat(rgb_list, dim=0)
                    rgb_pred = rgb_pred.reshape(-1, H, W, 3).cpu().detach().numpy()
                    rgb_pred = (255 * np.clip(rgb_pred, 0, 1)).astype(np.uint8)
                    val_save_path = f'validation_result'
                    if not os.path.exists(val_save_path):
                        os.makedirs(val_save_path)
                    for i, rgb in enumerate(rgb_pred):
                        plt.imsave(f'{val_save_path}/validation_image_epoch_{epoch+1}_{i}.png', rgb)
                            # 写入PSNR和epoch还有loss到txt文件
                    with open(val_save_path+'/psnr_loss.txt', 'a') as f:
                        f.write(f"Epoch {epoch+1} PSNR: {avg_psnr.item()} Loss: {avg_loss}\n")

    # 保存训练好的模型
    save_coarse_path = save_path.replace('.pth', '_coarse.pth')
    save_fine_path = save_path.replace('.pth', '_fine.pth')
    net_coarse.eval().cpu()
    net_fine.eval().cpu()
    torch.save(net_coarse.state_dict(), save_coarse_path)
    torch.save(net_fine.state_dict(), save_fine_path)
    print(f"模型已保存到 {save_coarse_path} 和 {save_fine_path}")
    return net_coarse,net_fine

def render_and_create_video(net_coarse:NeRF,net_fine:NeRF, poses, params:NeRFParams,device,video_path,image_path=None):
    net_coarse.eval()
    net_fine.eval()
    bound = params.bound
    N_samples = params.N_samples
    use_view = params.use_view
    # 假设 H, W, focal 已保存于 params 中
    H, W, focal = params.H, params.W, params.focal

    frames = []
    idx = 0
    for pose in tqdm(poses):
        with torch.no_grad():
            # 这里假设 pose 为一个4x4的相机矩阵
            rays_o, rays_d = sample_rays_np(H, W, focal, pose[:3, :4])
            rays_o = torch.reshape(torch.tensor(rays_o, device=device, dtype=torch.float32), [-1, 3])
            rays_d = torch.reshape(torch.tensor(rays_d, device=device, dtype=torch.float32), [-1, 3])
            
            # 分成多个batch进行渲染
            batch_size = 4096
            num_batches = (rays_o.shape[0] + batch_size - 1) // batch_size
            rgb_batches = []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, rays_o.shape[0])
                batch_rays_o = rays_o[start:end]
                batch_rays_d = rays_d[start:end]
                rays_od = (batch_rays_o, batch_rays_d)
                # 调用 coarse-to-fine 渲染函数
                rgb_batch, _, _ = render_rays_coarseTofine(net_coarse, net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view, use_hierarchy=True)
                rgb_batches.append(rgb_batch)
            
            # 拼接所有batch的结果
            rgb = torch.cat(rgb_batches, dim=0)

        # 重塑 rgb 为(H, W, 3)大小
        rgb = rgb.reshape(H, W, 3).cpu().numpy()
        H_new, W_new = rgb.shape[:2]
        # 如果高度或宽度不是16的倍数，则进行缩放
        if H_new % 16 != 0 or W_new % 16 != 0:
            new_H = (H_new // 16) * 16  # 将高度调整为16的倍数
            new_W = (W_new // 16) * 16  # 将宽度调整为16的倍数
            rgb = torch.tensor(rgb).unsqueeze(0).permute(0, 3, 1, 2).float()  # 转换为torch tensor并调整维度

            # 使用torch.nn.functional.interpolate进行缩放
            rgb_resized = F.interpolate(rgb, size=(new_H, new_W), mode='bilinear', align_corners=False)
            rgb = rgb_resized.squeeze(0).permute(1, 2, 0).numpy()  # 恢复原维度
        # 转换为 uint8 类型后保存
        frame_img = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)
        frames.append(frame_img)
        # 保存图片
        if image_path is not None:
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            imageio.imsave(f'{image_path}/image_{idx}.png', frame_img)
        idx += 1
    imageio.mimsave(video_path, frames, fps=30,quality=7)
    print(f"视频已保存到 {video_path}")
            

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def generate_circular_poses(radius, elevation, num_poses, returnc2w=True):
    """
    生成绕物体360度旋转的摄像机位姿。

    参数:
    - radius (float): 摄像机到物体的距离。
    - elevation (float): 摄像机的仰角（以度为单位）。
    - num_poses (int): 需要生成的位姿数量。
    - returnc2w (bool): 如果为 True，返回从相机坐标系到世界坐标系的变换矩阵。

    返回:
    - poses (list of np.ndarray): 生成的位姿矩阵列表。
    """
    poses = []
    for i in range(num_poses):
        theta = 360 * i / num_poses  # 偏航角
        phi = elevation  # 仰角

        # 使用球坐标生成相机位姿
        c2w = pose_spherical(theta, phi, radius)

        # 如果需要返回从相机坐标系到世界坐标系的变换矩阵，则取反转矩阵
        if returnc2w:
            poses.append(c2w)
        else:
            poses.append(np.linalg.inv(c2w))  # 返回w2c矩阵

    return poses

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(100)
    # 开启加速
    torch.backends.cudnn.benchmark = True

    # 创建参数实例
    params = NeRFParams()
    params.only_render = False
    params.nerf_coarse_path = 'tiny_nerf_coarse.pth'
    params.nerf_fine_path = 'tiny_nerf_fine.pth'
    params.epochs = 1
    params.batch_size = 512
    params.bound = (2., 6.)
    params.N_samples = (64, 128)
    params.use_view = True
    params.use_hierarchy = True
    params.n_train = 100
    params.n_val = 6

    if params.only_render:
        params.H = 128
        params.W = 128

        net_coarse = NeRF(input_ch=63,use_view_dirs=True)
        net_fine = NeRF(input_ch=63,use_view_dirs=True)
        net_coarse.load_state_dict(torch.load(params.nerf_coarse_path))
        net_fine.load_state_dict(torch.load(params.nerf_fine_path))
        net_coarse.to(device)
        net_fine.to(device)
        # 生成新的位姿数据，使摄像机绕物体旋转360度，共180个位置
        radius = 4.0        # 根据您的场景调整摄像机距离
        elevation = -30.0    # 摄像机的仰角
        num_poses = 120      # 每隔3度一个位姿，共120个位姿
        new_poses = generate_circular_poses(radius, elevation, num_poses)

        data_path = 'tiny_nerf_data.npz'
        if not os.path.exists(data_path):
            print(f"数据文件 {data_path} 未找到。")
            raise FileNotFoundError
        trainDataset = NeRFDataset(data_path, start_idx=0,n_train=params.n_train,transform=None,device=device)
        params.focal = trainDataset.focal
        print(f"已加载NeRF数据: focal={params.focal}, H={params.H}, W={params.W}")
        # 使用训练好的模型进行渲染并创建视频
        render_and_create_video(net_coarse, net_fine,new_poses, params=params, device=device, video_path='rendered_video_1.mp4',image_path='./test_images')
    else:
        basedir = os.path.join(os.path.dirname(__file__),
                        '../dataset/nerf_synthetic-20230812T151944Z-001/nerf_synthetic/lego')
        if not os.path.exists(basedir):
            print(f"数据文件 {basedir} 未找到。")
            raise FileNotFoundError
        # 使用NeRFDatasetUnified当作数据集
        trainDataset = NeRFDatasetUnified(basedir=basedir, dataset_type='blender', split='train', half_res=False,
                testskip=1, device='cpu')
        # dataloader的num_workers如果要>0，必须在main函数中调用，此外Dataset的device必须在cpu上
        trainDataloader = DataLoader(trainDataset, batch_size=params.batch_size, shuffle=True,num_workers=4)
        # 验证集的dataloader一定不能shuffle，因为我们需要按顺序渲染图片
        validationDataset = NeRFDatasetUnified(basedir=basedir, dataset_type='blender', split='val', half_res=False,
                testskip=1, device='cpu')
        validationDataloader = DataLoader(validationDataset, batch_size=params.batch_size, shuffle=False)

        images,poses,focal,H,W = trainDataset.imgs,trainDataset.poses,trainDataset.hwf[2],trainDataset.hwf[0],trainDataset.hwf[1]

        print(f"已加载NeRF数据: images.shape={images.shape}, poses.shape={poses.shape}, focal={focal}, H={H}, W={W}")
        params.H = H
        params.W = W
        params.focal = focal
        # 训练NeRF模型
        net_coarse,net_fine = train_nerf(trainDataloader,validationDataloader,params=params, device=device, epochs=params.epochs, save_path='tiny_nerf.pth')
        net_coarse.to(device) 
        net_fine.to(device)
        torch.cuda.empty_cache()
        # # 使用训练好的模型进行渲染并创建视频
        # render_and_create_video(net_coarse, net_fine, poses, params=params, device=device, video_path='rendered_video_0.mp4',image_path='./test_images')

        # # 生成新的位姿数据，使摄像机绕物体旋转360度，共180个位置
        # radius = 4.0        # 根据您的场景调整摄像机距离
        # elevation = 30.0    # 摄像机的仰角
        # # num_poses = 180      # 每隔2度一个位姿，共180个位姿
        # num_poses = 60
        # new_poses = generate_circular_poses(radius, elevation, num_poses)
        
        # # 使用训练好的模型进行渲染并创建视频
        # render_and_create_video(net_coarse, net_fine, new_poses, params=params, device=device, video_path='rendered_video_1.mp4',image_path='./test_images')
