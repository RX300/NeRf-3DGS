
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from tqdm import tqdm
from torch.utils.data import Dataset

#“npz”数据包含相机的焦距、相机拍摄的图像和相机拍摄时的姿势
# if not os.path.exists('tiny_nerf_data.npz'):
    # !wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
#torch统一默认使用float32
torch.set_default_dtype(torch.float32)
device = torch.device("cuda")
torch.manual_seed(999) #通过指定seed值，可以保证每次执行 torch.manual_seed()后生成的随机数都相同
np.random.seed(666)
torch.backends.cudnn.benchmark=True #提升卷积神经网络的运行速度
#加载数据，位姿，焦距
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print("images.shape:", images.shape)
print("poses.shape:", poses.shape)
print("focal:", focal)

n_train = 100 # use n_train images for training

test_img, test_pose = images[101], poses[101]
images = images[:n_train]
poses = poses[:n_train]

# plt.imshow(test_img)
# plt.show()

class NeRF(nn.Module):
    #
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=24, skip=4, use_view_dirs=True,name="NeRF"):
        super(NeRF, self).__init__()
        self.name = name
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skip = skip
        self.use_view_dirs = use_view_dirs

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
        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        alpha = F.relu(self.alpha_linear(h))
        feature = self.feature_linear(h)

        if self.use_view_dirs:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, alpha

# net = NeRF(use_view_dirs=True)
# print(net)


def PE(x, L):#位置编码 低频信息转换为高频信息-  x代表输入给编码器的数据维度，也就是3，2,5， l为数学公式中的L
    #这个函数对x坐标的3个值和向量d 的2个值都进行了编码。实验中设置了 L=10 for y(x)，L=4 for y（d）
#这里为了方便统一处理，应该会影响最后效果
  pai = 3.14
  pe = []
  for i in range(L):
    for fn in [torch.sin, torch.cos]:#依次 先后读取sin cos 函数
      pe.append(fn(2.**i * x ))
  return torch.cat(pe, -1)  #对tensor 进行拼接

def sample_rays_np(H, W, f, c2w):#
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5+.5)/f, -(j-H*.5+.5)/f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    #c2w[:3,-1]提取了最后一列的前三行元素，也就是相机的位置
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def uniform_sample_point(tn, tf, N_samples, device): # 归一化 采样的点，可以理解为把光线的原点移到near平面
    k = torch.rand([N_samples], device=device) / float(N_samples)
    # 在 0-1 之间均匀采样 N_samples+1 个点，第一个点是0，最后一个点是1
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value

# Hierarchical sampling (section 5.2)
#大概步骤 1根据pdf 求 cdf 2 做0-1的均匀采样 3求采样点值在cdf中的对应分块和传播时间 4求解采样点对应的z_vals
def sample_pdf_point(bins, weights, N_samples, device):
    '''
    bins => (总的采样数-1)，每个 bin 代表一个子区间的中点
    weights => (batchsize. 总的采样数-2)
    N_samples => 用于重要性采样的采样数，和总的采样数不同
    返回的是z值不是采样点
    '''
    pdf = F.normalize(weights, p=1, dim=-1)
    cdf = torch.cumsum(pdf, -1)
    # cdf => (batchsize, 总的采样数-2)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # 在最前面补上了0,cdf => (batchsize, 总的采样数-1)

    # uniform sampling
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device).contiguous()
    # u => (batchsize, N_samples)

    # invert
    #使用searchsorted函数并且设置right=True，可以找到cdf中最后一个大于等于u的位置，这个位置就是我们要找的位置
    ids = torch.searchsorted(cdf, u, right=True)
    # ids => (batchsize, N_samples)
    # 计算每个采样点在累积分布函数（CDF）中对应区间的下界索引，确保索引值不小于0。
    below = torch.max(torch.zeros_like(ids - 1, device=device), ids - 1)
    # 计算每个采样点在累积分布函数（CDF）中对应区间的上界索引，确保索引值不大于CDF的最大索引值。
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids, device=device), ids)
    ids_g = torch.stack([below, above], -1)
    # ids_g => (batch, N_samples, 2)

    # matched_shape : [batch, N_samples, bins]
    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
    # gather cdf value
    cdf_val = torch.gather(cdf.unsqueeze(1).expand(matched_shape), -1, ids_g)
    # gather z_val
    bins_val = torch.gather(bins[None, None, :].expand(matched_shape), -1, ids_g)
    # cdf_val, bins_val => (batch, N_samples, 2)

    # get z_val for the fine sampling
    cdf_d = (cdf_val[..., 1] - cdf_val[..., 0])
    #torch.where 函数将 cdf_d 中小于 1e-5 的元素替换为 1，否则保持不变
    cdf_d = torch.where(cdf_d < 1e-5, torch.ones_like(cdf_d, device=device), cdf_d)
    # t表示采样点在当前子区间的相对位置，在 0-1 之间
    t = (u - cdf_val[..., 0]) / cdf_d
    samples = bins_val[..., 0] + t * (bins_val[..., 1] - bins_val[..., 0])
    # samples => (batch, N_samples)
    return samples

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
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_d => tensor(Batch_Size, 3)
    # z_vals => tensor(Batch_Size, uniform_N)
    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    # pts_flat => tensor(Batch_Size*uniform_N, 3)
    pts_flat = PE(pts_flat, L=10)
    # pts_flat => tensor(Batch_Size*uniform_N, 3*2*10)
    dir_flat = None
    if use_view:
        dir_flat = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), p=2, dim=-1)
        dir_flat = PE(dir_flat, L=4)
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

# def render_rays(net, rays, bound, N_samples, device, noise_std=.0, use_view=False):
#     rays_o, rays_d = rays
#     bs = rays_o.shape[0]
#     near, far = bound
#     uniform_N, important_N = N_samples
#     z_vals = uniform_sample_point(near, far, uniform_N, device)
#     # z_vals => tensor(uniform_N)
#     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
#     # rays_o[..., None, :].shape => torch.Size([Batch_Size, 1, 3]),None是为了增加一个维度
#     # pts => tensor(Batch_Size, uniform_N, 3)
#     # rays_o, rays_d => tensor(Batch_Size, 3)

#     # Run network
#     if important_N is not None:
#         with torch.no_grad():
#             rgb, weights = get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)
#             # z_vals[..., 1:]代表子区间的右端点，z_vals[..., :-1]代表子区间的左端点
#             z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
#             samples = sample_pdf_point(z_vals_mid, weights[..., 1:-1], important_N, device)

#         z_vals = z_vals.unsqueeze(0).expand([bs, uniform_N])
#         z_vals, _ = torch.sort(torch.cat([z_vals, samples], dim=-1), dim=-1)
#         pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

#     rgb, weights = get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)

#     # rgb => tensor(Batch_Size, uniform_N, 3)
#     # weights => tensor(Batch_Size, uniform_N)
#     rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
#     depth_map = torch.sum(weights * z_vals, -1)
#     acc_map = torch.sum(weights, -1)

#     # rgb_map => tensor(Batch_Size, 3)
#     # depth_map => tensor(Batch_Size)
#     # acc_map => tensor(Batch_Size)
#     return rgb_map, depth_map, acc_map

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

print(torch.__version__)
print(torch.cuda.is_available())

print("Process rays data!")
rays_o_list = list()
rays_d_list = list()
rays_rgb_list = list()

for i in range(n_train):
    img = images[i]
    pose = poses[i]
    rays_o, rays_d = sample_rays_np(H, W, focal, pose)

    rays_o_list.append(rays_o.reshape(-1, 3))
    rays_d_list.append(rays_d.reshape(-1, 3))
    rays_rgb_list.append(img.reshape(-1, 3))

rays_o_npy = np.concatenate(rays_o_list, axis=0)
rays_d_npy = np.concatenate(rays_d_list, axis=0)
rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device)
# rays => tensor(N_rays, 9) => o+d+rgb


#############################
# training parameters
#############################   Batch_size = 4096
N = rays.shape[0]
Batch_size = 1024
iterations = N // Batch_size
print(f"There are {iterations} batches of rays and each batch contains {Batch_size} rays")

bound = (2., 6.)
N_samples = (64, 128)
use_view = True
epoch = 1
psnr_list = []
e_nums = []

#############################
# test data
#############################
test_rays_o, test_rays_d = sample_rays_np(H, W, focal, test_pose)
test_rays_o = torch.tensor(test_rays_o,dtype=torch.float32, device=device)
test_rays_d = torch.tensor(test_rays_d, dtype=torch.float32,device=device)
test_rgb = torch.tensor(test_img,dtype=torch.float32, device=device)

test_rays_o = test_rays_o.reshape(-1, 3)
test_rays_d = test_rays_d.reshape(-1, 3)
test_rgb = test_rgb.reshape(-1, 3)
#test_rays_o.shape => torch.Size([H*W, 3])
print(test_rays_o.shape, test_rays_d.shape, test_rgb.shape)

#############################
# training
#############################
net_coarse = NeRF(use_view_dirs=use_view,name="Coarse").to(device)
torch.manual_seed(0)
net_fine = NeRF(use_view_dirs=use_view,name="Fine").to(device)


optimizer = torch.optim.Adam([
    {'params': net_coarse.parameters(), 'lr': 5e-4},
    {'params': net_fine.parameters(), 'lr': 5e-4}
    ]
)
mse = torch.nn.MSELoss()
print("Start training!")
for e in (range(epoch)):
    # create iteration for training
    rays = rays[torch.randperm(N), :]
    train_iter = iter(torch.split(rays, Batch_size, dim=0))

    # render + mse
    avg_loss = 0
    with tqdm(total=iterations, desc=f"Epoch {e+1}", ncols=150, dynamic_ncols=True) as p_bar:
        for i in range(iterations):
            train_rays = next(train_iter)
            assert train_rays.shape == (Batch_size, 9)
            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb_coarse, _, _ = render_rays_coarseTofine(net_coarse,net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view,use_hierarchy=False)
            rgb_fine,_, _ = render_rays_coarseTofine(net_coarse,net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view,use_hierarchy=True)
            #打印是否需要梯度
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
    
    # validation
    with torch.no_grad():
        rgb_list = []  # 存储所有批次的RGB结果
        pixel_num = test_rays_o.shape[0]
        val_iterations =pixel_num // Batch_size+int(pixel_num % Batch_size != 0)# 计算验证批次数
        # 使用进度条显示验证进度
        with tqdm(total=val_iterations, desc=f"Validation Epoch {e+1}", ncols=100) as val_bar:
            for i in range(val_iterations):
                start = i * Batch_size
                end = min(start + Batch_size, pixel_num)

                # 获取当前验证批次的光线
                batch_rays_o = test_rays_o[start:end]
                batch_rays_d = test_rays_d[start:end]
                rays_od = (batch_rays_o, batch_rays_d)

                # 渲染当前验证批次的光线
                #rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)
                rgb, _, __ = render_rays_coarseTofine(net_coarse,net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view,use_hierarchy=True)
                rgb_list.append(rgb)

                val_bar.update(1)

        # 拼接所有批次的RGB结果
        rgb_pred = torch.cat(rgb_list, dim=0)
        # 计算验证损失
        #val_loss = mse(rgb_pred, test_rgb).cpu()
        mse_loss = mse(rgb_pred, test_rgb).cpu()
        # val_loss = torch.norm(rgb_pred - test_rgb, p=2).cpu()
        loss = mse_loss 
        print(f"Validation Loss after Epoch {e+1}: {loss.item():1.5f}")
        # 计算PSNR
        psnr = -10. * torch.log(loss) / torch.log(torch.tensor([10.]))
        print(f"PSNR after Epoch {e+1}: {psnr.item()}")
        #保存图片
        rgb_pred = rgb_pred.reshape(H, W, 3).cpu().detach().numpy()
        save_path = f'outputnerfipynbpy'
        if((e+1) % 1 == 0):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(save_path+f'/{e+1}.png', rgb_pred)
        # 写入PSNR和epoch还有loss到txt文件
        with open(save_path+'/psnr_loss.txt', 'a') as f:
            f.write(f"Epoch {e+1} PSNR: {psnr.item()} Loss: {loss.item()}\n")
        # 保存模型
        torch.save(net_fine.state_dict(), save_path+f'/model_{e+1}.pth')

print(' Traning Done')
#清空缓存
torch.cuda.empty_cache()

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

print('Start rendering video')
frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    with torch.no_grad():
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = sample_rays_np(H, W, focal, c2w[:3,:4])
        rays_o = torch.reshape(torch.tensor(rays_o, device=device, dtype=torch.float32), [-1, 3])
        rays_d = torch.reshape(torch.tensor(rays_d, device=device, dtype=torch.float32), [-1, 3])
        rays_od = (rays_o, rays_d)
        # rgb, depth, acc = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)
        rgb, depth, acc = render_rays_coarseTofine(net_coarse, net_fine, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=use_view)

    rgb = rgb.reshape(H, W, 3).cpu().numpy()
    # 确保图像的尺寸是16的倍数
    H_new, W_new = rgb.shape[:2]

    # 如果高度或宽度不是16的倍数，进行缩放
    if H_new % 16 != 0 or W_new % 16 != 0:
        new_H = (H_new // 16) * 16  # 将高度调整为16的倍数
        new_W = (W_new // 16) * 16  # 将宽度调整为16的倍数
        rgb = torch.tensor(rgb).unsqueeze(0).permute(0, 3, 1, 2).float()  # 转换为torch tensor并调整维度

        # 使用torch.nn.functional.interpolate进行缩放
        rgb_resized = F.interpolate(rgb, size=(new_H, new_W), mode='bilinear', align_corners=False)
        rgb_resized = rgb_resized.squeeze(0).permute(1, 2, 0).numpy()  # 恢复原维度

    # 将图像转换为uint8并保存
    frames.append((255 * np.clip(rgb_resized, 0, 1)).astype(np.uint8))

import imageio
f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

print('Done')