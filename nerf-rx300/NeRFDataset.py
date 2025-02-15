import os
import numpy as np
import torch
from torch.utils.data import Dataset

# 导入各数据加载函数（请根据实际情况调整路径）
from load_blender import load_blender_data
from load_llff import load_llff_data
from load_LINEMOD import load_LINEMOD_data
from load_deepvoxels import load_dv_data

# 从 run_tiny_nerf.py 中导入采样光线函数
from run_nerf_helpers import get_rays_np

def convert_pose(pose, ds_type):
    """
    将不同数据集中的 pose 转换为 4×4 的相机矩阵
    ds_type 可选：'blender'、'llff'、'linemod'、'deepvoxels'
    """
    if ds_type == 'blender':
        return pose  # 已经是 4×4
    elif ds_type == 'llff':
        # llff 的 pose shape 为 (3,5)，取前 4 列作为 [R | t]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = pose[:, :4]
        return c2w
    elif ds_type in ['deepvoxels']:
        # 假定 pose 为 (3,4)，需要扩展为 4×4
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = pose
        return c2w
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

class NeRFDatasetUnified(Dataset):
    """
    统一的 NeRF 数据集，支持 Blender、LLFF、LINEMOD 和 DeepVoxels 数据。
    载入数据后对每张图像预计算所有像素点对应的光线（光线原点与方向）以及 RGB 标签。
    """
    def __init__(self, basedir, dataset_type='blender', split='train', half_res=False,
                 testskip=1, device='cpu'):
        """
        :param basedir: 数据根目录
        :param dataset_type: 数据类型：'blender'、'llff'、'linemod'、'deepvoxels'
        :param split: 划分名称，常用 'train'、'val'、'test'
        :param half_res: 是否采用半分辨率（对 Blender、LINEMOD 有效）
        :param testskip: 控制非训练图像的采样间隔
        :param device: 数据所在设备
        """
        self.dataset_type = dataset_type
        self.split = split
        self.device = device
        if dataset_type == 'blender':
            imgs, poses, render_poses, hwf, i_split = load_blender_data(basedir, half_res, testskip)
            # 根据 split 选择对应图像索引
            if split == 'train':
                indices = i_split[0]
            elif split == 'val':
                indices = i_split[1]
            else:
                indices = i_split[2]
            self.imgs = imgs[indices]
            self.poses = poses[indices]  # 每个 pose 均为 4×4
            self.hwf = hwf  # [H, W, focal]
        elif dataset_type == 'llff':
            # load_llff_data 返回：images, poses, bds, render_poses, i_test
            images, poses, bds, render_poses, i_test = load_llff_data(basedir, factor=8,
                                                                       recenter=True, bd_factor=.75,
                                                                       spherify=False, path_zflat=False)
            if split == 'test':
                self.imgs = images[i_test:i_test+1]
                self.poses = poses[i_test:i_test+1]
            else:
                # 训练（或验证）时剔除 holdout 视角
                indices = [i for i in range(images.shape[0]) if i != i_test]
                self.imgs = images[indices]
                self.poses = poses[indices]
            H, W = self.imgs.shape[1:3]
            focal = 1.0  # llff 数据通常不直接提供 focal 值，可自行估计或传入正确焦距
            self.hwf = [H, W, focal]
        elif dataset_type == 'deepvoxels':
            # load_dv_data 返回：imgs, poses, render_poses, hwf, i_split
            imgs, poses, render_poses, hwf, i_split = load_dv_data(basedir, testskip=testskip)
            if split == 'train':
                indices = i_split[0]
            elif split == 'val':
                indices = i_split[1]
            else:
                indices = i_split[2]
            self.imgs = imgs[indices]
            self.poses = poses[indices]  # pose shape 假定为 (3,4)
            self.hwf = hwf
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        H, W, focal = self.hwf
        self.render_poses = render_poses
        # 对所有图像预计算每张图片所有像素点的光线及对应 RGB
        all_rays = []
        all_rgbs = []
        for i in range(self.imgs.shape[0]):
            # 将不同数据的 pose 转换为 4×4 矩阵
            c2w = convert_pose(self.poses[i], dataset_type)
            # 调用 sample_rays_np 采样全部光线，返回 (H, W, 3) 的光线原点和方向
            # 构造摄像机内参矩阵 K
            K = np.array([[focal, 0, W/2.0],
                        [0, focal, H/2.0],
                        [0, 0, 1]], dtype=np.float32)
            rays_o, rays_d = get_rays_np(H, W, K, c2w)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            # 获取图像 RGB（若存在 alpha 通道，仅取前 3 个通道）
            rgb = self.imgs[i][..., :3]
            rgbs = rgb.reshape(-1, 3)
            rays = np.concatenate([rays_o, rays_d], axis=-1)  # shape: (H*W, 6)
            all_rays.append(rays)
            all_rgbs.append(rgbs)
        all_rays = np.concatenate(all_rays, axis=0)
        all_rgbs = np.concatenate(all_rgbs, axis=0)
        self.all_rays = torch.from_numpy(all_rays).to(device).float()
        self.all_rgbs = torch.from_numpy(all_rgbs).to(device).float()

    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        """
        返回单个采样数据，包括：光线原点、光线方向、RGB颜色
        """
        ray = self.all_rays[idx]
        return ray[:3], ray[3:6], self.all_rgbs[idx]