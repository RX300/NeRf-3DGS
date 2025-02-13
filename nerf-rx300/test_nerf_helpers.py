import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from load_blender import *
from nerf_train import *


def test_load_blender_data():
    """
    测试 load_blender_data 函数，并打印返回值的详细信息。
    """
    # 设置数据集的路径
    basedir = os.path.join(os.path.dirname(__file__),
                           '../dataset/nerf_synthetic-20230812T151944Z-001/nerf_synthetic/lego')

    # 设置参数
    half_res = False
    testskip = 1

    # 检查 basedir 是否存在
    if not os.path.exists(basedir):
        print(f"Error: The directory '{basedir}' does not exist.")
        return

    # 调用 load_blender_data 函数
    print("Loading Blender data...")
    imgs, poses, render_poses, hwf, i_split = load_blender_data(
        basedir, half_res, testskip)
    H, W, focal = hwf

    # 打印图像信息
    print("\n=== Images ===")
    print(f"Type: {type(imgs)}")
    print(f"Shape: {imgs.shape}")  # (N, H, W, 4)
    print(f"Dtype: {imgs.dtype}")
    print(f"Pixel range: min={imgs.min()}, max={imgs.max()}")

    # 打印位姿信息
    print("\n=== Poses ===")
    print(f"Type: {type(poses)}")
    print(f"Shape: {poses.shape}")  # (N, 4, 4)
    print(f"Dtype: {poses.dtype}")
    print(f"First pose:\n{poses[0]}")
    print(f"Last pose:\n{poses[-1]}")

    # 打印渲染位姿信息
    print("\n=== Render Poses ===")
    print(f"Type: {type(render_poses)}")
    print(f"Shape: {render_poses.shape}")  # (40, 4, 4)
    print(f"Dtype: {render_poses.dtype}")
    print(f"First render pose:\n{render_poses[0]}")
    print(f"Last render pose:\n{render_poses[-1]}")

    # 打印图像尺寸和焦距
    print("\n=== Image Dimensions and Focal Length ===")
    print(f"Height: {H}")
    print(f"Width: {W}")
    print(f"Focal Length: {focal}")

    # 打印数据集划分信息
    print("\n=== Data Splits ===")
    split_names = ['train', 'val', 'test']
    for split, indices in zip(split_names, i_split):
        print(f"{split.capitalize()} set:")
        print(f"  Number of images: {len(indices)}")
        print(f"  Index range: {indices[0]} to {indices[-1]}")

    # 可选：打印部分图像和位姿以验证
    print("\n=== Sample Images and Poses ===")
    num_samples = min(3, imgs.shape[0])  # 打印前3张图像
    for i in range(num_samples):
        print(f"\nImage {i}:")
        print(f"  Shape: {imgs[i].shape}")
        print(f"  Dtype: {imgs[i].dtype}")
        print(f"  Pixel range: min={imgs[i].min()}, max={imgs[i].max()}")
        print(f"  Pose:\n{poses[i]}")

    print("\nTest completed successfully.")


def test_get_rays():
    # 图像尺寸
    H, W = 600, 800

    # 相机内参矩阵 (示例值)
    K = torch.tensor([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1]
    ], dtype=torch.float32)

    # 相机到世界的转换矩阵 (示例值)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    c2w[:3, -1] = torch.tensor([0, 0, 0], dtype=torch.float32)  # 相机位于世界坐标原点

    rays_o, rays_d = get_rays(H, W, K, c2w)

    print("射线起点 (rays_o) 的形状:", rays_o.shape)  # 应为 (600, 800, 3)
    print("射线方向 (rays_d) 的形状:", rays_d.shape)  # 应为 (600, 800, 3)
