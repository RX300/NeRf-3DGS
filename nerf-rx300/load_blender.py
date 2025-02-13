import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    #  数据集的索引划分，包含三个部分：
    # i_split[0]: 训练集的图像索引。
    # i_split[1]: 验证集的图像索引。
    # i_split[2]: 测试集的图像索引。
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # 用于渲染新视角的相机位姿，按照球形路径均匀分布
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                               for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


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
