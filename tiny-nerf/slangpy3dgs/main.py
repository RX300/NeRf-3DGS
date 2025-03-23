import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
from pathlib import Path
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import slangpy as spy
import pathlib

# Import the diff_gaussian_rasterization module
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, 
        GaussianRasterizer
    )
    DIFF_RASTER_AVAILABLE = True
except ImportError:
    DIFF_RASTER_AVAILABLE = False
    print("WARNING: diff_gaussian_rasterization module not found. Please install it first:")
    print("pip install submodules/diff-gaussian-rasterization")

# Set the random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

from typing import NamedTuple
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class GaussianSplatting:
    def __init__(self, num_gaussians=5000, learning_rate=5e-4):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        
        if not DIFF_RASTER_AVAILABLE:
            raise ImportError("diff_gaussian_rasterization is required but not available.")
        
        # Define SH degree and coefficients
        self.sh_degree = 3  # 3rd order spherical harmonics
        self.sh_dim = (self.sh_degree + 1) ** 2  # Number of SH coefficients
        
        # Load the NeRF dataset
        self.load_nerf_data('../tiny_nerf_data.npz')
        # Create an SGL device with the local folder for slangpy includes
        sgldevice = spy.create_device(include_paths=[
                pathlib.Path(__file__).parent.absolute(),
        ])

        # Load the module
        self.utilsmodule = spy.TorchModule.load_from_file(sgldevice, "spherical_harmonics.slang")
        self.shadermodule = spy.TorchModule.load_from_file(sgldevice, "vertex_shader.slang")
        # Initialize Gaussian parameters
        self.xyz = torch.empty(0)
        self.scales = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacities = torch.empty(0)
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        self.initialize_gaussians()
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # Setup optimizer
        self.optimizer = Adam(
            [
                {'params': self.means, 'lr': self.learning_rate},
                {'params': self.scales, 'lr': self.scaling_lr},
                {'params': self.rotations, 'lr': self.rotation_lr},
                {'params': self.opacities, 'lr': self.opacity_lr},
                {'params': self.features_dc, 'lr': self.feature_lr},
                {'params': self.features_rest, 'lr': self.feature_lr/20.0}
            ]
        )

    def load_nerf_data(self, filename):
        """Load data from the tiny_nerf_data.npz file."""
        data = np.load(filename)
        
        # Extract data
        self.images = data['images']  # [N, H, W, 3]
        self.poses = data['poses']    # [N, 4, 4]
        self.focal = float(data['focal'])
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).to(device)
        self.poses = torch.from_numpy(self.poses).to(device)
        
        # Get dimensions
        self.n_images, self.height, self.width, _ = self.images.shape
        print(f"Loaded {self.n_images} images of size {self.height}x{self.width}")
        # 处理点云
        path = "./results"
        # 如果path不存在
        if not os.path.exists(path):
            os.makedirs(path)
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # tinyNeRF 格式没有预先生成的点云，所以我们创建一个随机点云
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # 我们在合成场景的范围内创建随机点，范围与Blender场景相似
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            
            # 保存点云到PLY文件
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
        try:
            self.pcd = fetchPly(ply_path)
        except:
            self.pcd = None
        # Create rays for all pixels
        self.rays_o, self.rays_d = self.get_rays()
    
    def initialize_gaussians(self):
        """Initialize Gaussian parameters."""
        # self.features_dc.shape=>[num_gaussians,1,3]
        # self.features_rest.shape=>[num_gaussians,(max_sh_degree+1)**2-1,3]
        self.means = nn.Parameter(data=torch.rand(self.num_gaussians, 3, device=device), requires_grad=True)
        dist2 = torch.clamp_min(distCUDA2(self.means.float().cuda()), 0.0000001)
        self.scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.rotations = nn.Parameter(data=torch.zeros(self.num_gaussians, 4, device=device), requires_grad=True)
        opacities = inverse_sigmoid(0.1 * torch.ones(self.num_gaussians, 1, device=device))
        self.opacities = nn.Parameter(data=opacities, requires_grad=True)
        self.features_dc = nn.Parameter(data=torch.zeros(self.num_gaussians, 1, 3, device=device), requires_grad=True)
        self.features_rest = nn.Parameter(data=torch.zeros(self.num_gaussians, (self.sh_degree+1)**2-1, 3, device=device), requires_grad=True)   

    def get_rays(self):
        """Compute rays for all pixels in all images."""
        i, j = torch.meshgrid(
            torch.arange(self.height, device=device),
            torch.arange(self.width, device=device),
            indexing='ij'
        )
        
        i = i.flatten()
        j = j.flatten()
        
        # Map pixel coordinates to NDC space
        dirs = torch.stack([
            (j - self.width/2) / self.focal,   # x-direction
            -(i - self.height/2) / self.focal,  # y-direction
            -torch.ones_like(i, device=device)  # z-direction
        ], dim=-1)  # (H*W, 3)
        
        # Repeat for each image
        dirs = dirs.unsqueeze(0).repeat(self.n_images, 1, 1)  # (N, H*W, 3)
        
        # Rotate rays based on camera poses
        rays_d = torch.zeros_like(dirs)
        for i in range(self.n_images):
            # Extract rotation matrix from pose
            rot = self.poses[i, :3, :3]  # 3x3 rotation matrix
            # Rotate directions
            rays_d[i] = torch.matmul(dirs[i], rot.T)
        
        # Normalize directions
        rays_d = F.normalize(rays_d, p=2, dim=-1)
        
        # Origins from camera poses
        rays_o = self.poses[:, :3, 3].unsqueeze(1).repeat(1, self.height * self.width, 1)
        
        return rays_o, rays_d
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternions to rotation matrices."""
        q = F.normalize(q, p=2, dim=1)  # Ensure unit quaternions
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Construct rotation matrix
        rot = torch.zeros(q.shape[0], 3, 3, device=device)
        
        # Fill in rotation matrix elements
        rot[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        rot[:, 0, 1] = 2 * (x * y - w * z)
        rot[:, 0, 2] = 2 * (x * z + w * y)
        
        rot[:, 1, 0] = 2 * (x * y + w * z)
        rot[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        rot[:, 1, 2] = 2 * (y * z - w * x)
        
        rot[:, 2, 0] = 2 * (x * z - w * y)
        rot[:, 2, 1] = 2 * (y * z + w * x)
        rot[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return rot
    
    def build_covariance_from_scaling_rotation(self, scaling, rotation):
        """Build 3D covariance matrices from scaling and rotation."""
        # Convert quaternion to rotation matrix
        R = self.quaternion_to_rotation_matrix(rotation)
        
        # Create scaling matrix S
        S = torch.diag_embed(scaling)
        
        # Compute covariance matrix: Sigma = R * S * S * R^T
        S_squared = torch.matmul(S, S)
        covariance = torch.matmul(torch.matmul(R, S_squared), R.transpose(1, 2))
        
        return covariance
    
    def render_image_with_diff_raster(self, camera_pose, height=None, width=None, scaling_modifier=1.0):
        """Render an image using diff_gaussian_rasterization."""
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_pose.cpu().numpy()
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        near = 0.01
        far = 5
        fov = focal2fov(self.focal, width)
        tanfovx = math.tan(fov * 0.5)
        tanfovy = math.tan(fov * 0.5)
        projection_matrix = getProjectionMatrix(near, far, fov, fov).transpose(0,1).to(device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        # Background color
        bg_color = torch.tensor([0,0,0],dtype=torch.float32,device=device,requires_grad=False)  # White background
        
        # Create rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=3,  # Using 3rd order SH
            campos=camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=True
        )
        
        # Create rasterizer
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Create a dummy screenspace tensor for gradients
        screenspace_points = torch.zeros_like(self.means, device=device, requires_grad=True)+0

        # Get parameters for rasterization
        #将means3D转换为tensor
        means3D =  torch.ones_like(self.means) * self.means
        scales = torch.exp(self.scales)  # Apply activation to scales
        rotations = torch.nn.functional.normalize(self.rotations)  # Ensure unit quaternions
        opacity = torch.sigmoid(self.opacities)  # Apply activation to opacities
        # 1. 确保参数形状正确
        means3D = means3D.contiguous()
        scales = scales.contiguous()
        rotations = rotations.contiguous()
        opacity = opacity.contiguous()
        # 2. 将世界视图变换矩阵转为合适格式
        world_view_transform = world_view_transform.float().contiguous()
        full_proj_transform = full_proj_transform.float().contiguous()
        camera_center = camera_center.float().contiguous()
        # 3. 创建输出参数
        N = self.num_gaussians
        out_radii = np.zeros(shape=(N), dtype=np.int32, device='cpu')
        out_xyz_vs = torch.ones(N, 3, dtype=torch.float32, device=device)
        out_depths = torch.zeros(N, dtype=torch.float32, device=device)
        out_cov3Ds = torch.zeros(N, 6, dtype=torch.float32, device=device)  # 3x3 对称矩阵的上三角部分
        out_rgb = torch.zeros(N, 3, dtype=torch.float32, device=device)
        out_inv_cov_vs = torch.zeros(N, 4, dtype=torch.float32, device=device)
        out_tiles_touched = np.zeros(shape=(N), dtype=np.int32, device='cpu')
        out_pixels_xy = torch.zeros(size=(N, 2), dtype=torch.float32, device=device)
        testPointsVS = torch.ones(N, 3, dtype=torch.float32, device=device)
        p_hom_test = torch.zeros(N, 4, dtype=torch.float32, device=device)
        conic_opacity = torch.zeros(N, 4, dtype=torch.float32, device=device)
        # means.shape => [num_gaussians, 3]
        # scales.shape => [num_gaussians, 3]
        # rotations.shape => [num_gaussians, 4]
        # opacity.shape => [num_gaussians, 1]
        #这里一定要在每次训练前重新拼接sh_coeffs
        # shs.shape => [num_gaussians, sh_dim, 3]
        shs = torch.cat((self.features_dc, self.features_rest), dim=1)
        # gaussian = torch.ones(size=(100000,48), dtype=torch.float32, device='cuda', requires_grad=True)
        # print(shs.shape)
        # self.utilsmodule.testNdArray(shs.reshape(self.num_gaussians,-1))

        self.shadermodule.preprocess_shader(
            spy.thread_id(),                    # g_idx - 线程ID
            means3D,                            # xyz_ws - 世界空间中高斯点的位置
            shs.reshape(self.num_gaussians, -1),  # sh_coeffs - 球谐系数
            opacity.squeeze(),                            # opacities - 不透明度
            rotations,                          # rotations - 旋转四元数
            scales,                             # scales - 缩放系数
            3,                                  # active_sh - 活动的球谐阶数
            1.0,                                # scale_modifier - 缩放修正系数
            world_view_transform.T.reshape(-1),               # world_view_transform - 世界到视图变换矩阵
            projection_matrix.T.reshape(-1),                # proj_mat - 投影矩阵
            camera_center,                      # cam_pos - 相机位置
            fov,                                # fovy - 视场角y
            fov,                                # fovx - 视场角x
            int(height),                        # image_height - 图像高度
            int(width),                         # image_width - 图像宽度
            False,                              # prefiltered - 是否预过滤
            True,                                # antialiasing - 是否抗锯齿
            out_radii,                          # 输出半径
            out_xyz_vs,                         # 输出视图空间位置  
            out_depths,                         # 输出深度
            out_cov3Ds,                         # 输出3D协方差矩阵
            out_rgb,                            # 输出RGB颜色
            out_inv_cov_vs,                     # 输出视图空间逆协方差矩阵
            out_tiles_touched,                   # 输出触及的瓦片数
            out_pixels_xy,                      # 输出像素坐标
            testPointsVS,
            p_hom_test
        )
        means3Dhomogeneous = torch.cat((means3D, torch.ones(N, 1, device=device)), dim=1)
        # gtPointsVS = torch.matmul(means3Dhomogeneous, world_view_transform.T)
        # print(f"gtPointsVS:{gtPointsVS.mean(dim=0)}")
        # print(f"testPointsVS:{testPointsVS.mean(dim=0)}")
        # meansHomo = torch.matmul(means3Dhomogeneous, full_proj_transform)
        # print(f"out_xyz_vs:{out_xyz_vs.mean(dim=0)}")
        print(f"out_pixels_xy:{out_pixels_xy.mean(dim=0)}")
        # print(f"meansHomo:{meansHomo.mean(dim=0)}")
        print(f"p_hom_test:{p_hom_test.mean(dim=0)}")
        print(f"out_radii:{out_radii.mean()}")
        print(f"out_depths:{out_depths.mean()}")
        print(f"out_inv_cov_vs:{out_inv_cov_vs.mean(dim=0)}")
        # Render
        rendered_image, radii, invdepths,means2D,meanshomo,conic_opacity = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,  # Using spherical harmonics
            colors_precomp=None,  # Not using precomputed colors
            opacities=opacity,
            scales=scales, 
            rotations=rotations,
            cov3D_precomp=None  # Will be computed from scales and rotations
        )
        # print(f"meanshomo.shape:{meanshomo.shape}")
        # print(f"meanshomo:{meanshomo.mean(dim=0)}")
        print(f"means2D.shape:{means2D.shape}")
        print(f"means2D:{means2D.mean(dim=0)}")
        print(f"radii.shape:{radii.shape}")
        print(f"radii:{radii.float().mean()}")
        #对每个invdepth求倒数
        depths = 1.0 / invdepths
        print(f"invdepths.shape:{invdepths.shape}")
        print(f"invdepths:{invdepths.mean()}")
        print(f"depths:{depths.mean()}")
        print(f"conic_opacity:{conic_opacity.mean(dim=0)}")
        exit()
        rendered_image = rendered_image.permute(1, 2, 0).contiguous()
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
        return rendered_image
    
    def debug_image_values(self, rendered_image):
        # Check for NaNs or Infs
        if torch.isnan(rendered_image).any():
            print("WARNING: NaN values detected in rendered image!")
            
        if torch.isinf(rendered_image).any():
            print("WARNING: Infinite values detected in rendered image!")
        
        # Check value statistics
        print(f"Min: {rendered_image.min().item()}, Max: {rendered_image.max().item()}")
        print(f"Mean: {rendered_image.mean().item()}, Std: {rendered_image.std().item()}")
        
        # Check if all values are very small
        if rendered_image.max().item() < 0.01:
            print("WARNING: Maximum value is very small, might appear black")
        
        # Check if all values are very large
        if rendered_image.min().item() > 0.99:
            print("WARNING: Minimum value is very large, might appear white")
            
        # Check if no variation
        if rendered_image.std().item() < 0.01:
            print("WARNING: Very little variation in pixel values")
            
        # Check if any channel has unusual values
        for c in range(3):
            channel = rendered_image[..., c]
            print(f"Channel {c}: Min={channel.min().item():.4f}, Max={channel.max().item():.4f}, Mean={channel.mean().item():.4f}")

    def render_image(self, pose, height=None, width=None):
        """Render an image from a given camera pose."""
        return self.render_image_with_diff_raster(pose, height, width)
    
    def train(self, num_epochs=100, batch_size=4096,save_dir=None):
        """Train the Gaussian Splatting model."""
        # Training loop
        losses = []
        mse = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            # Train on all images
            total_loss = 0.0
            num_batches = 0
            
            for img_idx in tqdm(range(self.n_images), desc=f"Epoch {epoch+1}/{num_epochs}"):
                self.sh_coeffs = torch.cat((self.features_dc, self.features_rest), dim=1)
                # Get image and pose for this view
                gt_image = self.images[img_idx]
                pose = self.poses[img_idx]

                self.optimizer.zero_grad()
                if torch.isnan(self.means).any():
                    print(f"在{epoch}次第{img_idx}张图片的训练means3D中有nan值")
                    #打印nan值数量
                    print(f"nan值数量：{torch.isnan(self.means).sum()}")
                # else:
                #     print(f"在{epoch}次第{img_idx}张图片的训练means3D中没有nan值")
                rendered_image = self.render_image_with_diff_raster(pose)
                if img_idx == self.n_images-1:
                    self.debug_image_values(rendered_image)

                # Compute loss (MSE)
                gt_image_reshaped = gt_image.reshape(-1, 3)
                rendered_image_reshaped = rendered_image.reshape(-1, 3)
                loss = mse(rendered_image, gt_image)
                
                # Backward pass
                loss.backward()
                #self.gaussians.optimizer.step()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Compute average loss
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # Visualize progress every epoch
            if (epoch + 1) % 50 == 0 or epoch == 0:
                test_pose = self.poses[0]  # Use first pose for visualization
                with torch.no_grad():
                    rendered_image = self.render_image(test_pose)
                    self.debug_image_values(rendered_image)
                    rendered_image_numpy = rendered_image.cpu().numpy()
                    rendered_image_numpy = (255*np.clip(rendered_image_numpy, 0.0, 1.0)).astype(np.uint8)
                    if save_dir is None:
                        save_dir = '3dgs_rendered_images'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.imsave(f"{save_dir}/rendered_image_{epoch+1}.png", rendered_image_numpy)
                
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(rendered_image_numpy)
                plt.title(f"Rendered (Epoch {epoch+1})")
                
                plt.subplot(1, 2, 2)
                plt.imshow(self.images[0].cpu().numpy())
                plt.title("Ground Truth")
                
                plt.suptitle(f"Loss: {avg_loss:.6f}")
                plt.tight_layout()
                plt.show()
        
        return losses
    
    def render_360_orbit(self, num_frames=30, radius=4.0, height=None, width=None):
        """Render a 360-degree orbit around the scene."""
        frames = []
        
        for i in range(num_frames):
            # Create camera pose
            angle = 2 * np.pi * i / num_frames
            x = radius * np.sin(angle)
            z = radius * np.cos(angle)
            
            # Look-at pose (camera at (x, 0, z) looking at origin)
            pose = torch.eye(4, device=device)
            
            # Camera position
            pose[0, 3] = x
            pose[1, 3] = 0
            pose[2, 3] = z
            
            # Camera orientation (look at origin)
            forward = -torch.tensor([x, 0, z], device=device)
            forward = F.normalize(forward, p=2, dim=0)
            
            # Use world up vector
            up = torch.tensor([0, 1, 0], device=device)
            
            # Compute camera basis
            right = torch.cross(forward, up)
            right = F.normalize(right, p=2, dim=0)
            
            up = torch.cross(right, forward)
            up = F.normalize(up, p=2, dim=0)
            
            # Set rotation matrix
            pose[0, :3] = right
            pose[1, :3] = up
            pose[2, :3] = -forward  # Negate because camera looks along -Z
            
            # Render frame
            with torch.no_grad():
                frame = self.render_image(pose, height, width)
                frames.append(frame.cpu().numpy())
        
        return frames

# Installation helper function
def check_and_install_diff_gaussian_rasterization():
    """Check if diff_gaussian_rasterization is installed, if not, attempt to install it."""
    try:
        import diff_gaussian_rasterization
        print("diff_gaussian_rasterization is already installed.")
        return True
    except ImportError:
        print("diff_gaussian_rasterization not found. Attempting to install...")
        try:
            # Try to find the submodule directory in the repository
            import os
            if os.path.exists("submodules/diff-gaussian-rasterization"):
                import subprocess
                result = subprocess.run(
                    ["pip", "install", "-e", "submodules/diff-gaussian-rasterization"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("Successfully installed diff_gaussian_rasterization.")
                    return True
                else:
                    print(f"Failed to install: {result.stderr}")
                    return False
            else:
                print("Could not find diff-gaussian-rasterization in the submodules directory.")
                print("Please install manually using:")
                print("  git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git")
                print("  pip install -e ./diff-gaussian-rasterization")
                return False
        except Exception as e:
            print(f"Error during installation: {e}")
            return False

# Main execution
if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Check if diff_gaussian_rasterization is installed
    if not check_and_install_diff_gaussian_rasterization():
        print("Failed to install diff_gaussian_rasterization. Please install manually.")
        exit(1)
    
    # Create Gaussian Splatting model
    model = GaussianSplatting(num_gaussians=100000, learning_rate=1e-3)

    # Train the model
    losses = model.train(num_epochs=1000, batch_size=4096,save_dir='3dgs_rendered_images')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()
    
    # Render a test image
    test_pose = model.poses[0]  # Use first pose for testing
    with torch.no_grad():
        rendered_image = model.render_image(test_pose)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rendered_image.cpu().numpy())
    plt.title("Rendered")
    
    plt.subplot(1, 2, 2)
    plt.imshow(model.images[0].cpu().numpy())
    plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.show()
    
    # Render a 360-degree orbit
    print("Rendering 360-degree orbit...")
    frames = model.render_360_orbit(num_frames=30)
    
    # Display a few frames
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(frames[i*7])
        plt.title(f"Frame {i*7}")
    
    plt.tight_layout()
    plt.show()
    
    print("Done!")