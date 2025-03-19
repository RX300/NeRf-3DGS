import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
from NeRFDataset import NeRFDatasetUnified
import imageio

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

def debug_image_values(rendered_image):
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

class GaussianSplatting:
    def __init__(self, dataset:NeRFDatasetUnified, num_gaussians=5000, learning_rate=5e-4):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        
        if not DIFF_RASTER_AVAILABLE:
            raise ImportError("diff_gaussian_rasterization is required but not available.")
        
        # Define SH degree and coefficients
        self.sh_degree = 3  # 3rd order spherical harmonics
        self.sh_dim = (self.sh_degree + 1) ** 2  # Number of SH coefficients
        
        # Load dataset
        self.dataset = dataset
        self.setup_from_dataset()
        
        # Initialize Gaussian parameters
        self.initialize_gaussians()
        
        # Setup optimizer
        self.optimizer = SGD(
            [
                {'params': self.means, 'lr': self.learning_rate},
                {'params': self.scales, 'lr': self.learning_rate},
                {'params': self.rotations, 'lr': self.learning_rate},
                {'params': self.opacities, 'lr': self.learning_rate},
                {'params': self.sh_coeffs, 'lr': self.learning_rate},
            ]
        )
    
    def setup_from_dataset(self):
        """Setup the renderer using the dataset properties."""
        # Extract data from dataset
        self.images = self.dataset.imgs  # [N, H, W, 3]
        self.poses = self.dataset.poses  # [N, 4, 4]
        self.hwf = self.dataset.hwf      # [H, W, focal]
        self.render_poses = self.dataset.render_poses
        
        # Extract dimensions
        self.height, self.width = int(self.hwf[0]), int(self.hwf[1])
        self.focal = self.hwf[2]
        self.n_images = len(self.images)
        
        # Move to device
        if not isinstance(self.images, torch.Tensor):
            self.images = torch.from_numpy(self.images).float().to(device)
        if not isinstance(self.poses, torch.Tensor):
            self.poses = torch.from_numpy(self.poses).float().to(device)
        
        print(f"Loaded {self.n_images} images of size {self.height}x{self.width}")
        print(f"Focal length: {self.focal}")
    
    def initialize_gaussians(self):
        """Initialize Gaussian parameters with better strategy."""
        # 在一个更合理的范围内初始化点位置
        # 例如，在[-1, 1]^3的立方体中均匀分布
        x = torch.linspace(0, 5, int(np.cbrt(self.num_gaussians))+1)
        y = torch.linspace(0, 5, int(np.cbrt(self.num_gaussians))+1)
        z = torch.linspace(0, 5, int(np.cbrt(self.num_gaussians))+1)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # 随机选择num_gaussians个点
        if points.shape[0] >= self.num_gaussians:
            idx = torch.randperm(points.shape[0])[:self.num_gaussians]
            points = points[idx]
        else:
            # 如果网格点不够，添加额外的随机点
            extra_points = torch.rand(self.num_gaussians - points.shape[0], 3) * 2 - 1
            points = torch.cat([points, extra_points], dim=0)
        
        # 添加少量噪声，打破规则性
        points = points + 0.1 * torch.randn_like(points)
        
        self.means = nn.Parameter(points.to(device))
        # Initialize scales
        #self.scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.scales = nn.Parameter(0.1*(torch.rand(self.num_gaussians, 3)).to(device).float())

        # Initialize rotations as quaternions (w, x, y, z)
        # Initialize with identity rotation (w=1, x=y=z=0)
        quat = torch.zeros(self.num_gaussians, 4, device=device).float()
        quat[:, 0] = 1.0  # w=1
        self.rotations = nn.Parameter(quat)

        # # Determine scene bounds from poses
        # # For Blender datasets, we know most content is within a unit sphere
        # near, far = 2.0, 6.0
        
        # # Initialize gaussians in a sphere with density increasing toward center
        # radius = 1.5  # Radius for gaussian placement
        
        # # Generate evenly distributed points on a sphere using Fibonacci spiral
        # n_points = self.num_gaussians
        # indices = torch.arange(0, n_points, dtype=torch.float32, device=device)
        # phi = torch.acos(1 - 2 * indices / n_points)
        # theta = torch.pi * (1 + 5**0.5) * indices
        
        # # Convert to Cartesian coordinates
        # x = torch.cos(theta) * torch.sin(phi)
        # y = torch.sin(theta) * torch.sin(phi)
        # z = torch.cos(phi)
        
        # # Scale by random radius (more points near center)
        # r = radius * torch.pow(torch.rand(n_points, device=device), 1/3)
        # points = torch.stack([x * r, y * r, z * r], dim=1)
        
        # # Add some noise to break symmetry
        # points += 0.05 * torch.randn_like(points)
        
        # # Set as parameter
        # self.means = nn.Parameter(points)
        
        # # Initialize scales with small values but log-parameterized
        # log_scales = -2.0 + 0.1 * torch.randn(self.num_gaussians, 3, device=device)
        # self.scales = nn.Parameter(log_scales)
        
        # # Initialize rotations as quaternions (w, x, y, z)
        # # Start near identity with some noise
        # quat = torch.zeros(self.num_gaussians, 4, device=device)
        # quat[:, 0] = 1.0  # w=1 for identity rotation
        # quat += 0.1 * torch.randn_like(quat)
        # self.rotations = nn.Parameter(F.normalize(quat, p=2, dim=1))
        
        # Initialize opacities as low values (logarithmic parameterization)
        self.opacities = nn.Parameter(-3.0 + 0.1 * torch.randn(self.num_gaussians, 1, device=device))
        
        # Initialize SH coefficients with variance decreasing for higher degrees
        sh_coeffs = torch.zeros(self.num_gaussians, 3, self.sh_dim, device=device)
        
        # Base colors with more diverse initialization - use more vibrant colors
        sh_coeffs[:, 0, 0] = torch.rand(self.num_gaussians, device=device) * 0.5 + 0.2  # Red channel
        sh_coeffs[:, 1, 0] = torch.rand(self.num_gaussians, device=device) * 0.5 + 0.2  # Green channel
        sh_coeffs[:, 2, 0] = torch.rand(self.num_gaussians, device=device) * 0.5 + 0.2  # Blue channel
        
        # Higher order coefficients with decreasing variance
        for i in range(1, self.sh_dim):
            scale = 0.5 / (1.0 + i)  # Decreasing scale for higher order terms
            sh_coeffs[:, :, i] = scale * torch.randn(self.num_gaussians, 3, device=device)
        
        self.sh_coeffs = nn.Parameter(sh_coeffs)
    
    def render_image_with_diff_raster(self, camera_pose, height=None, width=None, scaling_modifier=1.0):
        """Render an image using diff_gaussian_rasterization."""
        if height is None:
            height = self.height
        if width is None:
            width = self.width
            
        # Create camera intrinsics and extrinsics
        tanfovx = math.tan(0.5 * 2 * math.atan(width / (2 * self.focal)))
        tanfovy = math.tan(0.5 * 2 * math.atan(height / (2 * self.focal)))
        
        # Extract camera world to view transform from pose
        c2w = camera_pose.cpu().numpy()
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        world_view_transform = getWorld2View2(R, T)
        world_view_transform = torch.tensor(world_view_transform, device=device)
        
        # Create projection matrix (perspective)
        near = 0.01
        far = 4.0
        fov = focal2fov(self.focal, width)
        proj_matrix = getProjectionMatrix(near, far, fov, fov).to(device)
        
        # Camera center
        camera_center = camera_pose[:3, 3]
        
        # Background color (black)
        bg_color = torch.zeros(3, device=device)
        
        # Apply activations to parameters
        means3D = self.means
        scales = torch.exp(self.scales)  # Exponential activation for scales
        rotations = F.normalize(self.rotations, p=2, dim=1)  # Normalize quaternions
        opacities = torch.sigmoid(self.opacities)  # Sigmoid activation for opacity
        
        # Create rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=proj_matrix,
            sh_degree=self.sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=True  # Enable antialiasing for better quality
        )
        
        # Create rasterizer
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Create a dummy screenspace tensor for gradients
        screenspace_points = torch.zeros_like(means3D, device=device, requires_grad=True)
        
        # Process SH coefficients for the rasterizer
        # From [N, 3, SH_dim] to [N, SH_dim*3]
        shs = self.sh_coeffs.permute(0, 2, 1)
        
        # Render
        rendered_image, radii, invdepths = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales, 
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Reshape from [C, H, W] to [H, W, C]
        rendered_image = rendered_image.permute(1, 2, 0)
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
        
        return rendered_image
    
    def debug_image_values(self, rendered_image):
        """Analyze and print information about the rendered image."""
        # Check for NaNs or Infs
        if torch.isnan(rendered_image).any():
            print("WARNING: NaN values detected in rendered image!")
            
        if torch.isinf(rendered_image).any():
            print("WARNING: Infinite values detected in rendered image!")
        
        # Check value statistics
        print(f"Min: {rendered_image.min().item():.6f}, Max: {rendered_image.max().item():.6f}")
        print(f"Mean: {rendered_image.mean().item():.6f}, Std: {rendered_image.std().item():.6f}")
        
        # Check if values are in a good range
        if rendered_image.max().item() < 0.01:
            print("WARNING: Maximum value is very small, might appear black")
        
        if rendered_image.min().item() > 0.99:
            print("WARNING: Minimum value is very large, might appear white")
            
        if rendered_image.std().item() < 0.01:
            print("WARNING: Very little variation in pixel values")
            
        # Check individual channels
        for c in range(3):
            channel = rendered_image[..., c]
            print(f"Channel {c}: Min={channel.min().item():.4f}, Max={channel.max().item():.4f}, Mean={channel.mean().item():.4f}")
    
    def render_image(self, pose, height=None, width=None):
        """Render an image from a given camera pose."""
        return self.render_image_with_diff_raster(pose, height, width)
    
    def parameters(self):
        """Return all parameters of the model."""
        return [self.means, self.scales, self.rotations, self.opacities, self.sh_coeffs]
    
    def train_epoch(self, epoch, iterations_per_epoch):
        """训练一个epoch，顺序使用数据集中的图像。
        
        Args:
            epoch (int): 当前epoch编号
            iterations_per_epoch (int): 每轮迭代次数，通常等于数据集大小
            
        Returns:
            float: 本轮平均损失
        """
        epoch_loss = 0.0
        
        # 决定是否随机打乱训练顺序
        use_random_order = False
        if use_random_order:
            # 随机顺序
            indices = torch.randperm(self.n_images)[:iterations_per_epoch]
        else:
            # 按顺序使用所有图像，可能会重复或截断
            indices = torch.arange(min(iterations_per_epoch, self.n_images))
        
        with tqdm(total=len(indices), desc=f"Epoch {epoch+1}", ncols=100) as pbar:
            for i, idx in enumerate(indices):
                # 获取这张图像
                gt_image = self.images[idx]
                if gt_image.shape[2] == 4:
                    gt_image = gt_image[:, :, :3]  # 去掉alpha通道 [H, W, 3]
                    
                pose = self.poses[idx]
                
                # 前向传播 - 渲染图像
                rendered_image = self.render_image_with_diff_raster(pose) # [H, W, 3]
                
                # 计算损失 (MSE)
                gt_image_reshaped = gt_image.reshape(-1, 3)
                rendered_image_reshaped = rendered_image.reshape(-1, 3)
                from loss_utils import ssim,l1_loss
                mse = torch.nn.functional.mse_loss(rendered_image_reshaped, gt_image_reshaped)
                l1 = l1_loss(rendered_image_reshaped, gt_image_reshaped)
                # 额外的SSIM损失分量以改善结构相似性
                ssim_loss = 0.0
                if rendered_image.shape[0] > 10 and rendered_image.shape[1] > 10:  # SSIM的最小尺寸
                    try:
                        # rendered_batch = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        # gt_batch = gt_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        # ssim_val = ssim(rendered_batch, gt_batch)
                        # ssim_loss = 0.2 * (1.0 - ssim_val)
                        rendered_batch = rendered_image.permute(2, 0, 1)  # [3, H, W]
                        gt_batch = gt_image.permute(2, 0, 1) # [3, H, W]
                        #ssim_loss = ssim(rendered_batch, gt_batch)
                    except:
                        pass  # 如果loss_utils不可用则跳过
                
                # 组合损失
                loss = mse
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                epoch_loss += loss.item()
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.6f}', 'mse': f'{mse.item():.6f}'})
        
        # 计算平均损失
        avg_loss = epoch_loss / len(indices)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        return avg_loss
    
    def train(self, num_epochs=100, iterations_per_epoch=None, save_dir='results'):
        """Train the Gaussian Splatting model.
        
        Args:
            num_epochs (int): 训练轮数
            iterations_per_epoch (int, optional): 每轮迭代次数。如果为None，则使用数据集大小
            save_dir (str): 保存结果的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 如果未指定iterations_per_epoch，则使用整个数据集大小
        if iterations_per_epoch is None:
            iterations_per_epoch = self.n_images
            
        print(f"Training for {num_epochs} epochs with {iterations_per_epoch} images per epoch")
        
        # 训练循环
        losses = []
        best_loss = float('inf')
        best_psnr = 0.0
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        
        for epoch in range(num_epochs):
            # 训练阶段 - 按顺序使用数据集中的所有图像
            epoch_loss = self.train_epoch(epoch, iterations_per_epoch)
            losses.append(epoch_loss)
            
            # 验证阶段
            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                avg_loss, avg_psnr = self.validate(epoch, save_dir)
                
                # 保存最佳模型
                if avg_psnr > best_psnr:
                    best_loss = avg_loss
                    best_psnr = avg_psnr
                    self.save_checkpoint(f"{save_dir}/best_model.pt")
                    print(f"Saved new best model with PSNR: {best_psnr:.2f}")
            
            # 更新学习率
            scheduler.step()
            
            # 每50个epoch保存一次检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"{save_dir}/model_epoch_{epoch+1}.pt")
        
        # 保存最终模型
        self.save_checkpoint(f"{save_dir}/final_model.pt")
        
        return losses
    
    def validate(self, epoch, save_dir):
        """Validate on the test set."""
        val_losses = []
        val_psnrs = []
        
        # Create validation directory
        val_dir = os.path.join(save_dir, f"validation_epoch_{epoch+1}")
        os.makedirs(val_dir, exist_ok=True)
        
        # Use a small subset of training poses for validation
        val_indices = list(range(0, min(self.n_images, 5)))
        
        with torch.no_grad():
            for idx in tqdm(val_indices, desc="Validating", ncols=100):
                # Render from this viewpoint
                pose = self.poses[idx]
                gt_image = self.images[idx]
                if gt_image.shape[2] == 4:
                    gt_image = gt_image[:, :, :3]
                
                # Render
                rendered_image = self.render_image(pose)
                debug_image_values(rendered_image)
                # Compute metrics
                mse = F.mse_loss(rendered_image, gt_image)
                psnr = -10 * torch.log10(mse)
                
                val_losses.append(mse.item())
                val_psnrs.append(psnr.item())
                
                # Save the rendered image
                rendered_np = rendered_image.cpu().numpy()
                gt_np = gt_image.cpu().numpy()
                
                # Create comparison image
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(rendered_np)
                axes[0].set_title(f"Rendered (PSNR: {psnr.item():.2f})")
                axes[1].imshow(gt_np)
                axes[1].set_title("Ground Truth")
                plt.tight_layout()
                plt.savefig(os.path.join(val_dir, f"comparison_{idx}.png"))
                plt.close()
        
        # Compute average metrics
        avg_loss = sum(val_losses) / len(val_losses)
        avg_psnr = sum(val_psnrs) / len(val_psnrs)
        
        print(f"Validation - Avg Loss: {avg_loss:.6f}, Avg PSNR: {avg_psnr:.2f}")
        
        # Save metrics to file
        with open(os.path.join(save_dir, "validation_metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}, PSNR = {avg_psnr:.2f}\n")
        
        return avg_loss, avg_psnr
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'means': self.means,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': self.opacities,
            'sh_coeffs': self.sh_coeffs,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        self.means = checkpoint['means']
        self.scales = checkpoint['scales']
        self.rotations = checkpoint['rotations']
        self.opacities = checkpoint['opacities']
        self.sh_coeffs = checkpoint['sh_coeffs']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def render_spiral(self, num_frames=120, save_path=None, radius=4.0, zrate=0.5, render_factor=1):
        """Render a spiral trajectory around the scene."""
        frames = []
        
        # If save_path is provided, create directory if it doesn't exist
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # If render poses already exist
        if hasattr(self, 'render_poses') and self.render_poses is not None:
            render_poses = self.render_poses.to(device)
            num_frames = len(render_poses)
            
            for i in tqdm(range(num_frames), desc="Rendering spiral"):
                pose = render_poses[i]
                with torch.no_grad():
                    rendered_image = self.render_image(pose, 
                                                       self.height//render_factor, 
                                                       self.width//render_factor)
                    img_np = rendered_image.cpu().numpy()
                    img_np = (255 * np.clip(img_np, 0, 1)).astype(np.uint8)
                    frames.append(img_np)
                    
                    # Save individual frame if needed
                    if save_path is not None and '.' in os.path.basename(save_path):
                        frame_path = save_path.replace('.mp4', f'_{i:03d}.png')
                        imageio.imsave(frame_path, img_np)
        else:
            # Generate spiral path
            theta = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
            
            for i in tqdm(range(num_frames), desc="Rendering spiral"):
                # Compute camera pose for this frame
                angle = theta[i]
                
                # Create camera-to-world matrix
                cam2world = np.zeros((4, 4))
                cam2world[3, 3] = 1
                
                # Camera position on a circle in the xz-plane
                cam2world[0, 3] = radius * np.cos(angle)  # x
                cam2world[1, 3] = 0.5 * zrate * np.sin(2 * angle)  # Small up/down motion
                cam2world[2, 3] = radius * np.sin(angle)  # z
                
                # Camera orientation - looking at origin
                forward = -cam2world[:3, 3].copy()  # Look toward the origin
                forward = forward / np.linalg.norm(forward)
                
                right = np.cross([0, 1, 0], forward)
                right = right / np.linalg.norm(right)
                
                up = np.cross(forward, right)
                
                cam2world[:3, 0] = right
                cam2world[:3, 1] = up
                cam2world[:3, 2] = forward
                
                pose = torch.from_numpy(cam2world).float().to(device)
                
                # Render
                with torch.no_grad():
                    rendered_image = self.render_image(pose, 
                                                       self.height//render_factor, 
                                                       self.width//render_factor)
                    img_np = rendered_image.cpu().numpy()
                    img_np = (255 * np.clip(img_np, 0, 1)).astype(np.uint8)
                    frames.append(img_np)
                    
                    # Save individual frame if needed
                    if save_path is not None and '.' in os.path.basename(save_path):
                        frame_path = save_path.replace('.mp4', f'_{i:03d}.png')
                        imageio.imsave(frame_path, img_np)
        
        # Save video if path is provided
        if save_path is not None:
            imageio.mimwrite(save_path, frames, fps=30)
            print(f"Video saved to {save_path}")
        
        return frames

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    # Create results directory
    results_dir = "3dgs_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if diff_gaussian_rasterization is available
    if not DIFF_RASTER_AVAILABLE:
        print("diff_gaussian_rasterization is required. Please install it first.")
        return
    
    print("Loading dataset...")
    
    # Choose a dataset path (e.g., hotdog)
    dataset_path = "./hotdog"  # Change this to your dataset path
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return
    
    # Load dataset
    dataset = NeRFDatasetUnified(
        basedir=dataset_path,
        dataset_type='blender',
        split='train',
        half_res=False,
        testskip=1,
        device='cpu'  # Initially load on CPU, then move to device as needed
    )
    
    print(f"Dataset loaded with {len(dataset.imgs)} images of size {dataset.hwf[0]}x{dataset.hwf[1]}")
    
    # Create Gaussian Splatting model
    num_gaussians = 100000  # Start with a larger number for complex scenes
    model = GaussianSplatting(dataset, num_gaussians=num_gaussians, learning_rate=5e-4)
    
    # Set training parameters
    num_epochs = 300
    iterations_per_epoch = None  # Adjust based on dataset size
    
    # Train the model
    print("Starting training...")
    losses = model.train(num_epochs=num_epochs, 
                        iterations_per_epoch=iterations_per_epoch,
                        save_dir=results_dir)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{results_dir}/training_loss.png")
    
    # Render a spiral video
    print("Rendering spiral video...")
    model.render_spiral(save_path=f"{results_dir}/spiral_render.mp4")
    
    print("Training and rendering complete!")

if __name__ == "__main__":
    main()