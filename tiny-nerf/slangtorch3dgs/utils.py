import numpy as np
import torch
from plyfile import PlyData, PlyElement
import math
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