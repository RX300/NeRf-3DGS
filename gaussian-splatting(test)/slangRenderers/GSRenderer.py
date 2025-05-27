import slangtorch
import torch
import torch.nn as nn
from pathlib import Path
import os
import math

def get_cuda_compute_capability_string():
    """
    获取当前 PyTorch 可用 CUDA 设备的计算能力字符串 (例如 '8.6')。
    如果没有可用的 CUDA 设备，则返回 None。
    """
    if torch.cuda.is_available():
        # 获取第一个可用 GPU 的计算能力
        # get_device_capability() 返回一个元组 (major, minor)
        major, minor = torch.cuda.get_device_capability(0) # 0 代表第一个 GPU
        return f"{major}.{minor}"
    else:
        return None

def sort_by_keys_torch(keys, values):
  """Sorts a values tensor by a corresponding keys tensor."""
  sorted_keys, idxs = torch.sort(keys)
  sorted_val = values[idxs]
  return sorted_keys, sorted_val

class GSRenderer:
    def __init__(self,image_height, image_width, tile_width=16, tile_height=16):
        # 获取当前目录
        self.DIR = Path(__file__).parent
        # 获取计算能力
        compute_capability = get_cuda_compute_capability_string()
        if compute_capability:
            print(f"检测到 CUDA 计算能力: {compute_capability}")
            # 设置环境变量 (如果你确实需要这样做的话)
            os.environ['TORCH_CUDA_ARCH_LIST'] = compute_capability
            print(f"TORCH_CUDA_ARCH_LIST 已设置为: {os.environ['TORCH_CUDA_ARCH_LIST']}")
        else:
            raise RuntimeError("No CUDA device available. Please ensure you have a compatible GPU and PyTorch with CUDA support installed.")
        # 加载着色器
        print("Compiling shaders...")
        self.init_vertex_shader()
        self.init_tile_shader()
        self.image_height = image_height
        self.image_width = image_width
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.grid_width = (image_width + tile_width - 1) // tile_width
        self.grid_height = (image_height + tile_height - 1) // tile_height
        self.init_fragment_shader(self.tile_height, self.tile_width)
        print("Compiling shaders... done")
    def init_vertex_shader(self):
        self.vertex_shader = slangtorch.loadModule(os.path.join(self.DIR, "shader/vertex_shader.slang"), skipNinjaCheck=True)
    def init_tile_shader(self):
        self.tile_shader = slangtorch.loadModule(os.path.join(self.DIR, "shader/tile_shader.slang"), skipNinjaCheck=True)
    def init_fragment_shader(self, tile_height, tile_width):
        self.fragment_shader = slangtorch.loadModule(os.path.join(self.DIR, "shader/alphablend_shader.slang"),
                                                    defines={"PYTHON_TILE_HEIGHT": tile_height, "PYTHON_TILE_WIDTH": tile_width},
                                                    skipNinjaCheck=True)
    def run_shader(self,xyz_ws, rotations, scales,opacity,sh_coeffs, active_sh,world_view_transform, proj_mat, cam_pos,fovy, fovx):
        n_points = xyz_ws.shape[0]
        tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb =VertexShader.apply(
            self,xyz_ws,sh_coeffs, rotations, scales, opacity, active_sh,world_view_transform, proj_mat, cam_pos,fovy, fovx)
        with torch.no_grad():
            index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
            total_size_index_buffer = index_buffer_offset[-1]
            unsorted_keys = torch.zeros((total_size_index_buffer,), device="cuda", dtype=torch.int64)
            unsorted_gauss_idx = torch.zeros((total_size_index_buffer,), device="cuda", dtype=torch.int32)
            self.tile_shader.generate_keys(xyz_vs=xyz_vs,
                                            rect_tile_space=rect_tile_space,
                                            index_buffer_offset=index_buffer_offset,
                                            out_unsorted_keys=unsorted_keys,
                                            out_unsorted_gauss_idx=unsorted_gauss_idx,
                                            grid_height=self.grid_height,
                                            grid_width=self.grid_width).launchRaw(
                    blockSize=(256, 1, 1),
                    gridSize=(math.ceil(n_points/256), 1, 1)
            )    
            sorted_keys, sorted_gauss_idx = sort_by_keys_torch(unsorted_keys, unsorted_gauss_idx)
            tile_ranges = torch.zeros((self.grid_height*self.grid_width, 2), 
                                        device="cuda",
                                        dtype=torch.int32)

            self.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,out_tile_ranges=tile_ranges).launchRaw(
                    blockSize=(256, 1, 1),gridSize=(math.ceil(total_size_index_buffer/256), 1, 1))
                # retain_grad fails if called with torch.no_grad() under evaluation
        try:
            xyz_vs.retain_grad()
        except:
            pass
        image_rgb = FragmentShader.apply(self,sorted_gauss_idx, tile_ranges,xyz_vs, inv_cov_vs, opacity, rgb)
        return image_rgb.permute(2,0,1)[:3, ...], xyz_vs, radii

class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, renderer:GSRenderer,xyz_ws,sh_coeffs, rotations, scales,opcities, active_sh,world_view_transform, proj_mat, cam_pos,
                fovy, fovx):
        n_points = xyz_ws.shape[0]
        tiles_touched = torch.zeros((n_points), device="cuda", dtype=torch.int32)
        rect_tile_space = torch.zeros((n_points, 4), device="cuda", dtype=torch.int32)
        radii = torch.zeros((n_points),device="cuda",dtype=torch.int32)
        xyz_vs = torch.zeros((n_points, 3),device="cuda", dtype=torch.float)
        inv_cov_vs = torch.zeros((n_points, 2, 2),device="cuda",dtype=torch.float)
        rgb = torch.zeros((n_points, 3),device="cuda",dtype=torch.float)

        renderer.vertex_shader.vertex_shader(xyz_ws=xyz_ws,
                                    sh_coeffs=sh_coeffs,
                                    rotations=rotations,
                                    scales=scales,
                                    opcities=opcities,
                                    active_sh=active_sh,
                                    world_view_transform=world_view_transform,
                                    proj_mat=proj_mat,
                                    cam_pos=cam_pos,
                                    out_tiles_touched=tiles_touched,
                                    out_rect_tile_space=rect_tile_space,
                                    out_radii=radii,
                                    out_xyz_vs=xyz_vs,
                                    out_inv_cov_vs=inv_cov_vs,
                                    out_rgb=rgb,
                                    fovy=fovy,
                                    fovx=fovx,
                                    image_height=renderer.image_height,
                                    image_width=renderer.image_width,
                                    grid_height=renderer.grid_height,
                                    grid_width=renderer.grid_width,
                                    tile_height=renderer.tile_height,
                                    tile_width=renderer.tile_width).launchRaw(
                                        blockSize=(256, 1, 1),gridSize=(math.ceil(n_points/256), 1, 1))

        ctx.save_for_backward(xyz_ws, sh_coeffs, rotations, scales,opcities, world_view_transform, proj_mat, cam_pos,
                                tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb)
        ctx.renderer = renderer
        ctx.fovy = fovy
        ctx.fovx = fovx
        ctx.active_sh = active_sh

        return tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_radii, grad_xyz_vs, grad_inv_cov_vs, grad_rgb):
        (xyz_ws, sh_coeffs, rotations, scales, opcities, world_view_transform, proj_mat, cam_pos,
        tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb) = ctx.saved_tensors
        fovy = ctx.fovy
        fovx = ctx.fovx
        active_sh = ctx.active_sh

        n_points = xyz_ws.shape[0]

        grad_xyz_ws = torch.zeros_like(xyz_ws)
        grad_rotations = torch.zeros_like(rotations)
        grad_scales = torch.zeros_like(scales)
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)

        ctx.renderer.vertex_shader.vertex_shader.bwd(xyz_ws=(xyz_ws, grad_xyz_ws),
                                                    rotations=(rotations, grad_rotations),
                                                    scales=(scales, grad_scales),
                                                    sh_coeffs=(sh_coeffs, grad_sh_coeffs),
                                                    opcities = opcities,
                                                    active_sh=active_sh,
                                                    world_view_transform=world_view_transform,
                                                    proj_mat=proj_mat,
                                                    cam_pos=cam_pos,
                                                    out_tiles_touched=tiles_touched,
                                                    out_rect_tile_space=rect_tile_space,
                                                    out_radii=radii,
                                                    out_xyz_vs=(xyz_vs, grad_xyz_vs),
                                                    out_inv_cov_vs=(inv_cov_vs, grad_inv_cov_vs),
                                                    out_rgb=(rgb, grad_rgb),
                                                    fovy=fovy,
                                                    fovx=fovx,
                                                    image_height=ctx.renderer.image_height,
                                                    image_width=ctx.renderer.image_width,
                                                    grid_height=ctx.renderer.grid_height,
                                                    grid_width=ctx.renderer.grid_width,
                                                    tile_height=ctx.renderer.tile_height,
                                                    tile_width=ctx.renderer.tile_width).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(math.ceil(n_points/256), 1, 1)
        )
        # backward的返回数目要与forward除ctx的输入数目一致
        return None, grad_xyz_ws, grad_sh_coeffs, grad_rotations, grad_scales, None, None, None, None, None, None, None

class FragmentShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, renderer:GSRenderer,sorted_gauss_idx, tile_ranges,xyz_vs, inv_cov_vs, opacity, rgb, device="cuda"):
        image_height = renderer.image_height
        image_width = renderer.image_width
        grid_height = renderer.grid_height
        grid_width = renderer.grid_width
        tile_height = renderer.tile_height
        tile_width = renderer.tile_width
        output_img = torch.zeros((image_height, image_width, 4), device=device)
        n_contributors = torch.zeros((image_height, image_width, 1),dtype=torch.int32, device=device)

        splat_kernel_with_args = renderer.fragment_shader.splat_tiled(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=xyz_vs, inv_cov_vs=inv_cov_vs, 
            opacity=opacity, rgb=rgb, 
            output_img=output_img,
            n_contributors=n_contributors,
            grid_height=grid_height,
            grid_width=grid_width,
            tile_height=tile_height,
            tile_width=tile_width
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(tile_width, 
                    tile_height, 1),
            gridSize=(grid_width, 
                    grid_height, 1)
        )

        ctx.save_for_backward(sorted_gauss_idx, tile_ranges,
                              xyz_vs, inv_cov_vs, opacity, rgb, 
                              output_img, n_contributors)
        ctx.renderer = renderer

        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_gauss_idx, tile_ranges, xyz_vs, inv_cov_vs, opacity, rgb, output_img, n_contributors) = ctx.saved_tensors
        renderer = ctx.renderer
        grid_height = renderer.grid_height
        grid_width = renderer.grid_width
        tile_height = renderer.tile_height
        tile_width = renderer.tile_width

        xyz_vs_grad = torch.zeros_like(xyz_vs)
        inv_cov_vs_grad = torch.zeros_like(inv_cov_vs)
        opacity_grad = torch.zeros_like(opacity)
        rgb_grad = torch.zeros_like(rgb)

        kernel_with_args = renderer.fragment_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=(xyz_vs, xyz_vs_grad),
            inv_cov_vs=(inv_cov_vs, inv_cov_vs_grad),
            opacity=(opacity, opacity_grad),
            rgb=(rgb, rgb_grad),
            output_img=(output_img, grad_output_img),
            n_contributors=n_contributors,
            grid_height=grid_height,
            grid_width=grid_width,
            tile_height=tile_height,
            tile_width=tile_width)
        
        kernel_with_args.launchRaw(
            blockSize=(tile_width, 
                    tile_height, 1),
            gridSize=(grid_width, 
                    grid_height, 1)
        )
        return None, None, None, xyz_vs_grad, inv_cov_vs_grad, opacity_grad, rgb_grad, None
