import torch
import numpy as np
import pathlib
import falcor
from torch.autograd import Function
from pathlib import Path

class FalcorGaussianRenderer:
    """封装高斯点云渲染相关的Falcor操作"""
    
    def __init__(self):
        # 创建包含窗口的 testbed 实例，用于显示渲染结果
        self.testbed = falcor.Testbed()
        self.testbed.show_ui = False
        self.device = self.testbed.device
        
        # 获取当前目录
        DIR = Path(__file__).parent
        
        # 加载计算通道
        self.generate_keys_pass = falcor.ComputePass(self.device, file=DIR/"tiles.cs.slang", cs_entry="generate_keys_main")
        self.compute_tile_ranges_pass = falcor.ComputePass(self.device, file=DIR/"tiles.cs.slang", cs_entry="compute_tile_ranges_main")
        self.fragment_pass = falcor.ComputePass(self.device, file=DIR/"fragment.cs.slang", cs_entry="splat_tiled_main")
    
    def create_buffer(self, struct_size, element_count, data=None, torch_tensor=None, numpy_array=None):
        """创建Falcor缓冲区并可选初始化数据"""
        buffer = self.device.create_structured_buffer(
            struct_size=struct_size,
            element_count=element_count,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                    | falcor.ResourceBindFlags.UnorderedAccess
                    | falcor.ResourceBindFlags.Shared,
        )
        
        if torch_tensor is not None:
            buffer.from_torch(torch_tensor.detach())
        elif numpy_array is not None:
            buffer.from_numpy(numpy_array)
        elif data is not None:
            if isinstance(data, torch.Tensor):
                buffer.from_torch(data.detach())
            elif isinstance(data, np.ndarray):
                buffer.from_numpy(data)
        
        return buffer
    
    def create_byte_buffer(self, size_in_bytes, data=None):
        """创建字节地址缓冲区并可选初始化数据"""
        buffer = self.device.create_structured_buffer(
            size=size_in_bytes,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                    | falcor.ResourceBindFlags.UnorderedAccess
                    | falcor.ResourceBindFlags.Shared,
        )
        
        if data is not None:
            buffer.from_numpy(data)
        else:
            # 初始化为全零
            zeros = np.zeros(size_in_bytes, dtype=np.byte)
            buffer.from_numpy(zeros)
        
        return buffer
    
    def create_texture(self, width, height, format=falcor.ResourceFormat.RGBA32Float, data=None):
        """创建Falcor纹理并可选初始化数据"""
        texture = self.device.create_texture(
            width=width,
            height=height,
            format=format,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                    | falcor.ResourceBindFlags.UnorderedAccess
                    | falcor.ResourceBindFlags.Shared,
        )
        
        if data is not None:
            texture.from_numpy(data)
        
        return texture
    
    def generate_keys(self, xyz_vs, rect_tile_space, index_buffer_offset, grid_height, grid_width):
        """执行生成键的操作"""
        # 创建输出缓冲区
        total_size = index_buffer_offset[-1]
        out_unsorted_keys_buffer = self.create_buffer(8, total_size)
        out_unsorted_gauss_idx_buffer = self.create_buffer(4, total_size)
        
        # 设置计算通道参数
        xyz_vs_buffer = self.create_buffer(12, xyz_vs.shape[0], torch_tensor=xyz_vs)
        rect_tile_space_buffer = self.create_buffer(16, rect_tile_space.shape[0], numpy_array=rect_tile_space)
        index_buffer_offset_buffer = self.create_buffer(4, index_buffer_offset.shape[0], numpy_array=index_buffer_offset)
        
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.generate_keys_pass.globals.tiles_paramter
        vars.xyz_vs = xyz_vs_buffer
        vars.rect_tile_space = rect_tile_space_buffer
        vars.index_buffer_offset = index_buffer_offset_buffer
        vars.out_unsorted_keys = out_unsorted_keys_buffer
        vars.out_unsorted_gauss_idx = out_unsorted_gauss_idx_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        
        # 执行计算通道
        self.generate_keys_pass.execute(threads_x=xyz_vs.shape[0])
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 获取结果
        out_unsorted_keys = out_unsorted_keys_buffer.to_torch([total_size], falcor.int64)
        out_unsorted_gauss_idx = out_unsorted_gauss_idx_buffer.to_torch([total_size], falcor.int32)
        
        return out_unsorted_keys, out_unsorted_gauss_idx
    
    def compute_tile_ranges(self, sorted_keys, grid_height, grid_width):
        """计算瓦片范围"""
        # 创建输出缓冲区
        tile_ranges = torch.zeros((grid_width*grid_height, 2), device="cuda", dtype=torch.int32)
        out_tile_ranges_buffer = self.create_buffer(8, tile_ranges.shape[0])
        
        # 设置计算通道参数
        sorted_keys_buffer = self.create_buffer(8, sorted_keys.shape[0], torch_tensor=sorted_keys)
        
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.compute_tile_ranges_pass.globals.tiles_paramter
        vars.sorted_keys = sorted_keys_buffer
        vars.out_tile_ranges = out_tile_ranges_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        
        # 执行计算通道
        self.compute_tile_ranges_pass.execute(threads_x=sorted_keys.shape[0])
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 获取结果
        out_tile_ranges = out_tile_ranges_buffer.to_torch([grid_width*grid_height, 2], falcor.int32)
        print(f"out_tile_ranges.mean(): {out_tile_ranges[:50,:].float().mean(dim=0)}")
        self.out_tile_rangesmean = out_tile_ranges[:50,:].float().mean(dim=0)
        return out_tile_ranges, out_tile_ranges_buffer
    
    def fragment_render(self, sorted_gauss_idx, tile_ranges, xyz_vs, inv_cov_vs, opacity, rgb, width, height, grid_height, grid_width):
        """执行片段渲染"""
        # 创建缓冲区
        sorted_gauss_idx_buffer = self.create_buffer(4, sorted_gauss_idx.shape[0], torch_tensor=sorted_gauss_idx)
        xyz_vs_buffer = self.create_buffer(12, xyz_vs.shape[0], torch_tensor=xyz_vs)
        inv_cov_vs_buffer = self.create_buffer(16, inv_cov_vs.shape[0], torch_tensor=inv_cov_vs)
        opacity_buffer = self.create_buffer(4, opacity.shape[0], torch_tensor=opacity)
        rgb_buffer = self.create_buffer(12, rgb.shape[0], torch_tensor=rgb)
        # 创建输出纹理
        out_image_numpy = np.zeros((height, width, 4), dtype=np.float32)
        output_img_texture = self.create_texture(width, height, data=out_image_numpy)
        
        n_contributors_numpy = np.zeros((height, width), dtype=np.uint32)
        n_contributors_texture = self.create_texture(
            width, height, 
            format=falcor.ResourceFormat.R32Uint, 
            data=n_contributors_numpy
        )
        
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.fragment_pass.globals.fragment_parameter
        vars.sorted_gauss_idx = sorted_gauss_idx_buffer
        vars.tile_ranges = tile_ranges
        vars.xyz_vs = xyz_vs_buffer
        vars.inv_cov_vs = inv_cov_vs_buffer
        vars.opacity = opacity_buffer
        vars.gaussian_rgb = rgb_buffer
        vars.output_img = output_img_texture
        vars.n_contributors = n_contributors_texture
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        vars.tile_height = 16
        vars.tile_width = 16
        
        # 执行计算通道
        self.fragment_pass.execute(threads_x=width, threads_y=height)
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 获取结果
        out_img = output_img_texture.to_numpy()
        # 转换为PyTorch张量
        out_img_torch = torch.from_numpy(out_img).to("cuda")
        
        return out_img_torch

    def backward_fragment_render(self, sorted_gauss_idx, tile_ranges, xyz_vs, inv_cov_vs, opacity, rgb, 
                                grad_output, width, height, grid_height, grid_width):
        """执行反向传播计算梯度"""
        # 创建输入缓冲区
        sorted_gauss_idx_buffer = self.create_buffer(4, sorted_gauss_idx.shape[0], torch_tensor=sorted_gauss_idx)
        xyz_vs_buffer = self.create_buffer(12, xyz_vs.shape[0], torch_tensor=xyz_vs)
        inv_cov_vs_buffer = self.create_buffer(16, inv_cov_vs.shape[0], torch_tensor=inv_cov_vs)
        opacity_buffer = self.create_buffer(4, opacity.shape[0], torch_tensor=opacity)
        rgb_buffer = self.create_buffer(12, rgb.shape[0], torch_tensor=rgb)
        tile_ranges_buffer = self.create_buffer(8, tile_ranges.shape[0], torch_tensor=tile_ranges)
        # 创建输出图像纹理
        output_img_numpy = np.zeros((height, width, 4), dtype=np.float32)
        output_img_texture = self.create_texture(width, height, data=output_img_numpy)
        
        # 创建n_contributors纹理
        n_contributors_numpy = np.zeros((height, width), dtype=np.uint32)
        n_contributors_texture = self.create_texture(
            width, height, 
            format=falcor.ResourceFormat.R32Uint, 
            data=n_contributors_numpy
        )
        
        # 创建梯度输入纹理
        grad_output_texture = self.create_texture(width, height, data=grad_output.cpu().numpy())
        
        # 计算每个元素的字节大小
        N = xyz_vs.shape[0]
        xyz_vs_stride = 12  # float3 (4字节 * 3)
        inv_cov_vs_stride = 16  # float4 (4字节 * 4)
        opacity_stride = 4  # float (4字节)
        rgb_stride = 12  # float3 (4字节 * 3)
        
        # 创建字节地址缓冲区
        grad_xyz_vs_buffer = self.create_buffer(xyz_vs_stride, N)
        grad_inv_cov_vs_buffer = self.create_buffer(inv_cov_vs_stride, N)
        grad_opacity_buffer = self.create_buffer(opacity_stride, N)
        grad_rgb_buffer = self.create_buffer(rgb_stride, N)
        grad_xyz_vs_buffer.from_torch(torch.zeros([N, 3], device="cuda"))
        grad_inv_cov_vs_buffer.from_torch(torch.zeros([N, 4], device="cuda"))
        grad_opacity_buffer.from_torch(torch.zeros([N], device="cuda"))
        grad_rgb_buffer.from_torch(torch.zeros([N, 3], device="cuda"))
        grad_test_buffer = self.create_buffer(4, 1)
        grad_test_buffer.from_torch(torch.tensor([1], dtype=torch.float32,device="cuda"))
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 加载反向传播计算通道
        if not hasattr(self, 'fragment_backward_pass'):
            self.fragment_backward_pass = falcor.ComputePass(
                self.device, 
                file=Path(__file__).parent/"fragment.cs.slang", 
                cs_entry="splat_tiled_backward_main"
            )
        
        # 设置参数并执行
        vars = self.fragment_backward_pass.globals.fragment_parameter
        vars.sorted_gauss_idx = sorted_gauss_idx_buffer
        vars.tile_ranges = tile_ranges_buffer
        vars.xyz_vs = xyz_vs_buffer
        vars.inv_cov_vs = inv_cov_vs_buffer
        vars.opacity = opacity_buffer
        vars.gaussian_rgb = rgb_buffer
        vars.output_img = output_img_texture
        vars.n_contributors = n_contributors_texture
        vars.grad_output = grad_output_texture
        vars.grad_xyz_vs = grad_xyz_vs_buffer
        vars.grad_inv_cov_vs = grad_inv_cov_vs_buffer
        vars.grad_opacity = grad_opacity_buffer
        vars.grad_rgb = grad_rgb_buffer
        vars.grad_out_test = grad_test_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        vars.tile_height = 16
        vars.tile_width = 16
        tile_ranges_buffer = self.create_buffer(8, tile_ranges.shape[0], torch_tensor=tile_ranges)
        vars.tile_ranges_out = tile_ranges_buffer
        tile_ranges_out_tensor = tile_ranges_buffer.to_torch([grid_width*grid_height, 2], falcor.int32)
        print(f"tile_ranges_out_tensor.mean(): {tile_ranges_out_tensor[:50,:].float().mean(dim=0)}")
        # # 设置字节步长
        # vars.grad_xyz_vs_stride = xyz_vs_stride
        # vars.grad_inv_cov_vs_stride = inv_cov_vs_stride
        # vars.grad_opacity_stride = opacity_stride
        # vars.grad_rgb_stride = rgb_stride
        
        # 执行计算通道
        self.fragment_backward_pass.execute(threads_x=width, threads_y=height)
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 转换为Torch张量
        grad_xyz_vs = grad_xyz_vs_buffer.to_torch([N, 3], falcor.float32)
        grad_inv_cov_vs = grad_inv_cov_vs_buffer.to_torch([N, 4], falcor.float32)
        grad_opacity = grad_opacity_buffer.to_torch([N], falcor.float32)
        grad_rgb = grad_rgb_buffer.to_torch([N, 3], falcor.float32)
        tile_ranges = tile_ranges_buffer.to_torch([grid_width*grid_height, 2], falcor.int32)
        print(f"tile_ranges.mean(): {tile_ranges[:50,:].float().mean(dim=0)}")
        print(f"grad_rgb.mean: {grad_rgb.mean(dim=0)}")
        grad_test = grad_test_buffer.to_torch([1], falcor.float32)
        print(f"grad_test: {grad_test}")
        self.device.render_context.wait_for_cuda()
        exit()
        return grad_xyz_vs, grad_inv_cov_vs, grad_opacity, grad_rgb


class GaussianRenderFunction(Function):
    """可微分的高斯点云渲染函数"""
    
    @staticmethod
    def forward(ctx, renderer:FalcorGaussianRenderer, xyz_vs, inv_cov_vs, opacity, rgb, rect_tile_space, index_buffer_offset, width, height):
        """前向传播：执行高斯点云渲染"""
        # 获取网格尺寸
        grid_height = (height + 15) // 16
        grid_width = (width + 15) // 16
        
        # 执行生成键的操作
        out_unsorted_keys, out_unsorted_gauss_idx = renderer.generate_keys(
            xyz_vs, rect_tile_space, index_buffer_offset, grid_height, grid_width
        )
        
        # 排序键和索引
        sorted_keys, sorted_indices = torch.sort(out_unsorted_keys)
        sorted_gauss_idx = out_unsorted_gauss_idx[sorted_indices]
        
        # 计算瓦片范围
        out_tile_ranges, tile_ranges_buffer = renderer.compute_tile_ranges(sorted_keys, grid_height, grid_width)
        
        # 执行片段渲染
        rendered_image = renderer.fragment_render(
            sorted_gauss_idx, tile_ranges_buffer, xyz_vs, inv_cov_vs, opacity, rgb, 
            width, height, grid_height, grid_width
        )
        
        # 保存上下文信息用于反向传播
        ctx.save_for_backward(xyz_vs, inv_cov_vs, opacity, rgb, sorted_gauss_idx, out_tile_ranges)
        ctx.renderer = renderer
        ctx.width = width
        ctx.height = height
        ctx.grid_width = grid_width
        ctx.grid_height = grid_height
        
        return rendered_image
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算参数的梯度"""
        # 获取保存的张量
        xyz_vs, inv_cov_vs, opacity, rgb, sorted_gauss_idx, out_tile_ranges = ctx.saved_tensors
        print(f"backward:out_tile_ranges.mean(): {out_tile_ranges[:50,:].float().mean(dim=0)}")
        # 获取渲染器和其他参数
        renderer = ctx.renderer
        width = ctx.width
        height = ctx.height
        grid_width = ctx.grid_width
        grid_height = ctx.grid_height
        
        # 计算梯度
        grad_xyz_vs, grad_inv_cov_vs, grad_opacity, grad_rgb = renderer.backward_fragment_render(
            sorted_gauss_idx, out_tile_ranges, xyz_vs, inv_cov_vs, opacity, rgb,
            grad_output, width, height, grid_height, grid_width
        )
        # 打印梯度
        print(f"grad_xyz_vs.mean: {grad_xyz_vs.mean(dim=0)}")
        print(f"grad_inv_cov_vs.mean: {grad_inv_cov_vs.mean(dim=0)}")
        print(f"grad_opacity.mean: {grad_opacity.mean()}")
        print(f"grad_rgb.mean: {grad_rgb.mean(dim=0)}")
        # 返回与forward参数数量一致的梯度
        return (None,                # renderer
                grad_xyz_vs,         # xyz_vs
                grad_inv_cov_vs,     # inv_cov_vs
                grad_opacity,        # opacity
                grad_rgb,            # rgb
                None,                # rect_tile_space
                None,                # index_buffer_offset
                None,                # width
                None)                # height


def render_gaussian_image(renderer, xyz_vs, inv_cov_vs, opacity, rgb, rect_tile_space, index_buffer_offset, width, height):
    """渲染高斯点云的便捷函数"""
    return GaussianRenderFunction.apply(
        renderer, xyz_vs, inv_cov_vs, opacity, rgb, rect_tile_space, index_buffer_offset, width, height
    )