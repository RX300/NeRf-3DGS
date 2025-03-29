import torch
import numpy as np
import pathlib
import falcor
from torch.autograd import Function
from pathlib import Path

class FalcorGaussianRenderer:
    """封装高斯点云渲染相关的Falcor操作"""
    
    def __init__(self,gaussian_num, width, height,grid_width, grid_height):
        # 创建包含窗口的 testbed 实例，用于显示渲染结果
        self.testbed = falcor.Testbed()
        self.testbed.show_ui = False
        self.device = self.testbed.device
       
        # 获取当前目录
        self.DIR = Path(__file__).parent
        self.init_generate_keys_pass(gaussian_num)
        self.init_compute_tile_ranges_pass(grid_width,grid_height)
        self.init_fragment_pass(gaussian_num, width, height)
        self.init_backward_fragment_render(gaussian_num, width, height)
        # # 加载计算通道
        # self.generate_keys_pass = falcor.ComputePass(self.device, file=DIR/"tiles.cs.slang", cs_entry="generate_keys_main")
        # self.compute_tile_ranges_pass = falcor.ComputePass(self.device, file=DIR/"tiles.cs.slang", cs_entry="compute_tile_ranges_main")
        # self.fragment_pass = falcor.ComputePass(self.device, file=DIR/"fragment.cs.slang", cs_entry="splat_tiled_main")
    
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
    def init_generate_keys_pass(self,gaussian_num):
        """初始化生成键的计算通道"""
        self.generate_keys_pass = falcor.ComputePass(self.device, file=self.DIR/"tiles.cs.slang", cs_entry="generate_keys_main")
        self.xyz_vs_buffer = self.create_buffer(12, gaussian_num)
        self.rect_tile_space_buffer = self.create_buffer(16, gaussian_num)
        self.index_buffer_offset_buffer = self.create_buffer(4, gaussian_num)
        # self.out_unsorted_gauss_idx_buffer = self.create_buffer(4, 100000)
        # self.out_unsorted_keys_buffer = self.create_buffer(8, 100000)
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        self.b_init = False
    def generate_keys(self, xyz_vs, rect_tile_space:np.ndarray, index_buffer_offset:np.ndarray, grid_height, grid_width):
        """执行生成键的操作"""
        # self.generate_keys_pass = falcor.ComputePass(self.device, file=self.DIR/"tiles.cs.slang", cs_entry="generate_keys_main")
        # 创建输出缓冲区
        self.total_size = index_buffer_offset[-1]
        # #如果没有self.out_unsorted_keys_buffer，才创建
        # if  hasattr(self, 'out_unsorted_keys_buffer'):
        #     self.out_unsorted_keys_buffer.release_cuda_memory()
        #     self.device.render_context.wait_for_cuda()
        #     del self.out_unsorted_keys_buffer
        #     self.out_unsorted_keys_buffer = self.create_buffer(8, self.total_size)
        # if not hasattr(self, 'out_unsorted_keys_buffer'):
        #     self.out_unsorted_keys_buffer = self.create_buffer(8, self.total_size)
        # if not hasattr(self, 'out_unsorted_gauss_idx_buffer'):
        #     self.out_unsorted_gauss_idx_buffer = self.create_buffer(4, self.total_size)

        self.out_unsorted_keys_buffer = self.create_buffer(8, self.total_size)
        self.out_unsorted_gauss_idx_buffer = self.create_buffer(4, self.total_size)

        # 设置计算通道参数
        # xyz_vs = xyz_vs.contiguous()
        xyz_vs_detached = xyz_vs.detach()
        # print(f"xyz_vs.shape: {xyz_vs.shape}")
        # print(type(xyz_vs))
        self.xyz_vs_buffer.from_torch(xyz_vs_detached)
        self.rect_tile_space_buffer.from_numpy(rect_tile_space)
        self.index_buffer_offset_buffer.from_numpy(index_buffer_offset)
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        # 设置参数并执行
        vars = self.generate_keys_pass.globals.tiles_paramter
        # vars = self.generate_keys_pass.root_var.tiles_paramter
        vars.xyz_vs = self.xyz_vs_buffer
        vars.rect_tile_space = self.rect_tile_space_buffer
        vars.index_buffer_offset = self.index_buffer_offset_buffer
        vars.out_unsorted_keys = self.out_unsorted_keys_buffer
        vars.out_unsorted_gauss_idx = self.out_unsorted_gauss_idx_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        # 执行计算通道
        self.generate_keys_pass.execute(threads_x=xyz_vs.shape[0])
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()

        # 获取结果
        out_unsorted_keys = self.out_unsorted_keys_buffer.to_torch([self.total_size], falcor.int64)
        out_unsorted_gauss_idx = self.out_unsorted_gauss_idx_buffer.to_torch([self.total_size], falcor.int32)
        self.out_unsorted_keys_buffer.release_cuda_memoryV5()
        print("test3")
        self.device.render_context.wait_for_cuda()
        return out_unsorted_keys, out_unsorted_gauss_idx
    def init_compute_tile_ranges_pass(self,grid_width, grid_height):
        """初始化计算瓦片范围的计算通道"""
        self.compute_tile_ranges_pass = falcor.ComputePass(self.device, file=self.DIR/"tiles.cs.slang", cs_entry="compute_tile_ranges_main")
        # 创建输出缓冲区
        self.out_tile_ranges_buffer = self.create_buffer(8, grid_width*grid_height)
    def compute_tile_ranges(self, sorted_keys, grid_height, grid_width):
        """计算瓦片范围"""
        if not hasattr(self, 'sorted_keys_buffer'):
            # 设置计算通道参数
            self.sorted_keys_buffer = self.create_buffer(8, sorted_keys.shape[0], torch_tensor=sorted_keys)
        
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.compute_tile_ranges_pass.globals.tiles_paramter
        vars.sorted_keys = self.sorted_keys_buffer
        vars.out_tile_ranges = self.out_tile_ranges_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        
        # 执行计算通道
        self.compute_tile_ranges_pass.execute(threads_x=sorted_keys.shape[0])
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 获取结果
        out_tile_ranges = self.out_tile_ranges_buffer.to_torch([grid_width*grid_height, 2], falcor.int32)
        print(f"out_tile_ranges.mean(): {out_tile_ranges[:50,:].float().mean(dim=0)}")
        self.out_tile_rangesmean = out_tile_ranges[:50,:].float().mean(dim=0)
        return out_tile_ranges, self.out_tile_ranges_buffer
    def init_fragment_pass(self,gaussian_num, width, height):
        """初始化片段渲染的计算通道"""
        self.fragment_pass = falcor.ComputePass(self.device, file=self.DIR/"fragment.cs.slang", cs_entry="splat_tiled_main")
        self.inv_cov_vs_buffer = self.create_buffer(16, gaussian_num)
        self.opacity_buffer = self.create_buffer(4, gaussian_num)
        self.gaussain_rgb_buffer = self.create_buffer(12, gaussian_num)
        self.output_img_texture = self.create_texture(width, height, data=np.zeros((height, width, 4), dtype=np.float32))
        self.n_contributors_texture = self.create_texture(width, height, format=falcor.ResourceFormat.R32Uint, data=np.zeros((height, width), dtype=np.uint32))
    def fragment_render(self, sorted_gauss_idx, tile_ranges, xyz_vs, inv_cov_vs, opacity, gaussain_rgb, width, height, grid_height, grid_width):
        """执行片段渲染"""
        # 更新缓冲区
        if not hasattr(self, 'sorted_gauss_idx_buffer'):
            self.sorted_gauss_idx_buffer = self.create_buffer(4, sorted_gauss_idx.shape[0], torch_tensor=sorted_gauss_idx)
        xyz_vs_detached = xyz_vs.detach()
        inv_cov_vs_detached = inv_cov_vs.detach()
        opacity_detached = opacity.detach()
        gaussain_rgb_detached = gaussain_rgb.detach()
        self.xyz_vs_buffer.from_torch(xyz_vs_detached)
        self.inv_cov_vs_buffer.from_torch(inv_cov_vs_detached)
        self.opacity_buffer.from_torch(opacity_detached)
        self.gaussain_rgb_buffer.from_torch(gaussain_rgb_detached)
        
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.fragment_pass.globals.fragment_parameter
        vars.sorted_gauss_idx = self.sorted_gauss_idx_buffer
        vars.tile_ranges = tile_ranges
        vars.xyz_vs = self.xyz_vs_buffer
        vars.inv_cov_vs = self.inv_cov_vs_buffer
        vars.opacity = self.opacity_buffer
        vars.gaussian_rgb = self.gaussain_rgb_buffer
        vars.output_img = self.output_img_texture
        vars.n_contributors = self.n_contributors_texture
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        vars.tile_height = 16
        vars.tile_width = 16
        
        # 执行计算通道
        self.fragment_pass.execute(threads_x=width, threads_y=height)
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 获取结果
        n_contributors_numpy = self.n_contributors_texture.to_numpy()
        out_img = self.output_img_texture.to_numpy()
        # 转换为PyTorch张量
        out_img_torch = torch.from_numpy(out_img).to("cuda")
        print(n_contributors_numpy.dtype)
        n_contributors_numpy = n_contributors_numpy.astype(np.uint32)
        n_contributors_torch = torch.from_numpy(n_contributors_numpy).to("cuda")
        return out_img_torch,out_img,n_contributors_torch

    def init_backward_fragment_render(self,gaussian_num,width,height):
        self.backward_fragment_render_pass = falcor.ComputePass(self.device, file=self.DIR/"fragment.cs.slang", cs_entry="splat_tiled_backward_main")
        xyz_vs_stride = 12  # float3 (4字节 * 3)
        inv_cov_vs_stride = 16  # float4 (4字节 * 4)
        opacity_stride = 4  # float (4字节)
        gaussain_rgb_stride = 12  # float3 (4字节 * 3)
        self.grad_xyz_vs_buffer = self.create_buffer(xyz_vs_stride, gaussian_num,torch_tensor=torch.zeros([gaussian_num, 3], dtype=torch.float32, device="cuda"))
        self.grad_inv_cov_vs_buffer = self.create_buffer(inv_cov_vs_stride, gaussian_num,torch_tensor=torch.zeros([gaussian_num, 4], dtype=torch.float32, device="cuda"))
        self.grad_opacity_buffer = self.create_buffer(opacity_stride, gaussian_num,torch_tensor=torch.zeros([gaussian_num], dtype=torch.float32, device="cuda"))
        self.grad_rgb_buffer = self.create_buffer(gaussain_rgb_stride, gaussian_num,torch_tensor=torch.zeros([gaussian_num, 3], dtype=torch.float32, device="cuda"))
        self.grad_output_texture = self.create_texture(width, height)
    def backward_fragment_render(self, sorted_gauss_idx, tile_ranges,rendered_image_texture,n_contributors_torch, xyz_vs, inv_cov_vs, opacity, rgb, 
                                grad_output, width, height, grid_height, grid_width):

        # 创建梯度输入纹理
        self.grad_output_texture.from_numpy(grad_output.cpu().numpy())
        # 创建字节地址缓冲区
        self.grad_xyz_vs_buffer.from_torch(torch.zeros([xyz_vs.shape[0], 3], device="cuda"))
        self.grad_inv_cov_vs_buffer.from_torch(torch.zeros([xyz_vs.shape[0], 4], device="cuda"))
        self.grad_opacity_buffer.from_torch(torch.zeros([xyz_vs.shape[0]], device="cuda"))
        self.grad_rgb_buffer.from_torch(torch.zeros([xyz_vs.shape[0], 3], device="cuda"))
        self.grad_test_buffer.from_torch(torch.zeros([1], device="cuda"))
        # 等待CUDA处理结束
        self.device.render_context.wait_for_cuda()
        
        # 设置参数并执行
        vars = self.backward_fragment_render_pass.globals.fragment_parameter
        vars.sorted_gauss_idx = self.sorted_gauss_idx_buffer
        vars.tile_ranges = self.out_tile_ranges_buffer
        vars.xyz_vs = self.xyz_vs_buffer
        vars.inv_cov_vs = self.inv_cov_vs_buffer
        vars.opacity = self.opacity_buffer
        vars.gaussian_rgb = self.gaussain_rgb_buffer
        vars.output_img = self.output_img_texture
        vars.n_contributors = self.n_contributors_texture
        vars.grad_output = self.grad_output_texture
        vars.grad_xyz_vs = self.grad_xyz_vs_buffer
        vars.grad_inv_cov_vs = self.grad_inv_cov_vs_buffer
        vars.grad_opacity = self.grad_opacity_buffer
        vars.grad_rgb = self.grad_rgb_buffer
        vars.grid_height = grid_height
        vars.grid_width = grid_width
        vars.tile_height = 16
        vars.tile_width = 16
        
        # 执行计算通道
        self.backward_fragment_render_pass.execute(threads_x=width, threads_y=height)
        
        # 等待Falcor处理结束
        self.device.render_context.wait_for_falcor()
        
        # 转换为Torch张量
        N = xyz_vs.shape[0]
        grad_xyz_vs = self.grad_xyz_vs_buffer.to_torch([N, 3], falcor.float32)
        grad_inv_cov_vs = self.grad_inv_cov_vs_buffer.to_torch([N, 4], falcor.float32)
        grad_opacity = self.grad_opacity_buffer.to_torch([N], falcor.float32)
        grad_rgb = self.grad_rgb_buffer.to_torch([N, 3], falcor.float32)
        # tile_ranges = self.tile_ranges_buffer.to_torch([grid_width*grid_height, 2], falcor.int32)
        # print(f"tile_ranges.mean(): {tile_ranges[:50,:].float().mean(dim=0)}")
        # print(f"grad_rgb.mean: {grad_rgb.mean(dim=0)}")
        # grad_test = grad_test_buffer.to_torch([1], falcor.float32)
        # print(f"grad_test: {grad_test}")
        # self.device.render_context.wait_for_cuda()
        # exit()
        return grad_xyz_vs, grad_inv_cov_vs, grad_opacity, grad_rgb


class GaussianRenderFunction(torch.autograd.Function):
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
        rendered_image_torch,rendered_image_texture,n_contributors_torch = renderer.fragment_render(
            sorted_gauss_idx, tile_ranges_buffer, xyz_vs, inv_cov_vs, opacity, rgb, 
            width, height, grid_height, grid_width
        )
        
        # 保存上下文信息用于反向传播
        ctx.save_for_backward(xyz_vs, inv_cov_vs, opacity, rgb, sorted_gauss_idx, out_tile_ranges,n_contributors_torch)
        ctx.renderer = renderer
        ctx.width = width
        ctx.height = height
        ctx.grid_width = grid_width
        ctx.grid_height = grid_height
        ctx.rendered_image_texture = rendered_image_texture
        torch.cuda.empty_cache()
        renderer.device.end_frame()
        return rendered_image_torch
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算参数的梯度"""
        # 获取保存的张量
        xyz_vs, inv_cov_vs, opacity, rgb, sorted_gauss_idx, out_tile_ranges,n_contributors_torch = ctx.saved_tensors
        print(f"backward:out_tile_ranges.mean(): {out_tile_ranges[:50,:].float().mean(dim=0)}")
        # 获取渲染器和其他参数
        renderer = ctx.renderer
        width = ctx.width
        height = ctx.height
        grid_width = ctx.grid_width
        grid_height = ctx.grid_height
        rendered_image_texture = ctx.rendered_image_texture
        print(f"grad_output.shape: {grad_output.shape}")
        print(f"grad_output.mean: {grad_output.reshape(-1,4).mean(dim=0)}")
        # 计算梯度
        grad_xyz_vs, grad_inv_cov_vs, grad_opacity, grad_rgb = renderer.backward_fragment_render(
            sorted_gauss_idx, out_tile_ranges,rendered_image_texture,n_contributors_torch, xyz_vs, inv_cov_vs, opacity, rgb,
            grad_output, width, height, grid_height, grid_width
        )
        # 打印梯度
        print(f"grad_xyz_vs.mean: {grad_xyz_vs.mean(dim=0)}")
        print(f"grad_inv_cov_vs.mean: {grad_inv_cov_vs.mean(dim=0)}")
        print(f"grad_opacity.mean: {grad_opacity.mean()}")
        print(f"grad_rgb.mean: {grad_rgb.mean(dim=0)}")
        torch.cuda.empty_cache()
        renderer.device.end_frame()
        #exit()
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