"""
这是一个使用 Falcor + DiffSlang + PyTorch 的简单示例，
实现 3D 高斯 splatting 的渲染与梯度计算。
与 2D 版本类似，但这里每个 blob 表示的是一个三维高斯：
    - pos: 3D 位置 (float3)
    - radius: 高斯半径 (float)
    - color: 颜色 (float3)
    - opacity: 不透明度 (float)
总共 8 个 float，每个 blob 占 32 字节。
Falcor compute pass 会将每个 blob 投影到图像平面，
计算其对像素的贡献。
"""

import falcor                    # 导入 Falcor 渲染库
from pathlib import Path         # 用于文件路径操作
import torch                     # 导入 PyTorch
import numpy as np               # 导入 numpy
from PIL import Image           # 用于图像处理

# 获取当前文件所在目录
DIR = Path(__file__).parent

# 目标图像路径（可参照 tiny_nerf_data.npz 中的目标图像）
TARGET_IMAGE = DIR / "./monalisa.jpg"

# 定义 3D 高斯 blob 数量、输出图像分辨率及优化迭代次数
BLOB_COUNT = 1024 * 8                   # 3D 高斯 blob 数量
RESOLUTION = 512                        # 输出图像分辨率（假设生成2D渲染结果）
ITERATIONS = 4000                       # 优化迭代次数

  
class Splat3D:
    """
    Splat3D 类封装了 Falcor 渲染管线中正向与反向（梯度）计算所需的缓冲区和 compute pass，
    实现 3D 高斯 splatting 渲染。每个 blob 表示一个三维高斯。
    """
    def __init__(self, device: falcor.Device):
        # 保存 Falcor 设备对象
        self.device = device

        # 创建用于存储 3D 高斯 blob 参数的结构化缓冲区（每个 blob 占 32 字节）
        self.blobs_buf = device.create_structured_buffer(
            struct_size=32,
            element_count=BLOB_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                      | falcor.ResourceBindFlags.UnorderedAccess
                      | falcor.ResourceBindFlags.Shared,
        )

        # 创建用于存储 3D 高斯 blob 梯度的缓冲区
        self.grad_blobs_buf = device.create_structured_buffer(
            struct_size=32,
            element_count=BLOB_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                      | falcor.ResourceBindFlags.UnorderedAccess
                      | falcor.ResourceBindFlags.Shared,
        )

        # 创建用于存储渲染图像的缓冲区，每个像素 12 字节（RGB 每个 float）
        self.image_buf = device.create_structured_buffer(
            struct_size=12,
            element_count=RESOLUTION * RESOLUTION,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                      | falcor.ResourceBindFlags.UnorderedAccess
                      | falcor.ResourceBindFlags.Shared,
        )

        # 创建用于存储渲染图像梯度的缓冲区
        self.grad_image_buf = device.create_structured_buffer(
            struct_size=12,
            element_count=RESOLUTION * RESOLUTION,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
                      | falcor.ResourceBindFlags.UnorderedAccess
                      | falcor.ResourceBindFlags.Shared,
        )

        # 创建正向计算的 compute pass，使用 Slang 着色器入口 "forward_main"
        self.forward_pass = falcor.ComputePass(
            device, file=DIR / "splat3d.cs.slang", cs_entry="forward_main"
        )

        # 创建反向传播 compute pass，使用入口 "backward_main"
        self.backward_pass = falcor.ComputePass(
            device, file=DIR / "splat3d.cs.slang", cs_entry="backward_main"
        )

    def forward(self, blobs, view_matrix, proj_matrix):
        """
        正向传播：利用 3D 高斯 blob 参数渲染出图像。
        参数：
            blobs: blob 参数张量 (形状: [BLOB_COUNT, 8])
            view_matrix: 摄像机视图矩阵 (4x4)
            proj_matrix: 投影矩阵 (4x4)
        返回：
            渲染后的图像 Tensor (形状: [RESOLUTION, RESOLUTION, 3])
        """
        self.blobs_buf.from_torch(blobs.detach())
        self.device.render_context.wait_for_cuda()
        # 设置 compute pass 全局变量
        vars = self.forward_pass.globals.forward
        vars.blobs = self.blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.image = self.image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        # 设置摄像机矩阵（注意：falcor 预期 float4x4）
        vars.view = view_matrix
        vars.proj = proj_matrix
        # 执行 compute pass，线程数与输出分辨率一致
        self.forward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        self.device.render_context.wait_for_falcor()
        return self.image_buf.to_torch([RESOLUTION, RESOLUTION, 3], falcor.float32)

    def backward(self, blobs, grad_intensities):
        """
        反向传播：根据渲染图像梯度计算 blob 参数梯度。
        参数：
            blobs: 当前 blob 参数（detach 后的张量）
            grad_intensities: 渲染图像梯度
        返回：
            blob 梯度 Tensor (形状: [BLOB_COUNT, 8])
        """
        self.grad_blobs_buf.from_torch(torch.zeros([BLOB_COUNT, 8]).cuda())
        self.blobs_buf.from_torch(blobs.detach())
        self.grad_image_buf.from_torch(grad_intensities.detach())
        self.device.render_context.wait_for_cuda()
        vars = self.backward_pass.globals.backward
        vars.blobs = self.blobs_buf
        vars.grad_blobs = self.grad_blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.grad_image = self.grad_image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        self.backward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        self.device.render_context.wait_for_falcor()
        return self.grad_blobs_buf.to_torch([BLOB_COUNT, 8], falcor.float32)


class Splat3DFunction(torch.autograd.Function):
    """
    自定义 autograd Function，前向调用 Splat3D.forward，
    反向调用 Splat3D.backward，通过 Falcor compute pass 完成渲染与梯度计算。
    """
    @staticmethod
    def forward(ctx, blobs, view_matrix, proj_matrix):
        image = splat3d.forward(blobs, view_matrix, proj_matrix)
        ctx.save_for_backward(blobs)
        return image

    @staticmethod
    def backward(ctx, grad_intensities):
        blobs = ctx.saved_tensors[0]
        grad_blobs = splat3d.backward(blobs, grad_intensities)
        return grad_blobs, None, None


class Splat3DModule(torch.nn.Module):
    """
    将 Splat3DFunction 封装为 PyTorch Module，
    方便在训练过程中调用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, blobs, view_matrix, proj_matrix):
        return Splat3DFunction.apply(blobs, view_matrix, proj_matrix)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(999)
    np.random.seed(666)

    # 数据加载（参照 tiny_nerf_data.npz）
    data = np.load('tiny_nerf_data.npz')
    images = data['images']        # (N, H, W, 3)
    poses = data['poses']          # (N, 4, 4)
    focal = data['focal'].item()   # 标量
    H, W = images.shape[1:3]
    n_train = 100
    test_img, test_pose = images[101], poses[101]
    images = images[:n_train]
    poses = poses[:n_train]

    # 创建窗口、纹理，并加载目标图像（此处仅用于调试渲染效果）
    testbed = falcor.Testbed(create_window=True, width=RESOLUTION, height=RESOLUTION)
    testbed.show_ui = False
    device = testbed.device

    testbed.render_texture = device.create_texture(
        format=falcor.ResourceFormat.RGB32Float,
        width=RESOLUTION,
        height=RESOLUTION,
        mip_levels=1,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )

    # 初始化 Splat3D 渲染模块
    splat3d = Splat3D(device)
    Splat3DFunction.splat3d = splat3d

    # 示例：初始化随机 3D 高斯 blob 参数
    # 每个 blob 包含: [pos_x, pos_y, pos_z, radius, color_r, color_g, color_b, opacity]
    blobs = torch.rand([BLOB_COUNT, 8]).cuda()
    # 例如，将 radius 调整到较小范围，opacity 映射至 [0,1]
    blobs[:, 3] = blobs[:, 3] * 0.05 + 0.005
    blobs[:, 4:7] = blobs[:, 4:7] * 0.5 + 0.25

    # 此处假设已有摄像机的 view 与 projection 矩阵，可根据 tiny_nerf_data.npz 得到 poses 与 focal 后构造
    # 这里仅生成单位矩阵作示例
    view_matrix = torch.eye(4, device=device).float()
    proj_matrix = torch.eye(4, device=device).float()

    # 执行一次前向渲染
    image = splat3d.forward(blobs, view_matrix, proj_matrix)
    # 将结果显示在窗口
    testbed.render_texture.from_numpy(image.cpu().numpy())
    testbed.frame()