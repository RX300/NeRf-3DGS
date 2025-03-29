"""
这是一个使用 Falcor + DiffSlang + PyTorch 的简单示例。
它通过学习一组（累加式）二维高斯分布来表示一张图像。
- Falcor compute pass 用于将高斯分布渲染到图像上（正向传播）。
- DiffSlang 用于计算高斯关于图像的梯度（反向传播）。
- PyTorch 用于优化高斯使其匹配目标图像。
"""

import falcor                                # 导入 Falcor 渲染库
from pathlib import Path                     # 用于文件路径操作
import torch                                 # 导入 PyTorch
import numpy as np                           # 导入 numpy
from PIL import Image                        # 导入 PIL 用于图像处理

# 获取当前文件的目录
DIR = Path(__file__).parent

# 目标图像文件路径，这里使用 monalisa.jpg
TARGET_IMAGE = DIR / "./monalisa.jpg"

# 定义高斯 blob 数量、输出图像分辨率及优化迭代次数
BLOB_COUNT = 1024 * 40                      # 高斯 blob 的数量
RESOLUTION = 1024                          # 输出图像的分辨率
ITERATIONS = 4000                          # 优化迭代次数

  
class Splat2D:
    """
    Splat2D 类封装了 Falcor 渲染管线中正向和反向（梯度）计算所需的缓冲区及 compute pass。
    """
    def __init__(self, device: falcor.Device):
        # 保存 Falcor 设备对象
        self.device = device

        # 创建用于存储高斯 blob 参数的结构化缓冲区
        self.blobs_buf = device.create_structured_buffer(
            struct_size=32,                # 每个 blob 占用 32 字节
            element_count=BLOB_COUNT,        # blob 数量
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        # 创建用于存储高斯 blob 梯度的缓冲区
        self.grad_blobs_buf = device.create_structured_buffer(
            struct_size=32,
            element_count=BLOB_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )

        # 创建用于存储渲染图像的缓冲区，每个像素 12 字节（RGB，每个为 float）
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

        # 创建正向计算的 compute pass，使用指定的 Slang 着色器入口 "forward_main"
        self.forward_pass = falcor.ComputePass(
            device, file=DIR / "splat2d.cs.slang", cs_entry="forward_main"
        )

        # 创建反向传播（梯度计算）的 compute pass，入口为 "backward_main"
        self.backward_pass = falcor.ComputePass(
            device, file=DIR / "splat2d.cs.slang", cs_entry="backward_main"
        )

    def forward(self, blobs):
        """
        正向传播：用当前 highgauss 参数渲染出图像。
        参数 blobs 为 blob 参数，调用 Falcor 渲染管线，返回渲染后的图像 Tensor。
        """
        # 将 blobs 数据从 PyTorch 传入 Falcor 的缓冲区（detach 防止梯度传播）
        self.blobs_buf = device.create_structured_buffer(
            struct_size=32,                # 每个 blob 占用 32 字节
            element_count=BLOB_COUNT,        # blob 数量
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )
        self.blobs_buf.from_torch(blobs.detach())
        # 等待 CUDA 处理结束
        self.device.render_context.wait_for_cuda()
        # 设置 compute pass 全局变量
        vars = self.forward_pass.globals.forward
        vars.blobs = self.blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.image = self.image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        # 执行 compute pass，启动的线程数与输出分辨率相同
        self.forward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        # 等待 Falcor 处理结束
        self.device.render_context.wait_for_falcor()
        # 将渲染结果从 Falcor 缓冲区转为 PyTorch Tensor (形状为 [RESOLUTION, RESOLUTION, 3])
        return self.image_buf.to_torch([RESOLUTION, RESOLUTION, 3], falcor.float32)

    def backward(self, blobs, grad_intensities):
        """
        反向传播：根据渲染图像的梯度 grad_intensities 计算每个 blob 参数的梯度。
        """
        # 初始化梯度缓冲区为全 0
        self.grad_blobs_buf.from_torch(torch.zeros([BLOB_COUNT, 8]).cuda())
        # 将当前 blobs 参数传给缓冲区
        self.blobs_buf.from_torch(blobs.detach())
        # 将渲染图像梯度传入对应缓冲区
        self.grad_image_buf.from_torch(grad_intensities.detach())
        self.device.render_context.wait_for_cuda()
        # 设置反向传播 compute pass 的全局变量
        vars = self.backward_pass.globals.backward
        vars.blobs = self.blobs_buf
        vars.grad_blobs = self.grad_blobs_buf
        vars.blob_count = BLOB_COUNT
        vars.grad_image = self.grad_image_buf
        vars.resolution = falcor.uint2(RESOLUTION, RESOLUTION)
        # 执行反向传播的 compute pass
        self.backward_pass.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)
        self.device.render_context.wait_for_falcor()
        # 将计算得到的 blob 梯度转为 PyTorch Tensor (形状为 [BLOB_COUNT, 8])
        return self.grad_blobs_buf.to_torch([BLOB_COUNT, 8], falcor.float32)


class Splat2DFunction(torch.autograd.Function):
    """
    自定义 autograd Function，把前向和反向传播都定义在这里，
    利用 Falcor compute pass 实现渲染和梯度计算。
    """
    @staticmethod
    def forward(ctx, blobs):
        # 调用正向传播函数，生成图像
        image = splat2d.forward(blobs)
        # 保存 blobs 用于反向传播
        ctx.save_for_backward(blobs)
        return image

    @staticmethod
    def backward(ctx, grad_intensities):
        # 从上下文中恢复 blobs 参数
        blobs = ctx.saved_tensors[0]
        # 调用反向传播函数，计算 blobs 的梯度
        grad_blobs = splat2d.backward(blobs, grad_intensities)
        return grad_blobs


class Splat2DModule(torch.nn.Module):
    """
    封装 Splat2DFunction 到一个 Module 中，方便在 PyTorch 中调用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, blobs):
        return Splat2DFunction.apply(blobs)


# 创建包含窗口的 testbed 实例，用于显示渲染结果
testbed = falcor.Testbed(create_window=True, width=RESOLUTION, height=RESOLUTION)
testbed.show_ui = False
device = testbed.device

# 为显示结果创建一个纹理，格式为 RGB32Float
testbed.render_texture = device.create_texture(
    format=falcor.ResourceFormat.RGB32Float,
    width=RESOLUTION,
    height=RESOLUTION,
    mip_levels=1,
    bind_flags=falcor.ResourceBindFlags.ShaderResource,
)

# 初始化 Splat2D 渲染函数，并将其绑定到 Splat2DFunction 类中
splat2d = Splat2D(device)
Splat2DFunction.splat2d = splat2d

# 设置高斯参数：位置、尺度、旋转以及颜色
blob_positions = torch.rand([BLOB_COUNT, 2]).cuda()        # 随机生成 blob 位置
blob_scales = torch.log(torch.ones([BLOB_COUNT, 2]).cuda() * 0.005)   # 使用对数尺度
blob_rotations = torch.rand([BLOB_COUNT, 1]).cuda() * (2 * np.pi)       # 随机旋转角度
blob_colors = torch.rand([BLOB_COUNT, 3]).cuda() * 0.5 + 0.25            # 颜色在 [0.25, 0.75] 范围内

# 将这些参数设置为需要梯度计算
params = (blob_positions, blob_scales, blob_colors, blob_rotations)
for param in params:
    param.requires_grad = True

# 加载目标图像，并进行预处理（调整分辨率、归一化到[0,1]）
image = Image.open(TARGET_IMAGE).resize([RESOLUTION, RESOLUTION]).convert("RGB")
target = np.asarray(image).astype(np.float32) / 255.0
target_cuda = torch.from_numpy(target).cuda()

# 实例化神经网络模块，封装 Splat2DFunction
model = Splat2DModule()


def optimize():
    """
    优化过程：使用 Adam 优化器迭代更新高斯参数，使渲染图像与目标图像之间的损失最小。
    """
    optimizer = torch.optim.Adam(params, lr=0.01)   # 使用 Adam 优化器
    sigmoid = torch.nn.Sigmoid()                      # 使用 Sigmoid 对颜色参数进行映射
    for iteration in range(ITERATIONS):
        optimizer.zero_grad()

        # 拼接 blob 参数以构造 blobs 张量
        blobs = torch.concat(
            (
                blob_positions,
                torch.exp(blob_scales),    # 将对数尺度转为尺度
                sigmoid(blob_colors),      # 将 blob_colors 映射到 [0,1]
                blob_rotations,
            ),
            dim=1,
        )
        # 正向传播得到图像
        image = model.forward(blobs)
        # 计算损失：这里使用 L1 损失，可根据需要改为 MSE 损失
        loss = torch.nn.functional.l1_loss(image, target_cuda)
        # loss = torch.nn.functional.mse_loss(image, target_cuda)
        loss.backward()              # 反向传播计算梯度
        optimizer.step()             # 更新参数

        # 每 5 次迭代打印一次当前损失，并将渲染图像显示在窗口上
        if iteration % 5 == 0:
            print(f"iteration={iteration}, loss={loss.item()}")
            render_image = image.detach()
            # 对渲染图像进行伽马校正
            render_image = torch.pow(render_image, 2.2)
            # 将渲染图像传给渲染纹理，并更新显示
            testbed.render_texture.from_numpy(render_image.cpu().numpy())
            testbed.frame()
            if testbed.should_close:
                break


# 运行优化过程
optimize()

# 优化完成后，持续显示图像直到窗口关闭
while not testbed.should_close:
    testbed.frame()
