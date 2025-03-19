#include "Utils/NVAPI.slangh" // 引入 NVAPI 辅助头文件

// 定义跳过距离参数，当绘制点与像素之间的距离大于该值时，可以跳过计算
static const float kSkipDistance = 0.1;

// 定义 Blob 结构，表示单个高斯 blob
// 继承自 IDifferentiable 接口，支持反向传播（梯度计算）
struct Blob : IDifferentiable {
    float2 pos;       // 位置
    float2 scale;     // 尺度（缩放）
    float3 color;     // 颜色（RGB）
    float rotation;   // 旋转角度
};

// 以下函数均标注了 [BackwardDifferentiable] 属性，表示它们支持自动微分（可用于反向传播计算梯度）

// 计算 2x2 矩阵的逆矩阵
[BackwardDifferentiable]
float2x2 inverse(float2x2 M)
{
    float2x2 inv;
    float invdet = 1.0f / determinant(M);  // 计算行列式的倒数
    inv[0][0] = M[1][1] * invdet;
    inv[1][1] = M[0][0] * invdet;
    inv[0][1] = -M[0][1] * invdet;
    inv[1][0] = -M[1][0] * invdet;
    return inv;
}

// 生成旋转矩阵，根据给定的角度计算二维旋转矩阵
[BackwardDifferentiable]
float2x2 rotation_matrix(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2x2(c, -s, s, c);
}

// 生成缩放矩阵，根据给定的缩放向量构造二维对角缩放矩阵
[BackwardDifferentiable]
float2x2 scale_matrix(float2 scale)
{
    return float2x2(scale.x, 0, 0, scale.y);
}

// 计算二维高斯函数，输入 x 和协方差矩阵 sigma
[BackwardDifferentiable]
float gaussian(float2 x, float2x2 sigma)
{
    float2x2 sigma_inv = inverse(sigma);   // 计算协方差矩阵的逆
    return exp(-0.5 * dot(x, mul(sigma_inv, x)));  // exp(-0.5 * x^T * sigma_inv * x)
}

// 根据像素位置和 Blob 参数计算一个高斯 blob 的颜色贡献
// no_diff 标记表示 pos 在反向传播时不参与微分
[BackwardDifferentiable]
float3 eval(no_diff float2 pos, Blob blob) {
    // 计算旋转矩阵
    float2x2 R = rotation_matrix(blob.rotation);
    // 计算缩放矩阵，并确保缩放不小于最低值 0.001
    float2x2 S = scale_matrix(max(blob.scale, 0.001));
    // 组合得到协方差矩阵 sigma = R * S * (S^T) * (R^T)
    float2x2 sigma = mul(mul(R, S), mul(transpose(S), transpose(R)));
    // 计算高斯值并乘以颜色，作为该 blob 对当前像素的贡献
    return gaussian(pos - blob.pos, sigma) * blob.color;
}

// 前向传播结构体，用于将多个 blob 渲染成图像
struct Forward {
    StructuredBuffer<Blob> blobs;       // Blob 参数缓冲区
    uint blob_count;                    // Blob 数量
    RWStructuredBuffer<float3> image;   // 可读写的图像输出缓冲区（RGB）
    uint2 resolution;                   // 输出图像分辨率

    // 前向计算函数，对当前像素索引 idx 进行处理
    void forward(uint2 idx)
    {
        // 如果像素索引超出图像分辨率范围，则直接返回
        if (any(idx >= resolution))
            return;

        // 计算当前像素在归一化空间中的位置（范围 [0,1]）
        float2 pos = float2(idx) / resolution;
        float3 color = float3(0);  // 初始化颜色累加值为 0
        for (uint i = 0; i < blob_count; ++i) {
            // 可选：根据跳过距离判断是否跳过当前 blob
            // if (kSkipDistance > 0 && distance(pos, blobs[i].pos) > kSkipDistance)
            //     continue;
            // 将每个 blob 的颜色贡献累加
            color += eval(pos, blobs[i]);
        }
        // 将计算好的颜色写入图像缓冲区中对应位置
        image[idx.y * resolution.x + idx.x] = color;
    }
};

// 反向传播结构体，用于计算 blob 参数的梯度
struct Backward {
    StructuredBuffer<Blob> blobs;       // Blob 参数缓冲区
    RWByteAddressBuffer grad_blobs;       // 存储 blob 参数梯度的缓冲区（以字节地址访问）
    uint blob_count;                    // Blob 数量
    StructuredBuffer<float3> grad_image;  // 渲染图像梯度缓冲区
    uint2 resolution;                   // 图像分辨率

    // 辅助函数：将计算得到的梯度写入指定偏移位置
    void write_grad(uint offset, float value)
    {
        float sum = WaveActiveSum(value);  // 对一个 wave 内的值进行求和（汇总计算）
        if (WaveIsFirstLane())             // 仅在 wave 中第一个线程执行写入操作，避免重复写入
            grad_blobs.InterlockedAddF32(offset, sum);
    }

    // 反向计算函数，对当前像素索引 idx 的梯度进行处理
    void backward(uint2 idx)
    {
        // 如果像素索引超出图像分辨率，返回不处理
        if (any(idx >= resolution))
            return;

        // 计算当前像素的归一化位置
        no_diff float2 pos = float2(idx) / resolution;

        // 遍历所有 blob
        for (uint i = 0; i < blob_count; ++i) {
            // 创建 blob 的差分对，存储正向传播与梯度信息
            var d_blob = diffPair(blobs[i]);
            // 获取当前像素的梯度值
            float3 d_image = grad_image[idx.y * resolution.x + idx.x];
            // 如果距离过大则跳过该 blob 的反向传播
            if (kSkipDistance > 0 && distance(pos, blobs[i].pos) > kSkipDistance)
                continue;
            // 计算 blob 参数在当前像素处的梯度
            bwd_diff(eval)(pos, d_blob, d_image);
            // 将不同参数的梯度写入 grad_blobs 缓冲区，不同参数间间隔 32 字节
            write_grad(i * 32, d_blob.d.pos.x);
            write_grad(i * 32 + 4, d_blob.d.pos.y);
            write_grad(i * 32 + 8, d_blob.d.scale.x);
            write_grad(i * 32 + 12, d_blob.d.scale.y);
            write_grad(i * 32 + 16, d_blob.d.color.r);
            write_grad(i * 32 + 20, d_blob.d.color.g);
            write_grad(i * 32 + 24, d_blob.d.color.b);
            write_grad(i * 32 + 28, d_blob.d.rotation);
        }
    }
};

// 定义 ParameterBlock，用于将 Forward 和 Backward 结构体传递给着色器
ParameterBlock<Forward> forward;
ParameterBlock<Backward> backward;

// 定义计算着色器入口函数，设置线程组大小为 16x16，每个线程处理一个像素。
// forward_main 对应前向传播，将图像渲染输出
[numthreads(16, 16, 1)]
void forward_main(uint3 tid: SV_DispatchThreadID)
{
    forward.forward(tid.xy);
}

// backward_main 对应反向传播，计算每个 blob 参数的梯度
[numthreads(16, 16, 1)]
void backward_main(uint3 tid: SV_DispatchThreadID)
{
    backward.backward(tid.xy);
}
