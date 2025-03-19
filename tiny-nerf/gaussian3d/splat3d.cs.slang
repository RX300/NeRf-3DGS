#include "Utils/NVAPI.slangh" // 引入 NVAPI 辅助头文件

// 定义跳过距离参数（可选）
// 当 blob 在图像平面上的投影距离当前像素超过该值时，可跳过计算
static const float kSkipDistance = 0.1;

// 定义 Blob 结构，表示单个 3D 高斯 blob
// 每个 blob 包含：3D 位置 (float3)、半径 (float)、颜色 (float3)、不透明度 (float)
struct Blob {
    float3 pos;      // 3D 位置
    float radius;    // 高斯半径
    float3 color;    // 颜色
    float opacity;   // 不透明度
};

// 全局前向传播结构体，用于传递参数到着色器
struct Forward {
    StructuredBuffer<Blob> blobs;        // Blob 参数缓冲区
    uint blob_count;                     // Blob 数量
    RWStructuredBuffer<float3> image;    // 输出图像缓冲区（RGB，每个像素 float3）
    uint2 resolution;                    // 图像分辨率
    float4x4 view;                       // 摄像机视图矩阵
    float4x4 proj;                       // 投影矩阵

    // 前向渲染函数，对每个像素索引 idx 进行计算
    void forward(uint2 idx)
    {
        // 判断像素是否在图像范围内
        if (any(idx >= resolution))
            return;

        // 计算当前像素坐标（归一化空间 [0,1]）
        float2 uv = (float2(idx) + 0.5f) / resolution;
        float3 accumColor = float3(0);
        // 遍历所有 blob
        for (uint i = 0; i < blob_count; ++i) {
            Blob blob = blobs[i];
            // 将 blob 的 3D 位置投影到屏幕上：
            // 首先转换到摄像机空间
            float4 camPos = view * float4(blob.pos, 1.0);
            // 再应用投影矩阵
            float4 projPos = proj * camPos;
            projPos /= projPos.w;  // 透视除法
            // 转换到归一化屏幕坐标 [0,1]
            float2 projUV = projPos.xy * 0.5f + 0.5f;
            // 可选：跳过距离较远的 blob
            if (kSkipDistance > 0 && distance(uv, projUV) > kSkipDistance)
                continue;
            // 计算高斯核（利用 blob.radius 调整分布宽度）
            float err = distance(uv, projUV) / blob.radius;
            float weight = blob.opacity * exp(-0.5f * err * err);
            // 累加颜色贡献
            accumColor += blob.color * weight;
        }
        // 写入输出图像
        image[idx.y * resolution.x + idx.x] = accumColor;
    }
};

// 反向传播结构体（梯度计算部分，这里仅示意，不做具体实现）
// 实际应用中需要根据 3D 高斯的参数求导
struct Backward {
    // 省略实现——类似于 forward 结构，只需对每个 blob 的参数求梯度，
    // 并通过 InterlockedAddF32 写入到 grad_blobs 缓冲区
    void backward(uint2 idx)
    {
        // 这里为示例，不提供详细实现
    }
};

// 定义 ParameterBlock，将 Forward 与 Backward 结构传递给着色器
ParameterBlock<Forward> forward;
ParameterBlock<Backward> backward;

// 入口函数：前向渲染，每个线程处理一个像素
[numthreads(16, 16, 1)]
void forward_main(uint3 tid : SV_DispatchThreadID)
{
    forward.forward(tid.xy);
}

// 入口函数：反向传播，每个线程处理一个像素的梯度计算
[numthreads(16, 16, 1)]
void backward_main(uint3 tid : SV_DispatchThreadID)
{
    backward.backward(tid.xy);
}