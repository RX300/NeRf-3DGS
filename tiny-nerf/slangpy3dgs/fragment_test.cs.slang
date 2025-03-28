#include "Utils/NVAPI.slangh" // 引入 NVAPI 辅助头文件

import utils;

static const uint TILE_HEIGHT = 16;
static const uint TILE_WIDTH = 16;

groupshared Splat_2D_AlphaBlend collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct FragmentParameter:IDifferentiablePtrType
{
    typealias Differential = FragmentParameter;
    StructuredBuffer<int32_t> sorted_gauss_idx;
    StructuredBuffer<int32_t2> tile_ranges;
    RWStructuredBuffer<int32_t2>tile_ranges_out;
    RWStructuredBuffer<float3> xyz_vs;
    RWStructuredBuffer<float4> inv_cov_vs;
    RWStructuredBuffer<float> opacity;
    RWStructuredBuffer<float3> gaussian_rgb;
    RWTexture2D<float4> output_img;
    RWTexture2D<uint32_t> n_contributors;

    uint grid_height;
    uint grid_width;
    uint tile_height;
    uint tile_width;
};
// 为 xyz_vs (float3) 实现 load 函数
[BackwardDerivative(load_xyz_vs_bwd)]
float3 load_xyz_vs(FragmentParameter parameter, uint index)
{
    return parameter.xyz_vs[index];
}

// xyz_vs 的反向导数
void load_xyz_vs_bwd(DifferentialPtrPair<FragmentParameter> p, uint index, float3 dOut)
{
    // 计算字节偏移量
    uint xyz_offset_x = index * 12; // 3个float，每个4字节
    uint xyz_offset_y = xyz_offset_x + 4;
    uint xyz_offset_z = xyz_offset_x + 8;

    p.d.xyz_vs[index] += dOut; // 直接赋值，避免原子操作
    // 使用字节地址缓冲区原子累加梯度
    p.d.grad_xyz_vs.InterlockedAddF32(xyz_offset_x, dOut.x);
    p.d.grad_xyz_vs.InterlockedAddF32(xyz_offset_y, dOut.y);
    p.d.grad_xyz_vs.InterlockedAddF32(xyz_offset_z, dOut.z);
}

// 为 inv_cov_vs (float4) 实现 load 函数
[BackwardDerivative(load_inv_cov_vs_bwd)]
float4 load_inv_cov_vs(FragmentParameter parameter, uint index)
{
    return parameter.inv_cov_vs[index];
}

// inv_cov_vs 的反向导数
void load_inv_cov_vs_bwd(DifferentialPtrPair<FragmentParameter> p, uint index, float4 dOut)
{
    // 计算字节偏移量
    uint inv_cov_offset_x = index * 16; // 4个float，每个4字节
    uint inv_cov_offset_y = inv_cov_offset_x + 4;
    uint inv_cov_offset_z = inv_cov_offset_x + 8;
    uint inv_cov_offset_w = inv_cov_offset_x + 12;

    // 使用字节地址缓冲区原子累加梯度
    p.d.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_x, dOut.x);
    p.d.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_y, dOut.y);
    p.d.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_z, dOut.z);
    p.d.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_w, dOut.w);
}

// 为 opacity (float) 实现 load 函数
[BackwardDerivative(load_opacity_bwd)]
float load_opacity(FragmentParameter parameter, uint index)
{
    return parameter.opacity[index];
}

// opacity 的反向导数
void load_opacity_bwd(DifferentialPtrPair<FragmentParameter> p, uint index, float dOut)
{
    // 计算字节偏移量
    uint opacity_offset = index * 4; // 1个float，4字节

    // 使用字节地址缓冲区原子累加梯度
    p.d.grad_opacity.InterlockedAddF32(opacity_offset, dOut);
}

// 为 gaussian_rgb (float3) 实现 load 函数
[BackwardDerivative(load_gaussian_rgb_bwd)]
float3 load_gaussian_rgb(FragmentParameter parameter, uint index)
{
    return parameter.gaussian_rgb[index];
}

// gaussian_rgb 的反向导数
void load_gaussian_rgb_bwd(DifferentialPtrPair<FragmentParameter> p, uint index, float3 dOut)
{
    // 计算字节偏移量
    uint rgb_offset_x = index * 12; // 3个float，每个4字节
    uint rgb_offset_y = rgb_offset_x + 4;
    uint rgb_offset_z = rgb_offset_x + 8;

    // 使用字节地址缓冲区原子累加梯度
    p.d.grad_rgb.InterlockedAddF32(rgb_offset_x, dOut.x);
    p.d.grad_rgb.InterlockedAddF32(rgb_offset_y, dOut.y);
    p.d.grad_rgb.InterlockedAddF32(rgb_offset_z, dOut.z);
}
ParameterBlock<FragmentParameter> fragment_parameter_fwd;
ParameterBlock<FragmentParameter> fragment_parameter_bwd;

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (1 - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}
// 根据update_pixel_state反向求pixel_state_t_nm1
float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    //  计算移除当前高斯点后的透明度（时间步n-1的透明度）
    //  transmittance_t_nm1对应于pixel_state_t_nm1.a
    float transmittance_t_nm1 = pixel_state_t_n.a / (1 - gauss_rgba_t_n.a);
    // 时间步n-1的颜色值：当前颜色值减去当前高斯点的颜色值乘以透明度
    // color_t_nm1对应于pixel_state_t_nm1.rgb
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

[Differentiable]
bool process_gaussian_contribution(
    float4 curr_pixel_state, // 当前像素状态（输入/输出参数）
    inout int local_n_contrib,     // 贡献计数（输入/输出参数）
    float2 center_pix_coord,       // 像素中心坐标
    float3 position,               // 高斯点位置（包含 NDC 坐标）
    float4 inv_covariance,         // 高斯点反协方差矩阵 (存储为 float4，对应 2x2 矩阵)
    float opacity_value,           // 高斯点不透明度
    float3 color,                  // 高斯点颜色
    uint32_t H,                    // 图像高度
    uint32_t W,                    // 图像宽度
    inout float4 next_pixel_state,   // 下一个像素状态
    out bool is_skip_curgauss      // 是否跳过当前高斯点
)
{
    next_pixel_state = curr_pixel_state;
    is_skip_curgauss = false;
    // 增加贡献计数
    local_n_contrib++;

    // 计算像素坐标与高斯中心的差值向量
    float2 d = {
        center_pix_coord.x - ndc2pix(position.x, W),
        center_pix_coord.y - ndc2pix(position.y, H)
    };

    // 计算高斯点在像素位置的评估值
    // 使用反协方差矩阵计算二次型
    float power = -0.5f * (inv_covariance[0] * d.x * d.x +
                           inv_covariance[3] * d.y * d.y +
                           (inv_covariance[1] + inv_covariance[2]) * d.x * d.y);

    // 计算 alpha 值，并限制到 0.99
    float alpha = min(0.99f, opacity_value * exp(power));

    // 计算预乘 alpha 的颜色
    float3 premult_rgb = color * alpha;
    float4 gauss_rgba = float4(premult_rgb, alpha);

    // 跳过贡献微小的高斯点
    if (gauss_rgba.a < 1.0f / 255.0f)
    {
        is_skip_curgauss = true;           // 跳过当前高斯点
        return true;                       // 继续处理下一个高斯点
    }

    // 更新像素状态
    float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);

    // 检查是否终止处理（透明度过低）
    if (new_pixel_state.a < 0.0001f)
    {
        // 此高斯点未实际贡献，在终止前减去计数
        local_n_contrib--;
        return false; // 停止处理更多高斯点
    }

    // 更新当前像素状态
    next_pixel_state = new_pixel_state;
    return true; // 继续处理下一个高斯点
}

// [BackwardDerivative(bwd_alpha_blend)]
[Differentiable]
float4 alpha_blend(FragmentParameter parameter,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t H,
                   uint32_t W,
                   uint3 tid,         //线程的全局三维索引
                   uint thread_rank  //线程在线程组内的一维索引
                   )
{
    float2 center_pix_coord = pix_coord;
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    float4 next_pixel_state = float4(0.f, 0.f, 0.f, 1.f);

    uint32_t block_size = parameter.tile_height * parameter.tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    if(!is_inside)
        return curr_pixel_state;
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    [MaxIters(1000)]
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(parameter.sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_splat_alphablendv2(
                parameter.xyz_vs[coll_id], parameter.inv_cov_vs[coll_id],
                parameter.opacity[coll_id], parameter.gaussian_rgb[coll_id]);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            [MaxIters(16*16)]
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                uint32_t coll_id = uint32_t(parameter.sorted_gauss_idx[tile_idx_start + splat_pointer_offset - j]);
                Splat_2D_AlphaBlend g = collected_splats[j];

                // 从Splat_2D_AlphaBlend中提取参数
                float3 position = g.xyz_vs;
                float4 inv_covariance = float4(g.inv_cov_vs[0][0], g.inv_cov_vs[0][1], g.inv_cov_vs[1][0], g.inv_cov_vs[1][1]);
                float opacity_value = g.opacity;
                float3 color = g.rgb;
                
                bool is_skip_curgauss = false;
                // 使用process_gaussian_contribution代替原有代码
                curr_pixel_state = next_pixel_state;
                bool continue_processing = process_gaussian_contribution(
                    curr_pixel_state,
                    local_n_contrib,
                    center_pix_coord,
                    position,
                    inv_covariance,
                    opacity_value,
                    color,
                    H, W,
                    next_pixel_state,
                    is_skip_curgauss
                );
                if (is_skip_curgauss)
                    continue;
                if (!continue_processing) {
                    thread_active = false;
                    break;
                }

            }
        }
        splats_left_to_process -= block_size;
    }

    parameter.n_contributors[uint2(uint32_t(pix_coord.x), uint32_t(pix_coord.y))] = local_n_contrib;

    return curr_pixel_state;
}

[Differentiable]
void forward_splat_tiled_main(
    inout FragmentParameter parameter, uint3 tid, uint groupthreadIndex)
{
    uint thread_rank = groupthreadIndex;
    uint32_t3 globalIdx = tid;
    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = globalIdx.y / parameter.tile_height * parameter.grid_width + globalIdx.x / parameter.tile_width;
    uint32_t tile_idx_start = uint32_t(parameter.tile_ranges[tile_idx][0]);
    uint32_t tile_idx_end = uint32_t(parameter.tile_ranges[tile_idx][1]);

    // 获取图像尺寸
    uint3 image_size;
    parameter.output_img.GetDimensions(image_size.x, image_size.y);

    bool is_inside = (pix_coord.x < image_size.x && pix_coord.y < image_size.y);

    float4 pixel_state = alpha_blend(parameter,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     image_size.y,
                                     image_size.x,
                                     tid,
                                     thread_rank);

    if (is_inside) {
        parameter.output_img[pix_coord] = pixel_state;
    }
}
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void splat_tiled_main(uint3 tid: SV_DispatchThreadID, uint groupthreadIndex: SV_GroupIndex)
{
    forward_splat_tiled_main(fragment_parameter_fwd, tid, groupthreadIndex);
}

[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void splat_tiled_backward_main(uint3 tid: SV_DispatchThreadID, uint3 groupthreadId: SV_GroupThreadID, uint groupthreadIndex: SV_GroupIndex, uint3 gid: SV_GroupID)
{
    bwd_diff(forward_splat_tiled_main)(DifferentialPtrPair<FragmentParameter>(fragment_parameter_fwd,fragment_parameter_bwd), tid, groupthreadIndex);
}