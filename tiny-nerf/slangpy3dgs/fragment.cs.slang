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

    // 梯度缓冲区改为ByteAddressBuffer
    no_diff RWTexture2D<float4> grad_output; // 输出图像的梯度
    no_diff RWByteAddressBuffer grad_xyz_vs; // 位置梯度 (使用字节缓冲区)
    no_diff RWByteAddressBuffer grad_inv_cov_vs; // 反协方差梯度 (使用字节缓冲区)
    no_diff RWByteAddressBuffer grad_opacity;    // 不透明度梯度 (使用字节缓冲区)
    no_diff RWByteAddressBuffer grad_rgb;        // 颜色梯度 (使用字节缓冲区)
    no_diff RWByteAddressBuffer grad_out_test;        // 输出测试梯度 (使用字节缓冲区)
    uint grid_height;
    uint grid_width;
    uint tile_height;
    uint tile_width;
};
ParameterBlock<FragmentParameter> fragment_parameter;

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
float4 alpha_blend(uint32_t2 pix_coord,
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

    uint32_t block_size = fragment_parameter.tile_height * fragment_parameter.tile_width;
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
            uint32_t coll_id = uint32_t(fragment_parameter.sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_splat_alphablendv2(
                fragment_parameter.xyz_vs[coll_id], fragment_parameter.inv_cov_vs[coll_id],
                fragment_parameter.opacity[coll_id], fragment_parameter.gaussian_rgb[coll_id]);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            [MaxIters(16*16)]
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                uint32_t coll_id = uint32_t(fragment_parameter.sorted_gauss_idx[tile_idx_start + splat_pointer_offset - j]);
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

                // local_n_contrib++;
                // Splat_2D_AlphaBlend g = collected_splats[j];
                // float4 gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);
                
                // // Skip Splats that have a tiny contribution.
                // if (gauss_rgba.a < 1.0f / 255.0f)
                //     continue;

                // float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                // if (new_pixel_state.a < 0.0001f) {
                //     // This Splat never registred so we subtract it before we break.
                //     local_n_contrib--;
                //     thread_active = false;
                //     break;
                // }
                // curr_pixel_state = new_pixel_state;
            }
        }
        splats_left_to_process -= block_size;
    }

    fragment_parameter.n_contributors[uint2(uint32_t(pix_coord.x), uint32_t(pix_coord.y))] = local_n_contrib;

    return curr_pixel_state;
}

// void bwd_alpha_blend(uint32_t2 pix_coord,
//                      uint32_t tile_idx_start,
//                      uint32_t tile_idx_end,
//                      uint32_t H,
//                      uint32_t W,
//                      uint3 tid,
//                      uint thread_rank,
//                      float4 d_current_pixel_state)
// {
//     // Load the final pixel state.
//     bool is_inside = (pix_coord.x < W && pix_coord.y < H);
//     uint32_t block_size = fragment_parameter.tile_height * fragment_parameter.tile_width;
//     const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

//     int splats_left_to_process = tile_idx_end - tile_idx_start;
//     uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

//     float4 current_pixel_state;
//     int32_t n_contrib_fwd;
//     if (is_inside) {
//         current_pixel_state = fragment_parameter.output_img[pix_coord];
//         n_contrib_fwd = fragment_parameter.n_contributors[uint2(uint32_t(pix_coord.x), uint32_t(pix_coord.y))];
//     }
//     float2 center_pix_coord = pix_coord;
//     DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);
//     float4 d = d_current_pixel_state;
//     for (int i = 0; i < rounds; i++)
//     {
//         // Collectively fetch per-Gaussian data from global to shared
//         AllMemoryBarrierWithGroupSync();
//         // 当前处理的gaussian是当前block中的第几个
//         int progress = i * block_size + thread_rank;
//         if (tile_idx_start + progress < tile_idx_end)
//         {
//             uint32_t coll_id = uint32_t(fragment_parameter.sorted_gauss_idx[tile_idx_end - progress - 1]);
//             collected_idx[thread_rank] = coll_id;
//             collected_splats[thread_rank] = load_splat_alphablendv2(
//                 fragment_parameter.xyz_vs[coll_id], fragment_parameter.inv_cov_vs[coll_id],
//                 fragment_parameter.opacity[coll_id], fragment_parameter.gaussian_rgb[coll_id]);
//         }
//         AllMemoryBarrierWithGroupSync();
//         if (is_inside) {
//             for (int j = 0; j < min(block_size, splats_left_to_process); j++)
//             {
//                 current_splat_offset--;
//                 if (current_splat_offset >= n_contrib_fwd)
//                     continue;
//                 uint32_t g_idx = collected_idx[j];
//                 Splat_2D_AlphaBlend g = collected_splats[j];

//                 float4 gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);

//                 if (gauss_rgba.a < 1.0f / 255.0f)
//                     continue;

//                 // 从Splat_2D_AlphaBlend中提取参数
//                 float3 position = g.xyz_vs;
//                 float4 inv_covariance = float4(g.inv_cov_vs[0][0], g.inv_cov_vs[0][1], g.inv_cov_vs[1][0], g.inv_cov_vs[1][1]);
//                 float opacity_value = g.opacity;
//                 float3 color = g.rgb;

//                 DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state, d_current_pixel_state);
//                 DifferentialPair<float4> dp_last_pixel_state = diffPair(undo_pixel_state(current_pixel_state, gauss_rgba));
//                 DifferentialPair<float3> dp_xyz_vs = diffPair(fragment_parameter.xyz_vs[g_idx]);
//                 DifferentialPair<float4> dp_inv_cov_vs = diffPair(fragment_parameter.inv_cov_vs[g_idx]);
//                 DifferentialPair<float3> dp_gauss_rgb = diffPair(fragment_parameter.gaussian_rgb[g_idx]);
//                 DifferentialPair<float> dp_opacity = diffPair(fragment_parameter.opacity[g_idx]);
//                 bwd_diff(process_gaussian_contribution)(dp_last_pixel_state, n_contrib_fwd, dp_center_pix_coord, dp_xyz_vs, dp_inv_cov_vs, dp_opacity, dp_gauss_rgb, H, W, dp_current_pixel_state);
//                 d_current_pixel_state = dp_last_pixel_state.getDifferential();
//                 // current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);
//                 // 计算字节偏移量
//                 uint xyz_offset_x = g_idx * 12; // 3个float，每个4字节
//                 uint xyz_offset_y = xyz_offset_x + 4;
//                 uint xyz_offset_z = xyz_offset_x + 8;

//                 uint inv_cov_offset_x = g_idx * 16; // 4个float，每个4字节
//                 uint inv_cov_offset_y = inv_cov_offset_x + 4;
//                 uint inv_cov_offset_z = inv_cov_offset_x + 8;
//                 uint inv_cov_offset_w = inv_cov_offset_x + 12;

//                 uint opacity_offset = g_idx * 4; // 1个float，4字节

//                 uint rgb_offset_x = g_idx * 12; // 3个float，每个4字节
//                 uint rgb_offset_y = rgb_offset_x + 4;
//                 uint rgb_offset_z = rgb_offset_x + 8;

//                 // 使用NVAPI提供的浮点原子操作
//                 // xyz_vs梯度
//                 fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_x, dp_xyz_vs.d.x);
//                 fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_y, dp_xyz_vs.d.y);
//                 fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_z, dp_xyz_vs.d.z);

//                 // inv_cov_vs梯度
//                 fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_x, dp_inv_cov_vs.d.x);
//                 fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_y, dp_inv_cov_vs.d.y);
//                 fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_z, dp_inv_cov_vs.d.z);
//                 fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_w, dp_inv_cov_vs.d.w);

//                 // opacity梯度
//                 fragment_parameter.grad_opacity.InterlockedAddF32(opacity_offset, dp_opacity.d);

//                 // Gaussian的rgb梯度
//                 fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_x, dp_gauss_rgb.d.x);
//                 fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_y, dp_gauss_rgb.d.y);
//                 fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_z, dp_gauss_rgb.d.z);
//             }
//         }
//         splats_left_to_process -= block_size;
//     }
// }

// //https://zhuanlan.zhihu.com/p/30654608000

void bwd_alpha_blend(uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t H,
                     uint32_t W,
                     uint3 tid,
                     uint thread_rank,
                     float4 d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = fragment_parameter.tile_height * fragment_parameter.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);


    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = fragment_parameter.output_img[pix_coord];
        n_contrib_fwd = fragment_parameter.n_contributors[uint2(uint32_t(pix_coord.x), uint32_t(pix_coord.y))];
    }
    float2 center_pix_coord = pix_coord;
    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);
    // current_pixel_state =float4(1.0f,1.0f,1.0f,1.0f);
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        // 当前处理的gaussian是当前block中的第几个
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(fragment_parameter.sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat_alphablendv2(
                fragment_parameter.xyz_vs[coll_id], fragment_parameter.inv_cov_vs[coll_id],
                fragment_parameter.opacity[coll_id], fragment_parameter.gaussian_rgb[coll_id]);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd)
                    continue;
                uint32_t g_idx = collected_idx[j];
                Splat_2D_AlphaBlend g = collected_splats[j];

                float4 gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);

                if (gauss_rgba.a < 1.0f / 255.0f)
                    continue;
                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);

                // Back-prop automatically through blending and gaussian evaluation.
                DifferentialPair<Splat_2D_AlphaBlend> dp_g = diffPair(g);
                DifferentialPair<float3> dp_xyz_vs = diffPair(fragment_parameter.xyz_vs[g_idx]);
                DifferentialPair<float4> dp_inv_cov_vs = diffPair(fragment_parameter.inv_cov_vs[g_idx]);
                DifferentialPair<float3> dp_gauss_rgb = diffPair(fragment_parameter.gaussian_rgb[g_idx]);
                DifferentialPair<float> dp_opacity = diffPair(fragment_parameter.opacity[g_idx]);
                DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat)(dp_g, dp_center_pix_coord, H, W, dp_gauss_rgba.d);
                bwd_diff(load_splat_alphablendv2)(dp_xyz_vs, dp_inv_cov_vs, dp_opacity, dp_gauss_rgb, dp_g.d);

                // 计算字节偏移量
                uint xyz_offset_x = g_idx * 12; // 3个float，每个4字节
                uint xyz_offset_y = xyz_offset_x + 4;
                uint xyz_offset_z = xyz_offset_x + 8;

                uint inv_cov_offset_x = g_idx * 16; // 4个float，每个4字节
                uint inv_cov_offset_y = inv_cov_offset_x + 4;
                uint inv_cov_offset_z = inv_cov_offset_x + 8;
                uint inv_cov_offset_w = inv_cov_offset_x + 12;

                uint opacity_offset = g_idx * 4; // 1个float，4字节

                uint rgb_offset_x = g_idx * 12; // 3个float，每个4字节
                uint rgb_offset_y = rgb_offset_x + 4;
                uint rgb_offset_z = rgb_offset_x + 8;

                // 使用NVAPI提供的浮点原子操作
                // xyz_vs梯度
                fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_x, dp_xyz_vs.d.x);
                fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_y, dp_xyz_vs.d.y);
                fragment_parameter.grad_xyz_vs.InterlockedAddF32(xyz_offset_z, dp_xyz_vs.d.z);

                // inv_cov_vs梯度
                fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_x, dp_inv_cov_vs.d.x);
                fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_y, dp_inv_cov_vs.d.y);
                fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_z, dp_inv_cov_vs.d.z);
                fragment_parameter.grad_inv_cov_vs.InterlockedAddF32(inv_cov_offset_w, dp_inv_cov_vs.d.w);

                // opacity梯度
                fragment_parameter.grad_opacity.InterlockedAddF32(opacity_offset, dp_opacity.d);

                // Gaussian的rgb梯度
                fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_x, dp_gauss_rgb.d.x);
                fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_y, dp_gauss_rgb.d.y);
                fragment_parameter.grad_rgb.InterlockedAddF32(rgb_offset_z, dp_gauss_rgb.d.z);
            }
        }
        splats_left_to_process -= block_size;
    }
}

[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void splat_tiled_main(uint3 tid: SV_DispatchThreadID, uint3 groupthreadId: SV_GroupThreadID, uint groupthreadIndex: SV_GroupIndex, uint3 gid: SV_GroupID)
{
    uint thread_rank = groupthreadIndex;
    uint32_t3 globalIdx = tid;
    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = globalIdx.y / fragment_parameter.tile_height * fragment_parameter.grid_width + globalIdx.x / fragment_parameter.tile_width;
    uint32_t tile_idx_start = uint32_t(fragment_parameter.tile_ranges[tile_idx][0]);
    uint32_t tile_idx_end = uint32_t(fragment_parameter.tile_ranges[tile_idx][1]);

    uint3 image_size;
    fragment_parameter.output_img.GetDimensions(image_size.x, image_size.y);

    bool is_inside = (pix_coord.x < image_size.x &&
                      pix_coord.y < image_size.y);

    float4 pixel_state = alpha_blend(pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     image_size.y,
                                     image_size.x,
                                     tid,
                                     thread_rank);

    if (is_inside) {
        fragment_parameter.output_img[pix_coord] = pixel_state;
    }
}

[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void splat_tiled_backward_main(uint3 tid: SV_DispatchThreadID, uint3 groupthreadId: SV_GroupThreadID, uint groupthreadIndex: SV_GroupIndex, uint3 gid: SV_GroupID)
{
    fragment_parameter.grad_out_test.InterlockedAddF32(0, 0.005f);
    uint thread_rank = groupthreadIndex;
    uint32_t3 globalIdx = tid;
    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = globalIdx.y / fragment_parameter.tile_height * fragment_parameter.grid_width + globalIdx.x / fragment_parameter.tile_width;
    uint32_t tile_idx_start = uint32_t(fragment_parameter.tile_ranges[tile_idx][0]);
    uint32_t tile_idx_end = uint32_t(fragment_parameter.tile_ranges[tile_idx][1]);
    fragment_parameter.tile_ranges_out[tile_idx] = int2(tile_idx_start, tile_idx_end);
    // 获取图像尺寸
    uint3 image_size;
    fragment_parameter.output_img.GetDimensions(image_size.x, image_size.y);

    bool is_inside = (pix_coord.x < image_size.x && pix_coord.y < image_size.y);

    // 对于图像内的有效像素，获取梯度并执行反向传播
    if (is_inside) {
        // 读取当前像素的梯度
        float4 d_current_pixel_state = fragment_parameter.grad_output[pix_coord];

        // 如果梯度为零，则跳过计算
        if (all(d_current_pixel_state == 0))
            return;
        // 调用反向传播函数计算梯度
        bwd_alpha_blend(pix_coord,
                       tile_idx_start,
                       tile_idx_end,
                       image_size.y,
                       image_size.x,
                       tid,
                       thread_rank,
                       d_current_pixel_state);
    }
}