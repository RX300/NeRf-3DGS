// https://github.com/hbb1/diff-surfel-rasterization/blob/e0ed0207b3e0669960cfad70852200a4a5847f61/cuda_rasterizer/forward.cu

import utils_2dgs;
// 在def init_fragment_shader(self, tile_height, tile_width):中有定义PYTHON_TILE_HEIGHT
static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;
static const uint BLOCK_SIZE = TILE_HEIGHT * TILE_WIDTH;
groupshared Splat_2D_AlphaBlend collected_splats[BLOCK_SIZE];
groupshared uint32_t collected_idx[BLOCK_SIZE];
groupshared Splat_2D_NormalOpacity_UVW collected_normalopacity_Tuvw[BLOCK_SIZE];
[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (1 - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.a / (1 - gauss_rgba_t_n.a);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
float4 alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   DiffTensorView xyz_vs,
                   DiffTensorView inv_cov_vs,
                   DiffTensorView opacity,
                   DiffTensorView rgb,
                   DiffTensorView final_pixel_state,
                   DiffTensorView transMats,
                   DiffTensorView normal_opacity,
                   TensorView<int32_t> n_contributors,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    // 线程块内的一维线性索引
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // 线程块内的所有线程共同从全局内存加载高斯数据到共享内存
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            // 通过排序后的键值对的值数组(储存的是高斯全局索引)获取当前高斯点在高斯数组中的全局索引
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
            collected_normalopacity_Tuvw[thread_rank] = load_splat_alphablend_2(coll_id, normal_opacity, transMats);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                // First compute two homogeneous planes, See Eq. (8)
                Splat_2D_AlphaBlend g = collected_splats[j];
                Splat_2D_NormalOpacity_UVW trans_data = collected_normalopacity_Tuvw[j];
                const float3 Tu = trans_data.TU;
                const float3 Tv = trans_data.TV;
                const float3 Tw = trans_data.TW;
                float4 gauss_rgba = evaluate_splat_2dgs(g, center_pix_coord, H, W, Tu, Tv, Tw, trans_data.normal_opacity);
                
                // Skip Splats that have a tiny contribution.
                if (gauss_rgba.a < 1.0f / 255.0f)
                    continue;

                float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                if (new_pixel_state.a < 0.0001f) {
                    // This Splat never registred so we subtract it before we break.
                    local_n_contrib--;
                    thread_active = false;
                    break;
                }
                curr_pixel_state = new_pixel_state;
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return curr_pixel_state;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     DiffTensorView xyz_vs,
                     DiffTensorView inv_cov_vs,
                     DiffTensorView opacity,
                     DiffTensorView rgb,
                     DiffTensorView final_pixel_state,
                     DiffTensorView transMats,
                     DiffTensorView normal_opacity,
                     TensorView<int32_t> n_contributors,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     uint32_t H,
                     uint32_t W,
                     float4 d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    float2 center_pix_coord = pix_coord;

    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);


    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
            collected_normalopacity_Tuvw[thread_rank] = load_splat_alphablend_2(coll_id, normal_opacity, transMats);
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

                // Load transformation matrices for 2DGS
                Splat_2D_NormalOpacity_UVW trans_data = collected_normalopacity_Tuvw[j];
                const float3 Tu = trans_data.TU;
                const float3 Tv = trans_data.TV;
                const float3 Tw = trans_data.TW;
                float4 normal_opacity_val = trans_data.normal_opacity;

                float4 gauss_rgba = evaluate_splat_2dgs(g, center_pix_coord, H, W, Tu, Tv, Tw, normal_opacity_val);

                if (gauss_rgba.a < 1.0f / 255.0f)
                    continue;

                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);

                // Back-prop automatically through blending and gaussian evaluation.
                DifferentialPair<Splat_2D_AlphaBlend> dp_g = diffPair(g);
                DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);
                DifferentialPair<float3> dp_Tu = diffPair(Tu);
                DifferentialPair<float3> dp_Tv = diffPair(Tv);
                DifferentialPair<float3> dp_Tw = diffPair(Tw);
                DifferentialPair<float4> dp_normal_opacity_val = diffPair(normal_opacity_val);                
                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat_2dgs)(dp_g, dp_center_pix_coord, H, W, dp_Tu, dp_Tv, dp_Tw, dp_normal_opacity_val, dp_gauss_rgba.d);
                bwd_diff(load_splat_alphablend)(g_idx, xyz_vs, inv_cov_vs, opacity, rgb, dp_g.d);

                // Propagate gradients through load_splat_alphablend_2
                Splat_2D_NormalOpacity_UVW d_trans_data = {};
                d_trans_data.TU = dp_Tu.d;
                d_trans_data.TV = dp_Tv.d;
                d_trans_data.TW = dp_Tw.d;
                d_trans_data.normal_opacity = dp_normal_opacity_val.d;
                bwd_diff(load_splat_alphablend_2)(g_idx, normal_opacity, transMats, d_trans_data);
                
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz_vs,
                 DiffTensorView inv_cov_vs,
                 DiffTensorView opacity,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 DiffTensorView transMats,
                 DiffTensorView normal_opacity,
                 TensorView<int32_t> n_contributors,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));    float4 pixel_state = alpha_blend(sorted_gauss_idx,
                                     xyz_vs,
                                     inv_cov_vs,
                                     opacity,
                                     rgb,
                                     output_img,
                                     transMats,
                                     normal_opacity,
                                     n_contributors,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     output_img.size(0),
                                     output_img.size(1));

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_state.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_state.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_state.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
    }
}
