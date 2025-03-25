#include "Utils/NVAPI.slangh" // 引入 NVAPI 辅助头文件

import utils;

static const uint TILE_HEIGHT = 16;
static const uint TILE_WIDTH = 16;

groupshared Splat_2D_AlphaBlend collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct FragmentParameter
{
    StructuredBuffer<int32_t> sorted_gauss_idx;
    StructuredBuffer<int32_t2> tile_ranges;
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
ParameterBlock<FragmentParameter> fragment_parameter;

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

[BackwardDerivative(bwd_alpha_blend)]
float4 alpha_blend(uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t H,
                   uint32_t W,
                   uint3 tid,
                   uint thread_rank)
{
    float2 center_pix_coord = pix_coord;
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    uint32_t block_size = fragment_parameter.tile_height * fragment_parameter.tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
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
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                Splat_2D_AlphaBlend g = collected_splats[j];
                float4 gauss_rgba = evaluate_splat(g, center_pix_coord, H, W);
                
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
        fragment_parameter.n_contributors[uint2(uint32_t(pix_coord.x), uint32_t(pix_coord.y))] = local_n_contrib;

    return curr_pixel_state;
}
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

    uint32_t thread_rank = tid.y * W + tid.x;

    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
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
                DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat)(dp_g, dp_center_pix_coord, H, W, dp_gauss_rgba.d);
                //bwd_diff(load_splat_alphablendv2)(fragment_parameter.xyz_vs[g_idx], fragment_parameter.inv_cov_vs[g_idx], fragment_parameter.opacity[g_idx], fragment_parameter.bg_color[tid.xy].xyz, dp_g.d);
            }
        }
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
        // fragment_parameter.output_img[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))] = pixel_state.r;
        // fragment_parameter.output_img[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))] = pixel_state.g;
        // fragment_parameter.output_img[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))] = pixel_state.b;
        // fragment_parameter.output_img[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))] = pixel_state.a;
    }
}
