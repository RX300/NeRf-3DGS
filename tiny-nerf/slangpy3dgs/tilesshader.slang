import utils;
import spherical_harmonics;
import "slangpy";

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<int32_t> rect_tile_space,
                   TensorView<int32_t> index_buffer_offset,
                   TensorView<int64_t> out_unsorted_keys,
                   TensorView<int32_t> out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= xyz_vs.size(0))
        return;

    float3 ndc_xyz = {
        xyz_vs[uint2(globalIdx, 0)],
        xyz_vs[uint2(globalIdx, 1)],
        xyz_vs[uint2(globalIdx, 2)]
    };

    // Find this Gaussian's offset in buffer for writing keys/values.
    int32_t offset;
    if (globalIdx == 0)
        offset = 0;
    else
        offset = index_buffer_offset[globalIdx - 1];

    int32_t rect_min_x = rect_tile_space[uint2(globalIdx, 0)];
    int32_t rect_min_y = rect_tile_space[uint2(globalIdx, 1)];
    int32_t rect_max_x = rect_tile_space[uint2(globalIdx, 2)];
    int32_t rect_max_y = rect_tile_space[uint2(globalIdx, 3)];

    for (int32_t y = rect_min_y; y < rect_max_y; y++)
    {
        for (int32_t x = rect_min_x; x < rect_max_x; x++)
        {
            uint64_t key = y * grid_width + x;
            key <<= 32;
            key = key | reinterpret<int32_t>(ndc_xyz.z);
            out_unsorted_keys[offset] = key;
            out_unsorted_gauss_idx[offset] = globalIdx;
            offset++;
        }
    }
}

void generate_keysv2(float3 xyz_vs,
                   int32_t4 rect_tile_space,
                   int32_t index_buffer_offset,
                   out int64_t out_unsorted_keys,
                   out int32_t out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width)
{

    float3 ndc_xyz = xyz_vs;
    // Find this Gaussian's offset in buffer for writing keys/values.
    int32_t offset;

    int32_t rect_min_x = rect_tile_space[0];
    int32_t rect_min_y = rect_tile_space[1];
    int32_t rect_max_x = rect_tile_space[2];
    int32_t rect_max_y = rect_tile_space[3];

    for (int32_t y = rect_min_y; y < rect_max_y; y++)
    {
        for (int32_t x = rect_min_x; x < rect_max_x; x++)
        {
            uint64_t key = y * grid_width + x;
            key <<= 32;
            key = key | reinterpret<int32_t>(ndc_xyz.z);
            out_unsorted_keys = key;
            out_unsorted_gauss_idx = index_buffer_offset;
            offset++;
        }
    }
}

// sorted_keys: 已排序的64位整数张量,[N,1]，每个键包含瓦片ID（高32位）和高斯点的索引（低32位）
// out_tile_ranges: [N,2]张量，每行包含一个瓦片的起始和结束索引
[AutoPyBindCUDA]
[CUDAKernel]
void compute_tile_ranges(TensorView<int64_t> sorted_keys,
                         TensorView<int32_t> out_tile_ranges)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= sorted_keys.size(0))
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint32_t currtile = uint32_t(uint64_t(sorted_keys[globalIdx]) >> 32);
    if (globalIdx == 0)
        out_tile_ranges[uint2(currtile, 0)] = 0;
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);
        if (currtile != prevtile)
        {
            out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }
    if (globalIdx == sorted_keys.size(0) - 1)
        out_tile_ranges[uint2(currtile, 1)] = sorted_keys.size(0);
}

void compute_tile_rangesv2(no_diff inout int64_t sorted_keys,
                           no_diff out int32_t out_tile_ranges)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;


    // Read tile ID from key. Update start/end of tile range if at limit.
    uint32_t currtile = uint32_t(uint64_t(sorted_keys) >> 32);
    if (globalIdx == 0)
        out_tile_ranges[uint2(currtile, 0)] = 0;
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);
        if (currtile != prevtile)
        {
            out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }
    if (globalIdx == sorted_keys.size(0) - 1)
        out_tile_ranges[uint2(currtile, 1)] = sorted_keys.size(0);
}