#include "Utils/NVAPI.slangh" // 引入 NVAPI 辅助头文件

struct TilesParamter
{
    StructuredBuffer<float3> xyz_vs;
    StructuredBuffer<int32_t4> rect_tile_space;
    StructuredBuffer<int32_t> index_buffer_offset;
    RWStructuredBuffer<int64_t> out_unsorted_keys;
    RWStructuredBuffer<int32_t> out_unsorted_gauss_idx;
    StructuredBuffer<int64_t> sorted_keys;
    RWStructuredBuffer<int32_t2> out_tile_ranges;
    uint grid_height;
    uint grid_width;
};
ParameterBlock<TilesParamter> tiles_paramter;
[numthreads(32, 1, 1)]
void generate_keys_main(uint3 tid: SV_DispatchThreadID)
{

    int globalIdx = tid.x;

    if (globalIdx >= tiles_paramter.xyz_vs.getCount())
        return;

    float3 ndc_xyz = tiles_paramter.xyz_vs[globalIdx];

    // Find this Gaussian's offset in buffer for writing keys/values.
    int offset;
    if (globalIdx == 0)
        offset = 0;
    else
        offset = tiles_paramter.index_buffer_offset[globalIdx - 1];

    int32_t4 rect_tile = tiles_paramter.rect_tile_space[globalIdx];

    int32_t rect_min_x = rect_tile.x;
    int32_t rect_min_y = rect_tile.y;
    int32_t rect_max_x = rect_tile.z;
    int32_t rect_max_y = rect_tile.w;

    for (int y = rect_min_y; y < rect_max_y; y++)
    {
        for (int x = rect_min_x; x < rect_max_x; x++)
        {
            uint64_t key = y * tiles_paramter.grid_width + x;
            key <<= 32;
            key = key | reinterpret<int32_t>(ndc_xyz.z);
            tiles_paramter.out_unsorted_keys[offset] = key;
            tiles_paramter.out_unsorted_gauss_idx[offset] = globalIdx;
            offset++;
        }
    }
}

[numthreads(32, 1, 1)]
void compute_tile_ranges_main(uint3 tid: SV_DispatchThreadID)
{
    int globalIdx = tid.x;

    if (globalIdx >= tiles_paramter.sorted_keys.getCount())
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint32_t currtile = uint32_t(uint64_t(tiles_paramter.sorted_keys[globalIdx]) >> 32);
    if (globalIdx == 0)
        tiles_paramter.out_tile_ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(tiles_paramter.sorted_keys[globalIdx - 1]) >> 32);
        if (currtile != prevtile)
        {
            tiles_paramter.out_tile_ranges[prevtile].y = globalIdx;
            tiles_paramter.out_tile_ranges[currtile].x = globalIdx;
        }
    }
    if (globalIdx == tiles_paramter.sorted_keys.getCount() - 1)
        tiles_paramter.out_tile_ranges[currtile].y = tiles_paramter.sorted_keys.getCount();
}