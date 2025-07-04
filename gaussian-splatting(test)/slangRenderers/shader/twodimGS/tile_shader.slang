// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import utils_2dgs;
// generate_keys 对应 duplicateWithKeys
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
            key <<= 32;                                     // 高 32 位用于存储 tile id
            key = key | reinterpret<int32_t>(ndc_xyz.z);    // 低 32 位用于存储 深度 值
            out_unsorted_keys[offset] = key;                // 键 = tile id + 深度值
            out_unsorted_gauss_idx[offset] = globalIdx;     // 值 = 高斯点的索引 idx
            offset++;
        }
    }

}

// compute_tile_ranges 对应 identifyTileRanges
// out_tile_ranges 的格式为 [tile_id, 0] = start_idx, [tile_id, 1] = end_idx
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
    // 如果是第一个元素，初始化当前 tile 的起始索引为 0
    if (globalIdx == 0)
        out_tile_ranges[uint2(currtile, 0)] = 0;
    else
    {
        // 如果当前 tile ID 与前一个 tile ID 不同，更新前一个 tile 的结束索引
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);
        if (currtile != prevtile)
        {   // 更新前一个 tile 的结束索引
            out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            // 初始化当前 tile 的起始索引
            out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }
    // 如果是最后一个元素，更新当前 tile 的结束索引
    if (globalIdx == sorted_keys.size(0) - 1)
        out_tile_ranges[uint2(currtile, 1)] = sorted_keys.size(0);
}