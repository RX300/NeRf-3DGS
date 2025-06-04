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

import "../utils";


// __device__ inline uint32_t processTiles(
//     const float4 con_o, const float disc, const float t, const float2 p,
//     float2 bbox_min, float2 bbox_max,
//     float2 bbox_argmin, float2 bbox_argmax,
//     int2 rect_min, int2 rect_max,
//     const dim3 grid, const bool isY,
//     uint32_t idx, uint32_t off, float depth,
//     uint64_t *gaussian_keys_unsorted,
//     uint32_t *gaussian_values_unsorted
//     )
// {

//     // ---- AccuTile Code ---- //

//     // Set variables based on the isY flag
//     float BLOCK_U = isY ? BLOCK_Y : BLOCK_X;
//     float BLOCK_V = isY ? BLOCK_X : BLOCK_Y;

//     if (isY) {
//         rect_min = { rect_min.y, rect_min.x };
//         rect_max = { rect_max.y, rect_max.x };

//         bbox_min = { bbox_min.y, bbox_min.x };
//         bbox_max = { bbox_max.y, bbox_max.x };

//         bbox_argmin = { bbox_argmin.y, bbox_argmin.x };
//         bbox_argmax = { bbox_argmax.y, bbox_argmax.x };
//     }

//     uint32_t tiles_count = 0;
//     float2 intersect_min_line, intersect_max_line;
//     float ellipse_min, ellipse_max;
//     float min_line, max_line;

//     // Initialize max line
//     // Just need the min to be >= all points on the ellipse
//     // and  max to be <= all points on the ellipse
//     intersect_max_line = { bbox_max.y, bbox_min.y };

//     min_line = rect_min.x * BLOCK_U;
//     // Initialize min line intersections.
//     if (bbox_min.x <= min_line) {
//         // Boundary case
//         intersect_min_line = computeEllipseIntersection(
//             con_o, disc, t, p, isY, rect_min.x * BLOCK_U);
//     } else {
//         // Same as max line
//         intersect_min_line = intersect_max_line;
//     }

//     // Loop over either y slices or x slices based on the `isY` flag.
//     for (int u = rect_min.x; u < rect_max.x; ++u)
//     {
//         // Starting from the bottom or left, we will only need to compute
//         // intersections at the next line.
//         max_line = min_line + BLOCK_U;
//         if (max_line <= bbox_max.x) {
//             intersect_max_line = computeEllipseIntersection(
//                 con_o, disc, t, p, isY, max_line);
//         }

//         // If the bbox min is in this slice, then it is the minimum
//         // ellipse point in this slice. Otherwise, the minimum ellipse
//         // point will be the minimum of the intersections of the min/max lines.
//         if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
//             ellipse_min = bbox_min.y;
//         } else {
//             ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
//         }

//         // If the bbox max is in this slice, then it is the maximum
//         // ellipse point in this slice. Otherwise, the maximum ellipse
//         // point will be the maximum of the intersections of the min/max lines.
//         if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
//             ellipse_max = bbox_max.y;
//         } else {
//             ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
//         }

//         // Convert ellipse_min/ellipse_max to tiles touched
//         // First map back to tile coordinates, then subtract.
//         int min_tile_v = max(rect_min.y,
//                              min(rect_max.y, (int)(ellipse_min / BLOCK_V))
//             );
//         int max_tile_v = min(rect_max.y,
//                              max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
//             );

//         tiles_count += max_tile_v - min_tile_v;
//         // Only update keys array if it exists.
//         if (gaussian_keys_unsorted != nullptr) {
//             // Loop over tiles and add to keys array
//             for (int v = min_tile_v; v < max_tile_v; v++)
//             {
//                 // For each tile that the Gaussian overlaps, emit a
//                 // key/value pair. The key is |  tile ID  |      depth      |,
//                 // and the value is the ID of the Gaussian. Sorting the values
//                 // with this key yields Gaussian IDs in a list, such that they
//                 // are first sorted by tile and then by depth.
//                 uint64_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
//                 key <<= 32;
//                 key |= *((uint32_t *)&depth);
//                 gaussian_keys_unsorted[off] = key;
//                 gaussian_values_unsorted[off] = idx;
//                 off++;
//             }
//         }
//         // Max line of this tile slice will be min lin of next tile slice
//         intersect_min_line = intersect_max_line;
//         min_line = max_line;
//     }
//     return tiles_count;
// }

// __device__ inline uint32_t duplicateToTilesTouched(
//     const float2 p, const float4 con_o, const dim3 grid,
//     uint32_t idx, uint32_t off, float depth,
//     uint64_t *gaussian_keys_unsorted,
//     uint32_t *gaussian_values_unsorted
//     )
// {

//     //  ---- SNUGBOX Code ---- //

//     // Calculate discriminant
//     float disc = con_o.y * con_o.y - con_o.x * con_o.z;

//     // If ill-formed ellipse, return 0
//     if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) {
//         return 0;
//     }

//     // Threshold: opacity * Gaussian = 1 / 255
//     float t = 2.0f * log(con_o.w * 255.0f);

//     float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
//     x_term = (con_o.y < 0) ? x_term : -x_term;
//     float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
//     y_term = (con_o.y < 0) ? y_term : -y_term;

//     float2 bbox_argmin = { p.y - y_term, p.x - x_term };
//     float2 bbox_argmax = { p.y + y_term, p.x + x_term };

//     float2 bbox_min = {
//         computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
//         computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
//     };
//     float2 bbox_max = {
//         computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
//         computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
//     };

//     // Rectangular tile extent of ellipse
//     int2 rect_min = {
//         max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
//         max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y)))
//     };
//     int2 rect_max = {
//         max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
//         max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1)))
//     };

//     int y_span = rect_max.y - rect_min.y;
//     int x_span = rect_max.x - rect_min.x;

//     // If no tiles are touched, return 0
//     if (y_span * x_span == 0) {
//         return 0;
//     }

//     // If fewer y tiles, loop over y slices else loop over x slices
//     bool isY = y_span < x_span;
//     return processTiles(
//         con_o, disc, t, p,
//         bbox_min, bbox_max,
//         bbox_argmin, bbox_argmax,
//         rect_min, rect_max,
//         grid, isY,
//         idx, off, depth,
//         gaussian_keys_unsorted,
//         gaussian_values_unsorted
//     );
// }

inline uint32_t processTiles(
    const float3 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const int2 grid, const int2 tile, const bool isY,
    uint32_t idx, uint32_t off, float depth,
    TensorView<int64_t> gaussian_keys_unsorted,
    TensorView<int32_t> gaussian_values_unsorted
    )
{

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = isY ? tile.y : tile.x;
    float BLOCK_V = isY ? tile.x : tile.y;

    if (isY) {
        rect_min = { rect_min.y, rect_min.x };
        rect_max = { rect_max.y, rect_max.x };

        bbox_min = { bbox_min.y, bbox_min.x };
        bbox_max = { bbox_max.y, bbox_max.x };

        bbox_argmin = { bbox_argmin.y, bbox_argmin.x };
        bbox_argmax = { bbox_argmax.y, bbox_argmax.x };
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    // Just need the min to be >= all points on the ellipse
    // and  max to be <= all points on the ellipse
    intersect_max_line = { bbox_max.y, bbox_min.y };

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
        // Boundary case
        intersect_min_line = computeEllipseIntersection(
            con_o, disc, t, p, isY, rect_min.x * BLOCK_U);
    } else {
        // Same as max line
        intersect_min_line = intersect_max_line;
    }

    // Loop over either y slices or x slices based on the `isY` flag.
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
            intersect_max_line = computeEllipseIntersection(
                con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/max lines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/max lines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v = max(rect_min.y,
                             min(rect_max.y, (int)(ellipse_min / BLOCK_V))
            );
        int max_tile_v = min(rect_max.y,
                             max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
            );

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        // Loop over tiles and add to keys array
        for (int v = min_tile_v; v < max_tile_v; v++)
        {
            // For each tile that the Gaussian overlaps, emit a
            // key/value pair. The key is |  tile ID  |      depth      |,
            // and the value is the ID of the Gaussian. Sorting the values
            // with this key yields Gaussian IDs in a list, such that they
            // are first sorted by tile and then by depth.
            uint64_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
            key <<= 32;
            key |= *((int32_t *)&depth);
            gaussian_keys_unsorted[off] = key;
            gaussian_values_unsorted[off] = idx;
            off++;
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}

uint32_t duplicateToTilesTouched(
    const float2 p, const float opacity,const float3 con_o, const int2 grid,
    const int2 tile, uint32_t idx, uint32_t off, float depth,
    TensorView<int64_t> gaussian_keys_unsorted,
    TensorView<int32_t> gaussian_values_unsorted
    )
{
    
    //  ---- SNUGBOX Code ---- //

    // Calculate discriminant
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;

    // If ill-formed ellipse, return 0
    if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) {
        return 0;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(opacity * 255.0f);

    float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
    x_term = (con_o.y < 0) ? x_term : -x_term;
    float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
    y_term = (con_o.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = { p.y - y_term, p.x - x_term };
    float2 bbox_argmax = { p.y + y_term, p.x + x_term };

    float2 bbox_min = {
        computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
        computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
        computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
        computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    int2 rect_min = {
        max(0, min((int)grid.x, (int)(bbox_min.x / tile.x))),
        max(0, min((int)grid.y, (int)(bbox_min.y / tile.y)))
    };
    int2 rect_max = {
        max(0, min((int)grid.x, (int)(bbox_max.x / tile.x + 1))),
        max(0, min((int)grid.y, (int)(bbox_max.y / tile.y + 1)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return 0
    if (y_span * x_span == 0) {
        return 0;
    }

    // If fewer y tiles, loop over y slices else loop over x slices
    bool isY = y_span < x_span;
    return processTiles(
        con_o, disc, t, p,
        bbox_min, bbox_max,
        bbox_argmin, bbox_argmax,
        rect_min, rect_max,
        grid, tile, isY,
        idx, off, depth,
        gaussian_keys_unsorted,
        gaussian_values_unsorted
    );
}

// generate_keys 对应 duplicateWithKeys
[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<float> opacity,
                   TensorView<float> inv_cov_vs,
                   TensorView<int32_t> rect_tile_space,
                   TensorView<int32_t> index_buffer_offset,
                   TensorView<int64_t> out_unsorted_keys,
                   TensorView<int32_t> out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width,
                   uint tile_height,
                   uint tile_width,
                   uint image_height,
                   uint image_width)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= xyz_vs.size(0))
        return;

    float3 ndc_xyz = {
        xyz_vs[uint2(globalIdx, 0)],
        xyz_vs[uint2(globalIdx, 1)],
        xyz_vs[uint2(globalIdx, 2)]
    };
    float2 pixelspace_xy = {
        ndc2pix(ndc_xyz.x, image_width),
        ndc2pix(ndc_xyz.y, image_height)
    };
    float3 invCov2D = {
        inv_cov_vs[uint3(globalIdx, 0, 0)],
        inv_cov_vs[uint3(globalIdx, 0, 1)],
        inv_cov_vs[uint3(globalIdx, 1, 1)]
    };


    // Find this Gaussian's offset in buffer for writing keys/values.
    int32_t offset;
    if (globalIdx == 0)
        offset = 0;
    else
        offset = index_buffer_offset[globalIdx - 1];

    // int32_t rect_min_x = rect_tile_space[uint2(globalIdx, 0)];
    // int32_t rect_min_y = rect_tile_space[uint2(globalIdx, 1)];
    // int32_t rect_max_x = rect_tile_space[uint2(globalIdx, 2)];
    // int32_t rect_max_y = rect_tile_space[uint2(globalIdx, 3)];

    // for (int32_t y = rect_min_y; y < rect_max_y; y++)
    // {
    //     for (int32_t x = rect_min_x; x < rect_max_x; x++)
    //     {
    //         uint64_t key = y * grid_width + x;
    //         key <<= 32;                                     // 高 32 位用于存储 tile id
    //         key = key | reinterpret<int32_t>(ndc_xyz.z);    // 低 32 位用于存储 深度 值
    //         out_unsorted_keys[offset] = key;                // 键 = tile id + 深度值
    //         out_unsorted_gauss_idx[offset] = globalIdx;     // 值 = 高斯点的索引 idx
    //         offset++;
    //     }
    // }
    duplicateToTilesTouched(
        pixelspace_xy, opacity[globalIdx],
        invCov2D, int2(grid_width, grid_height), int2(tile_width, tile_height),
        globalIdx, offset, ndc_xyz.z,
        out_unsorted_keys, out_unsorted_gauss_idx
    );
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