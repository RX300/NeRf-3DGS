import utils_2dgs;
import spherical_harmonics;

// pixel_xy是中心点坐标，radius是半径
rectangle get_rectangle_tile_space(
    float2 pixel_xy,
    float radius,
    uint grid_height,
    uint grid_width,
    uint tile_height,
    uint tile_width) {

    rectangle rect_tile_space;
    // 计算tile空间中的坐标范围，计算的是一个正方形
    rect_tile_space.min_x = int32_t(floor(clip((pixel_xy.x - radius) / tile_width, 0, grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip((pixel_xy.y - radius) / tile_height, 0, grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip((pixel_xy.x + radius) / tile_width, 0, grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip((pixel_xy.y + radius) / tile_height, 0, grid_height)));

    //TODO: Clean this up, unintuivie math and code
    //rect_tile_space.max_x = clip((ndc_xy.x + radius + tile_width - 1)/tile_width, 0.0, grid_width);
    //rect_tile_space.max_y = clip((ndc_xy.y + radius + tile_height - 1)/tile_height, 0.0, grid_height);

    return rect_tile_space;
}


[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void vertex_shader(DiffTensorView xyz_ws,
                   DiffTensorView sh_coeffs,
                   DiffTensorView rotations,
                   DiffTensorView scales,
                   DiffTensorView opcities,
                   uint active_sh,
                   TensorView<float> world_view_transform,
                   TensorView<float> proj_mat,
                   TensorView<float> cam_pos,
                   TensorView<int32_t> out_tiles_touched,
                   TensorView<int32_t> out_rect_tile_space,
                   TensorView<int32_t> out_radii,
                   DiffTensorView out_transMat,
                   DiffTensorView out_xyz_vs,
                   DiffTensorView out_inv_cov_vs,
                   DiffTensorView out_rgb,
                   DiffTensorView out_normal,
                   no_diff float fovy,
                   no_diff float fovx,
                   uint image_height,
                   uint image_width,
                   uint grid_height,
                   uint grid_width,
                   uint tile_height,
                   uint tile_width)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= xyz_ws.size(0))
        return;

    Camera cam = no_diff load_camera(world_view_transform, proj_mat, cam_pos, fovy, fovx, image_height, image_width);
    Gaussian_3D gauss = load_gaussian(g_idx, xyz_ws, sh_coeffs, rotations, scales, active_sh);
    Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam, active_sh);
    if (splat.xyz_vs.z <= 0.2) {
        return;
    }

    // float det = compute_det(splat.cov_vs);

    // if (det == 0.0f)
    //     return;
    // float radius = splat_radius(splat.cov_vs, det);

    // float2x2 g_inv_cov_vs = float2x2(splat.cov_vs[1][1], -splat.cov_vs[0][1], -splat.cov_vs[1][0], splat.cov_vs[0][0]) / det;
    
    // float2 pixelspace_xy = { ndc2pix(splat.xyz_vs.x, image_width), ndc2pix(splat.xyz_vs.y, image_height) };
    // rectangle rect_tile_space = get_rectangle_tile_space(pixelspace_xy,radius, grid_height, grid_width, tile_height, tile_width);
    // int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    // if (n_tiles == 0) {
    //     return;
    // }

    float3x3 T;
    float3 normal;
    compute_transmat(gauss.xyz_ws, gauss.scales.xy, 1.0f, gauss.rotations, cam.proj_mat, cam.world_view_transform, image_width, image_height, T, normal);
    out_transMat[uint2(g_idx, 0)] = T[0][0];
    out_transMat[uint2(g_idx, 1)] = T[0][1];
    out_transMat[uint2(g_idx, 2)] = T[0][2];
    out_transMat[uint2(g_idx, 3)] = T[1][0];
    out_transMat[uint2(g_idx, 4)] = T[1][1];
    out_transMat[uint2(g_idx, 5)] = T[1][2];
    out_transMat[uint2(g_idx, 6)] = T[2][0];
    out_transMat[uint2(g_idx, 7)] = T[2][1];
    out_transMat[uint2(g_idx, 8)] = T[2][2];

    float cutoff = 3.0f;
    float2 point_image;
    float radius;
    float2 extent;
    const float FilterSize = 0.707106; // sqrt(2) / 2
    bool ok = compute_aabb(T, cutoff, point_image, extent);
    if (!ok) return;
    radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));

    rectangle rect_tile_space = get_rectangle_tile_space(point_image,radius, grid_height, grid_width, tile_height, tile_width);
    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

    if (n_tiles == 0) {
        return;
    }

    out_radii[g_idx] = (uint32_t)radius;
    out_tiles_touched[g_idx] = n_tiles;
    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    out_xyz_vs.storeOnce(uint2(g_idx, 0), point_image.x);
    out_xyz_vs.storeOnce(uint2(g_idx, 1), point_image.y);
    out_xyz_vs.storeOnce(uint2(g_idx, 2), splat.xyz_vs.z);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 0), g_inv_cov_vs[0][0]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 1), g_inv_cov_vs[0][1]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 0), g_inv_cov_vs[1][0]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 1), g_inv_cov_vs[1][1]);
    out_rgb.storeOnce(uint2(g_idx, 0), splat.rgb.r);
    out_rgb.storeOnce(uint2(g_idx, 1), splat.rgb.g);
    out_rgb.storeOnce(uint2(g_idx, 2), splat.rgb.b);

    out_normal[uint2(g_idx, 0)] = normal.x;
    out_normal[uint2(g_idx, 1)] = normal.y;
    out_normal[uint2(g_idx, 2)] = normal.z;
    out_normal[uint2(g_idx, 3)] = opcities[uint(g_idx)];
}
