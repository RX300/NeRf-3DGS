import utils;
import spherical_harmonics;
import "slangpy";

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float radius,
    uint grid_height,
    uint grid_width,
    uint tile_height,
    uint tile_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = int32_t(floor(clip((ndc_xy.x - radius) / tile_width, 0, grid_width)));
    rect_tile_space.min_y = int32_t(floor(clip((ndc_xy.y - radius) / tile_height, 0, grid_height)));
    rect_tile_space.max_x = int32_t(ceil(clip((ndc_xy.x + radius) / tile_width, 0, grid_width)));
    rect_tile_space.max_y = int32_t(ceil(clip((ndc_xy.y + radius) / tile_height, 0, grid_height)));

    //TODO: Clean this up, unintuivie math and code
    //rect_tile_space.max_x = clip((ndc_xy.x + radius + tile_width - 1)/tile_width, 0.0, grid_width);
    //rect_tile_space.max_y = clip((ndc_xy.y + radius + tile_height - 1)/tile_height, 0.0, grid_height);

    return rect_tile_space;
}

// [AutoPyBindCUDA]
// [CUDAKernel]
// [Differentiable]
// void vertex_shader(DiffTensorView xyz_ws,
//                    DiffTensorView sh_coeffs,
//                    DiffTensorView rotations,
//                    DiffTensorView scales,
//                    uint active_sh,
//                    TensorView<float> world_view_transform,
//                    TensorView<float> proj_mat,
//                    TensorView<float> cam_pos,
//                    TensorView<int32_t> out_tiles_touched,
//                    TensorView<int32_t> out_rect_tile_space,
//                    TensorView<int32_t> out_radii,
//                    float3 out_xyz_vs,
//                    DiffTensorView out_inv_cov_vs,
//                    float3 out_rgb,
//                    no_diff float fovy,
//                    no_diff float fovx,
//                    uint image_height,
//                    uint image_width,
//                    uint grid_height,
//                    uint grid_width,
//                    uint tile_height,
//                    uint tile_width)
// {
//     uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

//     if (g_idx >= xyz_ws.size(0))
//         return;

//     Camera cam = no_diff load_camera(world_view_transform, proj_mat, cam_pos, fovy, fovx, image_height, image_width);
//     Gaussian_3D gauss = load_gaussian(g_idx, xyz_ws, sh_coeffs, rotations, scales, active_sh);
//     Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam, active_sh);
//     if (splat.xyz_vs.z <= 0.2) {
//         return;
//     }

//     float det = compute_det(splat.cov_vs);

//     if (det == 0.0f)
//         return;
//     float radius = splat_radius(splat.cov_vs, det);


//     float2 pixelspace_xy = { ndc2pix(splat.xyz_vs.x, image_width), ndc2pix(splat.xyz_vs.y, image_height) };
//     rectangle rect_tile_space = get_rectangle_tile_space(pixelspace_xy,
//                                                          radius, grid_height, grid_width, tile_height, tile_width);
//     int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) * (rect_tile_space.max_y - rect_tile_space.min_y);

//     if (n_tiles == 0) {
//         return;
//     }

//     float2x2 g_inv_cov_vs = float2x2(splat.cov_vs[1][1], -splat.cov_vs[0][1], -splat.cov_vs[1][0], splat.cov_vs[0][0]) / det;

//     out_radii[g_idx] = (uint32_t)radius;
//     out_tiles_touched[g_idx] = n_tiles;
//     out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
//     out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
//     out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
//     out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

//     out_xyz_vs.storeOnce(uint2(g_idx, 0), splat.xyz_vs.x);
//     out_xyz_vs.storeOnce(uint2(g_idx, 1), splat.xyz_vs.y);
//     out_xyz_vs.storeOnce(uint2(g_idx, 2), splat.xyz_vs.z);
//     out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 0), g_inv_cov_vs[0][0]);
//     out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 1), g_inv_cov_vs[0][1]);
//     out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 0), g_inv_cov_vs[1][0]);
//     out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 1), g_inv_cov_vs[1][1]);
//     out_rgb.storeOnce(uint2(g_idx, 0), splat.rgb.r);
//     out_rgb.storeOnce(uint2(g_idx, 1), splat.rgb.g);
//     out_rgb.storeOnce(uint2(g_idx, 2), splat.rgb.b);
// }


[Differentiable]
void preprocess_shader(
    // 输入参数
    no_diff int32_t g_idx,                  // 高斯点索引
    float3 xyz_ws,                          // 世界空间中高斯点的位置
    float[48] sh_coeffs,                    // 球谐系数
    float4 rotations,                       // 旋转四元数
    float3 scales,                          // 缩放系数
    no_diff uint active_sh,                 // 活动的球谐阶数
    no_diff float scale_modifier,           // 缩放修正系数
    no_diff float[16] world_view_transform, // 世界到视图变换矩阵
    no_diff float[16] proj_mat,             // 投影矩阵
    no_diff float3 cam_pos,                 // 相机位置
    no_diff float fovy,                     // 视场角y
    no_diff float fovx,                     // 视场角x
    no_diff uint image_height,              // 图像高度
    no_diff uint image_width,               // 图像宽度
    no_diff bool prefiltered,               // 是否预过滤
    no_diff bool antialiasing,              // 是否抗锯齿
    // 输出参数
    out no_diff int32_t out_radii, // 输出半径
    out float3 out_xyz_ndc,         // 输出ndc空间位置
    out float out_depths,          // 输出深度
    out float[6] out_cov3Ds,       // 输出3D协方差矩阵
    out float3 out_rgb,            // 输出RGB颜色
    out float4 out_inv_cov_vs,     // 输出视图空间逆协方差矩阵
    out no_diff int32_t out_tiles_touched, // 输出触及的瓦片数
    out float2 pixelspace_uv, // 输出像素空间坐标
    out float3 testPointsVS,          // 输出测试点
    out float4 p_hom_test,              // 输出测试点
)
{
    // uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    // // 检查索引是否有效
    // if (g_idx >= xyz_ws.size(0))
    //     return;

    // 初始化半径和触及的瓦片数为0
    // out_radii[g_idx] = 0;
    // out_tiles_touched[g_idx] = 0;
    out_radii = 0;
    out_tiles_touched = 0;

    // 加载相机
    Camera cam = no_diff load_camerav2(
        world_view_transform,
        proj_mat,
        cam_pos,
        fovy,
        fovx,
        image_height,
        image_width
    );
    float4x4 world_view_transformv1 = float4x4(
        world_view_transform[0], world_view_transform[1], world_view_transform[2], world_view_transform[3],
        world_view_transform[4], world_view_transform[5], world_view_transform[6], world_view_transform[7],
        world_view_transform[8], world_view_transform[9], world_view_transform[10], world_view_transform[11],
        world_view_transform[12], world_view_transform[13], world_view_transform[14], world_view_transform[15]
    );
    float4x4 proj_matv1 = float4x4(
        proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3],
        proj_mat[4], proj_mat[5], proj_mat[6], proj_mat[7],
        proj_mat[8], proj_mat[9], proj_mat[10], proj_mat[11],
        proj_mat[12], proj_mat[13], proj_mat[14], proj_mat[15]
    );
    testPointsVS = mul(world_view_transformv1, float4(xyz_ws, 1.0f)).xyz;
    float3 p_orig = xyz_ws;
    float4 p_hom = mul(mul(proj_matv1,world_view_transformv1),float4(xyz_ws, 1.0f));
    p_hom_test = p_hom;
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 加载高斯点
    Gaussian_3D gauss = load_gaussianv2(g_idx, xyz_ws, sh_coeffs, rotations, scales, active_sh);

    // 应用缩放修正
    gauss.scales *= scale_modifier;

    // 投影高斯点到相机空间
    Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam, active_sh);

    // 近平面剔除
    if (splat.xyz_vs.z <= 0.2) {
        return;
    }

    // 计算协方差行列式
    float det = compute_det(splat.cov_vs);

    // 计算抗锯齿的h卷积缩放因子
    constexpr float h_var = 0.3f;
    float2x2 cov_plus_h = splat.cov_vs;
    cov_plus_h[0][0] += h_var;
    cov_plus_h[1][1] += h_var;
    splat.cov_vs = cov_plus_h;
    float det_cov_plus_h = compute_det(cov_plus_h);
    float h_convolution_scaling = 1.0f;

    if (antialiasing)
        h_convolution_scaling = sqrt(max(0.000025f, det / det_cov_plus_h)); // 确保数值稳定性
    det = det_cov_plus_h;
    // 如果行列式为零，说明投影后的高斯退化了，跳过
    if (det == 0.0f)
        return;
    // 计算高斯点的半径
    float radius = splat_radius(cov_plus_h, det_cov_plus_h);

    // 计算像素空间坐标
    float2 pixelspace_xy = {
        ndc2pix(splat.xyz_vs.x, image_width),
        ndc2pix(splat.xyz_vs.y, image_height)
    };
    pixelspace_uv = pixelspace_xy;
    // 计算矩形覆盖的瓦片区域
    rectangle rect_tile_space = get_rectangle_tile_space(
        pixelspace_xy,
        radius,
        image_height / 16, // 假设grid_height为image_height/16
        image_width / 16,  // 假设grid_width为image_width/16
        16, 16             // 假设tile尺寸为16x16
    );

    int32_t n_tiles = (rect_tile_space.max_x - rect_tile_space.min_x) *
                      (rect_tile_space.max_y - rect_tile_space.min_y);

    // 如果没有覆盖任何瓦片，跳过
    if (n_tiles == 0) {
        return;
    }

    // 计算视图空间协方差矩阵的逆
    float det_inv = 1.0f / det_cov_plus_h;
    float2x2 inv_cov_vs = float2x2(
        cov_plus_h[1][1] * det_inv, -cov_plus_h[0][1] * det_inv,
        -cov_plus_h[1][0] * det_inv, cov_plus_h[0][0] * det_inv
    );

    // 计算3D协方差矩阵并存储
    float3x3 cov3D = get_covariance_from_quat_scales(gauss.rotations, gauss.scales);

    // 将结果写入输出缓冲区
    // out_radii[g_idx] = int32_t(radius);
    // out_tiles_touched[g_idx] = n_tiles;
    float mid = 0.5f * (splat.cov_vs[0][0] + splat.cov_vs[1][1]);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    out_radii = int32_t(radius);
    out_tiles_touched = n_tiles;

    // 存储视图空间位置
    out_xyz_ndc=splat.xyz_vs;
    // out_xyz_vs.storeOnce(uint2(g_idx, 0), splat.xyz_vs.x);
    // out_xyz_vs.storeOnce(uint2(g_idx, 1), splat.xyz_vs.y);
    // out_xyz_vs.storeOnce(uint2(g_idx, 2), splat.xyz_vs.z);

    // 存储深度
    out_depths = splat.xyz_vs.z;
    //out_depths.storeOnce(uint2(g_idx, 0), splat.xyz_vs.z);

    // 存储3D协方差矩阵（只存上三角部分）
    out_cov3Ds[0] = cov3D[0][0];
    out_cov3Ds[1] = cov3D[0][1];
    out_cov3Ds[2] = cov3D[0][2];
    out_cov3Ds[3] = cov3D[1][1];
    out_cov3Ds[4] = cov3D[1][2];
    out_cov3Ds[5] = cov3D[2][2];
    // out_cov3Ds.storeOnce(uint2(g_idx, 0), cov3D[0][0]);
    // out_cov3Ds.storeOnce(uint2(g_idx, 1), cov3D[0][1]);
    // out_cov3Ds.storeOnce(uint2(g_idx, 2), cov3D[0][2]);
    // out_cov3Ds.storeOnce(uint2(g_idx, 3), cov3D[1][1]);
    // out_cov3Ds.storeOnce(uint2(g_idx, 4), cov3D[1][2]);
    // out_cov3Ds.storeOnce(uint2(g_idx, 5), cov3D[2][2]);

    // 存储2D逆协方差矩阵
    out_inv_cov_vs = float4(inv_cov_vs[0][0], inv_cov_vs[0][1], inv_cov_vs[1][0], inv_cov_vs[1][1]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 0), inv_cov_vs[0][0]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 1), inv_cov_vs[0][1]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 0), inv_cov_vs[1][0]);
    // out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 1), inv_cov_vs[1][1]);

    // 存储RGB颜色（应用h_convolution_scaling）
    float opacity = h_convolution_scaling; // 在完整实现中，应该使用实际的不透明度
    out_rgb = splat.rgb;
    // out_rgb.storeOnce(uint2(g_idx, 0), splat.rgb.r);
    // out_rgb.storeOnce(uint2(g_idx, 1), splat.rgb.g);
    // out_rgb.storeOnce(uint2(g_idx, 2), splat.rgb.b);
}

//===============================================fs shader===============================================
// // 定义块大小常量，与CUDA版本相同
// #ifndef BLOCK_X
// #define BLOCK_X 16
// #endif

// #ifndef BLOCK_Y
// #define BLOCK_Y 16
// #endif

// #define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
// #define NUM_CHANNELS 3

// // 主要渲染函数，每个线程块协作处理一个瓦片，每个线程处理一个像素
// [AutoPyBindCUDA]
// [CUDAKernel]
// [Differentiable]
// void render_shader<let CHANNELS = 3>(
//     // 输入参数
//     TensorView<uint2> ranges,           // 每个瓦片的高斯点范围
//     TensorView<uint32_t> point_list,    // 排序后的高斯点列表
//     uint image_width,                   // 图像宽度
//     uint image_height,                  // 图像高度
//     TensorView<float2> points_xy_image, // 2D投影点坐标
//     DiffTensorView features,            // 颜色特征
//     TensorView<float4> conic_opacity,   // 二次锥和不透明度
//     TensorView<float> depths,           // 深度值
//     TensorView<float> bg_color,         // 背景颜色

//     // 输出参数
//     DiffTensorView out_color,       // 输出颜色
//     TensorView<float> final_T,      // 最终透明度
//     TensorView<uint32_t> n_contrib, // 贡献高斯点数
//     DiffTensorView out_invdepth     // 输出深度
// )
// {
//     // 确定当前瓦片和对应的像素范围
//     uint32_t horizontal_blocks = (image_width + BLOCK_X - 1) / BLOCK_X;
//     uint2 pix_min = uint2(cudaBlockIdx().x * BLOCK_X, cudaBlockIdx().y * BLOCK_Y);
//     uint2 pix_max = uint2(min(pix_min.x + BLOCK_X, image_width), min(pix_min.y + BLOCK_Y, image_height));
//     uint2 pix = uint2(pix_min.x + cudaThreadIdx().x, pix_min.y + cudaThreadIdx().y);
//     uint32_t pix_id = image_width * pix.y + pix.x;
//     float2 pixf = float2(float(pix.x), float(pix.y));

//     // 检查此线程是否与有效像素关联
//     bool inside = pix.x < image_width && pix.y < image_height;
//     bool done = !inside;

//     // 加载要处理的ID范围
//     uint2 range = ranges[cudaBlockIdx().y * horizontal_blocks + cudaBlockIdx().x];
//     const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     int toDo = range.y - range.x;

//     // 分配存储空间，用于批量获取数据
//     __shared__ int collected_id[BLOCK_SIZE];
//     __shared__ float2 collected_xy[BLOCK_SIZE];
//     __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

//     // 初始化辅助变量
//     float T = 1.0f;
//     uint32_t contributor = 0;
//     uint32_t last_contributor = 0;
//     float C[CHANNELS];
//     for (int ch = 0; ch < CHANNELS; ch++)
//         C[ch] = 0.0f;

//     float expected_invdepth = 0.0f;

//     // 迭代批次直到全部完成或范围处理完毕
//     for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
//     {
//         // 如果整个块投票表示已完成渲染，则结束
//         int num_done = __syncthreads_count(done);
//         if (num_done == BLOCK_SIZE)
//             break;

//         // 集体将每个高斯点数据从全局内存获取到共享内存
//         int progress = i * BLOCK_SIZE + cudaThreadIdx().x + cudaThreadIdx().y * BLOCK_X;
//         if (range.x + progress < range.y)
//         {
//             int coll_id = point_list[range.x + progress];
//             collected_id[cudaThreadIdx().x + cudaThreadIdx().y * BLOCK_X] = coll_id;
//             collected_xy[cudaThreadIdx().x + cudaThreadIdx().y * BLOCK_X] = points_xy_image[coll_id];
//             collected_conic_opacity[cudaThreadIdx().x + cudaThreadIdx().y * BLOCK_X] = conic_opacity[coll_id];
//         }
//         __syncthreads();

//         // 迭代当前批次
//         for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
//         {
//             // 跟踪当前范围中的位置
//             contributor++;

//             // 使用二次锥矩阵重采样
//             float2 xy = collected_xy[j];
//             float2 d = float2(xy.x - pixf.x, xy.y - pixf.y);
//             float4 con_o = collected_conic_opacity[j];
//             // con_o.x:cov2d_inv[0,0], con_o.y:cov2d_inv[0,1], con_o.z:cov2d_inv[1,1], con_o.w:opacity
//             float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
//             if (power > 0.0f)
//                 continue;

//             // 计算alpha值（高斯不透明度和指数衰减的乘积）
//             float alpha = min(0.99f, con_o.w * exp(power));
//             if (alpha < 1.0f / 255.0f)
//                 continue;
//             float test_T = T * (1 - alpha);
//             if (test_T < 0.0001f)
//             {
//                 done = true;
//                 continue;
//             }

//             // 累积颜色贡献
//             for (int ch = 0; ch < CHANNELS; ch++)
//                 C[ch] += features.loadAs<float>(collected_id[j] * CHANNELS + ch) * alpha * T;

//             // 计算逆深度
//             expected_invdepth += (1.0f / depths[collected_id[j]]) * alpha * T;

//             T = test_T;

//             // 跟踪最后一个对该像素有贡献的高斯点
//             last_contributor = contributor;
//         }
//     }

//     // 所有处理有效像素的线程将最终渲染数据写入帧和辅助缓冲区
//     if (inside)
//     {
//         final_T[pix_id] = T;
//         n_contrib[pix_id] = last_contributor;
//         for (int ch = 0; ch < CHANNELS; ch++)
//             out_color.storeOnce(uint2(pix_id, ch), C[ch] + T * bg_color[ch]);

//         // 存储逆深度
//         out_invdepth.storeOnce(uint2(pix_id, 0), expected_invdepth);
//     }
// }

// // 特化版本，用于在编译时确定通道数
// [AutoPyBindCUDA]
// [CUDAKernel]
// [Differentiable]
// void render_shader_rgb(
//     // 输入参数
//     TensorView<uint2> ranges,
//     TensorView<uint32_t> point_list,
//     uint image_width,
//     uint image_height,
//     TensorView<float2> points_xy_image,
//     DiffTensorView features,
//     TensorView<float4> conic_opacity,
//     TensorView<float> depths,
//     TensorView<float> bg_color,

//     // 输出参数
//     DiffTensorView out_color,
//     TensorView<float> final_T,
//     TensorView<uint32_t> n_contrib,
//     DiffTensorView out_invdepth
// )
// {
//     render_shader<3>(
//         ranges,
//         point_list,
//         image_width,
//         image_height,
//         points_xy_image,
//         features,
//         conic_opacity,
//         depths,
//         bg_color,
//         out_color,
//         final_T,
//         n_contrib,
//         out_invdepth
//     );
// }
