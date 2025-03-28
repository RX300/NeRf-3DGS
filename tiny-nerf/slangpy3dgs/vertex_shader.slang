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
    rect_tile_space.min_x = int32_t(min(grid_width, max(0, int32_t((ndc_xy.x - radius) / tile_width))));
    rect_tile_space.min_y = int32_t(min(grid_height, max(0, int32_t((ndc_xy.y - radius) / tile_height))));
    rect_tile_space.max_x = int32_t(min(grid_width, max(0, int32_t((ndc_xy.x + radius + tile_width - 1) / tile_width))));
    rect_tile_space.max_y = int32_t(min(grid_height, max(0, int32_t((ndc_xy.y + radius + tile_height - 1) / tile_height))));
    // rect_tile_space.min_x = int32_t(floor(clip((ndc_xy.x - radius) / tile_width, 0, grid_width)));
    // rect_tile_space.min_y = int32_t(floor(clip((ndc_xy.y - radius) / tile_height, 0, grid_height)));
    // rect_tile_space.max_x = int32_t(ceil(clip((ndc_xy.x + radius) / tile_width, 0, grid_width)));
    // rect_tile_space.max_y = int32_t(ceil(clip((ndc_xy.y + radius) / tile_height, 0, grid_height)));

    //TODO: Clean this up, unintuivie math and code
    //rect_tile_space.max_x = clip((ndc_xy.x + radius + tile_width - 1)/tile_width, 0.0, grid_width);
    //rect_tile_space.max_y = clip((ndc_xy.y + radius + tile_height - 1)/tile_height, 0.0, grid_height);

    return rect_tile_space;
}

[Differentiable]
void preprocess_shader(
    // 输入参数
    no_diff int32_t g_idx,                  // 高斯点索引
    in float3 xyz_ws,                       // 世界空间中高斯点的位置
    in float[48] sh_coeffs,                 // 球谐系数
    inout float opacity,                    // 不透明度
    in float4 rotations,                    // 旋转四元数
    in float3 scales,                       // 缩放系数
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
    inout no_diff int32_t out_radii,              // 输出半径
    out float3 out_xyz_ndc,                     // 输出ndc空间位置
    out float out_depths,                       // 输出深度
    out float[6] out_cov3Ds,                    // 输出3D协方差矩阵
    out float3 out_rgb,                         // 输出gaussian的RGB颜色
    out float4 out_inv_cov_vs,                  // 输出视图空间逆协方差矩阵
    inout no_diff int32_t out_tiles_touched,      // 输出触及的瓦片数
    inout no_diff int32_t4 out_rect_tile_space, // 输出瓦片区域
    inout no_diff float2 pixelspace_uv,         // 输出像素空间坐标
    inout no_diff float3 testPointsVS,          // 输出测试点
    inout no_diff float4 p_hom_test,              // 输出测试点
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

    // 计算高斯均值投影到二维平面上的像素空间坐标
    float2 pixelspace_xy = {
        ndc2pix(splat.xyz_vs.x, image_width),
        ndc2pix(splat.xyz_vs.y, image_height)
    };
    pixelspace_uv = detach(pixelspace_xy);
    // 计算高斯覆盖的tile区域,用矩形表示
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
    out_rect_tile_space = int32_t4(rect_tile_space.min_x, rect_tile_space.min_y, rect_tile_space.max_x, rect_tile_space.max_y);
    // 存储视图空间位置
    out_xyz_ndc=splat.xyz_vs;

    // 存储深度
    out_depths = splat.xyz_vs.z;

    // 存储3D协方差矩阵（只存上三角部分）
    out_cov3Ds[0] = cov3D[0][0];
    out_cov3Ds[1] = cov3D[0][1];
    out_cov3Ds[2] = cov3D[0][2];
    out_cov3Ds[3] = cov3D[1][1];
    out_cov3Ds[4] = cov3D[1][2];
    out_cov3Ds[5] = cov3D[2][2];

    // 存储2D逆协方差矩阵
    out_inv_cov_vs = float4(inv_cov_vs[0][0], inv_cov_vs[0][1], inv_cov_vs[1][0], inv_cov_vs[1][1]);
    p_hom_test = detach(float4(inv_cov_vs[0][0], inv_cov_vs[0][1], inv_cov_vs[1][1], opacity*h_convolution_scaling));

    // 存储RGB颜色（应用h_convolution_scaling）
    opacity *= h_convolution_scaling; // 在完整实现中，应该使用实际的不透明度
    out_rgb = splat.rgb;
}

[Differentiable]
void preprocess_shader_release(
    // 输入参数
    no_diff int32_t g_idx,                  // 高斯点索引
    float3 xyz_ws,                          // 世界空间中高斯点的位置
    float[48] sh_coeffs,                    // 球谐系数
    inout float opacity,                    // 不透明度
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
    out no_diff int32_t out_radii,            // 输出半径
    out float3 out_xyz_ndc,                   // 输出ndc空间位置
    out float[6] out_cov3Ds,                  // 输出3D协方差矩阵
    out float3 out_rgb,                       // 输出RGB颜色
    out float4 out_inv_cov_vs,                // 输出视图空间逆协方差矩阵
    out no_diff int32_t out_tiles_touched,    // 输出触及的瓦片数
    out no_diff int32_t4 out_rect_tile_space, // 输出瓦片区域
)
{

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

    float3 p_orig = xyz_ws;
    float4 p_hom = mul(mul(proj_matv1, world_view_transformv1), float4(xyz_ws, 1.0f));
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

    // 计算高斯均值投影到二维平面上的像素空间坐标
    float2 pixelspace_xy = {
        ndc2pix(splat.xyz_vs.x, image_width),
        ndc2pix(splat.xyz_vs.y, image_height)
    };

    // 计算高斯覆盖的tile区域,用矩形表示
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
    out_xyz_ndc = splat.xyz_vs;

    // 存储3D协方差矩阵（只存上三角部分）
    out_cov3Ds[0] = cov3D[0][0];
    out_cov3Ds[1] = cov3D[0][1];
    out_cov3Ds[2] = cov3D[0][2];
    out_cov3Ds[3] = cov3D[1][1];
    out_cov3Ds[4] = cov3D[1][2];
    out_cov3Ds[5] = cov3D[2][2];

    // 存储2D逆协方差矩阵
    out_inv_cov_vs = float4(inv_cov_vs[0][0], inv_cov_vs[0][1], inv_cov_vs[1][0], inv_cov_vs[1][1]);
    // 存储RGB颜色（应用h_convolution_scaling）
    opacity *= h_convolution_scaling; // 在完整实现中，应该使用实际的不透明度
    out_rgb = splat.rgb;
}
