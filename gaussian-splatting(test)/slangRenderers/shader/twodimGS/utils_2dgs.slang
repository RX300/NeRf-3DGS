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

import spherical_harmonics;

static const float eps = 1e-7;

struct rectangle {
    int32_t min_x;
    int32_t min_y;
    int32_t max_x;
    int32_t max_y;
};

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}

[Differentiable]
float ndc2pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

[Differentiable]
float pix2ndc(float v, int S)
{
    return (2.0 * v + 1.0) / S - 1.0;
}

float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}


struct Camera : IDifferentiable
{
    float4x4 world_view_transform;
    float4x4 proj_mat;
    float3 position;
    float fovy;
    float fovx;
    int H;
    int W;
}

Camera load_camera(TensorView<float> world_view_transform_t, TensorView<float> proj_mat_t, TensorView<float> position_t, no_diff float fovy, no_diff float fovx, uint H, uint W) {
    float4x4 world_view_transform = float4x4(world_view_transform_t[uint2(0, 0)], world_view_transform_t[uint2(0, 1)], world_view_transform_t[uint2(0, 2)], world_view_transform_t[uint2(0, 3)],
                                             world_view_transform_t[uint2(1, 0)], world_view_transform_t[uint2(1, 1)], world_view_transform_t[uint2(1, 2)], world_view_transform_t[uint2(1, 3)],
                                             world_view_transform_t[uint2(2, 0)], world_view_transform_t[uint2(2, 1)], world_view_transform_t[uint2(2, 2)], world_view_transform_t[uint2(2, 3)],
                                             world_view_transform_t[uint2(3, 0)], world_view_transform_t[uint2(3, 1)], world_view_transform_t[uint2(3, 2)], world_view_transform_t[uint2(3, 3)]);

    float4x4 proj_mat = float4x4(proj_mat_t[uint2(0, 0)], proj_mat_t[uint2(0, 1)], proj_mat_t[uint2(0, 2)], proj_mat_t[uint2(0, 3)],
                                 proj_mat_t[uint2(1, 0)], proj_mat_t[uint2(1, 1)], proj_mat_t[uint2(1, 2)], proj_mat_t[uint2(1, 3)],
                                 proj_mat_t[uint2(2, 0)], proj_mat_t[uint2(2, 1)], proj_mat_t[uint2(2, 2)], proj_mat_t[uint2(2, 3)],
                                 proj_mat_t[uint2(3, 0)], proj_mat_t[uint2(3, 1)], proj_mat_t[uint2(3, 2)], proj_mat_t[uint2(3, 3)]);
    float3 position = float3(position_t[0], position_t[1], position_t[2]);

    return { world_view_transform, proj_mat, position, fovy, fovx, H, W};
}

[Differentiable]
float3 geom_transform_points(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz / (p_out.w + eps);
}

[Differentiable]
float3 geom_transform_points2(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz;
}

[Differentiable]
float3 project_point(float3 point, Camera cam) {
    float3 proj_point = geom_transform_points(point, mul(cam.proj_mat, cam.world_view_transform));
    float3 view_point = geom_transform_points2(point, cam.world_view_transform);
    proj_point.z = view_point.z;
    return proj_point;
}

[Differentiable]
float3x3 compute_jacobian(float3 xyz_ws, Camera cam) {
    float tan_half_fovx = tan(cam.fovx / 2.0);
    float tan_half_fovy = tan(cam.fovy / 2.0);
    float h_x = cam.W / (2.0 * tan_half_fovx);
    float h_y = cam.H / (2.0 * tan_half_fovy);

    float3 t = geom_transform_points(xyz_ws, cam.world_view_transform);

    // TODO: Clean this up, quite unintuive math and code
    const float limx = 1.3f * tan_half_fovx;
    const float limy = 1.3f * tan_half_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    float3x3 J = float3x3(h_x / t.z, 0.0, -(h_x * t.x) / (t.z * t.z),
                          0.0, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
                          0.0, 0.0, 0.0);

    return J;
}

[Differentiable]
float2x2 covariance_3d_to_2d(Camera cam, float3 xyz_ws, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));
    cov_vs[0][0] += 0.3;
    cov_vs[1][1] += 0.3;

    return float2x2(cov_vs);
}

struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
};

[Differentiable]
Gaussian_3D load_gaussian(int32_t g_idx,
                          DiffTensorView xyz_ws,
                          DiffTensorView sh_coeffs,
                          DiffTensorView rotations,
                          DiffTensorView scales,
                          uint active_sh)
{
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);
    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh);
    float4 g_rotations = read_t4_float4(g_idx, rotations);
    float3 g_scales = read_t3_float3(g_idx, scales);

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales };
}


struct Splat_2D_Vertex : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float2x2 cov_vs;
};

struct Splat_2D_NormalOpacity_UVW : IDifferentiable
{
    float3 TU;
    float3 TV;
    float3 TW;
    float4 normal_opacity; // normal in world space, opacity
};

float splat_radius(float2x2 cov_vs, float det) {
    float mid = 0.5f * (cov_vs[0][0] + cov_vs[1][1]);
    float eigen_val_1 = mid + sqrt(max(0.1f, mid * mid - det));
    float eigen_val_2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(eigen_val_1, eigen_val_2)));

    return radius;
}

[Differentiable]
float compute_det(float2x2 M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}


[Differentiable]
Splat_2D_Vertex load_splat_vertex(int32_t g_idx,
                                  DiffTensorView xyz_vs,
                                  DiffTensorView cov_vs,
                                  DiffTensorView rgb)
{
    float3 g_xyz_vs = read_t3_float3(g_idx, xyz_vs);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float2x2 g_cov_vs = read_t2x2_float2x2(g_idx, cov_vs);

    return { g_xyz_vs, g_rgb, g_cov_vs };
}

[Differentiable]
Splat_2D_Vertex project_gaussian_to_camera(Gaussian_3D g, Camera cam, uint active_sh) {
    float3 xyz_vs = project_point(g.xyz_ws, cam);
    if (xyz_vs.z <= 0.2) {
        return {float3(0.0), float3(0.0), float2x2(0.0)};
    }
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, g.xyz_ws, cam.position, active_sh);
    float3x3 cov_ws = get_covariance_from_quat_scales(g.rotations, g.scales);
    float2x2 cov_vs = covariance_3d_to_2d(cam, g.xyz_ws, cov_ws);

    return { xyz_vs, rgb, cov_vs };
}

struct Splat_2D_AlphaBlend : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float opacity;
    float2x2 inv_cov_vs;
};

[Differentiable]
Splat_2D_AlphaBlend load_splat_alphablend(int32_t g_idx,
                                          DiffTensorView xyz_vs,
                                          DiffTensorView inv_cov_vs,
                                          DiffTensorView opacity,
                                          DiffTensorView rgb)
{
    float3 g_xyz_vs = read_t3_float3(g_idx, xyz_vs);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float g_opacity = read_t1_float(g_idx, opacity);
    float2x2 g_inv_cov = read_t2x2_float2x2(g_idx, inv_cov_vs);

    return { g_xyz_vs, g_rgb, g_opacity, g_inv_cov };
}

[Differentiable]
Splat_2D_NormalOpacity_UVW load_splat_alphablend_2(int32_t g_idx,
                                                   DiffTensorView normal_opacity,
                                                   DiffTensorView transMats)
{
    // Load normal and opacity (4 components: normal.xyz + opacity)
    float4 g_normal_opacity = read_t4_float4(g_idx, normal_opacity);
    float3 g_Tu = float3(transMats[9 * g_idx + 0],
                         transMats[9 * g_idx + 1],
                         transMats[9 * g_idx + 2]);

    float3 g_Tv = float3(transMats[9 * g_idx + 3],
                         transMats[9 * g_idx + 4],
                         transMats[9 * g_idx + 5]);

    float3 g_Tw = float3(transMats[9 * g_idx + 6],
                         transMats[9 * g_idx + 7],
                         transMats[9 * g_idx + 8]);

    return { g_Tu, g_Tv, g_Tw, g_normal_opacity };
}

[Differentiable]
float4 evaluate_splat(Splat_2D_AlphaBlend g, 
                      float2 pix_coord,
                      uint32_t H,
                      uint32_t W)
{
    //这里是通过高斯公式计算
    float3 g_xyz = g.xyz_vs;
    float3 g_rgb = g.rgb;
    float g_opacity = g.opacity;
    float2x2 g_inv_cov_vs = g.inv_cov_vs;

    float2 d = { pix_coord.x - ndc2pix(g_xyz.x, W),
                 pix_coord.y - ndc2pix(g_xyz.y, H) };
    float power = -0.5f * (g_inv_cov_vs[0][0] * d.x * d.x +
                           g_inv_cov_vs[1][1] * d.y * d.y + (g_inv_cov_vs[0][1] + g_inv_cov_vs[1][0]) * d.x * d.y);
    float alpha = min(0.99f, g_opacity * exp(power));
    float3 premult_rgb = g_rgb * alpha;
    
    return float4(premult_rgb, alpha);
}

[Differentiable]
float4 evaluate_splat_2dgs(Splat_2D_AlphaBlend g,
                           float2 pix_coord,
                           uint32_t H,
                           uint32_t W,
                           const float3 Tu,
                           const float3 Tv,
                           const float3 Tw,
                           const float4 normal_opacity)
{
    // Fisrt compute two homogeneous planes, See Eq. (8)
    // Transform the two planes into local u-v system.
    float3 k = (float)pix_coord.x * Tw - Tu;
    float3 l = (float)pix_coord.y * Tw - Tv;
    // Cross product of two planes is a line, Eq. (9)
    float3 p = cross(k, l);
    if (p.z == 0.0) 
        return float4(0.0f, 0.0f, 0.0f, 0.0f); // 主要是让最后一个分量为0，跳过这个高斯的计算
    // Perspective division to get the intersection (u,v), Eq. (10)
    float2 s = { p.x / p.z, p.y / p.z };
    float rho3d = (s.x * s.x + s.y * s.y);
    // Add low pass filter
    float2 d = { g.xyz_vs.x - (float)pix_coord.x, g.xyz_vs.y - (float)pix_coord.y };
    const float FilterInvSquare = 2.0f;
    float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
    float rho = min(rho3d, rho2d);
    // compute intersection and depth
    float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
    if (depth < 0.2f)
        return float4(0.0f, 0.0f, 0.0f, 0.0f); // 主要是让最后一个分量为0，跳过这个高斯的计算
    float4 nor_o = normal_opacity;
    float normal[3] = { nor_o.x, nor_o.y, nor_o.z };
    float opa = nor_o.w;
    float power = -0.5f * rho;
    float alpha = min(0.99f, opa * exp(power));
    float3 premult_rgb = g.rgb * alpha;
    return float4(premult_rgb, alpha);
}

[Differentiable]
float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}

[Differentiable]
float3x3 quat_to_rotmat(float4 q) {
    float r = q[0], x = q[1], y = q[2], z = q[3];

    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));
}

[Differentiable]
float3x3 scale_to_mat(float2 scale, float mod) {
    return float3x3(
        scale.x * mod, 0, 0,
        0, scale.y * mod, 0,
        0, 0, 1.0f
    );
}

// [Differentiable]
// void compute_transmat(
//     float3 p_orig,
//     float2 scale,
//     float mod,
//     float4 rot,
//     float4x4 projmatrix,
//     float4x4 viewmatrix,
//     int W,
//     int H,
//     out float3x3 T,
//     out float3 normal
// ) {
//     // rot转换为旋转矩阵R
//     float3x3 R = quat_to_rotmat(rot);

//     // 函数将缩放向量 `scale` 和缩放因子 `mod` 转换为缩放矩阵 `S`
//     float3x3 S = scale_to_mat(scale, mod);

//     // 计算变换矩阵 `L`
//     float3x3 L = mul(R, S);

//     // 计算世界坐标到 NDC 的矩阵
//     // splat2world矩阵 将 Gaussians 的中心从局部坐标转换为相机坐标。
//     // 这里的splat2world对应论文中的H变换
//     float4x3 splat2world = float4x3(
//         L[0][0], L[0][1], p_orig.x,
//         L[1][0], L[1][1], p_orig.y,
//         L[2][0], L[2][1], p_orig.z,
//         0.0, 0.0, 1.0
//     );

//     // world2ndc矩阵 将世界坐标转换为 NDC坐标
//     // projmatrix已经是float4x4格式，直接使用
//     // world2ndc*ndc2pix对应论文中的W变换
//     float4x4 world2ndc = projmatrix;

//     // 矩阵将 NDC坐标转换为像素坐标
//     float4x3 ndc2pix = float4x3(
//         float(W)/2.0, 0.0, 0.0,
//         0.0, float(H)/2.0, 0.0,
//         0.0, 0.0, 0.0,
//         float(W-1)/2.0, float(H-1)/2.0, 1.0
//     );

//     // 计算传输矩阵 T
//     T = mul(transpose(ndc2pix), mul(world2ndc, splat2world));
    
//     // 计算法线 normal
//     float3 local_normal = float3(L[0][2], L[1][2], L[2][2]);
//     normal = mul(viewmatrix, float4(local_normal,0.0f)).xyz;
// }

[Differentiable]
void compute_transmat(
    float3 p_orig,
    float2 scale,
    float mod,
    float4 rot,
    float4x4 projmatrix,
    float4x4 viewmatrix,
    int W,
    int H,
    out float3x3 T,
    out float3 normal
) {
    // rot转换为旋转矩阵R
    float3x3 R = quat_to_rotmat(rot);

    // 函数将缩放向量 `scale` 和缩放因子 `mod` 转换为缩放矩阵 `S`
    float3x3 S = scale_to_mat(scale, mod);

    // 计算变换矩阵 `L`
    float3x3 L = mul(R, S);

    // 计算世界坐标到 NDC 的矩阵
    // splat2world矩阵 将 Gaussians 的中心从局部坐标转换为相机坐标。
    // 这里的splat2world对应论文中的H变换
    float4x4 splat2world = float4x4(
        L[0][0], L[0][1], 0.0, p_orig.x,
        L[1][0], L[1][1], 0.0, p_orig.y,
        L[2][0], L[2][1], 0.0, p_orig.z,
        0.0, 0.0, 0.0, 1.0
    );

    // world2ndc矩阵 将世界坐标转换为 NDC坐标
    // projmatrix已经是float4x4格式，直接使用
    // world2ndc*ndc2pix对应论文中的W变换
    float4x4 world2ndc = projmatrix;

    // 矩阵将 NDC坐标转换为像素坐标
    float4x4 ndc2pix = float4x4(
        float(W) / 2.0, 0.0, 0.0, float(W - 1) / 2.0,
        0.0, float(H) / 2.0, 0.0, float(H - 1) / 2.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    float4x4 temp = mul(ndc2pix, mul(world2ndc, splat2world));
    //float4x4 temp = mul(splat2world, mul(world2ndc, ndc2pix));
    T = float3x3(temp[0][0], temp[0][1], temp[0][3],
                 temp[1][0], temp[1][1], temp[1][3],
                 temp[3][0], temp[3][1], temp[3][3]);
    // T[0] = temp[0].xyw;
    // T[1] = temp[1].xyw;
    // T[2] = temp[3].xyw;
    // 计算法线 normal
    float3 local_normal = float3(L[0][2], L[1][2], L[2][2]);
    normal = mul(viewmatrix, float4(local_normal, 0.0f)).xyz;
}

// --------------------------------------------------------------------------------------------------------------------
// 该代码计算 2D 高斯分布的边界框和中心，并将中心用于创建低通滤波器
// --------------------------------------------------------------------------------------------------------------------
/*
`T`： 3x3 矩阵，用于转换图像坐标到特征空间。上一个函数已经计算了
* `cutoff`： 高斯分布的截止频率。
* `point_image`： 指标图像上的点坐标。
* `extent`： bounding box 的扩展。
*/
[Differentiable]
bool compute_aabb(
    float3x3 T,
    float cutoff,
    out float2 point_image,
    out float2 extent
)
{
    float3 t = float3(cutoff * cutoff, cutoff * cutoff, -1.0f);
    float d = dot(t, T[2] * T[2]);
    if (d == 0.0f) {
        return false; // 如果 d 为零，则无法计算 AABB
    }
    float3 f = (1 / d) * t;

    point_image = float2(
        dot(f, T[0] * T[2]),
        dot(f, T[1] * T[2])
    );
    float2 h0 = point_image * point_image - float2(dot(f, T[0] * T[0]), dot(f, T[1] * T[1]));
    float2 h = sqrt(max(h0, float2(0.00001f)));
    extent = float2(
        h.x,
        h.y
    );
    return true;
}