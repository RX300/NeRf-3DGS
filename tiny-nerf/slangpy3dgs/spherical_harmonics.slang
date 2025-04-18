﻿// Copyright 2024 Google LLC
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

static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
static const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

struct SpherHarmCoeffs : IDifferentiable
{
    // Band 0
    float3 coeff0;

    // Band 1
    float3 coeff1;
    float3 coeff2;
    float3 coeff3;

    // Band2
    float3 coeff4;
    float3 coeff5;
    float3 coeff6;
    float3 coeff7;
    float3 coeff8;

    // Band3
    float3 coeff9;
    float3 coeff10;
    float3 coeff11;
    float3 coeff12;
    float3 coeff13;
    float3 coeff14;
    float3 coeff15;
};

[Differentiable]
SpherHarmCoeffs read_spherical_harmonics_coeffs(uint32_t g_idx, DiffTensorView sh_coeffs, uint32_t active_sh) {

    SpherHarmCoeffs g_sh_coeffs;
    g_sh_coeffs.coeff0 = float3(sh_coeffs[uint3(g_idx, 0, 0)], sh_coeffs[uint3(g_idx, 0, 1)], sh_coeffs[uint3(g_idx, 0, 2)]);
    
    if (active_sh > 0) {
      g_sh_coeffs.coeff1 = float3(sh_coeffs[uint3(g_idx, 1, 0)], sh_coeffs[uint3(g_idx, 1, 1)], sh_coeffs[uint3(g_idx, 1, 2)]);
      g_sh_coeffs.coeff2 = float3(sh_coeffs[uint3(g_idx, 2, 0)], sh_coeffs[uint3(g_idx, 2, 1)], sh_coeffs[uint3(g_idx, 2, 2)]);
      g_sh_coeffs.coeff3 = float3(sh_coeffs[uint3(g_idx, 3, 0)], sh_coeffs[uint3(g_idx, 3, 1)], sh_coeffs[uint3(g_idx, 3, 2)]);
      
      if (active_sh > 1) {
        g_sh_coeffs.coeff4 = float3(sh_coeffs[uint3(g_idx, 4, 0)], sh_coeffs[uint3(g_idx, 4, 1)], sh_coeffs[uint3(g_idx, 4, 2)]);
        g_sh_coeffs.coeff5 = float3(sh_coeffs[uint3(g_idx, 5, 0)], sh_coeffs[uint3(g_idx, 5, 1)], sh_coeffs[uint3(g_idx, 5, 2)]);
        g_sh_coeffs.coeff6 = float3(sh_coeffs[uint3(g_idx, 6, 0)], sh_coeffs[uint3(g_idx, 6, 1)], sh_coeffs[uint3(g_idx, 6, 2)]);
        g_sh_coeffs.coeff7 = float3(sh_coeffs[uint3(g_idx, 7, 0)], sh_coeffs[uint3(g_idx, 7, 1)], sh_coeffs[uint3(g_idx, 7, 2)]);
        g_sh_coeffs.coeff8 = float3(sh_coeffs[uint3(g_idx, 8, 0)], sh_coeffs[uint3(g_idx, 8, 1)], sh_coeffs[uint3(g_idx, 8, 2)]);

        if (active_sh > 2) {
          g_sh_coeffs.coeff9 = float3(sh_coeffs[uint3(g_idx, 9, 0)], sh_coeffs[uint3(g_idx, 9, 1)], sh_coeffs[uint3(g_idx, 9, 2)]);
          g_sh_coeffs.coeff10 = float3(sh_coeffs[uint3(g_idx, 10, 0)], sh_coeffs[uint3(g_idx, 10, 1)], sh_coeffs[uint3(g_idx, 10, 2)]);
          g_sh_coeffs.coeff11 = float3(sh_coeffs[uint3(g_idx, 11, 0)], sh_coeffs[uint3(g_idx, 11, 1)], sh_coeffs[uint3(g_idx, 11, 2)]);
          g_sh_coeffs.coeff12 = float3(sh_coeffs[uint3(g_idx, 12, 0)], sh_coeffs[uint3(g_idx, 12, 1)], sh_coeffs[uint3(g_idx, 12, 2)]);
          g_sh_coeffs.coeff13 = float3(sh_coeffs[uint3(g_idx, 13, 0)], sh_coeffs[uint3(g_idx, 13, 1)], sh_coeffs[uint3(g_idx, 13, 2)]);
          g_sh_coeffs.coeff14 = float3(sh_coeffs[uint3(g_idx, 14, 0)], sh_coeffs[uint3(g_idx, 14, 1)], sh_coeffs[uint3(g_idx, 14, 2)]);
          g_sh_coeffs.coeff15 = float3(sh_coeffs[uint3(g_idx, 15, 0)], sh_coeffs[uint3(g_idx, 15, 1)], sh_coeffs[uint3(g_idx, 15, 2)]);
        }
      }
    }
    return g_sh_coeffs;
}

[Differentiable]
SpherHarmCoeffs testNdArray(float[48] a)
{
    SpherHarmCoeffs g_sh_coeffs;
    return g_sh_coeffs;
}

[Differentiable]
SpherHarmCoeffs read_spherical_harmonics_coeffsv2(uint32_t g_idx, float[48] sh_coeffs, uint32_t active_sh) {

    SpherHarmCoeffs g_sh_coeffs;
    g_sh_coeffs.coeff0 = float3(sh_coeffs[0], sh_coeffs[1], sh_coeffs[2]);
    if (active_sh > 0)
    {
        g_sh_coeffs.coeff1 = float3(sh_coeffs[3], sh_coeffs[4], sh_coeffs[5]);
        g_sh_coeffs.coeff2 = float3(sh_coeffs[6], sh_coeffs[7], sh_coeffs[8]);
        g_sh_coeffs.coeff3 = float3(sh_coeffs[9], sh_coeffs[10], sh_coeffs[11]);
        if (active_sh > 1)
        {
            g_sh_coeffs.coeff4 = float3(sh_coeffs[12], sh_coeffs[13], sh_coeffs[14]);
            g_sh_coeffs.coeff5 = float3(sh_coeffs[15], sh_coeffs[16], sh_coeffs[17]);
            g_sh_coeffs.coeff6 = float3(sh_coeffs[18], sh_coeffs[19], sh_coeffs[20]);
            g_sh_coeffs.coeff7 = float3(sh_coeffs[21], sh_coeffs[22], sh_coeffs[23]);
            g_sh_coeffs.coeff8 = float3(sh_coeffs[24], sh_coeffs[25], sh_coeffs[26]);
            if (active_sh > 2)
            {
                g_sh_coeffs.coeff9 = float3(sh_coeffs[27], sh_coeffs[28], sh_coeffs[29]);
                g_sh_coeffs.coeff10 = float3(sh_coeffs[30], sh_coeffs[31], sh_coeffs[32]);
                g_sh_coeffs.coeff11 = float3(sh_coeffs[33], sh_coeffs[34], sh_coeffs[35]);
                g_sh_coeffs.coeff12 = float3(sh_coeffs[36], sh_coeffs[37], sh_coeffs[38]);
                g_sh_coeffs.coeff13 = float3(sh_coeffs[39], sh_coeffs[40], sh_coeffs[41]);
                g_sh_coeffs.coeff14 = float3(sh_coeffs[42], sh_coeffs[43], sh_coeffs[44]);
                g_sh_coeffs.coeff15 = float3(sh_coeffs[45], sh_coeffs[46], sh_coeffs[47]);
            }
        }
    }
    return g_sh_coeffs;
}

[Differentiable]
float3 compute_color_from_sh_coeffs(SpherHarmCoeffs sh, float3 g_xyz_ws, float3 cam_pos, uint32_t active_sh) {
    float3 dir = g_xyz_ws - cam_pos;
    dir = normalize(dir);

    float3 rgb = SH_C0 * sh.coeff0;
    if (active_sh > 0) {
        rgb = rgb - SH_C1 * dir.y * sh.coeff1 + SH_C1 * dir.z * sh.coeff2 - SH_C1 * dir.x * sh.coeff3;
        if (active_sh > 1)
        {
            float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
            float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;
            rgb = rgb +
                  SH_C2[0] * xy * sh.coeff4 +
                  SH_C2[1] * yz * sh.coeff5 +
                  SH_C2[2] * (2.0f * zz - xx - yy) * sh.coeff6 +
                  SH_C2[3] * xz * sh.coeff7 +
                  SH_C2[4] * (xx - yy) * sh.coeff8;

            if (active_sh > 2)
            {
                rgb = rgb +
                      SH_C3[0] * dir.y * (3.0f * xx - yy) * sh.coeff9 +
                      SH_C3[1] * xy * dir.z * sh.coeff10 +
                      SH_C3[2] * dir.y * (4.0f * zz - xx - yy) * sh.coeff11 +
                      SH_C3[3] * dir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh.coeff12 +
                      SH_C3[4] * dir.x * (4.0f * zz - xx - yy) * sh.coeff13 +
                      SH_C3[5] * dir.z * (xx - yy) * sh.coeff14 +
                      SH_C3[6] * dir.x * (xx - 3.0f * yy) * sh.coeff15;
            }
        }
    }

    rgb += 0.5;

    return max(rgb, 0.0);
}