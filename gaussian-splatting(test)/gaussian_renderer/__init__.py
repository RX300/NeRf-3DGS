#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import Camera
from slangRenderers.GSRenderer import GSRenderer

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#         antialiasing=pipe.antialiasing
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             if separate_sh:
#                 dc, shs = pc.get_features_dc, pc.get_features_rest
#             else:
#                 shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     if separate_sh:
#         rendered_image, radii, depth_image = rasterizer(
#             means3D = means3D,
#             means2D = means2D,
#             dc = dc,
#             shs = shs,
#             colors_precomp = colors_precomp,
#             opacities = opacity,
#             scales = scales,
#             rotations = rotations,
#             cov3D_precomp = cov3D_precomp)
#     else:
#         rendered_image, radii, depth_image = rasterizer(
#             means3D = means3D,
#             means2D = means2D,
#             shs = shs,
#             colors_precomp = colors_precomp,
#             opacities = opacity,
#             scales = scales,
#             rotations = rotations,
#             cov3D_precomp = cov3D_precomp)
        
#     # Apply exposure to rendered image (training only)
#     if use_trained_exp:
#         exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
#         rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0).nonzero(),
#         "radii": radii,
#         "depth" : depth_image
#         }
    
#     return out

# 全局变量缓存GSRenderer实例
_gs_renderer_cache = None

def get_gs_renderer(image_height, image_width):
    """
    获取GSRenderer实例，如果尺寸匹配则复用，否则重新创建
    """
    global _gs_renderer_cache
    
    if (_gs_renderer_cache is None or 
        _gs_renderer_cache.image_height != image_height or 
        _gs_renderer_cache.image_width != image_width):
        
        print(f"Initializing GSRenderer with dimensions: {image_height}x{image_width}")
        _gs_renderer_cache = GSRenderer(
            image_height=int(image_height),
            image_width=int(image_width)
        )
    
    return _gs_renderer_cache

# 使用GSRenderer替换原先的cuda渲染
def render(viewpoint_camera:Camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene using GSRenderer.
    
    Background tensor (bg_color) must be on GPU!

    Notes on behavior compared to the original CUDA rasterizer:
    - This implementation uses GSRenderer from slangRenderers.GSRenderer.
    - 'pipe.convert_SHs_python': This flag is not directly used by GSRenderer for its internal SH evaluation.
      GSRenderer takes pc.get_features() as sh_coeffs.
    - 'pipe.compute_cov3D_python': This flag is not used as GSRenderer computes covariance internally based on scales and rotations.
    - 'override_color': If 'override_color' (N,3 RGB) is provided, it cannot be directly used as a substitute for 
      SH coefficients in GSRenderer's current pipeline, which expects SH coefficients (pc.get_features()).
      The behavior with 'override_color' will differ; this implementation will use pc.get_features() regardless.
      A more sophisticated handling of override_color would require changes to GSRenderer or pre-processing.
    - 'separate_sh': This flag is not used by GSRenderer as it takes combined SH features (pc.get_features()).
    - 'bg_color': The provided 'bg_color' is not explicitly passed to GSRenderer in this integration,
      as GSRenderer's run_shader API does not currently accept it. The background will likely be black or
      depend on GSRenderer's internal default.
    - 'radii' and 'depth': These are returned as dummy/placeholder values. GSRenderer.run_shader
      does not directly return radii or a depth map. The 'visibility_filter' is also based on dummy radii.
    """
 
    # Create zero tensor for viewspace_points output, consistent with original behavior.
    # This tensor might be used for gradient computation elsewhere if needed.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Initialize GSRenderer
    # GSRenderer is initialized with image dimensions.
    gs_renderer = get_gs_renderer(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width)
    )

    # Prepare inputs for GSRenderer
    means3D = pc.get_xyz
    rotations = pc.get_rotation
    scales = pc.get_scaling
    
    # Apply scaling_modifier directly to scales
    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier
        
    opacity = pc.get_opacity
    
    # GSRenderer expects SH coefficients. override_color is not directly compatible
    # without changing GSRenderer or how SH features are prepared.
    # For now, we always pass pc.get_features().
    # If override_color is present, its original intent (bypassing SH) is not met here.
    sh_coeffs = pc.get_features 
    if override_color is not None:
        # Log or handle this case if specific behavior for override_color is needed with GSRenderer
        # For now, it's ignored in favor of sh_coeffs from pc.get_features
        pass

    active_sh_degree = pc.active_sh_degree
    
    # Render using GSRenderer
    # GSRenderer.run_shader returns the rendered image (C, H, W)
    rendered_image, screenspace_points, radii = gs_renderer.run_shader(
        xyz_ws=means3D,
        rotations=rotations,
        scales=scales,
        opacity=opacity,
        sh_coeffs=sh_coeffs, 
        active_sh=active_sh_degree, # Corrected: pass active_sh_degree
        world_view_transform=viewpoint_camera.world_view_transform.T,
        proj_mat=viewpoint_camera.projection_matrix.T,
        cam_pos=viewpoint_camera.camera_center,
        fovy=viewpoint_camera.FoVy,
        fovx=viewpoint_camera.FoVx
    ) # Output is (C, H, W)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure_data = pc.get_exposure_from_name(viewpoint_camera.image_name)
        if exposure_data is not None:
            # Ensure rendered_image is (H, W, C) for matmul, then permute back
            # rendered_image is (C, H, W), permute to (H, W, C)
            rendered_image_c_last = rendered_image.permute(1, 2, 0) 
            # Apply 3x3 color transformation and offset
            exposed_image = torch.matmul(rendered_image_c_last, exposure_data[:3, :3]) + exposure_data[:3, 3] 
            rendered_image = exposed_image.permute(2, 0, 1) # Permute back to (C, H, W)

    # Clamp the rendered image to [0, 1]
    rendered_image = rendered_image.clamp(0, 1)
    
    # Visibility filter: based on dummy_radii, all Gaussians are marked as visible.
    # Original was (radii > 0).nonzero()
    visibility_filter = (radii > 0).nonzero(as_tuple=False) # Ensure (N,1) shape
    
    # Depth map: GSRenderer does not provide a depth map. Create a dummy one.
    # Original depth_image was (1, H, W)
    dummy_depth_image = torch.zeros((1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), 
                                    device=means3D.device, dtype=torch.float32)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # Still provide this as per original contract
        "visibility_filter" : visibility_filter,
        "radii": radii,
        "depth" : dummy_depth_image
        }
    
    return out
