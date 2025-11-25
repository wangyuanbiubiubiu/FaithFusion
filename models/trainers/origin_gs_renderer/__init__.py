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
from modified_diff_gaussian_rasterization import GaussianRasterizer as ModifiedGaussianRasterizer, GaussianRasterizationSettings
from models.gaussians.basics import *

def getProjectionMatrix(znear, zfar, fovX, fovY, cx, cy):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = cx
    P[1, 2] = cy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
    

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf values found in {name}")

def modified_render(gs: dataclass_gs, cam: dataclass_camera, opaticy_mask: torch.Tensor, 
                    is_train_set: bool, override_color = None, zfar=0.01, znear=100.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs.means, dtype=gs.means.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    FovX = focal2fov(cam.Ks[0, 0], int(cam.W))
    FovY = focal2fov(cam.Ks[1, 1], int(cam.H))
    cx = (cam.Ks[0, 2] - int(cam.W) / 2) / int(cam.W) * 2
    cy = (cam.Ks[1, 2] - int(cam.H) / 2) / int(cam.H) * 2

    # Set up rasterization configuration
    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    world_view_transform = cam.camtoworlds.inverse().transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY,
                                            cx=cx, cy=cy).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.H),
        image_width=int(cam.W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=1,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

    means3D = gs.means
    means2D = screenspace_points
    opacity = gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze()
    opacity = opacity.unsqueeze(-1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    cov3D_precomp = None
    scales = gs.scales
    rotations = gs.quats

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if is_train_set or override_color is None: #因为要求H矩阵，所以必须要shs形式计算
        shs = gs.shs
        shs.retain_grad()
    else:
        if override_color is not None:
            # to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1, ), dtype=x.dtype, device=x.device)], dim=-1)
            # pts3d_homo = to_homo(means3D)
            # pts3d_cam = pts3d_homo @ world_view_transform
            # gaussian_depths = pts3d_cam[:, 2, None]
            # colors_precomp = override_color * gaussian_depths.clamp(min=0)
            colors_precomp = override_color
        else:
            colors_precomp = gs.rgbs

    means3D.retain_grad()
    opacity.retain_grad()
    scales.retain_grad()
    rotations.retain_grad()
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, radii, pixel_gaussian_counter = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    params_output = {
        'means': means3D,
        'rotations': rotations,
        'scales': scales,
        'opacities': opacity,
        'shs': shs,
    }
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "pixel_gaussian_counter": pixel_gaussian_counter,
            "opacity": depth,
            "params_output": params_output
            }
