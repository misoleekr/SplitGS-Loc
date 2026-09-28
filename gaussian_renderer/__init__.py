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

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization

from scene.gaussian_model import GaussianModel


def render_gsplat_with_info(
    pc: GaussianModel, 
    viewpoint_camera, 
    width, 
    height, 
    **rasterize_args
):
    scales = pc.get_scaling
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    rotations = pc.get_rotation
    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).cuda()
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
        **rasterize_args
    )
    
    colors.sum().backward()
    render_visible_mask = means3D.grad.norm(dim=-1) > 0
    means3D.grad.zero_()
    
    weight_info = {
        "isect_to_gs": info["flatten_ids"].detach().long(),
        "isect_weights": info["weights"].detach(),
        "isect_pix_ids": info["pix_ids"].detach()
    }
    return {
        "render_visible_mask": render_visible_mask, # [N_GS, ]
        "viewspace_points": info["means2d"].detach().squeeze(0), # [N, 2],
        "weight_info": weight_info
    }


def render_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    override_color=None,
    rgb_only=True,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    longest_edge=640,
    **rasterize_args
):
    """
    Render the 3DGS scene.
    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).cuda() # [4, 4]
    
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")

    # calculate intrinsic matrix
    width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
    max_edge = max(width, height)
    if max_edge > longest_edge:
        factor = longest_edge / max_edge
        width, height = int(width * factor), int(height * factor)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # render color
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        **rasterize_args
    )
    
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image
    radii = info["radii"].squeeze(0)  # [N,]
    visible_mask = radii > 0
    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    # render feature map
    if rgb_only is False:
        loc_feature = pc.get_loc_feature[visible_mask].squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, alphas, meta = rasterization(
            means=means3D[visible_mask],  # [N, 3]
            quats=rotations[visible_mask],  # [N, 4]
            scales=scales[visible_mask],  # [N, 3]
            opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
            colors=loc_feature,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)
    else:
        feat_map = None

    return {
        "render": color,
        "feature_map": feat_map,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": render_alphas,
    }
