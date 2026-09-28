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
import torch.nn.functional as F
import faiss

from gaussian_renderer import render_gsplat_with_info
from scene.gaussian_model import GaussianModel
from utils.loss_utils import *


@torch.no_grad()
def get_visible_center(
    means2d,
    H,
    W,
    render_visible_mask=None,
    fg_mask=None,
):
    # visible center
    xy = means2d
    visible_center = (
            (xy[:, 0] >= 0) & (xy[:, 0] < W) &
            (xy[:, 1] >= 0) & (xy[:, 1] < H)
    )
    
    # render mask
    if render_visible_mask is not None:
        visible_center = visible_center & render_visible_mask # (num_gs, ) bool

    # object mask
    if visible_center.any() and fg_mask is not None: # (1, H, W)
        x_in = torch.floor(xy[visible_center, 0]).to(torch.long)
        y_in = torch.floor(xy[visible_center, 1]).to(torch.long)
        img_mask_expand = torch.zeros_like(visible_center, dtype=torch.bool) # (num_gs, )
        img_mask_expand[visible_center] = fg_mask[0, y_in, x_in] # (num_visible, )
        visible_center = visible_center & img_mask_expand
        
    return visible_center


@torch.no_grad()
def get_weight_and_pix(
    num_gs,
    visible_center_mask: torch.Tensor, # (num_gs, ) bool
    weight_info: dict # {"isect_to_gs": (num_isect,), "isect_weights": (num_isect,), "isect_pix_ids": (isect_pix_ids,)}
):
    device = visible_center_mask.device
    isect_to_gs = weight_info["isect_to_gs"]
    isect_weights = weight_info["isect_weights"]
    isect_pix_ids = weight_info["isect_pix_ids"]

    # valid intersection
    valid = (isect_pix_ids != -1) & visible_center_mask[isect_to_gs]
    w_out = torch.zeros(num_gs, dtype=isect_weights.dtype, device=device)
    p_out = torch.full((num_gs,), -1, dtype=torch.int32, device=device)
    if not valid.any():
        return w_out, p_out
    g = isect_to_gs[valid]
    w = isect_weights[valid]
    p = isect_pix_ids[valid]

    # unique & max weight
    order = torch.argsort(w, descending=True, stable=True)
    g, w, p = g[order], w[order], p[order]
    order = torch.argsort(g, stable=True) # stable=True: keep the first max weight
    g, w, p = g[order], w[order], p[order]

    first = torch.ones_like(g, dtype=torch.bool)
    first[1:] &= (g[1:] != g[:-1])
    g, w, p = g[first], w[first], p[first]

    w_out[g] = w
    p_out[g] = p
    return w_out, p_out


def render_weight_and_pix(
    gaussians,
    cam,
    H,
    W,
    masks=None,
):
    # render
    render_pkg = render_gsplat_with_info(gaussians, cam, W, H)
    render_visible_mask = render_pkg["render_visible_mask"]
    means2d = render_pkg["viewspace_points"]
    weight_info = render_pkg["weight_info"]
    
    # object mask
    fg_mask = None
    if masks is not None:
        object_mask = masks[cam.image_name][0].cuda()[None]
        distort_mask = masks[cam.image_name][2].cuda()[None]
        mask = object_mask & distort_mask
        fg_mask = (
            F.interpolate(
                mask[None].float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5
        )

    # visible center mask
    visible_center_mask = get_visible_center(
        means2d,
        H, W,
        render_visible_mask=render_visible_mask,
        fg_mask=fg_mask,
    )
    
    # weight & pixel index
    num_gs = gaussians.get_xyz.shape[0]
    weight, pix_idx = get_weight_and_pix(
        num_gs,
        visible_center_mask,
        weight_info
    )
    
    return weight, pix_idx


def update_lse(
    lse_max, 
    lse_sum, 
    score_sum, 
    score_updated, 
    weight, 
    gs_idx
):
    # update LSE
    w = weight[gs_idx]
    m = lse_max[gs_idx]
    s = lse_sum[gs_idx]
    update = w > m

    # case1: w <= m
    lse_sum[gs_idx[~update]] = s[~update] + torch.exp(w[~update] - m[~update]) # w <= m

    # case2: w > m
    scale = torch.exp(m[update] - w[update])
    lse_sum[gs_idx[update]] = s[update] * scale + 1.0
    lse_max[gs_idx[update]] = w[update]
    
    # for scoring
    score_sum[gs_idx] += w
    score_updated[gs_idx] += 1
    return


def random_knn_score_faiss(points, npoints, score, k=32, gpu=True):
    sampled_idx = torch.randperm(points.shape[0], device=points.device)[:npoints]
    sampled_points = points[sampled_idx]
    points_np = points.detach().cpu().numpy()
    sampled_np = sampled_points.detach().cpu().numpy()

    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(3) 
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(points_np) # (N,3)
        _, knn_idx = gpu_index.search(sampled_np, k) # (npoints, k)
        knn_idx = torch.from_numpy(knn_idx).cuda()    
    else:
        index = faiss.IndexFlatL2(3)
        index.add(points_np) # (N,3)
        _, knn_idx = index.search(sampled_np, k) # (npoints, k)


    knn_score = score[knn_idx] # (npoints, k)
    score_knn_sort_idx = torch.argsort(knn_score, descending=True, dim=-1)
    
    final_sampled_idx = []
    used = set()
    valid_mask = score > 0
    for i in range(npoints):
        for j in score_knn_sort_idx[i]:
            idx = knn_idx[i, j].item()
            if valid_mask[idx] and idx not in used:
                used.add(idx)
                final_sampled_idx.append(idx)
                break
            
    return torch.tensor(list(final_sampled_idx)).cuda()


def sample_gaussians(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(3)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians
