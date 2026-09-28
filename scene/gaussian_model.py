#More actions
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn
import torch.nn.functional as F

from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._loc_feature = None
        self.num_keeps = []
        self.setup_functions()

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    def set_volume(self):
        scale = self.get_scaling
        # self.volume = (4.0 / 3.0) * torch.pi * scale[:, 0] * scale[:, 1] * scale[:, 2]
        self.volume = 8 * scale[:, 0] * scale[:, 1] * scale[:, 2] # axis-aligned box volume
        
    @property
    def get_volume(self):
        return self.volume
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def has_loc_feature(self) -> bool:
        return (isinstance(self._loc_feature, torch.Tensor))

    @property
    def get_loc_feature(self):
        return self._loc_feature  # 없으면 None
    
    def set_loc_feature(self, loc_feature):
        self._loc_feature = loc_feature.detach().contiguous().to(self._xyz.device)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add loc features
        if self.has_loc_feature():
            for i in range(self._loc_feature.shape[1]*self._loc_feature.shape[2]):
                l.append(f'loc_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        if self.has_loc_feature():
            loc_feature = self._loc_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # (N,D)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, loc_feature), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        prop_names = [p.name for p in plydata.elements[0].properties]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = sorted(
            [n for n in prop_names if n.startswith("f_rest_")],
            key=lambda x: int(x.split('_')[-1])
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        loc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("loc_")]
        if len(loc_names) > 0:
            loc_names = sorted(loc_names, key=lambda x: int(x.split('_')[-1]))
            loc_feature = np.stack([np.asarray(plydata.elements[0][n]) for n in loc_names], axis=1)  # (N,D)
            loc_feature = np.expand_dims(loc_feature, axis=-1)  # (N,D,1)
            self._loc_feature = torch.tensor(loc_feature, dtype=torch.float, device="cuda").transpose(1,2).contiguous()  # (N,1,D)
        else:
            self._loc_feature = None

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats_gsplat(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def prune_gaussians(self, mask):
        valid = ~mask

        self._xyz = nn.Parameter(self._xyz[valid].detach().contiguous(), requires_grad=True)
        self._features_dc = nn.Parameter(self._features_dc[valid].detach().contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(self._features_rest[valid].detach().contiguous(), requires_grad=True)
        self._opacity = nn.Parameter(self._opacity[valid].detach().contiguous(), requires_grad=True)
        self._scaling = nn.Parameter(self._scaling[valid].detach().contiguous(), requires_grad=True)
        self._rotation = nn.Parameter(self._rotation[valid].detach().contiguous(), requires_grad=True)

        # reset buffers
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        self._xyz = nn.Parameter(new_xyz, requires_grad=True)
        self._features_dc = nn.Parameter(new_features_dc, requires_grad=True)
        self._features_rest = nn.Parameter(new_features_rest, requires_grad=True)
        self._opacity = nn.Parameter(new_opacities, requires_grad=True)
        self._scaling = nn.Parameter(new_scaling, requires_grad=True)
        self._rotation = nn.Parameter(new_rotation, requires_grad=True)

        # reset buffers
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def collect_score(self, score_all):
        if not self.num_keeps:
            return self.get_xyz, score_all

        xyz_all = self.get_xyz
        orig_keep = self.num_keeps[0]
        xyz_new = xyz_all[:orig_keep]
        score_new = score_all[:orig_keep]
        collect_to_all = torch.arange(orig_keep, device=xyz_all.device)
        for depth, keep in enumerate(self.num_keeps):
            end = self.num_keeps[depth + 1] if depth + 1 < len(self.num_keeps) else xyz_all.shape[0]
            num_leaf = (end - keep) // 3
            assert (end - keep) % 3 == 0, (depth, keep, end)
            
            xyz_split = xyz_all[keep:end].reshape(num_leaf, 3, 3)
            xyz_split_center = xyz_split[:, 1, :] # center xyz == original xyz
            xyz_new = torch.cat([xyz_new, xyz_split_center], dim=0)
            
            score_split = score_all[keep:end].reshape(num_leaf, 3)
            score_split_avg = score_split.mean(dim=1)
            score_new = torch.cat([score_new, score_split_avg], dim=0)
            
            center_idx = keep + 1 + 3 * torch.arange(num_leaf, device=collect_to_all.device)
            collect_to_all = torch.cat([collect_to_all, center_idx], dim=0)
            
        self.collect_to_all = collect_to_all # for expand_idx() max Ncollect -> Nall
        return xyz_new, score_new
        
    def expand_idx(self, sampled_idx):
        if not self.num_keeps:
            return sampled_idx

        sampled_idx_center = self.collect_to_all[sampled_idx] # max Nnew -> max Nall
        keep_mask = sampled_idx_center < self.num_keeps[0]
        new_sampled_idx = sampled_idx_center[keep_mask]
        split_idx_center = sampled_idx_center[~keep_mask]
        
        split_idx_all = split_idx_center.unsqueeze(1).repeat(1, 3) # (Nsampled, 3)
        split_idx_all += torch.tensor([-1, 0, 1], device=sampled_idx.device, dtype=split_idx_all.dtype)
        split_idx_all = split_idx_all.reshape(-1)

        new_sampled_idx = torch.cat([new_sampled_idx, split_idx_all], dim=0)
        return new_sampled_idx
    
    def split_large(self, beta=1.4, center_lambda=2./3., v_thres=1e-3):
        # Select large gaussians
        self.set_volume()
        selected_pts_mask = self.get_volume >= v_thres
        
        # keep cursor
        if self.num_keeps: # not empty, split before
            selected_pts_mask[:self.num_keeps[-1]] = False # do not split already split ones (revived!)
        num_keep = (~selected_pts_mask).sum().item()
        if num_keep == self.get_xyz.shape[0]: # nothing to split
            return
        self.num_keeps.append(num_keep)

        # Get original attributes
        xyz = self.get_xyz # (N, 3)
        s_orig = self.get_scaling # (N, 3) scales (=stds)
        alpha_orig = self.get_opacity # (N, 1)
        R = build_rotation(self._rotation) # (N, 3, 3)
        
        # Keep original points that are not split
        keep_mask = ~selected_pts_mask
        keep_xyz = self._xyz[keep_mask]
        keep_features_dc = self._features_dc[keep_mask]
        keep_features_rest = self._features_rest[keep_mask]
        keep_rotation = self._rotation[keep_mask]
        keep_opacity = self._opacity[keep_mask]
        keep_scaling = self._scaling[keep_mask]
        
        # Select points to split
        xyz = xyz[selected_pts_mask] # (Nselect, 3)
        s_orig = s_orig[selected_pts_mask] # (Nselect, 3)
        alpha_orig = alpha_orig[selected_pts_mask] # (Nselect, 1)
        R = R[selected_pts_mask] # (Nselect, 3, 3)
        
        # Find major axis
        major_ax_indices = torch.argmax(s_orig, dim=1) # (Nselect,)
        onehot = F.one_hot(major_ax_indices, num_classes=3).to(s_orig.dtype) # (Nselect, 3)
        s_orig_major = torch.sum(s_orig * onehot, dim=1, keepdim=True) # (Nselect, 1)
        
        # Variables of Mixture of Gaussians
        side_lambda = (1. - center_lambda) / 2.
        new_var_factor = 1 - (1 - center_lambda) * (beta**2)
        assert new_var_factor > 0.0, f"beta must be in (0, sqrt(3)) to keep PSD, got {beta}"
        new_s_factor = new_var_factor ** 0.5
        
        # New opacity
        alpha_center = center_lambda * alpha_orig.clone()
        alpha_side = side_lambda * alpha_orig.clone()

        # New stds
        s_center = s_orig.clone()
        s_center = s_center * (1 - onehot) + new_s_factor * s_center * onehot # rescale major axis std, keep others
        s_side = s_center.clone()
        
        # New mean of side gaussians along major axis
        xyz_offsets = (beta * s_orig_major) * onehot # (Nselect, 1) * (Nselect, 3) = (Nselect, 3)
        xyz_offsets = torch.bmm(R, xyz_offsets.unsqueeze(-1)).squeeze(-1) # (Nselect, 3)
        
        # Create center & side gaussians that are split
        split_xyz = torch.stack((xyz - xyz_offsets, xyz, xyz + xyz_offsets), dim=1).reshape(-1, 3) # (3*Nselect, 3)
        split_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(3, dim=0)
        split_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(3, dim=0)
        split_rotation = self._rotation[selected_pts_mask].repeat_interleave(3, dim=0)
        split_alpha = torch.stack((alpha_side, alpha_center, alpha_side), dim=1).reshape(-1, 1) # (3*Nselect, 1)
        split_opacity = self.inverse_opacity_activation(split_alpha)
        split_scales = torch.stack((s_side, s_center, s_side), dim=1).reshape(-1, 3) # (3*Nselect, 3)
        split_scaling = self.scaling_inverse_activation(split_scales)
        
        # Concat
        new_xyz = torch.cat((keep_xyz, split_xyz), dim=0)
        new_features_dc = torch.cat((keep_features_dc, split_features_dc), dim=0)
        new_features_rest = torch.cat((keep_features_rest, split_features_rest), dim=0)
        new_rotation = torch.cat((keep_rotation, split_rotation), dim=0)
        new_opacity = torch.cat((keep_opacity, split_opacity), dim=0)
        new_scaling = torch.cat((keep_scaling, split_scaling), dim=0)
        
        # Replace all tensors & reset buffers
        self.replace_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
