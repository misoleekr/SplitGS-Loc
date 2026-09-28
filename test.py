import warnings
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.kpdetector import simple_nms
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose_return_info
from utils.test_utils import dual_softmax, mnn_match, topk_match, get_intrinsic


class SplitGSLoc:
    def __init__(self, landmark, config):
        self.landmarks = landmark
        self.config = config
        self.feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        self.longest_edge = config["longest_edge"]

    @torch.no_grad()
    def localize(self, query_image, fovx, fovy):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        query_fine_feature_map, scores = self.get_feature_map_scores(query_image)
        sparse_result = self.loc_sparse(query_fine_feature_map, scores, fovx, fovy)
        return {"sparse": sparse_result}

    @torch.no_grad()
    def loc_sparse(self, query_feature_map, heat_map, fovx, fovy):
        """
        feature_map: torch.Tensor, shape (C, H, W)
        """
        # detect
        H, W = query_feature_map.shape[-2:]

        kp_scores_after_nms = simple_nms(
            heat_map, self.config["sparse"].get("nms", 4)
        ).flatten()
        _, kp_ids = torch.topk(
            kp_scores_after_nms, self.config["sparse"].get("detect_num", 2048)
        )
        pos_mask = kp_scores_after_nms > 0
        kp_ids = kp_ids[pos_mask[kp_ids]]

        kp_mask = torch.zeros_like(kp_scores_after_nms, dtype=torch.bool)
        kp_mask[kp_ids] = True

        # sparse query features
        sampled_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[
            :, kp_mask
        ]

        # sparse landmark features
        landmark_features = F.normalize(
            self.landmarks.get_loc_feature.squeeze(), dim=-1
        )

        # sparse match
        corr_matrix = torch.matmul(sampled_features.T, landmark_features.T)

        # dual softmax
        if config["sparse"]["dual_softmax"] is True:
            corr_matrix = dual_softmax(
                corr_matrix=corr_matrix, temp=config["sparse"]["dual_softmap_temp"]
            )

        if config["sparse"]["mnn_match"] is True:
            # mnn match
            b_ids, im_idx, gs_ids = mnn_match(
                corr_matrix[None], thr=self.config["sparse"]["threshold"]
            )
        else:
            # topk match
            im_idx, gs_ids, val = topk_match(
                corr_matrix[None],
                self.config["sparse"]["topk"],
                thr=self.config["sparse"]["threshold"],
            )

        p2d = torch.stack([torch.arange(H * W) % W, torch.arange(H * W) // W], dim=1)

        p2d = p2d[kp_mask.cpu()][im_idx.cpu()].numpy()
        p3d = self.landmarks.get_xyz[gs_ids].cpu().numpy()

        K = get_intrinsic(fovx, fovy, W, H)

        pose_w2c, info = solve_pose_return_info(
            p2d + 0.5,
            p3d,
            K,
            self.config["sparse"]["solver"],
            self.config["sparse"]["reprojection_error"],
            self.config["sparse"]["confidence"],
            self.config["sparse"]["max_iterations"],
            self.config["sparse"]["min_iterations"],
        )
        inliers = info.get("inliers", np.array([]))
        
        # inlier's cosine sim
        corr_scores = corr_matrix[im_idx, gs_ids]
        cos_sim = corr_scores[inliers].mean().item() if inliers.shape[0] > 0 else 0.0

        return {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
            "cos_sim": cos_sim,
        }
        
    def get_feature_map_scores(self, image):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        fine_resolution = get_resolution_from_longest_edge(
            image.shape[-2], image.shape[-1], self.longest_edge
        )
        output = self.feature_extractor(image[None])
        feature_map = output["feature_map"]  # (1, C, H, W)
        scores = output["scores"] # (1, img_H, img_W)
        fine_feature_map = F.interpolate(
            feature_map, size=fine_resolution, mode="bilinear", align_corners=False
        )[0]
        fine_feature_map = F.normalize(fine_feature_map, p=2, dim=0)
        scores = F.interpolate(
            scores.unsqueeze(1), size=fine_resolution, mode="bilinear", align_corners=False
        )[0]
        
        return fine_feature_map, scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--prefix", default=None, type=str)
    parser.add_argument("--gs_path", default=None, type=str)
    parser.add_argument('--gpus', nargs="+", type=int, default=[0], help='device numbers of gpus to use')
    args = get_combined_args(parser)
    args.eval = True
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    print("GPU: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )

    scene_name = args.model_path.split("/")[-1]
    output_path = os.path.join("results", args.prefix, scene_name)
    os.makedirs(output_path, exist_ok=True)
    
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
        
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path

    yaml.dump(config, open(os.path.join(output_path, os.path.basename(args.cfg)), "w"))

    assert os.path.exists(args.gs_path), f"landmark path {args.gs_path} not exists"
    gaussian_path = args.gs_path
    
    scene = Scene(
        dataset,
        gaussians,
        gaussian_path=gaussian_path,
        shuffle=False,
        preload_cameras=False,
    )

    splitgsloc = SplitGSLoc(gaussians, config)

    test_cameras = scene.getTestCameras()

    results = []
    sparse_aes = []
    sparse_tes = []
    sparse_inliers = []

    for idx, camera_info in enumerate(tqdm(test_cameras, desc="SplitGS-Loc")):
        gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera_info.original_image.to("cuda")
        fovx = camera_info.FoVx
        fovy = camera_info.FoVy

        # localization
        loc_res = splitgsloc.localize(query_image, fovx, fovy)

        # evaluation
        sparse_ae, sparse_te = cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
        sparse_aes.append(sparse_ae)
        sparse_tes.append(sparse_te)
        sparse_inliers.append(loc_res["sparse"]["inliers"])
        
        loc_res["image_name"] = camera_info.image_name
        loc_res["sparse_AE"] = sparse_ae
        loc_res["sparse_TE"] = sparse_te

        loc_res["gt_pose_w2c"] = gt_w2c.tolist()
        results.append(loc_res)

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    sparse_inliers = np.array(sparse_inliers)

    results_summary = {
        "model_path": dataset.model_path,
        "sparse": {
            "median_ae": np.median(sparse_aes),
            "median_te": np.median(sparse_tes),
            "average_ae": np.mean(sparse_aes),
            "average_te": np.mean(sparse_tes),
            "recall_5m_10d": ((sparse_aes <= 10) & (sparse_tes <= 500)).sum()
            / len(sparse_aes),
            "recall_2m_5d": ((sparse_aes <= 5) & (sparse_tes <= 200)).sum()
            / len(sparse_aes),
            "recall_5cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 5)).sum()
            / len(sparse_aes),
            "recall_2cm_2d": ((sparse_aes <= 2) & (sparse_tes <= 2)).sum()
            / len(sparse_aes),
            "avg_inliers": sparse_inliers.mean(),
            
        },
    }

    print("Result Summary:")
    print(json.dumps(results_summary, indent=4))

    json.dump(
        results_summary, open(os.path.join(output_path, "summary.json"), "w"), indent=4
    )

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(output_path, "results.json"), "w"), indent=4)

    np.save(os.path.join(output_path, "sparse_aes.npy"), sparse_aes)
    np.save(os.path.join(output_path, "sparse_tes.npy"), sparse_tes)
    

    print("Result are saved in", output_path)
    print()