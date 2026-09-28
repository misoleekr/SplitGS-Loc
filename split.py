import warnings
import os
import sys
from argparse import ArgumentParser
import json
import logging
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

from arguments import ModelParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import seed_everything
from utils.aggregate_utils import render_weight_and_pix, update_lse, random_knn_score_faiss, sample_gaussians
from encoders.feature_extractor import FeatureExtractor

def aggregate(
    cameras,
    gaussians: GaussianModel,
    feature_extractor,
    masks=None,
    max_surv_gs=200000,
    side_beta = 1.4,
    w_thres=0.1,
    v_thres=1e-9,
    max_split=1,
    num_landmark_gs=32768,
    k=32,
):
    # ----------------------------------------------------------------------------
    # Stage 1-1: collecting weights and pixel indices & softmax
    logging.info(f"Stage 1: Collecting weights and pixel indices")
    _cameras = cameras.copy()
    num_img = len(_cameras)
    num_gs = gaussians.get_xyz.shape[0]
    H, W = _cameras[0].original_image.shape[1:3]
    logging.info("- Image size: {}x{}".format(W, H))
    logging.info(f"- Number of Gaussians: {num_gs}")
    
    # LogSumExp trick for gaussian-image weights matrix
    lse_max = torch.full((num_gs,), float('-inf')).cuda() # max(w_i)
    lse_sum = torch.zeros((num_gs,)).cuda() # sum(exp(w_i - c))
    score_sum = torch.zeros((num_gs,)).cuda() # sum(w_i)
    score_updated = torch.zeros((num_gs,)).cuda() # score = score_sum / cnt_updated
    weights = torch.zeros((max_surv_gs, num_img))
    gs_idxs = torch.full((max_surv_gs, num_img), -1, dtype=torch.int32)
    pix_idxs = torch.full((max_surv_gs, num_img), -1, dtype=torch.int32)
    
    for img_idx, cam in enumerate(tqdm(_cameras)):
        # render
        weight, pix_idx = render_weight_and_pix(
            gaussians=gaussians,
            cam=cam,
            W=W,
            H=H,
            masks=masks
        )
        
        # filter
        surv = (pix_idx != -1) & (weight > w_thres)
        surv_gs_idxs = torch.nonzero(surv, as_tuple=False).squeeze(1)
        num_surv = int(surv_gs_idxs.numel())
        
        # edge case: too many visible gaussians
        if num_surv > max_surv_gs:
            drop = num_surv - max_surv_gs
            surv_w = weight.index_select(0, surv_gs_idxs)
            if drop <= max_surv_gs:
                drop_local = torch.topk(surv_w, k=drop, largest=False, sorted=False).indices
                keep_local = torch.ones(num_surv, device=surv_gs_idxs.device, dtype=torch.bool)
                keep_local.index_fill_(0, drop_local, False)
                surv_gs_idxs = surv_gs_idxs[keep_local]
            else:
                keep_local = torch.topk(surv_w, k=max_surv_gs, largest=True, sorted=False).indices
                surv_gs_idxs = surv_gs_idxs.index_select(0, keep_local)
            num_surv = max_surv_gs
        
        # update LSE
        update_lse(
            lse_max, 
            lse_sum, 
            score_sum, 
            score_updated, 
            weight, 
            surv_gs_idxs
        )
        
        # update wieghts, gs_idxs, pix_idxs
        weights[:num_surv,img_idx] = weight.index_select(0, surv_gs_idxs).detach().cpu()
        gs_idxs[:num_surv,img_idx] = surv_gs_idxs.detach().cpu()
        pix_idxs[:num_surv,img_idx] = pix_idx.index_select(0, surv_gs_idxs).detach().cpu()
        
    # calculate lse
    lse = lse_max + torch.log(torch.clamp(lse_sum, min=1e-20)) # (num_gs, )
    total_surv = (score_updated > 0).sum().item()
    logging.info(f"- Weight > {w_thres}: {total_surv}/{num_gs} ({round(total_surv/num_gs*100)}%)")
    
    # ----------------------------------------------------------------------------
    # Stage 1-2: pruning gaussians
    logging.info(f"Stage 1-2: Pruning Gaussians with no weight updates")
    gaussians.prune_gaussians(score_updated == 0)
    
    # ----------------------------------------------------------------------------
    # Stage 2-0: Splitting
    logging.info(f"Stage 2-0: Split Gaussians into 3 along main axis")
    orig_num_gs = gaussians.get_xyz.shape[0]
    logging.info(f"- Number of Original Gaussians: {orig_num_gs}")
    
    split_cand = orig_num_gs
    for n_split in range(max_split):
        v_thres *= (10**n_split) # NOTE
        gaussians.split_large(beta=side_beta, center_lambda=2./3., v_thres=v_thres)
        # log
        num_split = (gaussians.get_xyz.shape[0] - gaussians.num_keeps[-1]) / 3
        split_gs_ratio = num_split / split_cand * 100
        logging.info(f"- [{n_split+1}] Number of Split Gaussians: {split_gs_ratio:.0f}%")
        split_cand = num_split
    
    num_gs = gaussians.get_xyz.shape[0]
    logging.info(f"- Total Number of Gaussians: {num_gs}")
    
    # ----------------------------------------------------------------------------
    # Stage 2-1: collecting weights and pixel indices & softmax
    logging.info(f"Stage 2-1: Collecting weights and pixel indices")
    if max_split  > 1:
        w_thres *= (0.5**(max_split-1)) # NOTE
    logging.info(f"- Weight threshold: {w_thres} after {max_split} splits")
    
    # LogSumExp trick for gaussian-image weights matrix
    lse_max = torch.full((num_gs,), float('-inf')).cuda() # max(w_i)
    lse_sum = torch.zeros((num_gs,)).cuda() # sum(exp(w_i - c))
    score_sum = torch.zeros((num_gs,)).cuda() # sum(w_i)
    score_updated = torch.zeros((num_gs,)).cuda() # score = score_sum / cnt_updated
    weights = torch.zeros((max_surv_gs, num_img))
    gs_idxs = torch.full((max_surv_gs, num_img), -1, dtype=torch.int32)
    pix_idxs = torch.full((max_surv_gs, num_img), -1, dtype=torch.int32)
    
    for img_idx, cam in enumerate(tqdm(_cameras)):
        # render
        weight, pix_idx = render_weight_and_pix(
            gaussians=gaussians,
            cam=cam,
            W=W,
            H=H,
            masks=masks
        )
        
        # filter
        surv = (pix_idx != -1) & (weight > w_thres)
        surv_gs_idxs = torch.nonzero(surv, as_tuple=False).squeeze(1)
        num_surv = int(surv_gs_idxs.numel())
        
        # edge case: too many visible gaussians
        if num_surv > max_surv_gs:
            drop = num_surv - max_surv_gs
            surv_w = weight.index_select(0, surv_gs_idxs)
            if drop <= max_surv_gs:
                drop_local = torch.topk(surv_w, k=drop, largest=False, sorted=False).indices
                keep_local = torch.ones(num_surv, device=surv_gs_idxs.device, dtype=torch.bool)
                keep_local.index_fill_(0, drop_local, False)
                surv_gs_idxs = surv_gs_idxs[keep_local]
            else:
                keep_local = torch.topk(surv_w, k=max_surv_gs, largest=True, sorted=False).indices
                surv_gs_idxs = surv_gs_idxs.index_select(0, keep_local)
            num_surv = max_surv_gs
        
        # update LSE
        update_lse(
            lse_max, 
            lse_sum, 
            score_sum, 
            score_updated, 
            weight, 
            surv_gs_idxs
        )
        
        # update wieghts, gs_idxs, pix_idxs
        weights[:num_surv,img_idx] = weight.index_select(0, surv_gs_idxs).detach().cpu()
        gs_idxs[:num_surv,img_idx] = surv_gs_idxs.detach().cpu()
        pix_idxs[:num_surv,img_idx] = pix_idx.index_select(0, surv_gs_idxs).detach().cpu()
        
    # calculate lse
    lse = lse_max + torch.log(torch.clamp(lse_sum, min=1e-20)) # (num_gs, )
    total_surv = (score_updated > 0).sum().item()
    total_updated = score_updated.sum().item()
    logging.info(f"- Weight > {w_thres}: {total_surv}/{num_gs} ({round(total_surv/num_gs*100)}%)")
    logging.info(f"- Weight Updates: {total_updated} ({round(total_updated/total_surv, 1)}feat/GS)")
    
    # average score
    score_updated[score_updated == 0] = 1 # avoid divide by zero
    score_avg = score_sum / score_updated  # (num_gs, )
    
    # ----------------------------------------------------------------------------
    # Stage 2-2: aggregate feature map into each gaussian
    logging.info(f"Stage 2-2: Aggregate feature map into each gaussian")
    feat_dim = feature_extractor.feature_dim
    gs_feats = torch.zeros((num_gs, feature_extractor.feature_dim)).cuda()
    for img_idx, cam in enumerate(tqdm(_cameras)):
        # load feature map
        gt_image = cam.original_image.cuda()
        gt_feature_map = feature_extractor(gt_image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0) # (256, 1080, 1920)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
        gt_feature_map = gt_feature_map.reshape(feat_dim, -1).transpose(1, 0) # (C, H, W) -> (H*W, C)
        
        # load weight and pixel index
        weight = weights[:, img_idx].cuda() # (max_surv_gs, )
        gs_idx = gs_idxs[:, img_idx].cuda() # (max_surv_gs, )
        pix_idx = pix_idxs[:, img_idx].cuda() # (max_surv_gs, )
        valid = (gs_idx != -1)
        
        if valid.any():
            w_softmax = weight[valid] # (num_valid, )
            g = gs_idx[valid] # (num_valid, )
            p = pix_idx[valid] # (num_valid, )
            w_softmax = torch.exp(w_softmax - lse[g]) # exp(w - logsumexp(w))
            gs_feats[g] += w_softmax.unsqueeze(1) * gt_feature_map[p] # (num_valid, C)
    
    # inject gs feats into real gs
    gaussians.set_loc_feature(gs_feats.unsqueeze(1)) # (num_gs, 1, C)
    logging.info(f"Aggregated features into {num_gs} Gaussians.")
    
    # ------------------------------------------------------------------------------
    # Stage 2-3: Sampling
    logging.info(f"Stage 2-3: Sample landmark gaussians")
    
    # free memory
    del _cameras, feature_extractor, masks
    del weights, gs_idxs, pix_idxs, lse_max, lse_sum, score_sum, score_updated, lse, gs_feats
    gc.collect()
    torch.cuda.empty_cache()
    
    # split -> origin
    xyz, score_avg = gaussians.collect_score(score_avg)

    # sampling
    logging.info(f"Sampling {num_landmark_gs} Gaussians with k={k}...")
    sampled_idx = random_knn_score_faiss(xyz, num_landmark_gs, score_avg, k=k, gpu=False)
    sampled_idx = torch.unique(sampled_idx)
    
    # origin -> split
    sampled_idx = gaussians.expand_idx(sampled_idx)
    return sampled_idx


if __name__ == "__main__":
    seed_everything(2026)
    parser = ArgumentParser(description="SplitGS-Loc mapping script")
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_dir", type=str, default="landmark")
    parser.add_argument("--max_surv_gs", type=int, default=200000)
    parser.add_argument("--side_beta", type=float, default=1.4)
    parser.add_argument("--weight_threshold", type=float, default=0.1)
    parser.add_argument("--volume_threshold", type=float, default=1e-9)
    parser.add_argument("--max_split", type=int, default=1)
    parser.add_argument("--num_landmark", type=int, default=32768)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument('--gpus', nargs="+", type=int, default=[0], help='device numbers of gpus to use')
    parser.add_argument("--log_level", type=str, default="info", help="logging level")
    args = parser.parse_args(sys.argv[1:])
    
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(
        os.path.join(args.save_dir, f"log.txt"), 
        mode="w", 
        encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        handlers=[file_handler, console_handler],
        force=True
    )
        
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    logging.info("Command: python " + " ".join(sys.argv))

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    masks = None
    masks_file = os.path.join(dataset.source_path, dataset.images, "masks.pkl")
    if os.path.exists(masks_file):
        logging.info(f"Loading masks from {masks_file}")
        masks = pickle.load(
            open(masks_file, "rb")
        )
    else:
        logging.info(f"No masks found in {masks_file}")

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        train_only=True
    )
    train_cameras = scene.getTrainCameras().copy()
    feature_extractor = FeatureExtractor(dataset.feature_type)
    
    # json
    cfg_path = os.path.join(args.model_path, "cfg_args.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_args_json = json.load(f)
        cfg_args_json["feature_type"] = dataset.feature_type
        with open(cfg_path, "w") as f:
            json.dump(cfg_args_json, f, indent=4)
    
    sampled_idx = aggregate(
        train_cameras,
        gaussians,
        feature_extractor,
        masks=masks,
        max_surv_gs=args.max_surv_gs,
        side_beta=args.side_beta,
        w_thres=args.weight_threshold,
        v_thres=args.volume_threshold,
        max_split=args.max_split,
        num_landmark_gs=args.num_landmark,
        k=args.landmark_k
    )
    logging.info(f"Sampled {len(sampled_idx)} landmarks.")
    
    logging.info("Saving feature gaussians...")
    os.makedirs(args.save_dir, exist_ok=True)
    gaussian_name = f"feat_gs.ply"
    gaussians.save_ply(os.path.join(args.save_dir, gaussian_name))
    logging.info(f"Saved to {os.path.join(args.save_dir, gaussian_name)}")
    
    logging.info("Saving landmark gaussians...")
    landmarks = sample_gaussians(gaussians, sampled_idx)
    landmark_name = f"landmark_gs.ply"
    landmarks.save_ply(os.path.join(args.save_dir, landmark_name))
    logging.info(f"Saved to {os.path.join(args.save_dir, landmark_name)}")
    
    logging.info("Finished.\n")