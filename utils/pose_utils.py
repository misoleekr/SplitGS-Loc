import numpy as np
import cv2
import poselib
import math

def solve_pose_return_info(
    p2d,
    p3d,
    K,
    solver="poselib",
    reprojection_error=8.0,
    confidence=0.9999,
    max_iterations=100000,
    min_iterations=1000,
):
    match_num = p2d.shape[0]
    if match_num < 4:
        print("[SKIP] No enough matches")
        return np.eye(4, dtype=np.float32), np.array([])

    if solver == "poselib":
        camera = {
            "model": "PINHOLE",
            "width": int(K[0, 2] * 2),
            "height": int(K[1, 2] * 2),
            "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        }

        max_reproj_error = reprojection_error
        confidence = confidence

        pose, info = poselib.estimate_absolute_pose(
            p2d,
            p3d,
            camera,
            {
                "max_iterations": max_iterations,
                "min_iterations": min_iterations,
                "max_reproj_error": max_reproj_error,
                "success_prob": confidence,
            },
            {
                "verbose": False,
            },
        )

        out_info = {}
        if info["num_inliers"] > 0:
            w2c = pose.Rt
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0).astype(
                np.float32
            )
            inliers = info["inliers"]
            indices = np.where(inliers)[0]
            inliers = indices.reshape(-1, 1).astype(np.int32)
            
            out_info["inliers"] = inliers.flatten()
            out_info["model_score"] = info["model_score"]
            
            return w2c, out_info

    return np.eye(4, dtype=np.float32), {}


def cal_pose_error(pred_w2c, gt_w2c):
    """
    Calculate the pose error between the predicted pose and the ground truth pose.
    """
    pred_R = pred_w2c[:3, :3]
    pred_t = np.linalg.inv(pred_w2c)[:3, -1]
    gt_R = gt_w2c[:3, :3]
    gt_t = np.linalg.inv(gt_w2c)[:3, -1]

    # calculate angle error
    r_err = np.matmul(gt_R, np.transpose(pred_R))
    r_err = cv2.Rodrigues(r_err)[0]
    # Extract the angle.
    ae = np.linalg.norm(r_err) * 180 / math.pi

    # calculate translation error
    te = np.linalg.norm(pred_t - gt_t) * 100

    return float(ae), float(te)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

