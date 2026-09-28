import numpy as np
import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal

def mnn_match(corr_matrix, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    mask = corr_matrix > thr
    mask = (
        mask
        * (corr_matrix == corr_matrix.max(dim=-1, keepdim=True)[0])
        * (corr_matrix == corr_matrix.max(dim=-2, keepdim=True)[0])
    )
    b_ids, i_ids, j_ids = torch.where(mask)
    return b_ids.squeeze(), i_ids.squeeze(), j_ids.squeeze()


def topk_match(corr_matrix, topk, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    N_im = corr_matrix.shape[-2]
    val, idx = torch.topk(corr_matrix, topk, dim=-1)
    val_flattened = val.flatten(1)
    idx_flattened = idx.flatten(1)
    mask = val_flattened > thr
    arange_tensor = torch.arange(N_im, device=corr_matrix.device)
    idx_im = arange_tensor[None].repeat(corr_matrix.shape[0], topk)[mask]
    idx_gs = idx_flattened[mask]
    val = val_flattened[mask]

    return idx_im, idx_gs, val


def dual_softmax(corr_matrix, temp=1):
    corr_matrix = corr_matrix / temp
    corr_matrix = F.softmax(corr_matrix, dim=-2) * F.softmax(corr_matrix, dim=-1)
    return corr_matrix


def get_intrinsic(fovx, fovy, width, height):
    focalX = fov2focal(fovx, width)
    focalY = fov2focal(fovy, height)
    K = np.array(
        [
            [focalX, 0.0, width / 2],
            [0.0, focalY, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K
