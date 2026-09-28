import torch
import torch.nn as nn

def simple_nms(scores, nms_radius: int):
    """
    Fast Non-maximum suppression to remove nearby points
    """
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    res = torch.where(max_mask, scores, zeros)
    return res

class KpDetector(torch.nn.Module):
    def __init__(self, in_dim):
        super(KpDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        x = self.cnn(feat_map)
        x = self.sigmoid(x)
        return x