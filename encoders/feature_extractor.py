import torch.nn as nn
import torch
from encoders.sp_encoder.export_image_embeddings import SuperPoint

class FeatureExtractor(nn.Module):
    def __init__(self, feature_type):
        super(FeatureExtractor, self).__init__()
        self.feature_type = feature_type
        if feature_type == "sp":
            print("Loading SuperPoint model...")
            self.model = SuperPoint().cuda().eval()
            self.feature_dim = 256
        else:
            raise ValueError("Foundation model not supported")


    @torch.no_grad()
    def forward(self, image):
        features, scores = self.model(image)
        return {
            "feature_map": features,
            "scores": scores
        }
