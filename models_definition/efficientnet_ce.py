import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2
import torch.nn.functional as F

class EfficientNetCE(nn.Module):
    def __init__(self, num_embeddings, num_identities):
        super().__init__()
        self.backbone = efficientnet_b2(weights=None)
        self.num_embeddings = num_embeddings
        self.num_identities = num_identities
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features=in_features,
                                                out_features=self.num_embeddings, bias=True)
        self.post_classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_embeddings, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=self.num_embeddings, out_features=self.num_identities)
        )

    def forward(self, x, return_embeddings=False):
        x = self.backbone(x)
        if return_embeddings:
            x = self.post_classifier[0](x)
            return F.normalize(x, p=2, dim=1)
        x = self.post_classifier(x)
        return x