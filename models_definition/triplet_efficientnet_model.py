import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

class TripletModel(nn.Module):
    def __init__(self, num_embeddings, num_identities):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        self.num_embeddings = num_embeddings
        self.num_identities = num_identities
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features=in_features,
                                                out_features=self.num_embeddings, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x