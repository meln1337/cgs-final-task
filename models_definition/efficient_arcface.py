import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

efficient_net_arcface = efficientnet_b2(weights=None)
efficient_net_arcface.classifier[1] = nn.Linear(in_features=efficient_net_arcface.classifier[1].in_features, out_features=512, bias=False)