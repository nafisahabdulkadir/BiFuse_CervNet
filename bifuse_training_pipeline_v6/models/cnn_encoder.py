# models/cnn_encoder.py
import torch
import torch.nn as nn
from torchvision import models
class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True, out_dim=512):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base          = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.proj     = nn.Linear(2048, out_dim)
        self._spatial = None
        self.features[-1].register_forward_hook(
            lambda m, i, o: setattr(self, '_spatial', o))
    def forward(self, x):
        feat   = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out    = self.proj(pooled)
        return out, self._spatial
