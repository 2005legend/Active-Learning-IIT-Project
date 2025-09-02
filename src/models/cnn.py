import torch
import torch.nn as nn
from torchvision import models

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        n = m.fc.in_features
        m.fc = nn.Linear(n, 2)
        self.model = m
    
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        # features from penultimate layer (before final FC)
        modules = list(self.model.children())[:-1]  # upto avgpool
        backbone = nn.Sequential(*modules)
        with torch.no_grad():
            feats = backbone(x)
            feats = feats.view(feats.size(0), -1)
        return feats