import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # returns logits score per sample (higher = more likely to pick)
        return self.net(feats).squeeze(-1)