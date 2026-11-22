"""Tiny temporal CNN over landmark features: (B, T, FEAT_DIM) -> (B, num_classes)."""
import torch, torch.nn as nn
from . import T_FRAMES
from .features import FEAT_DIM


class SignNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 192, drop: float = 0.3):
        super().__init__()
        def blk(i, o, k, d=1):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=d * (k - 1) // 2, dilation=d),
                                 nn.BatchNorm1d(o), nn.ReLU())
        self.net = nn.Sequential(blk(FEAT_DIM, width, 3), blk(width, width, 3, 2),
                                 blk(width, width, 3, 4), nn.Dropout(drop))
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))
