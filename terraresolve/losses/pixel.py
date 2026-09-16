"""Plain pixel-level reconstruction losses -- the 'are the pixel
values close?' judge from the project's loss framework."""
import torch.nn as nn

from terraresolve.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("l1")
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.L1Loss()

    def forward(self, pred, target):
        return self.fn(pred, target)


@LOSS_REGISTRY.register("l2")
class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.MSELoss()

    def forward(self, pred, target):
        return self.fn(pred, target)
