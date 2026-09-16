"""Spectral Angle Mapper as a *training* loss -- protects band
ratios (e.g. the NIR/Red balance NDVI depends on) that a plain pixel
loss doesn't care about at all. This is the piece that backs up the
project's 'spectrally trustworthy, not just sharp' claim during
training itself, not only at evaluation time."""
import torch
import torch.nn as nn

from terraresolve.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("sam")
class SpectralAngleLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = (pred * target).sum(dim=1)
        pred_norm = pred.norm(dim=1).clamp_min(self.eps)
        target_norm = target.norm(dim=1).clamp_min(self.eps)
        cos_angle = (dot / (pred_norm * target_norm)).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle).mean()
