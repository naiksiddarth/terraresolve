"""EDSR -- Enhanced Deep Residual Network (Lim et al., CVPRW 2017),
the CNN baseline. Simplified/kept light on purpose, and generalised
to N input bands (default 4: Sentinel-2 B2/B3/B4/B8) instead of RGB
only, since spectral fidelity across all four bands is the point."""
from __future__ import annotations
import torch
import torch.nn as nn

from terraresolve.registry import MODEL_REGISTRY


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class PixelShuffleUpsampler(nn.Sequential):
    """Decomposes an arbitrary integer scale into factors of 2 (and,
    if needed, one factor of 3) so x2/x3/x4/x6/x8... all work."""

    def __init__(self, scale: int, n_feats: int):
        layers = []
        s = scale
        while s % 2 == 0:
            layers += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
            s //= 2
        if s == 3:
            layers += [nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1), nn.PixelShuffle(3)]
        elif s != 1:
            raise ValueError(f"Unsupported scale factor: {scale}")
        super().__init__(*layers)


@MODEL_REGISTRY.register("edsr")
class EDSR(nn.Module):
    def __init__(self, in_channels: int = 4, n_feats: int = 64,
                 n_resblocks: int = 16, scale: int = 4, res_scale: float = 0.1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.upsample = PixelShuffleUpsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body(feat)
        feat = self.upsample(feat)
        return self.tail(feat)
