"""SSIM as a training loss -- the 'does the structure look similar?'
judge. Expressed as (1 - SSIM) so, like every other loss term here,
lower is better."""
import torch.nn as nn

from terraresolve.registry import LOSS_REGISTRY

try:
    from pytorch_msssim import SSIM as _SSIM
except ImportError:
    _SSIM = None


@LOSS_REGISTRY.register("ssim")
class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0, channels: int = 4):
        super().__init__()
        if _SSIM is None:
            raise ImportError(
                "pytorch-msssim is required for the SSIM loss: "
                "pip install pytorch-msssim"
            )
        self.ssim = _SSIM(data_range=data_range, channel=channels, size_average=True)

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred, target)
