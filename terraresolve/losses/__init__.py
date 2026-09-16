from terraresolve.losses.composite import CompositeLoss, build_loss
from terraresolve.losses.pixel import L1Loss, L2Loss
from terraresolve.losses.spectral import SpectralAngleLoss
from terraresolve.losses.structural import SSIMLoss

__all__ = [
    "L1Loss", "L2Loss", "SSIMLoss", "SpectralAngleLoss",
    "CompositeLoss", "build_loss",
]
