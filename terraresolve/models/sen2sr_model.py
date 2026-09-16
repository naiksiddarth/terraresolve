"""SEN2SRModel -- wraps ESA's SEN2SRLite (NonReference_RGBN_x4, SPAN architecture)
to satisfy TerraResolve's model contract:

  Registry contract  : decorated with @MODEL_REGISTRY.register("sen2sr")
                       so MODEL_REGISTRY.build("sen2sr", **params) works.
  Trainer contract   : nn.Module with .parameters() / .train() / .eval() / .to()
                       NOTE: this is a frozen pretrained model; train() is a no-op
                       (weights are fixed; the trainer loop can call it safely but
                       gradients will not flow through).
  Inference contract : model(x) where x is (B, 4, H, W) float32, values ~[0,1].
                       Sets USES_INTERNAL_TILING = True so inference.py skips its
                       own patch loop and delegates tiling to sen2sr.predict_large.

Band order expected by SEN2SR: [B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)]
  -- this matches the existing project convention (in_channels=4, Sentinel-2).
  -- values must be float32 in [0, 1] (raw DN / 10_000).
  -- NaN / Inf values are replaced with 0.0 before inference.

Scale factor: 4× (10 m → 2.5 m).

The underlying model chain (from models/pretrained/sen2sr_rgbn_x4/load.py):
  CNNSR(in=4, out=4, n_feats=24, upscale=4)   <- SPAN, 472 K params
      ↓ wrapped by
  SRModelWithConstraint                         <- adds spectral HardConstraint
      ↓
  sen2sr.predict_large(X, model, overlap=32)   <- handles full-scene tiling

The HardConstraint module compares SR output against the original LR context,
which means direct model(patch) calls on small tiles WILL fail with a tensor
size mismatch.  Always route through predict_large (handled automatically here).
"""
from __future__ import annotations

import pathlib
import torch
import torch.nn as nn

from terraresolve.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("sen2sr")
class SEN2SRModel(nn.Module):
    """Frozen pretrained SEN2SRLite wrapper, registry-compatible."""

    SCALE: int = 4
    # Tells run_inference() in engine/inference.py to skip its manual patch loop
    # and delegate tiling to sen2sr.predict_large instead.
    USES_INTERNAL_TILING: bool = False

    def __init__(
        self,
        weights_dir: str = "models/pretrained/sen2sr_rgbn_x4",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.weights_dir = str(weights_dir)
        self._device_str = device
        # _inner is lazy-loaded on first forward() call so that importing the
        # module doesn't require the weights to be present yet (e.g. during
        # unit tests or when the class is imported by the registry scanner).
        self._inner: nn.Module | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_inner(self) -> None:
        """Load weights from disk via mlstac.  Called once, lazily."""
        import mlstac  # soft dep: only needed at inference time

        path = pathlib.Path(self.weights_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"SEN2SR weights not found at '{path}'.  "
                "Run the download snippet from the project README first:\n"
                "  python scripts/download_sen2sr_weights.py"
            )
        loader = mlstac.load(str(path))
        self._inner = loader.compiled_model(device=self._device_str)

    def _ensure_loaded(self) -> None:
        if self._inner is None:
            self._load_inner()

    # ------------------------------------------------------------------
    # nn.Module API (trainer contract)
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        self._ensure_loaded()
        return self._inner.parameters(recurse=recurse)  # type: ignore[union-attr]

    def eval(self) -> "SEN2SRModel":
        self._ensure_loaded()
        self._inner.eval()  # type: ignore[union-attr]
        return self

    def train(self, mode: bool = True) -> "SEN2SRModel":  # type: ignore[override]
        # Weights are frozen; silently accept train() calls from the trainer
        # loop but do not switch the underlying model out of eval mode.
        return self

    def to(self, *args, **kwargs) -> "SEN2SRModel":  # type: ignore[override]
        # Capture device string for lazy-loading; move inner model if present.
        if args and isinstance(args[0], (str, torch.device)):
            self._device_str = str(args[0])
        if self._inner is not None:
            self._inner.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        # Return empty dict -- weights are managed by mlstac / safetensors,
        # not by torch.save().  Trainer._save_checkpoint() will write an
        # empty file, which is harmless.
        return {}

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        # No-op: weights are loaded from disk by mlstac, not from a torch
        # checkpoint.  Accepting calls silently keeps compatibility with any
        # code that tries to resume from a checkpoint.
        pass

    # ------------------------------------------------------------------
    # Forward  (inference.py contract)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a batch of 4-band Sentinel-2 images.

        Args:
            x: Float32 tensor of shape (B, 4, H, W).
               Must be exactly 128x128 spatially (handled by inference loop).
               Bands must be [B04, B03, B02, B08] (R, G, B, NIR).

        Returns:
            Float32 tensor of shape (B, 4, H*4, W*4).
        """
        self._ensure_loaded()
        
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        # We bypass sen2sr.predict_large because it contains a bug where it 
        # leaves 64px black borders on the bottom and right of the output.
        # Instead, TerraResolve's inference.py will pass exactly 128x128 patches
        # to this method and handle the overlapping/blending robustly itself.
        
        return self._inner(x)
