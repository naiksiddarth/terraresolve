"""Quick smoke test, no real data required. Confirms every registered
model can be built from config-shaped kwargs and run a forward pass,
and that the composite loss wires together correctly -- run this
after any change to verify the plug-and-play wiring before touching
real training data.

    python tests/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import terraresolve.losses  # noqa: F401
import terraresolve.models  # noqa: F401
from terraresolve.losses.composite import CompositeLoss
from terraresolve.registry import MODEL_REGISTRY


def test_models_forward():
    for name in MODEL_REGISTRY.names():
        model = MODEL_REGISTRY.build(name, in_channels=4, scale=4)
        x = torch.rand(1, 4, 32, 32)
        y = model(x)
        assert y.shape == (1, 4, 128, 128), f"{name} produced wrong shape: {tuple(y.shape)}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[ok] model '{name}' -> output {tuple(y.shape)}, {n_params:,} params")


def test_composite_loss():
    loss_fn = CompositeLoss({"l1": 1.0, "sam": 0.1})
    pred = torch.rand(2, 4, 32, 32, requires_grad=True)
    target = torch.rand(2, 4, 32, 32)
    total, parts = loss_fn(pred, target)
    total.backward()
    assert pred.grad is not None
    print(f"[ok] composite loss -> total={total.item():.4f}, parts={parts}")


if __name__ == "__main__":
    test_models_forward()
    test_composite_loss()
    print("\nAll smoke tests passed.")
