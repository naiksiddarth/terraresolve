"""Combines any registered loss terms with config-defined weights:

loss:
  terms:
    l1: 1.0
    ssim: 0.2
    sam: 0.1

Add a new loss anywhere in the codebase, register it, and it becomes
usable here purely by name -- this file never needs to change."""
from __future__ import annotations
from typing import Dict

import torch.nn as nn

from terraresolve.registry import LOSS_REGISTRY


class CompositeLoss(nn.Module):
    def __init__(self, terms: Dict[str, float], term_kwargs: Dict[str, dict] | None = None):
        super().__init__()
        term_kwargs = term_kwargs or {}
        self.weights = terms
        self.modules_by_name = nn.ModuleDict({
            name: LOSS_REGISTRY.build(name, **term_kwargs.get(name, {}))
            for name in terms
        })

    def forward(self, pred, target):
        total = 0.0
        parts = {}
        for name, weight in self.weights.items():
            value = self.modules_by_name[name](pred, target)
            parts[name] = float(value.detach())
            total = total + weight * value
        return total, parts


def build_loss(cfg: dict) -> CompositeLoss:
    return CompositeLoss(cfg["terms"], cfg.get("term_kwargs", {}))
