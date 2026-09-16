"""Model/loss/dataset-agnostic training loop. Everything it uses
comes out of a registry via config strings, so swapping the model
architecture, the loss recipe, or the dataset means editing the YAML
config -- this file never has to change."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from terraresolve.losses.composite import build_loss
from terraresolve.registry import DATASET_REGISTRY, MODEL_REGISTRY


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        want_cpu = cfg.get("device", "auto") == "cpu"
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not want_cpu) else "cpu")

        model_cfg = cfg["model"]
        self.model = MODEL_REGISTRY.build(model_cfg["name"], **model_cfg.get("params", {})).to(self.device)

        self.loss_fn = build_loss(cfg["loss"]).to(self.device)

        opt_cfg = cfg.get("optimizer", {"lr": 1e-4})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_cfg.get("lr", 1e-4))

        data_cfg = cfg["data"]
        train_ds = DATASET_REGISTRY.build(data_cfg["type"], **data_cfg.get("params", {}))
        train_cfg = cfg.get("train", {})
        self.loader = DataLoader(
            train_ds,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 2),
            drop_last=True,
        )

        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else None
        print(f"Training on device: {self.device}" + (f" ({gpu_name})" if gpu_name else ""))
        print(f"Dataset: {len(train_ds)} samples, {len(self.loader)} batches/epoch, "
              f"batch_size={self.loader.batch_size}")

    def fit(self, epochs: Optional[int] = None):
        epochs = epochs or self.cfg.get("train", {}).get("epochs", 10)
        self.model.train()
        n_batches = max(len(self.loader), 1)
        print_every = max(1, n_batches // 10)  # ~10 progress lines per epoch, regardless of dataset size

        for epoch in range(1, epochs + 1):
            running: dict = {}
            for batch_idx, (lr, hr) in enumerate(self.loader, start=1):
                lr, hr = lr.to(self.device), hr.to(self.device)
                pred = self.model(lr)
                loss, parts = self.loss_fn(pred, hr)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + v

                if batch_idx % print_every == 0 or batch_idx == n_batches:
                    so_far = ", ".join(f"{k}={v / batch_idx:.4f}" for k, v in running.items())
                    print(f"  epoch {epoch}/{epochs} -- batch {batch_idx}/{n_batches}: {so_far}", flush=True)

            summary = ", ".join(f"{k}={v / n_batches:.4f}" for k, v in running.items())
            print(f"[epoch {epoch}/{epochs}] {summary}")
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
        path = self.ckpt_dir / f"{self.cfg['model']['name']}_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), path)
