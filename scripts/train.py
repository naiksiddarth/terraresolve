"""Entry point.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --set model.name=swinir
    python scripts/train.py --config configs/default.yaml --set loss.terms.sam=0.3 train.epochs=5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import terraresolve.data     # noqa: F401  (registers datasets)
import terraresolve.losses   # noqa: F401  (registers losses)
import terraresolve.models   # noqa: F401  (registers models)
from terraresolve.engine.trainer import Trainer
from terraresolve.utils.config import apply_overrides, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.name=swinir")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
