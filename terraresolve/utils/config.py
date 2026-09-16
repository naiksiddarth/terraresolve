"""Tiny YAML config loader plus dotted-key CLI overrides, e.g.
`--set model.name=swinir loss.terms.sam=0.2` from the command line
without hand-editing the YAML file each time you want to try a swap."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key.path=value")
        key_path, value = item.split("=", 1)
        node = cfg
        keys = key_path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(value)
    return cfg


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
