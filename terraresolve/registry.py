"""
Generic name -> factory registry. This is the one mechanism that
makes the whole project plug-and-play: models, losses, metrics,
datasets and downstream tasks all register themselves into one of
the registries below, and every other part of the code asks for them
by string name (which comes straight out of the YAML config) instead
of importing a concrete class directly.

To add a new model/loss/metric/dataset/downstream task:
    1. Write the class or function.
    2. Decorate it: @MODEL_REGISTRY.register("my_new_model")
    3. Reference "my_new_model" in a config file.
No other file needs to change.
"""
from __future__ import annotations
from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._store: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        key = name.lower()

        def _decorator(obj):
            if key in self._store:
                raise KeyError(f"[{self.kind}] '{key}' is already registered")
            self._store[key] = obj
            return obj

        return _decorator

    def get(self, name: str) -> Callable[..., Any]:
        key = name.lower()
        if key not in self._store:
            available = ", ".join(sorted(self._store)) or "(none registered yet)"
            raise KeyError(
                f"[{self.kind}] Unknown name '{name}'. Available: {available}"
            )
        return self._store[key]

    def build(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._store)


MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
DATASET_REGISTRY = Registry("dataset")
DOWNSTREAM_REGISTRY = Registry("downstream")
