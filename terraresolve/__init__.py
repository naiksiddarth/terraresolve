"""TerraResolve: modular deep-learning super-resolution pipeline for
medium-resolution multispectral satellite imagery (SIH 2026, PS 26142).

Everything -- model, loss recipe, dataset source, evaluation metric,
downstream task -- is resolved by NAME from a YAML config through the
registries in `terraresolve.registry`. To add a new option: write it
and decorate it with the right @X_REGISTRY.register("name"). To use
it: put that name in your config. Nothing else in the codebase has
to change. See README.md for the full map of what's swappable where.
"""

__version__ = "0.1.0"
