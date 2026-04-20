"""YAML config loading and merging utilities."""
from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any

# Root of the repository — configs live under <ROOT>/configs/
_REPO_ROOT = Path(__file__).parent.parent.parent


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file and return as dict."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge *override* into *base* (override wins).  Returns a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_experiment_config(experiment_path: Path | str) -> dict:
    """Load a full experiment config, merging referenced sub-configs.

    The experiment YAML may contain a ``_defaults_`` list, e.g.::

        _defaults_:
          - data: default
          - features: default
          - model/vae: vae
          - calibration: default
          - evaluation: default

    Each entry ``group: name`` maps to ``configs/{group}/{name}.yaml``.
    Entries are merged in order; then the experiment-level keys override all.
    """
    exp_cfg = load_yaml(experiment_path)
    defaults = exp_cfg.pop("_defaults_", [])

    merged: dict = {}
    for entry in defaults:
        if isinstance(entry, dict):
            for group, name in entry.items():
                sub_path = _REPO_ROOT / "configs" / group / f"{name}.yaml"
                if sub_path.exists():
                    sub_cfg = load_yaml(sub_path)
                    # Nest under group key if the group contains a slash
                    group_key = group.replace("/", "_")
                    merged = merge_configs(merged, {group_key: sub_cfg})
                else:
                    import warnings
                    warnings.warn(f"Default config not found: {sub_path}")

    merged = merge_configs(merged, exp_cfg)
    return merged


def save_config(cfg: dict, path: Path | str) -> None:
    """Save config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


class Config:
    """Thin wrapper around a dict for attribute-style access.

    Nested dicts are recursively wrapped so ``cfg.model.latent_dim`` works.
    """

    def __init__(self, d: dict):
        object.__setattr__(self, "_data", {})
        for k, v in d.items():
            self._data[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise AttributeError(f"Config has no key '{key}'")

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        return data.get(key, default)

    def to_dict(self) -> dict:
        data = object.__getattribute__(self, "_data")
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in data.items()
        }

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
