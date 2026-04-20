"""Artifact saving / loading utilities."""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: Path | str, fmt: str = "parquet") -> None:
    """Save DataFrame as parquet (default) or csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    logger.debug(f"Saved DataFrame {df.shape} -> {path}")


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load DataFrame from parquet or csv (auto-detect by extension)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix in (".csv", ".txt"):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unknown file extension: {suffix}")


# ---------------------------------------------------------------------------
# NumPy arrays
# ---------------------------------------------------------------------------

def save_array(arr: np.ndarray, path: Path | str) -> None:
    """Save numpy array as .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.debug(f"Saved array {arr.shape} -> {path}")


def load_array(path: Path | str) -> np.ndarray:
    """Load numpy array from .npy."""
    return np.load(Path(path), allow_pickle=False)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class _FloatEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: dict, path: Path | str) -> None:
    """Save dict as JSON (handles numpy scalars and arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_FloatEncoder)
    logger.debug(f"Saved JSON -> {path}")


def load_json(path: Path | str) -> dict:
    """Load JSON file."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: Path | str) -> None:
    """Save arbitrary object as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved pickle -> {path}")


def load_pickle(path: Path | str) -> Any:
    """Load pickle file."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Create directory and all parents if not exists.  Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_path(output_dir: Path | str, name: str, suffix: str) -> Path:
    """Construct a standardised artifact path: ``output_dir/name.suffix``."""
    return Path(output_dir) / f"{name}.{suffix.lstrip('.')}"
