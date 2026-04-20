"""Train / inner-validation / test split utilities.

All splits are strictly temporal — no shuffling of time series,
no look-ahead.  The test set is touched ONLY during final OOS evaluation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """Indices and boundary dates for a train / val / test split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    total_n: int

    def __repr__(self) -> str:
        return (
            f"TemporalSplit(n={self.total_n}, "
            f"train={len(self.train_idx)} [{self.train_dates[0].date()} – {self.train_dates[-1].date()}], "
            f"val={len(self.val_idx)} [{self.val_dates[0].date()} – {self.val_dates[-1].date()}], "
            f"test={len(self.test_idx)} [{self.test_dates[0].date()} – {self.test_dates[-1].date()}])"
        )


def make_temporal_split(
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> TemporalSplit:
    """Create strict temporal split.

    Validates that ratios sum to ~1 and each split has at least 10 observations.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test ratios must sum to 1.0")

    n = len(dates)
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    for label, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if count < 10:
            raise ValueError(f"Split '{label}' has only {count} observations — too small.")

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    split = TemporalSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_dates=dates[train_idx],
        val_dates=dates[val_idx],
        test_dates=dates[test_idx],
        total_n=n,
    )
    logger.info(str(split))
    return split


def apply_split(
    data: pd.DataFrame | np.ndarray,
    split: TemporalSplit,
    subset: str,
) -> pd.DataFrame | np.ndarray:
    """Select train / val / test subset from data.

    Parameters
    ----------
    data   : DataFrame (DatetimeIndex) or ndarray (same length as split.total_n)
    subset : 'train', 'val', or 'test'
    """
    idx_map = {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}
    if subset not in idx_map:
        raise ValueError(f"subset must be 'train', 'val', or 'test', got '{subset}'")
    idx = idx_map[subset]

    if isinstance(data, pd.DataFrame):
        return data.iloc[idx]
    elif isinstance(data, pd.Series):
        return data.iloc[idx]
    else:
        return data[idx]


def make_rolling_windows_split(
    dates: pd.DatetimeIndex,
    window_years: float = 2.0,
    step_months: float = 3.0,
) -> list[TemporalSplit]:
    """Create multiple rolling-window train/test splits for robustness evaluation.

    Each split uses a fixed-length training window of *window_years* years
    and a test window of *step_months* months, stepping forward by *step_months*.
    """
    trading_days_per_year = 252
    trading_days_per_month = 21

    window_size = int(window_years * trading_days_per_year)
    step_size = int(step_months * trading_days_per_month)
    test_size = step_size

    splits = []
    n = len(dates)
    start = 0

    while start + window_size + test_size <= n:
        train_end = start + window_size
        test_end = min(train_end + test_size, n)

        train_idx = np.arange(start, train_end)
        val_idx = np.array([], dtype=int)  # no separate val in rolling eval
        test_idx = np.arange(train_end, test_end)

        # Use empty DatetimeIndex for val
        val_dates = dates[val_idx] if len(val_idx) > 0 else pd.DatetimeIndex([])

        splits.append(TemporalSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_dates=dates[train_idx],
            val_dates=val_dates,
            test_dates=dates[test_idx],
            total_n=n,
        ))
        start += step_size

    logger.info(f"Created {len(splits)} rolling-window splits")
    return splits
