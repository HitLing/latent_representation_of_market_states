"""Rolling window construction for market-state feature tensors.

Each window X_t ∈ R^{T × F} represents the market state at time t
using the preceding T trading days of F features.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowDataset:
    """Container for windowed time-series data.

    Attributes
    ----------
    windows : np.ndarray, shape (N, window_size, n_features)
    dates   : pd.DatetimeIndex, length N — date of the *last* bar in each window
    feature_names : list[str]
    window_size   : int
    step_size     : int
    """
    windows: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: list
    window_size: int
    step_size: int

    def __len__(self) -> int:
        return len(self.windows)

    def get_flat(self) -> np.ndarray:
        """Return windows reshaped to (N, window_size * n_features)."""
        return self.windows.reshape(len(self.windows), -1)

    def get_last_bar(self) -> np.ndarray:
        """Return only the last bar of each window: shape (N, n_features)."""
        return self.windows[:, -1, :]

    @property
    def n_features(self) -> int:
        return self.windows.shape[2]


def build_windows(
    features: pd.DataFrame,
    window_size: int,
    step_size: int = 1,
) -> WindowDataset:
    """Build rolling windows from a feature DataFrame.

    Parameters
    ----------
    features    : pd.DataFrame with DatetimeIndex and feature columns
    window_size : T — number of timesteps per window
    step_size   : stride (1 = every trading day)

    Returns
    -------
    WindowDataset where ``windows[i]`` == ``features.iloc[i : i+window_size]``
    """
    n_rows, n_feats = features.shape
    feature_names = features.columns.tolist()
    arr = features.to_numpy(dtype=np.float32)

    indices = list(range(0, n_rows - window_size + 1, step_size))
    n_windows = len(indices)

    if n_windows == 0:
        raise ValueError(
            f"window_size={window_size} is larger than dataset length {n_rows}"
        )

    windows = np.empty((n_windows, window_size, n_feats), dtype=np.float32)
    end_dates = []

    for out_idx, start in enumerate(indices):
        windows[out_idx] = arr[start : start + window_size]
        end_dates.append(features.index[start + window_size - 1])

    dates = pd.DatetimeIndex(end_dates)

    logger.info(
        f"Built {n_windows} windows: size={window_size}, stride={step_size}, "
        f"features={n_feats}, span {dates[0].date()} – {dates[-1].date()}"
    )
    return WindowDataset(
        windows=windows,
        dates=dates,
        feature_names=feature_names,
        window_size=window_size,
        step_size=step_size,
    )


def windows_to_flat_dataframe(dataset: WindowDataset) -> pd.DataFrame:
    """Convert windowed data to a flat DataFrame (last-bar features) for inspection."""
    df = pd.DataFrame(
        dataset.get_last_bar(),
        index=dataset.dates,
        columns=dataset.feature_names,
    )
    df.index.name = "date"
    return df
