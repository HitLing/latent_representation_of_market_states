"""Distribution partition into normal and stress regimes.

D = D_normal ∪ D_stress  (disjoint, exhaustive)

This partition object explicitly drives all downstream branches:
  D_normal → normal prototype layer (VAE + clustering)
  D_stress → tail extractor + conditional generator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimePartition:
    """Explicit partition of a dataset into normal and stress subsets.

    Every field is explicit and persistable.  This is not a passive filter —
    it is the formal partition that drives all downstream processing.
    """
    stress_score: pd.Series          # S_t values, DatetimeIndex
    regime_labels: pd.Series         # bool Series, True = stress
    alpha: float
    threshold: float

    normal_dates: pd.DatetimeIndex
    stress_dates: pd.DatetimeIndex

    n_total: int
    n_normal: int
    n_stress: int
    stress_fraction: float

    def __post_init__(self) -> None:
        assert self.n_normal + self.n_stress == self.n_total, (
            f"Partition sizes inconsistent: {self.n_normal} + {self.n_stress} "
            f"!= {self.n_total}"
        )

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def get_normal_mask(self) -> pd.Series:
        """Boolean mask aligned to stress_score index; True = normal."""
        return ~self.regime_labels

    def get_stress_mask(self) -> pd.Series:
        """Boolean mask aligned to stress_score index; True = stress."""
        return self.regime_labels

    # ------------------------------------------------------------------
    # Data selection
    # ------------------------------------------------------------------

    def filter_data(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray,
        regime: str,
    ) -> pd.DataFrame | pd.Series | np.ndarray:
        """Filter *data* to the normal or stress subset.

        Parameters
        ----------
        data   : DataFrame/Series with DatetimeIndex aligned to stress_score,
                 OR ndarray of length ``n_total`` (rows indexed positionally).
        regime : 'normal' or 'stress'
        """
        if regime not in ("normal", "stress"):
            raise ValueError(f"regime must be 'normal' or 'stress', got '{regime}'")
        dates = self.normal_dates if regime == "normal" else self.stress_dates

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.loc[data.index.intersection(dates)]
        else:
            # ndarray: select by boolean positional mask
            mask = self.regime_labels.values if regime == "stress" else ~self.regime_labels.values
            return data[mask]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return summary statistics as a dict."""
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_total": self.n_total,
            "n_normal": self.n_normal,
            "n_stress": self.n_stress,
            "stress_fraction": self.stress_fraction,
            "date_range_start": str(self.stress_score.index[0].date()),
            "date_range_end": str(self.stress_score.index[-1].date()),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with date, stress_score, and is_stress columns."""
        df = pd.DataFrame({
            "stress_score": self.stress_score,
            "is_stress": self.regime_labels,
        })
        df.index.name = "date"
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_partition(
    stress_score: pd.Series,
    regime_labels: pd.Series,
    alpha: float,
    threshold: float,
) -> RegimePartition:
    """Create a RegimePartition from stress index outputs.

    Validates input consistency before constructing the partition.
    """
    if not stress_score.index.equals(regime_labels.index):
        raise ValueError("stress_score and regime_labels must share the same index.")

    idx = stress_score.dropna().index
    stress_score = stress_score.loc[idx]
    regime_labels = regime_labels.loc[idx]

    normal_dates = stress_score.index[~regime_labels]
    stress_dates = stress_score.index[regime_labels]

    n_total = len(stress_score)
    n_stress = int(regime_labels.sum())
    n_normal = n_total - n_stress

    partition = RegimePartition(
        stress_score=stress_score,
        regime_labels=regime_labels,
        alpha=alpha,
        threshold=threshold,
        normal_dates=normal_dates,
        stress_dates=stress_dates,
        n_total=n_total,
        n_normal=n_normal,
        n_stress=n_stress,
        stress_fraction=n_stress / n_total if n_total > 0 else 0.0,
    )

    logger.info(
        f"Partition created: alpha={alpha:.3f}, threshold={threshold:.4f}, "
        f"n_normal={n_normal} ({n_normal/n_total:.1%}), "
        f"n_stress={n_stress} ({n_stress/n_total:.1%})"
    )
    return partition


def validate_partition_quality(partition: RegimePartition) -> dict:
    """Check partition for methodological validity.

    Returns a dict of check names → (passed: bool, message: str).
    """
    checks: dict[str, dict] = {}

    # 1. Stress fraction near alpha
    expected = partition.alpha
    actual = partition.stress_fraction
    diff = abs(actual - expected)
    checks["stress_fraction_near_alpha"] = {
        "passed": diff < 0.05,
        "message": f"Expected ≈{expected:.2%}, got {actual:.2%} (diff={diff:.2%})",
    }

    # 2. Enough stress obs for tail modeling
    min_stress = 30
    checks["enough_stress_obs"] = {
        "passed": partition.n_stress >= min_stress,
        "message": f"n_stress={partition.n_stress} (minimum={min_stress})",
    }

    # 3. Enough normal obs for VAE training
    min_normal = 100
    checks["enough_normal_obs"] = {
        "passed": partition.n_normal >= min_normal,
        "message": f"n_normal={partition.n_normal} (minimum={min_normal})",
    }

    for name, result in checks.items():
        level = logging.INFO if result["passed"] else logging.WARNING
        logger.log(level, f"Partition check '{name}': {result['message']}")

    return checks
