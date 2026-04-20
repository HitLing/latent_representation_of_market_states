"""Formal Stress Index — the central methodological object.

Composite score
---------------
S_t = (1/K) Σ_k normalised_indicator_k(t)   [equal weights, baseline]

Regime definition
-----------------
Stress  :  S_t > q_{1−α}(S)
Normal  :  S_t ≤ q_{1−α}(S)

α is a *structural parameter* — configured externally and calibrated on
the inner validation split, never chosen by visual inspection.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.stress_indicators import compute_all_indicators

logger = logging.getLogger(__name__)


class StressIndex:
    """Formal composite stress index.

    Every component is explicit and logged.  This is not a black-box score.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.alpha: float = cfg.get("regimes", {}).get("alpha", 0.10)
        self.threshold_: float | None = None
        self.indicator_weights_: dict[str, float] | None = None
        self._fit_returns: pd.DataFrame | None = None
        self._fit_weights: pd.Series | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        feature_cfg: dict,
    ) -> "StressIndex":
        """Fit the stress index on training data.

        1. Compute and normalise all partial indicators (fit CDF on *returns*).
        2. Aggregate with equal weights.
        3. Compute threshold = q_{1−α}.
        4. Store state for transform().
        """
        self._fit_returns = returns
        self._fit_weights = weights
        self._feature_cfg = feature_cfg

        indicators = compute_all_indicators(
            returns=returns,
            weights=weights,
            cfg=feature_cfg,
            fit_returns=returns,
        )

        # Store equal weights
        k = indicators.shape[1]
        self.indicator_weights_ = {col: 1.0 / k for col in indicators.columns}

        score = self._aggregate(indicators)
        score = score.dropna()

        self.threshold_ = float(np.quantile(score.values, 1.0 - self.alpha))

        n_stress = int((score > self.threshold_).sum())
        logger.info(
            f"StressIndex fitted: alpha={self.alpha}, threshold={self.threshold_:.4f}, "
            f"n_stress={n_stress} ({n_stress/len(score):.1%} of train obs)"
        )
        self._is_fitted = True
        return self

    def transform(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        feature_cfg: dict,
    ) -> tuple[pd.Series, pd.Series]:
        """Apply fitted index to data.

        Returns
        -------
        stress_score  : pd.Series[float] ∈ [0,1], DatetimeIndex
        regime_labels : pd.Series[bool],  True = stress
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        indicators = compute_all_indicators(
            returns=returns,
            weights=weights,
            cfg=feature_cfg,
            fit_returns=self._fit_returns,
        )
        score = self._aggregate(indicators)
        labels = score > self.threshold_
        labels.name = "is_stress"
        return score, labels

    def fit_transform(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        feature_cfg: dict,
    ) -> tuple[pd.Series, pd.Series]:
        """Fit and transform in one step."""
        self.fit(returns, weights, feature_cfg)
        return self.transform(returns, weights, feature_cfg)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        feature_cfg: dict,
        alpha_values: list[float],
    ) -> pd.DataFrame:
        """Analyse sensitivity of the regime partition to α.

        Returns a DataFrame with α as index and columns:
        [threshold, n_stress, stress_fraction].
        """
        indicators = compute_all_indicators(
            returns=returns, weights=weights, cfg=feature_cfg, fit_returns=returns
        )
        score = self._aggregate(indicators).dropna()

        rows = []
        for alpha in alpha_values:
            thr = float(np.quantile(score.values, 1.0 - alpha))
            n_stress = int((score > thr).sum())
            rows.append({
                "alpha": alpha,
                "threshold": thr,
                "n_stress": n_stress,
                "stress_fraction": n_stress / len(score),
            })
        return pd.DataFrame(rows).set_index("alpha")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate(self, indicators: pd.DataFrame) -> pd.Series:
        """Aggregate indicators to composite score.

        Baseline: equal weights (simple mean).
        Alternative weights can be set in cfg['regimes']['indicator_weights']
        — used only as a controlled ablation.
        """
        alt_weights = self.cfg.get("regimes", {}).get("indicator_weights", None)
        if alt_weights is not None:
            w = pd.Series(alt_weights)
            w = w.reindex(indicators.columns, fill_value=0.0)
            w = w / w.sum()
            score = indicators.multiply(w.values, axis=1).sum(axis=1)
        else:
            score = indicators.mean(axis=1)
        score.name = "stress_score"
        return score

    def get_indicator_contributions(self) -> dict[str, float]:
        """Return indicator names → weights (for interpretability)."""
        if self.indicator_weights_ is None:
            raise RuntimeError("Call fit() first.")
        return dict(self.indicator_weights_)

    def to_dict(self) -> dict:
        """Serialise fitted state."""
        return {
            "alpha": self.alpha,
            "threshold": self.threshold_,
            "indicator_weights": self.indicator_weights_,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: dict) -> "StressIndex":
        """Reconstruct from serialised state."""
        obj = cls(cfg)
        obj.alpha = d["alpha"]
        obj.threshold_ = d["threshold"]
        obj.indicator_weights_ = d["indicator_weights"]
        obj._is_fitted = d["is_fitted"]
        return obj
