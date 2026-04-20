"""Weighted scenario assembly — the central output connecting all branches.

S_final = S_normal_proto ∪ S_stress_hist ∪ S_stress_syn

This module assembles the final scenario set as an explicit, formally
well-defined discrete probability distribution.  Not ad-hoc concatenation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.assembly.weighting import (
    compute_normal_prototype_weights,
    compute_historical_stress_weights,
    compute_synthetic_stress_weights,
    renormalize_to_distribution,
    validate_weights,
)
from src.normal_model.prototypes import PrototypeModel
from src.regimes.partition import RegimePartition

logger = logging.getLogger(__name__)


@dataclass
class ScenarioAssembly:
    """The assembled scenario set with explicit probability weights.

    Each row in ``normal_scenarios`` corresponds to one prototype (M rows).
    Each row in ``hist_stress_scenarios`` is a historical stress observation.
    Each row in ``syn_stress_scenarios`` is a synthetic stress scenario.
    """
    normal_scenarios: pd.DataFrame
    hist_stress_scenarios: pd.DataFrame
    syn_stress_scenarios: pd.DataFrame

    normal_weights: np.ndarray
    hist_stress_weights: np.ndarray
    syn_stress_weights: np.ndarray

    m: float
    n_total_scenarios: int
    normal_fraction: float
    stress_fraction: float
    synthetic_mass: float
    historical_stress_mass: float

    def get_all_scenarios(self) -> pd.DataFrame:
        """Concatenate all scenario DataFrames."""
        frames = [self.normal_scenarios, self.hist_stress_scenarios]
        if len(self.syn_stress_scenarios) > 0:
            frames.append(self.syn_stress_scenarios)
        return pd.concat(frames, axis=0, ignore_index=True)

    def get_all_weights(self) -> np.ndarray:
        """Concatenated weight vector.  Sums to 1.0."""
        parts = [self.normal_weights, self.hist_stress_weights]
        if len(self.syn_stress_weights) > 0:
            parts.append(self.syn_stress_weights)
        return np.concatenate(parts)

    def get_source_labels(self) -> list[str]:
        """Source label for each scenario row."""
        labels = (
            ["normal_proto"] * len(self.normal_scenarios)
            + ["stress_hist"] * len(self.hist_stress_scenarios)
            + ["stress_syn"] * len(self.syn_stress_scenarios)
        )
        return labels

    def summary(self) -> dict:
        return {
            "n_normal_proto": len(self.normal_scenarios),
            "n_hist_stress": len(self.hist_stress_scenarios),
            "n_syn_stress": len(self.syn_stress_scenarios),
            "n_total": self.n_total_scenarios,
            "normal_mass": float(self.normal_weights.sum()),
            "hist_stress_mass": float(self.historical_stress_mass),
            "syn_stress_mass": float(self.synthetic_mass),
            "total_weight": float(self.get_all_weights().sum()),
            "m": self.m,
        }

    def validate(self) -> None:
        """Raise if weights are invalid."""
        total = self.get_all_weights().sum()
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Scenario weights sum to {total:.6f} ≠ 1.0")
        if np.any(self.get_all_weights() < -1e-9):
            raise ValueError("Negative scenario weights detected.")

    def to_dataframe(self) -> pd.DataFrame:
        """Combined DataFrame with scenario data, weights, and source labels."""
        df = self.get_all_scenarios().copy()
        df["weight"] = self.get_all_weights()
        df["source"] = self.get_source_labels()
        return df

    def save(self, path: Path | str) -> None:
        """Save assembly to parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path)
        logger.info(f"Saved ScenarioAssembly -> {path}")

    @classmethod
    def load(cls, path: Path | str) -> "ScenarioAssembly":
        """Load assembly from parquet.  Reconstructs weights and subsets."""
        df = pd.read_parquet(Path(path))
        normal_mask = df["source"] == "normal_proto"
        hist_mask = df["source"] == "stress_hist"
        syn_mask = df["source"] == "stress_syn"

        feature_cols = [c for c in df.columns if c not in ("weight", "source")]

        normal_s = df.loc[normal_mask, feature_cols]
        hist_s = df.loc[hist_mask, feature_cols]
        syn_s = df.loc[syn_mask, feature_cols]
        w_n = df.loc[normal_mask, "weight"].to_numpy()
        w_h = df.loc[hist_mask, "weight"].to_numpy()
        w_s = df.loc[syn_mask, "weight"].to_numpy()

        hist_mass = float(w_h.sum())
        syn_mass = float(w_s.sum())
        m = syn_mass / hist_mass if hist_mass > 0 else 0.0

        return cls(
            normal_scenarios=normal_s.reset_index(drop=True),
            hist_stress_scenarios=hist_s.reset_index(drop=True),
            syn_stress_scenarios=syn_s.reset_index(drop=True),
            normal_weights=w_n,
            hist_stress_weights=w_h,
            syn_stress_weights=w_s,
            m=m,
            n_total_scenarios=len(df),
            normal_fraction=float(w_n.sum()),
            stress_fraction=float(hist_mass + syn_mass),
            synthetic_mass=syn_mass,
            historical_stress_mass=hist_mass,
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def assemble_scenarios(
    partition: RegimePartition,
    prototype_model: PrototypeModel,
    normal_returns: pd.DataFrame,
    hist_stress_returns: pd.DataFrame,
    syn_stress_returns: pd.DataFrame,
    m: float,
    use_normal_prototypes: bool = True,
    asset_columns: list[str] | None = None,
) -> ScenarioAssembly:
    """Assemble the final scenario set.

    Parameters
    ----------
    use_normal_prototypes : if False (ablation), use raw normal obs, equal weight.
    """
    normal_frac = partition.n_normal / partition.n_total
    stress_frac = partition.n_stress / partition.n_total

    asset_cols = asset_columns or normal_returns.columns.tolist()

    # ---------------------------------------------------------------
    # Normal scenarios
    # ---------------------------------------------------------------
    if use_normal_prototypes:
        # One scenario per prototype centre (in latent/feature space)
        # We represent each prototype by the mean of assigned observations
        assignments = prototype_model.assignments

        # Align normal_returns to prototype assignments
        normal_arr = normal_returns[asset_cols].to_numpy()
        if len(normal_arr) != len(assignments):
            logger.warning(
                f"normal_returns length ({len(normal_arr)}) ≠ "
                f"prototype assignments ({len(assignments)}); truncating."
            )
            min_len = min(len(normal_arr), len(assignments))
            normal_arr = normal_arr[:min_len]
            assignments = assignments[:min_len]

        # Prototype scenarios = mean of assigned normal observations
        proto_scenarios = np.zeros((prototype_model.M, len(asset_cols)))
        for j in range(prototype_model.M):
            mask = assignments == j
            if mask.sum() > 0:
                proto_scenarios[j] = normal_arr[mask].mean(axis=0)

        normal_scenarios = pd.DataFrame(proto_scenarios, columns=asset_cols)
        w_n_raw = compute_normal_prototype_weights(
            prototype_model.assignments, prototype_model.M, normal_frac
        )
    else:
        # Ablation: use all normal observations with equal weight
        normal_scenarios = normal_returns[asset_cols].reset_index(drop=True)
        w_n_raw = np.full(len(normal_scenarios),
                          normal_frac / len(normal_scenarios))

    # ---------------------------------------------------------------
    # Historical stress scenarios
    # ---------------------------------------------------------------
    hist_scenarios = hist_stress_returns[asset_cols].reset_index(drop=True)
    w_h_raw = compute_historical_stress_weights(len(hist_scenarios), stress_frac)

    # ---------------------------------------------------------------
    # Synthetic stress scenarios
    # ---------------------------------------------------------------
    if len(syn_stress_returns) > 0:
        syn_asset_cols = [c for c in asset_cols if c in syn_stress_returns.columns]
        syn_scenarios = syn_stress_returns[syn_asset_cols].reset_index(drop=True)
        if set(syn_asset_cols) != set(asset_cols):
            # Reindex to add missing columns as 0
            syn_scenarios = syn_scenarios.reindex(columns=asset_cols, fill_value=0.0)
    else:
        syn_scenarios = pd.DataFrame(columns=asset_cols)
    w_s_raw = compute_synthetic_stress_weights(len(syn_scenarios), float(w_h_raw.sum()), m)

    # ---------------------------------------------------------------
    # Renormalise and validate
    # ---------------------------------------------------------------
    w_n, w_h, w_s = renormalize_to_distribution(w_n_raw, w_h_raw, w_s_raw)
    validate_weights(w_n, w_h, w_s, m)

    assembly = ScenarioAssembly(
        normal_scenarios=normal_scenarios,
        hist_stress_scenarios=hist_scenarios,
        syn_stress_scenarios=syn_scenarios,
        normal_weights=w_n,
        hist_stress_weights=w_h,
        syn_stress_weights=w_s,
        m=m,
        n_total_scenarios=len(normal_scenarios) + len(hist_scenarios) + len(syn_scenarios),
        normal_fraction=float(w_n.sum()),
        stress_fraction=float(w_h.sum() + w_s.sum()),
        synthetic_mass=float(w_s.sum()),
        historical_stress_mass=float(w_h.sum()),
    )
    assembly.validate()

    logger.info(
        f"ScenarioAssembly: {assembly.summary()}"
    )
    return assembly


def assemble_baseline_hs(
    returns: pd.DataFrame,
    asset_columns: list[str] | None = None,
) -> ScenarioAssembly:
    """Historical Simulation baseline — all observations equally weighted.

    Used for the baseline_hs ablation (no stress split, no prototypes).
    """
    asset_cols = asset_columns or returns.columns.tolist()
    scenarios = returns[asset_cols].reset_index(drop=True)
    N = len(scenarios)
    weights = np.full(N, 1.0 / N)

    empty_df = pd.DataFrame(columns=asset_cols)
    empty_w = np.array([], dtype=float)

    assembly = ScenarioAssembly(
        normal_scenarios=scenarios,
        hist_stress_scenarios=empty_df,
        syn_stress_scenarios=empty_df,
        normal_weights=weights,
        hist_stress_weights=empty_w,
        syn_stress_weights=empty_w,
        m=0.0,
        n_total_scenarios=N,
        normal_fraction=1.0,
        stress_fraction=0.0,
        synthetic_mass=0.0,
        historical_stress_mass=0.0,
    )
    assembly.validate()
    return assembly
