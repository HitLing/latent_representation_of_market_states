"""Historical Tail Extractor.

Purpose
-------
Organise D_stress observations into well-defined subsets for downstream use.
This module is SEPARATE from the stress index:
  - Stress index  → identifies which observations are stress
  - Tail extractor → organises stress observations for modelling and evaluation

Output subsets
--------------
D_stress_train_gen   : used to train the CVAE generator
D_stress_anchor      : historical tail anchor in final scenario assembly
D_stress_struct_eval : held-out structural validation subset (plausibility checks)

Critical: avoid circular validation — D_stress_struct_eval must NOT
overlap with D_stress_train_gen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.regimes.partition import RegimePartition
from src.tail.severity import assign_severity_buckets, compute_severity_stats

logger = logging.getLogger(__name__)


@dataclass
class TailExtract:
    """All tail extraction outputs in one place."""
    D_stress_train_gen: pd.DataFrame      # for CVAE training
    D_stress_anchor: pd.DataFrame          # historical tail anchor
    D_stress_struct_eval: pd.DataFrame     # structural evaluation hold-out

    severity_labels: pd.Series            # index=dates, values=bucket int
    severity_bucket_stats: pd.DataFrame   # statistics per severity bucket

    stress_correlation_matrix: pd.DataFrame
    normal_correlation_matrix: pd.DataFrame

    stress_portfolio_loss_summary: dict

    train_gen_fraction: float
    anchor_fraction: float
    struct_eval_fraction: float

    def summary(self) -> dict:
        return {
            "n_train_gen": len(self.D_stress_train_gen),
            "n_anchor": len(self.D_stress_anchor),
            "n_struct_eval": len(self.D_stress_struct_eval),
            "n_severity_buckets": self.severity_labels.nunique(),
            "train_gen_fraction": self.train_gen_fraction,
            "anchor_fraction": self.anchor_fraction,
            "struct_eval_fraction": self.struct_eval_fraction,
        }


def extract_tail(
    partition: RegimePartition,
    returns: pd.DataFrame,
    weights: pd.Series,
    cfg: dict,
) -> TailExtract:
    """Main tail extraction function.

    Steps
    -----
    1. Retrieve D_stress from partition (temporal ordering preserved).
    2. Assign severity buckets.
    3. Temporal split into train_gen / anchor / struct_eval (no shuffle).
    4. Compute correlation matrices (normal vs stress).
    5. Compute stress portfolio loss summary.
    6. Validate no circular overlap.
    """
    tail_cfg = cfg.get("tail", {})
    train_gen_frac: float = tail_cfg.get("train_gen_fraction", 0.60)
    anchor_frac: float = tail_cfg.get("anchor_fraction", 0.25)
    struct_eval_frac: float = tail_cfg.get("struct_eval_fraction", 0.15)
    n_buckets: int = tail_cfg.get("n_severity_buckets",
                                   cfg.get("model_cvae", {}).get("n_severity_buckets", 3))

    # Validate fractions
    total = train_gen_frac + anchor_frac + struct_eval_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Tail split fractions must sum to 1.0 (got {total:.4f})")

    # ---------------------------------------------------------------
    # 1. Get stress observations (temporal order preserved)
    # ---------------------------------------------------------------
    stress_returns = returns.loc[returns.index.intersection(partition.stress_dates)]
    if len(stress_returns) == 0:
        raise ValueError("No stress observations found in partition.")

    n_stress = len(stress_returns)
    logger.info(f"Tail extractor: {n_stress} stress observations")

    # ---------------------------------------------------------------
    # 2. Severity buckets
    # ---------------------------------------------------------------
    severity_labels = assign_severity_buckets(
        stress_score=partition.stress_score.loc[partition.stress_dates],
        stress_dates=partition.stress_dates,
        n_buckets=n_buckets,
    )

    w = weights.reindex(returns.columns, fill_value=0.0)
    w = w / w.sum()
    severity_stats = compute_severity_stats(stress_returns, w, severity_labels)

    # ---------------------------------------------------------------
    # 3. Temporal split (no shuffle — preserve crisis structure)
    # ---------------------------------------------------------------
    n_train = max(1, int(np.floor(n_stress * train_gen_frac)))
    n_anchor = max(1, int(np.floor(n_stress * anchor_frac)))
    n_eval = n_stress - n_train - n_anchor

    if n_eval <= 0:
        # If too few observations, give all to train and anchor, nothing to eval
        n_eval = 0
        n_anchor = n_stress - n_train

    idx = np.arange(n_stress)
    train_idx = idx[:n_train]
    anchor_idx = idx[n_train : n_train + n_anchor]
    eval_idx = idx[n_train + n_anchor :]

    D_train_gen = stress_returns.iloc[train_idx]
    D_anchor = stress_returns.iloc[anchor_idx]
    D_struct_eval = stress_returns.iloc[eval_idx] if len(eval_idx) > 0 else stress_returns.iloc[:0]

    # ---------------------------------------------------------------
    # 4. Correlation matrices
    # ---------------------------------------------------------------
    normal_returns = returns.loc[returns.index.intersection(partition.normal_dates)]
    stress_corr = stress_returns.corr() if len(stress_returns) > 1 else pd.DataFrame()
    normal_corr = normal_returns.corr() if len(normal_returns) > 1 else pd.DataFrame()

    # ---------------------------------------------------------------
    # 5. Portfolio loss summary
    # ---------------------------------------------------------------
    port_losses_stress = -(stress_returns @ w)
    port_losses_summary = {
        "mean_loss": float(port_losses_stress.mean()),
        "std_loss": float(port_losses_stress.std()),
        "max_loss": float(port_losses_stress.max()),
        "quantile_95": float(port_losses_stress.quantile(0.95)),
        "quantile_99": float(port_losses_stress.quantile(0.99)),
    }

    # ---------------------------------------------------------------
    # 6. Validate no overlap
    # ---------------------------------------------------------------
    validate_tail_extract_no_overlap(D_train_gen, D_anchor, D_struct_eval)

    extract = TailExtract(
        D_stress_train_gen=D_train_gen,
        D_stress_anchor=D_anchor,
        D_stress_struct_eval=D_struct_eval,
        severity_labels=severity_labels,
        severity_bucket_stats=severity_stats,
        stress_correlation_matrix=stress_corr,
        normal_correlation_matrix=normal_corr,
        stress_portfolio_loss_summary=port_losses_summary,
        train_gen_fraction=train_gen_frac,
        anchor_fraction=anchor_frac,
        struct_eval_fraction=struct_eval_frac,
    )

    logger.info(
        f"Tail extract: train_gen={len(D_train_gen)}, "
        f"anchor={len(D_anchor)}, struct_eval={len(D_struct_eval)}"
    )
    return extract


def validate_tail_extract_no_overlap(
    train_gen: pd.DataFrame,
    anchor: pd.DataFrame,
    struct_eval: pd.DataFrame,
) -> None:
    """Raise ValueError if any two subsets share dates (circular contamination)."""
    sets = [
        ("train_gen", set(train_gen.index)),
        ("anchor", set(anchor.index)),
        ("struct_eval", set(struct_eval.index)),
    ]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i][1] & sets[j][1]
            if overlap:
                raise ValueError(
                    f"Circular contamination: {sets[i][0]} and {sets[j][0]} "
                    f"share {len(overlap)} dates."
                )
