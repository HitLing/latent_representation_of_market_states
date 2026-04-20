"""CLI: Restricted calibration of alpha, m, M, b on inner validation.

Calibration is strictly confined to the INNER VALIDATION split of the
training data.  The final test set is NEVER touched here.

Usage
-----
python -m scripts.calibrate_method --config configs/experiment/full_method.yaml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logging
from src.utils.seeds import set_seed
from src.utils.config import load_experiment_config
from src.utils.io import load_dataframe, save_json, ensure_dir
from src.evaluation.calibration import (
    calibrate_parameters, make_inner_val_split,
    compute_calibration_metric, log_calibration_summary,
)
from src.regimes.stress_index import StressIndex
from src.regimes.partition import create_partition
from src.risk.weighted_risk_metrics import compute_risk_metrics
from src.risk.portfolio_loss import compute_scenario_losses
from src.assembly.scenario_assembly import assemble_baseline_hs

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None)
@click.option("--log-level", default="INFO")
def main(config: str, output_dir: str | None, log_level: str) -> None:
    """Grid-search calibration on inner validation split."""
    cfg = load_experiment_config(config)
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    seed = int(exp_cfg.get("seed", 42))
    out_dir = Path(output_dir or exp_cfg.get("output_dir", f"outputs/experiments/{exp_name}"))

    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name=exp_name)
    set_seed(seed)
    ensure_dir(out_dir / "metrics")

    # ------------------------------------------------------------------
    # Load train data
    # ------------------------------------------------------------------
    returns = load_dataframe(out_dir / "data" / "returns_clean.parquet")
    weights = load_dataframe(out_dir / "data" / "portfolio_weights.parquet").squeeze()
    with open(out_dir / "data" / "split_info.json") as f:
        split_info = json.load(f)

    train_returns = returns.iloc[split_info["train_idx"]]
    feat_cfg = cfg.get("features", cfg)
    cal_cfg = cfg.get("calibration", {})

    # Inner validation split
    inner_train, inner_val = make_inner_val_split(
        train_returns,
        inner_val_ratio=cal_cfg.get("inner_val_ratio", 0.20),
    )
    port_losses_val = -(inner_val @ weights.reindex(inner_val.columns, fill_value=0.0) /
                        weights.reindex(inner_val.columns, fill_value=0.0).sum())

    # ------------------------------------------------------------------
    # Define calibration objective
    # ------------------------------------------------------------------
    params_cfg = cal_cfg.get("parameters", {})
    alpha_search = params_cfg.get("alpha", {}).get("search_values", [0.10])
    m_search = params_cfg.get("m", {}).get("search_values", [0.50])

    def _eval_params(params: dict) -> float:
        alpha = params.get("alpha", 0.10)
        m = params.get("m", 0.50)
        try:
            si = StressIndex({"regimes": {"alpha": alpha}})
            score, labels = si.fit_transform(inner_train, weights, feat_cfg)
            partition = create_partition(score, labels, alpha, si.threshold_)

            hist_stress = partition.filter_data(inner_train, "stress")
            normal_ret = partition.filter_data(inner_train, "normal")

            # Simple assembly: prototypes = raw normal (no VAE for calibration speed)
            assembly = assemble_baseline_hs(inner_train)

            losses, w = compute_scenario_losses(assembly, weights)
            metrics = compute_risk_metrics(losses, w, [0.99])
            var_99 = metrics.get("VaR_0.99", float("nan"))
            es_99 = metrics.get("ES_0.99", float("nan"))

            if np.isnan(var_99):
                return float("inf")

            metric = compute_calibration_metric(
                port_losses_val.values, var_99, es_99, alpha=0.99
            )
        except Exception as e:
            logger.debug(f"Calibration trial failed: {e}")
            metric = float("inf")
        return metric

    # ------------------------------------------------------------------
    # Run grid search
    # ------------------------------------------------------------------
    best_params, results = calibrate_parameters(
        train_fn=_eval_params,
        param_grid={"alpha": alpha_search, "m": m_search},
    )

    log_calibration_summary(best_params, results)
    save_json(best_params, out_dir / "metrics" / "best_calibration_params.json")
    results.to_csv(out_dir / "metrics" / "calibration_grid_results.csv")

    logger.info(f"Calibration complete. Best params: {best_params}")


if __name__ == "__main__":
    main()
