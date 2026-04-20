"""CLI: Formal OOS evaluation on the held-out test set.

Loads pre-computed risk metrics and evaluates them against realised
test losses.  This script must only be run ONCE after all calibration
and modelling decisions are finalised.

Usage
-----
python -m scripts.evaluate_oos --experiment-dir outputs/experiments/full_method
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logging
from src.utils.io import load_dataframe, load_json, save_json
from src.risk.backtesting import run_full_backtest
from src.risk.weighted_risk_metrics import compute_quantile_loss_series
from src.utils.plotting import plot_backtesting_results

logger = logging.getLogger(__name__)


@click.command()
@click.option("--experiment-dir", "-d", required=True, type=click.Path(exists=True),
              help="Experiment output directory.")
@click.option("--log-level", default="INFO")
def main(experiment_dir: str, log_level: str) -> None:
    """Run formal OOS evaluation using pre-computed risk metrics."""
    out_dir = Path(experiment_dir)
    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name="oos_eval")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    returns = load_dataframe(out_dir / "data" / "returns_clean.parquet")
    weights = load_dataframe(out_dir / "data" / "portfolio_weights.parquet").squeeze()
    with open(out_dir / "data" / "split_info.json") as f:
        split_info = json.load(f)

    test_returns = returns.iloc[split_info["test_idx"]]
    port_losses_test = -(test_returns @ weights.reindex(test_returns.columns, fill_value=0.0) /
                         weights.reindex(test_returns.columns, fill_value=0.0).sum())
    port_losses_test.name = "portfolio_loss"

    # ------------------------------------------------------------------
    # Load risk metrics
    # ------------------------------------------------------------------
    metrics_path = out_dir / "metrics" / "risk_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Risk metrics not found at {metrics_path}. Run experiment first.")

    metrics = load_json(metrics_path)
    logger.info(f"Loaded risk metrics: {metrics}")

    # ------------------------------------------------------------------
    # Run backtests
    # ------------------------------------------------------------------
    conf_levels = [0.95, 0.99]
    all_results: dict = {}

    for cl in conf_levels:
        var_val = metrics.get(f"VaR_{cl}")
        es_val = metrics.get(f"ES_{cl}")
        if var_val is None:
            continue

        var_series = pd.Series(var_val, index=test_returns.index)
        es_series = pd.Series(es_val, index=test_returns.index)

        bt = run_full_backtest(port_losses_test, var_series, es_series, cl)
        ql = compute_quantile_loss_series(var_series, port_losses_test, cl)

        result = {
            "confidence_level": cl,
            "kupiec": {k: v for k, v in bt["kupiec"].items() if k != "exceedances"},
            "christoffersen": {k: v for k, v in bt["christoffersen"].items()},
            "es_backtest": bt["es_backtest"],
            "mean_quantile_loss": float(ql.mean()),
        }
        all_results[f"CL_{cl}"] = result

        # Plot
        fig = plot_backtesting_results(
            port_losses_test, var_series, bt["exceedances"], cl,
            save_path=out_dir / "figures" / f"oos_backtest_{cl}.png"
        )

        logger.info(
            f"OOS [{cl:.0%}]: exc_rate={bt['kupiec']['empirical_rate']:.3%} "
            f"(exp={bt['kupiec']['expected_rate']:.3%}), "
            f"Kupiec p={bt['kupiec']['p_value']:.3f}, "
            f"CC p={bt['christoffersen']['p_value_cc']:.3f}, "
            f"mean_QL={ql.mean():.5f}"
        )

    save_json(all_results, out_dir / "metrics" / "oos_evaluation_results.json")
    logger.info(f"OOS evaluation complete. Results saved to {out_dir / 'metrics'}")


if __name__ == "__main__":
    main()
