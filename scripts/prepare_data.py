"""CLI: Prepare and preprocess market data.

Usage
-----
python -m scripts.prepare_data --config configs/experiment/full_method.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from src.utils.logging_utils import setup_logging
from src.utils.seeds import set_seed
from src.utils.config import load_yaml, load_experiment_config
from src.utils.io import save_dataframe, ensure_dir, save_json
from src.data.loaders import load_market_data
from src.data.preprocessing import preprocess_returns
from src.features.market_features import compute_all_features
from src.features.portfolio_features import compute_portfolio_returns
from src.data.windowing import build_windows
from src.data.splits import make_temporal_split

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to experiment YAML config.")
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(config: str, output_dir: str | None, log_level: str) -> None:
    """Prepare and save market data, features, and windowed tensors."""
    cfg = load_experiment_config(config)
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    out_dir = Path(output_dir or exp_cfg.get("output_dir", f"outputs/experiments/{exp_name}"))

    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name=exp_name)
    set_seed(exp_cfg.get("seed", 42))

    ensure_dir(out_dir / "data")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", cfg)
    returns, weights = load_market_data(data_cfg)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    returns_clean = preprocess_returns(returns, data_cfg)
    save_dataframe(returns_clean, out_dir / "data" / "returns_clean.parquet")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    split = make_temporal_split(
        returns_clean.index,
        train_ratio=eval_cfg.get("train_ratio", 0.60),
        val_ratio=eval_cfg.get("val_ratio", 0.20),
        test_ratio=eval_cfg.get("test_ratio", 0.20),
    )

    # ------------------------------------------------------------------
    # 4. Compute features on train set (fit scalers on train only)
    # ------------------------------------------------------------------
    feat_cfg = cfg.get("features", cfg)
    train_returns = returns_clean.iloc[split.train_idx]
    features = compute_all_features(train_returns, weights, feat_cfg)
    features_full = compute_all_features(returns_clean, weights, feat_cfg)

    save_dataframe(features_full, out_dir / "data" / "features.parquet")

    # ------------------------------------------------------------------
    # 5. Build windows
    # ------------------------------------------------------------------
    features_filled = features_full.ffill().bfill()
    windows = build_windows(
        features_filled,
        window_size=feat_cfg.get("window_size", 21),
        step_size=feat_cfg.get("step_size", 1),
    )
    from src.utils.io import save_array, save_pickle
    save_array(windows.windows, out_dir / "data" / "windows.npy")
    save_pickle(windows.dates, out_dir / "data" / "window_dates.pkl")
    save_pickle(windows.feature_names, out_dir / "data" / "feature_names.pkl")

    # ------------------------------------------------------------------
    # 6. Save weights and split info
    # ------------------------------------------------------------------
    save_dataframe(weights.to_frame(), out_dir / "data" / "portfolio_weights.parquet")
    save_json(split.__dict__ | {
        "train_idx": split.train_idx.tolist(),
        "val_idx": split.val_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
    }, out_dir / "data" / "split_info.json")

    logger.info(f"Data preparation complete. Artifacts saved to {out_dir / 'data'}")


if __name__ == "__main__":
    main()
