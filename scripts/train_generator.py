"""CLI: Train the CVAE stress scenario generator.

Usage
-----
python -m scripts.train_generator --config configs/experiment/full_method.yaml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

from src.utils.logging_utils import setup_logging
from src.utils.seeds import set_seed
from src.utils.config import load_experiment_config
from src.utils.io import load_dataframe, load_pickle, save_json, ensure_dir
from src.regimes.stress_index import StressIndex
from src.regimes.partition import create_partition
from src.tail.extractor import extract_tail
from src.generator.cvae import CVAE
from src.generator.train import (
    prepare_stress_training_data, train_cvae, save_cvae_checkpoint
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None)
@click.option("--log-level", default="INFO")
def main(config: str, output_dir: str | None, log_level: str) -> None:
    """Train CVAE on D_stress_train_gen with severity conditioning."""
    cfg = load_experiment_config(config)
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    out_dir = Path(output_dir or exp_cfg.get("output_dir", f"outputs/experiments/{exp_name}"))

    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name=exp_name)
    set_seed(exp_cfg.get("seed", 42))

    data_dir = out_dir / "data"
    model_dir = out_dir / "models"
    ensure_dir(model_dir)

    # ------------------------------------------------------------------
    # Load prepared data
    # ------------------------------------------------------------------
    returns = load_dataframe(data_dir / "returns_clean.parquet")
    weights = load_dataframe(data_dir / "portfolio_weights.parquet").squeeze()
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    train_returns = returns.iloc[split_info["train_idx"]]

    # ------------------------------------------------------------------
    # Reconstruct partition
    # ------------------------------------------------------------------
    feat_cfg = cfg.get("features", cfg)
    si_state = json.load(open(model_dir / "stress_index_state.json"))
    si = StressIndex.from_dict(si_state, cfg)
    stress_score, regime_labels = si.transform(train_returns, weights, feat_cfg)
    partition = create_partition(stress_score, regime_labels, si.alpha, si.threshold_)

    # ------------------------------------------------------------------
    # Extract tail
    # ------------------------------------------------------------------
    tail = extract_tail(partition, train_returns, weights, cfg)
    logger.info(f"Tail extract: {tail.summary()}")

    # Save tail artifacts
    tail.D_stress_train_gen.to_parquet(model_dir / "D_stress_train_gen.parquet")
    tail.D_stress_anchor.to_parquet(model_dir / "D_stress_anchor.parquet")
    if len(tail.D_stress_struct_eval) > 0:
        tail.D_stress_struct_eval.to_parquet(model_dir / "D_stress_struct_eval.parquet")
    tail.severity_labels.to_frame().to_parquet(model_dir / "severity_labels.parquet")
    tail.severity_bucket_stats.to_parquet(model_dir / "severity_bucket_stats.parquet")

    # ------------------------------------------------------------------
    # Prepare CVAE training data
    # ------------------------------------------------------------------
    cvae_cfg = cfg.get("model_cvae", cfg.get("cvae", {}))
    n_buckets = int(cvae_cfg.get("n_severity_buckets", 3))

    X_train, C_train, scaler_params = prepare_stress_training_data(
        tail.D_stress_train_gen,
        tail.severity_labels,
        n_severity_buckets=n_buckets,
    )

    # Val split from training data
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_vl = X_train[:-n_val], X_train[-n_val:]
    C_tr, C_vl = C_train[:-n_val], C_train[-n_val:]

    # ------------------------------------------------------------------
    # Build and train CVAE
    # ------------------------------------------------------------------
    input_dim = X_train.shape[1]
    condition_dim = n_buckets
    if cvae_cfg.get("use_portfolio_context", False):
        condition_dim += 1  # add one portfolio context feature

    latent_dim = int(cvae_cfg.get("latent_dim", 16))
    hidden_dims = list(cvae_cfg.get("hidden_dims", [64, 64]))
    dropout = float(cvae_cfg.get("dropout", 0.1))

    cvae = CVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    history = train_cvae(
        cvae, X_tr, C_tr, cvae_cfg,
        X_val=X_vl, C_val=C_vl,
        checkpoint_dir=model_dir,
    )

    save_cvae_checkpoint(
        cvae, history, scaler_params, model_dir / "cvae_final.pt"
    )
    save_json(tail.summary(), model_dir / "tail_extract_summary.json")

    logger.info(f"Generator training complete. Checkpoint saved to {model_dir}")


if __name__ == "__main__":
    main()
