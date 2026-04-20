"""CLI: Run the full end-to-end experiment pipeline.

Stages
------
1. Data preparation
2. Stress index + partition
3. Normal VAE + prototypes
4. Tail extraction + CVAE training
5. Scenario assembly
6. Risk metric computation
7. OOS evaluation

Usage
-----
python -m scripts.run_full_experiment --config configs/experiment/full_method.yaml
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
from src.utils.io import (
    load_dataframe, save_dataframe, save_json, save_array,
    load_pickle, save_pickle, ensure_dir,
)
from src.utils.plotting import (
    plot_stress_index, plot_regime_distribution,
    plot_loss_distribution, plot_latent_space,
    plot_prototype_weights, save_figure,
)
from src.data.loaders import load_market_data
from src.data.preprocessing import preprocess_returns
from src.data.splits import make_temporal_split
from src.features.market_features import compute_all_features
from src.regimes.stress_index import StressIndex
from src.regimes.partition import create_partition, validate_partition_quality
from src.regimes.diagnostics import compute_regime_statistics, imbalance_report
from src.normal_model.vae import VAE, train_vae, encode_data, save_vae
from src.normal_model.prototypes import fit_prototypes, save_prototypes
from src.tail.extractor import extract_tail
from src.generator.cvae import CVAE
from src.generator.train import prepare_stress_training_data, train_cvae, save_cvae_checkpoint
from src.generator.inference import (
    generate_stress_scenarios, compute_generation_budget, validate_generated_scenarios,
)
from src.assembly.scenario_assembly import assemble_scenarios, assemble_baseline_hs
from src.risk.portfolio_loss import compute_scenario_losses, compute_portfolio_loss_series
from src.risk.weighted_risk_metrics import compute_risk_metrics
from src.risk.backtesting import run_full_backtest, compute_var_exceedances
from src.data.preprocessing import standardize_features
from src.features.portfolio_features import compute_portfolio_returns

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None)
@click.option("--log-level", default="INFO")
def main(config: str, output_dir: str | None, log_level: str) -> None:
    """Run full experiment pipeline end-to-end."""
    cfg = load_experiment_config(config)
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    seed = int(exp_cfg.get("seed", 42))
    out_dir = Path(output_dir or exp_cfg.get("output_dir", f"outputs/experiments/{exp_name}"))

    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name=exp_name)
    set_seed(seed)

    abl = cfg.get("ablation", {})
    use_stress_idx = abl.get("use_stress_index", True)
    use_prototypes = abl.get("use_normal_prototypes", True)
    use_generator = abl.get("use_stress_generator", True)

    for d in ["data", "models", "figures", "metrics", "artifacts"]:
        ensure_dir(out_dir / d)

    # ---------------------------------------------------------------
    # 1. Load and preprocess data
    # ---------------------------------------------------------------
    data_cfg = cfg.get("data", cfg)
    returns, weights = load_market_data(data_cfg)
    returns = preprocess_returns(returns, data_cfg)
    save_dataframe(returns, out_dir / "data" / "returns_clean.parquet")

    feat_cfg = cfg.get("features", cfg)
    eval_cfg = cfg.get("evaluation", cfg)
    split = make_temporal_split(
        returns.index,
        train_ratio=eval_cfg.get("train_ratio", 0.60),
        val_ratio=eval_cfg.get("val_ratio", 0.20),
        test_ratio=eval_cfg.get("test_ratio", 0.20),
    )
    save_json({
        "train_idx": split.train_idx.tolist(),
        "val_idx": split.val_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
    }, out_dir / "data" / "split_info.json")

    train_returns = returns.iloc[split.train_idx]
    test_returns = returns.iloc[split.test_idx]

    port_returns_all = compute_portfolio_returns(returns, weights)
    port_losses_test = -compute_portfolio_returns(test_returns, weights)

    # ---------------------------------------------------------------
    # 2. Stress index + partition
    # ---------------------------------------------------------------
    alpha = cfg.get("regimes", {}).get("alpha",
                    cfg.get("calibration", {}).get("parameters", {}).get("alpha", {}).get("default", 0.10))

    if use_stress_idx:
        si = StressIndex(cfg)
        si.alpha = alpha
        stress_score, regime_labels = si.fit_transform(train_returns, weights, feat_cfg)
        partition = create_partition(stress_score, regime_labels, si.alpha, si.threshold_)
        validate_partition_quality(partition)
        logger.info(imbalance_report(partition, train_returns))

        # Save regime artifacts
        partition.to_dataframe().to_parquet(out_dir / "artifacts" / "partition.parquet")
        stress_score.to_frame().to_parquet(out_dir / "artifacts" / "stress_index.parquet")
        save_json(si.to_dict(), out_dir / "models" / "stress_index_state.json")

        # Sensitivity analysis
        alpha_vals = cfg.get("evaluation", {}).get("sensitivity", {}).get("alpha_values",
                              [0.05, 0.08, 0.10, 0.12, 0.15])
        sens = si.sensitivity_analysis(train_returns, weights, feat_cfg, alpha_vals)
        sens.to_csv(out_dir / "metrics" / "alpha_sensitivity.csv")

        # Plot
        fig = plot_stress_index(stress_score, regime_labels, si.threshold_,
                                save_path=out_dir / "figures" / "stress_index.png")
        fig = plot_regime_distribution(regime_labels,
                                       save_path=out_dir / "figures" / "regime_distribution.png")
    else:
        # Baseline: no partition — treat all as "normal"
        partition = None

    # ---------------------------------------------------------------
    # 3. Normal VAE + prototypes
    # ---------------------------------------------------------------
    vae_cfg = cfg.get("model_vae", cfg.get("vae", {}))
    M = int(vae_cfg.get("M", 20))
    prototype_model = None

    if use_stress_idx and use_prototypes and partition is not None:
        normal_returns = partition.filter_data(train_returns, "normal")
        X_normal = normal_returns.to_numpy(dtype=np.float32)
        X_scaled, scaler_params = standardize_features(X_normal)
        save_pickle(scaler_params, out_dir / "models" / "normal_scaler.pkl")

        n_inner_val = max(1, int(len(X_scaled) * 0.15))
        X_vae_train = X_scaled[:-n_inner_val]
        X_vae_val = X_scaled[-n_inner_val:]

        input_dim = X_scaled.shape[1]
        vae = VAE(input_dim, list(vae_cfg.get("hidden_dims", [64, 32])),
                  int(vae_cfg.get("latent_dim", 8)), float(vae_cfg.get("dropout", 0.1)))
        train_vae(vae, X_vae_train, vae_cfg, val_data=X_vae_val)
        save_vae(vae, out_dir / "models" / "vae.pt")

        embeddings = encode_data(vae, X_scaled)
        save_array(embeddings, out_dir / "artifacts" / "normal_embeddings.npy")

        prototype_model = fit_prototypes(embeddings, M=M,
                                         method=vae_cfg.get("prototype_method", "kmeans"),
                                         seed=seed)
        save_prototypes(prototype_model, out_dir / "models" / "prototypes.npz")
        save_array(prototype_model.assignments,
                   out_dir / "artifacts" / "prototype_assignments.npy")

        fig = plot_latent_space(embeddings, prototype_model.assignments,
                                prototype_model.centers,
                                save_path=out_dir / "figures" / "latent_space.png")
        fig = plot_prototype_weights(prototype_model.weights,
                                     save_path=out_dir / "figures" / "prototype_weights.png")

    # ---------------------------------------------------------------
    # 4. Tail extraction + CVAE
    # ---------------------------------------------------------------
    cvae_cfg = cfg.get("model_cvae", cfg.get("cvae", {}))
    n_buckets = int(cvae_cfg.get("n_severity_buckets", 3))
    m_ratio = cfg.get("assembly", {}).get("m",
              cfg.get("calibration", {}).get("parameters", {}).get("m", {}).get("default", 0.50))

    syn_scenarios = pd.DataFrame()
    tail_extract = None

    if use_stress_idx and use_generator and partition is not None:
        tail_extract = extract_tail(partition, train_returns, weights, cfg)
        save_json(tail_extract.summary(), out_dir / "metrics" / "tail_extract_summary.json")
        tail_extract.D_stress_anchor.to_parquet(out_dir / "artifacts" / "D_stress_anchor.parquet")

        X_tr_cvae, C_tr_cvae, scaler_cvae = prepare_stress_training_data(
            tail_extract.D_stress_train_gen, tail_extract.severity_labels, n_buckets
        )
        n_cv = max(1, int(len(X_tr_cvae) * 0.15))
        cvae = CVAE(
            input_dim=X_tr_cvae.shape[1],
            condition_dim=n_buckets,
            hidden_dims=list(cvae_cfg.get("hidden_dims", [64, 64])),
            latent_dim=int(cvae_cfg.get("latent_dim", 16)),
            dropout=float(cvae_cfg.get("dropout", 0.1)),
        )
        train_cvae(cvae, X_tr_cvae[:-n_cv], C_tr_cvae[:-n_cv], cvae_cfg,
                   X_val=X_tr_cvae[-n_cv:], C_val=C_tr_cvae[-n_cv:],
                   checkpoint_dir=out_dir / "models")
        save_cvae_checkpoint(cvae, {}, scaler_cvae, out_dir / "models" / "cvae_final.pt")

        budget = compute_generation_budget(
            tail_extract.D_stress_anchor, m_ratio,
            tail_extract.severity_labels, n_buckets,
        )
        syn_scenarios = generate_stress_scenarios(
            cvae, budget, n_buckets, scaler_cvae,
            asset_names=train_returns.columns.tolist(), seed=seed,
        )
        syn_scenarios.to_parquet(out_dir / "artifacts" / "synthetic_scenarios.parquet")
        val_result = validate_generated_scenarios(syn_scenarios, tail_extract.D_stress_train_gen)
        save_json(val_result, out_dir / "metrics" / "generation_validation.json")

    # ---------------------------------------------------------------
    # 5. Assembly
    # ---------------------------------------------------------------
    asset_cols = train_returns.columns.tolist()
    if use_stress_idx and partition is not None:
        hist_stress = partition.filter_data(train_returns, "stress")
        normal_ret = partition.filter_data(train_returns, "normal")
    else:
        hist_stress = pd.DataFrame(columns=asset_cols)
        normal_ret = train_returns

    if len(syn_scenarios) > 0:
        syn_for_assembly = syn_scenarios[[c for c in asset_cols if c in syn_scenarios.columns]]
    else:
        syn_for_assembly = pd.DataFrame(columns=asset_cols)

    if use_stress_idx and partition is not None:
        assembly = assemble_scenarios(
            partition, prototype_model, normal_ret, hist_stress,
            syn_for_assembly, m=m_ratio,
            use_normal_prototypes=use_prototypes and prototype_model is not None,
            asset_columns=asset_cols,
        )
    else:
        assembly = assemble_baseline_hs(train_returns, asset_columns=asset_cols)

    assembly.save(out_dir / "artifacts" / "scenario_assembly.parquet")
    save_json(assembly.summary(), out_dir / "metrics" / "assembly_summary.json")

    # ---------------------------------------------------------------
    # 6. Risk metrics
    # ---------------------------------------------------------------
    conf_levels = eval_cfg.get("confidence_levels", [0.95, 0.99])
    losses, scenario_weights = compute_scenario_losses(assembly, weights)
    save_array(losses, out_dir / "artifacts" / "scenario_losses.npy")
    save_array(scenario_weights, out_dir / "artifacts" / "scenario_weights.npy")

    metrics = compute_risk_metrics(losses, scenario_weights, conf_levels)
    save_json(metrics, out_dir / "metrics" / "risk_metrics.json")

    var_levels = {str(cl): metrics[f"VaR_{cl}"] for cl in conf_levels if f"VaR_{cl}" in metrics}
    es_levels = {str(cl): metrics[f"ES_{cl}"] for cl in conf_levels if f"ES_{cl}" in metrics}
    fig = plot_loss_distribution(losses, scenario_weights, var_levels, es_levels,
                                 title=f"Loss Distribution — {exp_name}",
                                 save_path=out_dir / "figures" / "loss_distribution.png")

    # ---------------------------------------------------------------
    # 7. OOS evaluation (fixed VaR applied to test set)
    # ---------------------------------------------------------------
    for cl in conf_levels:
        var_val = metrics.get(f"VaR_{cl}")
        es_val = metrics.get(f"ES_{cl}")
        if var_val is None:
            continue
        # Fixed-VaR backtest: same VaR applied to all OOS days
        var_series = pd.Series(var_val, index=test_returns.index, name=f"VaR_{cl}")
        es_series = pd.Series(es_val, index=test_returns.index, name=f"ES_{cl}")
        bt = run_full_backtest(port_losses_test, var_series, es_series, cl)
        save_json(
            {k: v for k, v in bt.items() if k != "exceedances"},
            out_dir / "metrics" / f"backtest_{cl}.json"
        )
        bt["exceedances"].to_frame().to_parquet(
            out_dir / "artifacts" / f"exceedances_{cl}.parquet"
        )
        from src.utils.plotting import plot_backtesting_results
        fig = plot_backtesting_results(
            port_losses_test, var_series, bt["exceedances"], cl,
            save_path=out_dir / "figures" / f"backtest_{cl}.png"
        )
        logger.info(
            f"OOS backtest CL={cl}: "
            f"exc_rate={bt['kupiec']['empirical_rate']:.3%}, "
            f"Kupiec p={bt['kupiec']['p_value']:.3f}"
        )

    logger.info(f"\n=== Experiment '{exp_name}' complete. Artifacts in {out_dir} ===")
    logger.info(f"Risk metrics: {metrics}")


if __name__ == "__main__":
    main()
