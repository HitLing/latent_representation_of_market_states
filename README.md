# Tail-Risk Estimation via Latent Representation of Market States

**Diploma thesis research project**  
A formal, reproducible methodology for tail-risk estimation of a trading portfolio under regime shifts, using latent representations of market states.

---

## Research Motivation

Historical market data is imbalanced: normal market conditions dominate while stress episodes are rare. Traditional historical simulation may be unstable or inaccurate for tail-risk estimation precisely when it matters most — during or after stress regimes. This project implements a principled, step-by-step methodology that:

1. **Measures** stress explicitly via a formal composite stress index
2. **Partitions** the data into normal and stress regimes
3. **Stabilises** the bulk distribution using normal-state prototypes (VAE + clustering)
4. **Enriches** the tail with conditional synthetic stress scenarios (CVAE)
5. **Assembles** a weighted scenario set with explicit, formula-driven weights
6. **Estimates** VaR/ES from the weighted empirical loss distribution
7. **Evaluates** rigorously via three separated evaluation stages

---

## Method Overview

```
Raw market data (returns + portfolio weights)
          │
          ▼
┌─────────────────────────┐
│   Formal Stress Index   │  S_t = mean(normalised indicators)
│  return_shock           │  Threshold: q_{1-alpha}
│  vol_spike              │
│  correlation_stress     │
│  portfolio_loss_proxy   │
└────────────┬────────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
D_normal           D_stress
    │                  │
    ▼                  ▼
┌──────────────┐   ┌──────────────────────┐
│ VAE Encoder  │   │  Tail Extractor      │
│ (latent rep) │   │  D_stress_train_gen  │
│     +        │   │  D_stress_anchor     │
│  K-Means     │   │  D_stress_struct_eval│
│  Prototypes  │   └──────────┬───────────┘
│  (M centres) │              │
└──────┬───────┘              ▼
       │             ┌─────────────────┐
       │             │   CVAE (cond.)  │
       │             │  severity bucket│
       │             │  conditioning   │
       │             └──────┬──────────┘
       │                    │
       ▼                    ▼
   S_normal_proto     S_stress_syn
   (M weighted)     (m x hist mass)
       │                    │
       │         S_stress_hist (anchor)
       │                    │
       └────────┬───────────┘
                ▼
    ┌───────────────────────┐
    │  Scenario Assembly    │  Explicit weighted distribution
    │  w_total = 1.0        │  sum(w_syn) = m * sum(w_hist)
    └───────────┬───────────┘
                ▼
    ┌───────────────────────┐
    │  Portfolio Loss       │  loss_i = -sum_j w_j * r_{ij}
    │  Distribution Engine  │
    └───────────┬───────────┘
                ▼
    ┌───────────────────────┐
    │  Weighted VaR / ES    │  Transparent closed-form
    └───────────────────────┘
```

---

## Data Expectations

| Item | Format | Notes |
|------|--------|-------|
| Market returns | CSV or Parquet | DatetimeIndex, one column per asset |
| Portfolio weights | CSV or Parquet | Single column `weight`, index = asset names |
| Synthetic data | Auto-generated | If `use_synthetic: true` in config |

**Minimum requirements:** >= 500 trading days, >= 2 assets.

The synthetic data generator (`src/data/loaders.py`) produces realistic multivariate returns with stress episodes (Markov-chain regimes, Cholesky-correlated, crisis asymmetry). It is a transparent, documented fallback for testing the full pipeline without proprietary data.

---

## Setup

```bash
# Install with uv (recommended)
pip install uv
uv pip install -e ".[dev]"

# Or standard pip
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, PyTorch 2.1+

---

## Running the Pipeline

### Quickstart (full method, synthetic data)

```bash
python -m scripts.run_full_experiment \
    --config configs/experiment/full_method.yaml \
    --log-level INFO
```

### Step-by-step

```bash
# 1. Prepare data and features
python -m scripts.prepare_data --config configs/experiment/full_method.yaml

# 2. Train normal-state VAE + prototypes
python -m scripts.train_normal_model --config configs/experiment/full_method.yaml

# 3. Train CVAE stress generator
python -m scripts.train_generator --config configs/experiment/full_method.yaml

# 4. Calibrate alpha, m, M on inner validation (NEVER touches test set)
python -m scripts.calibrate_method --config configs/experiment/full_method.yaml

# 5. Run full experiment (assembly + risk + OOS backtest)
python -m scripts.run_full_experiment --config configs/experiment/full_method.yaml

# 6. Formal OOS evaluation (run only once, after all decisions are finalised)
python -m scripts.evaluate_oos --experiment-dir outputs/experiments/full_method
```

### Run ablations

```bash
python -m scripts.run_ablation --ablation baseline_hs
python -m scripts.run_ablation --ablation ablation_no_generator
python -m scripts.run_ablation --ablation ablation_no_proto
python -m scripts.run_ablation --ablation full_method
```

---

## Experiment Configuration

All settings are driven by YAML configs under `configs/`.

```yaml
# configs/experiment/full_method.yaml
experiment:
  name: "full_method"
  seed: 42
  output_dir: "outputs/experiments/full_method"

ablation:
  use_stress_index: true
  use_normal_prototypes: true
  use_stress_generator: true

_defaults_:
  data: default         # configs/data/default.yaml
  features: default     # configs/features/default.yaml
  model/vae: vae        # configs/model/vae.yaml
  model/cvae: cvae      # configs/model/cvae.yaml
  calibration: default
  evaluation: default
```

Key configurable parameters:

| Parameter | Config file | Role |
|-----------|-------------|------|
| `alpha` | `calibration/default.yaml` | Stress threshold — main structural param |
| `m` | `calibration/default.yaml` | Synthetic-to-historical mass ratio |
| `M` | `model/vae.yaml` | Number of normal prototypes |
| `n_severity_buckets` | `model/cvae.yaml` | CVAE condition dimension |
| `confidence_levels` | `evaluation/default.yaml` | VaR/ES levels |

---

## Baselines and Ablations

| Name | Stress index | Prototypes | Generator | Question answered |
|------|:---:|:---:|:---:|-------------------|
| `baseline_hs` | No | No | No | Pure historical simulation baseline |
| `ablation_no_generator` | Yes | Yes | No | Does synthetic tail enrichment improve OOS? |
| `ablation_no_proto` | Yes | No | Yes | What does prototype stabilisation add? |
| `full_method` | Yes | Yes | Yes | Combined effect of all components |

---

## Three-Level Evaluation Protocol

| Stage | Data used | Purpose |
|-------|-----------|---------|
| **A. Calibration** | Inner validation (train subset) | Select alpha, m, M, b |
| **B. Final OOS** | Held-out test set (touched once) | Kupiec, Christoffersen, ES backtest |
| **C. Structural** | D_stress_struct_eval + bootstrap | Plausibility, dependence, sensitivity |

---

## Saved Artifacts

All outputs land in `outputs/experiments/{experiment_name}/`:

```
data/
  returns_clean.parquet        Preprocessed returns
  features.parquet             Computed market features
  windows.npy                  Rolling window tensor (N, T, F)
  split_info.json              Train/val/test indices
  portfolio_weights.parquet    Portfolio weights

models/
  vae.pt                       Trained VAE state dict
  prototypes.npz               Prototype centres + weights
  cvae_final.pt                Trained CVAE checkpoint + scaler
  stress_index_state.json      Fitted stress index parameters

artifacts/
  stress_index.parquet         S_t time series
  partition.parquet            Regime labels (is_stress)
  normal_embeddings.npy        VAE latent embeddings for D_normal
  prototype_assignments.npy    Prototype assignment per observation
  D_stress_anchor.parquet      Historical tail anchor scenarios
  synthetic_scenarios.parquet  Generated synthetic scenarios
  scenario_assembly.parquet    Full weighted scenario set + labels
  scenario_losses.npy          Per-scenario portfolio losses
  scenario_weights.npy         Per-scenario probability weights
  exceedances_{cl}.parquet     VaR exceedance indicators

metrics/
  risk_metrics.json            VaR/ES estimates
  assembly_summary.json        Scenario assembly statistics
  backtest_{cl}.json           OOS backtest results per CL
  oos_evaluation_results.json  Combined OOS evaluation
  alpha_sensitivity.csv        Sensitivity of partition to alpha
  calibration_grid_results.csv Calibration grid search results

figures/
  stress_index.png
  regime_distribution.png
  latent_space.png
  prototype_weights.png
  loss_distribution.png
  backtest_{cl}.png
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: stress indicators, stress index, regime partition, prototypes, tail extractor, weighted VaR/ES, backtesting, and scenario assembly.

---

## Repository Structure

```
configs/           YAML configs (data, features, model, experiment, calibration, evaluation)
data/raw/          Raw input data (or auto-generated synthetic)
src/
  data/            Loaders, preprocessing, windowing, splits
  features/        Market features, portfolio features, stress indicators
  regimes/         Stress index, partition, diagnostics
  normal_model/    VAE, prototypes, stability analysis
  tail/            Tail extractor, severity buckets, dependence analysis
  generator/       CVAE model, training, inference, diagnostics
  assembly/        Weight computation, scenario assembly
  risk/            Portfolio loss engine, weighted VaR/ES, backtesting
  evaluation/      Calibration protocol, OOS evaluation, structural eval, ablations
  utils/           Logging, seeds, config loading, IO, plotting
scripts/           One CLI script per pipeline stage
tests/             Unit tests (pytest)
outputs/           Experiment results (git-ignored)
```

---

## Methodological Guarantees

- **No black-box risk prediction.** VaR and ES are computed from an explicit weighted scenario distribution via closed-form formulas.
- **Synthetic scenarios enrich, never replace.** The constraint `sum(w_syn) = m * sum(w_hist)` is enforced in code and verified at runtime.
- **Strict evaluation separation.** Calibration, OOS evaluation, and structural validation use non-overlapping data subsets.
- **Full reproducibility.** Random seeds are set globally at entry. All configs, scaler parameters, and model states are saved alongside artifacts.
- **Transparent stress index.** All indicator formulas, normalisation, and aggregation logic are explicit in `src/features/stress_indicators.py` and `src/regimes/stress_index.py`.
