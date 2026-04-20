"""Research plotting utilities — all plots save to file and return Figure."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def save_figure(fig: plt.Figure, path: Path | str) -> None:
    """Save figure, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved figure -> {path}")


def plot_stress_index(
    stress_index: pd.Series,
    regime_labels: pd.Series,
    threshold: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot stress index over time with threshold and stress shading."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(stress_index.index, stress_index.values, color="#2c7bb6", lw=1.2,
            label="Stress index $S_t$")
    ax.axhline(threshold, color="#d7191c", lw=1.5, ls="--",
               label=f"Threshold ({threshold:.3f})")

    # Shade stress periods
    stress_mask = regime_labels.reindex(stress_index.index, fill_value=False)
    in_stress = False
    t0 = None
    for date, is_stress in stress_mask.items():
        if is_stress and not in_stress:
            t0 = date
            in_stress = True
        elif not is_stress and in_stress:
            ax.axvspan(t0, date, alpha=0.15, color="#d7191c")
            in_stress = False
    if in_stress:
        ax.axvspan(t0, stress_index.index[-1], alpha=0.15, color="#d7191c")

    ax.set_title("Formal Stress Index $S_t$")
    ax.set_ylabel("Composite score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_regime_distribution(
    regime_labels: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot rolling regime share and overall bar chart."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Rolling share
    roll_30 = regime_labels.astype(int).rolling(30).mean()
    roll_90 = regime_labels.astype(int).rolling(90).mean()
    ax = axes[0]
    ax.plot(roll_30.index, roll_30.values, lw=1, label="30d rolling stress share")
    ax.plot(roll_90.index, roll_90.values, lw=1.5, label="90d rolling stress share")
    ax.set_title("Rolling stress regime share")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Bar chart
    ax2 = axes[1]
    counts = regime_labels.value_counts()
    labels = ["Normal", "Stress"]
    values = [counts.get(False, 0), counts.get(True, 0)]
    bars = ax2.bar(labels, values, color=["#2c7bb6", "#d7191c"], alpha=0.8)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{v}\n({v/sum(values):.1%})", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Regime counts")
    ax2.set_ylabel("Number of observations")

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_loss_distribution(
    losses: np.ndarray,
    weights: np.ndarray,
    var_levels: dict[str, float],
    es_levels: dict[str, float],
    title: str = "Portfolio Loss Distribution",
    save_path: Path | None = None,
) -> plt.Figure:
    """Weighted empirical loss distribution with VaR/ES markers."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort and cumulate
    order = np.argsort(losses)
    sl = losses[order]
    sw = weights[order]
    cdf = np.cumsum(sw)

    ax.plot(sl, cdf, color="#2c7bb6", lw=2, label="Weighted ECDF")
    ax.fill_between(sl, cdf, alpha=0.1, color="#2c7bb6")

    colors = ["#d7191c", "#fdae61", "#1a9641"]
    for i, (lbl, var_val) in enumerate(var_levels.items()):
        c = colors[i % len(colors)]
        ax.axvline(var_val, color=c, lw=1.5, ls="--", label=f"VaR {lbl}: {var_val:.4f}")
    for i, (lbl, es_val) in enumerate(es_levels.items()):
        c = colors[i % len(colors)]
        ax.axvline(es_val, color=c, lw=1.5, ls=":", label=f"ES {lbl}: {es_val:.4f}")

    ax.set_xlabel("Portfolio loss")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_latent_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototype_centers: np.ndarray | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """2D scatter of latent embeddings colored by prototype assignment.
    Uses PCA to reduce to 2D if latent_dim > 2.
    """
    _apply_style()

    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        emb2d = pca.fit_transform(embeddings)
        centers2d = pca.transform(prototype_centers) if prototype_centers is not None else None
        xlabel = "PC 1"
        ylabel = "PC 2"
    else:
        emb2d = embeddings
        centers2d = prototype_centers
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap="tab20",
                    s=10, alpha=0.5, linewidths=0)
    if centers2d is not None:
        ax.scatter(centers2d[:, 0], centers2d[:, 1], c="black", marker="X",
                   s=100, zorder=5, label="Prototypes")
        ax.legend()
    plt.colorbar(sc, ax=ax, label="Prototype assignment")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Normal-state latent space (prototype assignments)")

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_prototype_weights(
    weights: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of prototype weights."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(6, len(weights) // 2), 4))
    ax.bar(range(len(weights)), weights, color="#2c7bb6", alpha=0.8)
    ax.set_xlabel("Prototype index")
    ax.set_ylabel("Weight (empirical mass)")
    ax.set_title("Normal-prototype weights")

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_backtesting_results(
    realized_losses: pd.Series,
    var_series: pd.Series,
    exceedances: pd.Series,
    confidence_level: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Realized losses vs VaR estimate with exceedance markers."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(realized_losses.index, realized_losses.values, color="#555", lw=0.8,
            label="Realized loss", alpha=0.7)
    ax.plot(var_series.index, var_series.values, color="#d7191c", lw=1.5,
            label=f"VaR ({confidence_level:.0%})")

    exc_dates = exceedances[exceedances == 1].index
    ax.scatter(exc_dates, realized_losses.loc[exc_dates], color="#d7191c",
               s=30, zorder=5, label=f"Exceedances ({len(exc_dates)})")

    ax.set_title(f"VaR Backtesting — {confidence_level:.0%} confidence")
    ax.set_ylabel("Portfolio loss")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_sensitivity(
    param_values: list,
    metric_values: list,
    param_name: str,
    metric_name: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Sensitivity of a risk metric to a method parameter."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(param_values, metric_values, marker="o", color="#2c7bb6", lw=1.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Sensitivity: {metric_name} vs {param_name}")

    if save_path:
        save_figure(fig, save_path)
    return fig
