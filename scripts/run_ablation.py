"""CLI: Run a named ablation experiment.

Usage
-----
python -m scripts.run_ablation --ablation ablation_no_proto
python -m scripts.run_ablation --ablation baseline_hs
python -m scripts.run_ablation --ablation full_method
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from src.evaluation.ablations import ABLATION_CONFIGS, get_ablation_config
from src.utils.logging_utils import setup_logging
from src.utils.config import load_experiment_config

logger = logging.getLogger(__name__)


@click.command()
@click.option("--ablation", "-a", required=True,
              type=click.Choice(list(ABLATION_CONFIGS.keys())),
              help="Ablation variant to run.")
@click.option("--base-config", "-c",
              default="configs/experiment/full_method.yaml",
              type=click.Path(exists=True),
              help="Base experiment config (ablation flags will override).")
@click.option("--output-dir", "-o", default=None)
@click.option("--log-level", default="INFO")
def main(ablation: str, base_config: str, output_dir: str | None, log_level: str) -> None:
    """Run an ablation variant by injecting ablation flags into the base config."""
    abl_config = get_ablation_config(ablation)

    # Load base config
    cfg = load_experiment_config(base_config)

    # Override experiment name and ablation flags
    cfg["experiment"]["name"] = ablation
    cfg["experiment"]["description"] = abl_config.description
    default_out = f"outputs/experiments/{ablation}"
    cfg["experiment"]["output_dir"] = output_dir or default_out

    cfg["ablation"] = {
        "use_stress_index": abl_config.use_stress_index,
        "use_normal_prototypes": abl_config.use_normal_prototypes,
        "use_stress_generator": abl_config.use_stress_generator,
    }

    setup_logging(log_level, experiment_name=ablation)
    logger.info(f"Running ablation: {ablation}")
    logger.info(f"  Description: {abl_config.description}")
    logger.info(f"  Flags: {abl_config.to_dict()}")

    # Delegate to run_full_experiment logic
    from scripts.run_full_experiment import main as run_main
    from unittest.mock import patch
    import sys

    # Build synthetic argv for the delegated call
    args = [
        "--config", base_config,
        "--output-dir", cfg["experiment"]["output_dir"],
        "--log-level", log_level,
    ]

    # Re-invoke run_full_experiment with modified config written to temp file
    import tempfile, yaml
    from pathlib import Path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp_path = tmp.name

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_full_experiment",
         "--config", tmp_path,
         "--output-dir", cfg["experiment"]["output_dir"],
         "--log-level", log_level],
        check=False,
    )
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(f"Ablation '{ablation}' failed with returncode {result.returncode}")
    else:
        logger.info(f"Ablation '{ablation}' completed successfully.")


if __name__ == "__main__":
    main()
