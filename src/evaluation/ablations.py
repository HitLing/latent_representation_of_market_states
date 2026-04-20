"""Ablation experiment definitions.

Each ablation maps to a clear methodological question:

1. baseline_hs         : What does ignoring regime structure cost?
2. ablation_no_generator : Does tail enrichment improve OOS risk estimation?
3. ablation_no_proto   : What does the prototype layer add for bulk stabilisation?
4. full_method         : Combined effect of all components.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Flags controlling which methodology components are active."""
    name: str
    use_stress_index: bool
    use_normal_prototypes: bool
    use_stress_generator: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "use_stress_index": self.use_stress_index,
            "use_normal_prototypes": self.use_normal_prototypes,
            "use_stress_generator": self.use_stress_generator,
            "description": self.description,
        }


ABLATION_CONFIGS: dict[str, AblationConfig] = {
    "baseline_hs": AblationConfig(
        name="baseline_hs",
        use_stress_index=False,
        use_normal_prototypes=False,
        use_stress_generator=False,
        description="Historical Simulation Only — no stress index, no prototypes, no generation.",
    ),
    "ablation_no_generator": AblationConfig(
        name="ablation_no_generator",
        use_stress_index=True,
        use_normal_prototypes=True,
        use_stress_generator=False,
        description="Stress-aware HS without synthetic generation. Tests value of tail enrichment.",
    ),
    "ablation_no_proto": AblationConfig(
        name="ablation_no_proto",
        use_stress_index=True,
        use_normal_prototypes=False,
        use_stress_generator=True,
        description="Stress generation without prototype stabilisation. Tests value of prototype layer.",
    ),
    "full_method": AblationConfig(
        name="full_method",
        use_stress_index=True,
        use_normal_prototypes=True,
        use_stress_generator=True,
        description="Full method: all components active.",
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """Retrieve ablation config by name."""
    if name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation '{name}'. "
            f"Available: {list(ABLATION_CONFIGS.keys())}"
        )
    return ABLATION_CONFIGS[name]


def compare_ablation_results(
    results: dict[str, dict],
    metric_key: str = "VaR_0.99",
) -> pd.DataFrame:
    """Compare a single metric across all ablation variants.

    Parameters
    ----------
    results    : {ablation_name: metrics_dict}
    metric_key : which metric to compare

    Returns DataFrame with ablation as index.
    """
    rows = []
    for name, metrics in results.items():
        val = metrics.get(metric_key, float("nan"))
        row = {"ablation": name, metric_key: val}
        # Also include Kupiec test results if present
        kupiec = metrics.get("kupiec", {})
        if kupiec:
            row["exc_rate"] = kupiec.get("empirical_rate", float("nan"))
            row["kupiec_p"] = kupiec.get("p_value", float("nan"))
            row["kupiec_reject"] = kupiec.get("reject_H0", False)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("ablation")
    return df


def ablation_summary_table(
    results: dict[str, dict],
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Build a summary table across ablations and confidence levels."""
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rows = []
    for name, metrics in results.items():
        row = {"ablation": name}
        for cl in confidence_levels:
            var_key = f"VaR_{cl}"
            es_key = f"ES_{cl}"
            kupiec_key = f"kupiec_{cl}"
            row[var_key] = metrics.get(var_key, float("nan"))
            row[es_key] = metrics.get(es_key, float("nan"))

            kupiec = metrics.get(f"kupiec_CL_{cl}", metrics.get("kupiec", {}))
            row[f"exc_rate_{cl}"] = kupiec.get("empirical_rate", float("nan"))
            row[f"kupiec_p_{cl}"] = kupiec.get("p_value", float("nan"))

        rows.append(row)

    return pd.DataFrame(rows).set_index("ablation")
