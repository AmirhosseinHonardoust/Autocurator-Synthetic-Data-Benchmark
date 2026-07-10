"""Optional YAML configuration for the CLI.

A config file supplies defaults for any CLI option; explicit command-line
arguments always take precedence. Example::

    real: data/real.csv
    synthetic: data/synthetic.csv
    target: target
    task: classification
    out_dir: outputs/runs/example_run
    report: reports/example_run.html
    k: 5
    utility_model: linear
"""

from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "real": None,
    "synthetic": None,
    "target": None,
    "task": "classification",
    "out_dir": None,
    "report": None,
    "holdout": None,
    "k": 5,
    "utility_model": "linear",
}


def load_config(path: str) -> dict[str, Any]:
    """Load and validate a YAML config file into a plain dict."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    unknown = set(data) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return data


def resolve_settings(config: dict[str, Any], cli: dict[str, Any]) -> dict[str, Any]:
    """Merge settings with precedence: DEFAULTS < config file < CLI arguments."""
    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update({k: v for k, v in config.items() if v is not None})
    merged.update({k: v for k, v in cli.items() if v is not None})
    return merged
