import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocurator.cli import main


def _write_csv(path: Path, seed: int, n: int = 80) -> None:
    rng = np.random.RandomState(seed)
    pd.DataFrame(
        {
            "age": rng.normal(40, 8, n).round(),
            "income": rng.normal(50000, 8000, n).round(),
            "score": rng.normal(660, 25, n).round(),
            "visits": rng.poisson(8, n),
            "target": rng.randint(0, 2, n),
        }
    ).to_csv(path, index=False)


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["autocurator", *argv])
    main()


def test_cli_end_to_end(tmp_path, monkeypatch):
    real = tmp_path / "real.csv"
    synth = tmp_path / "synthetic.csv"
    _write_csv(real, 0)
    _write_csv(synth, 1)
    out_dir = tmp_path / "outputs" / "run"
    report = tmp_path / "reports" / "run.html"
    _run(
        [
            "--real",
            str(real),
            "--synthetic",
            str(synth),
            "--target",
            "target",
            "--out_dir",
            str(out_dir),
            "--report",
            str(report),
        ],
        monkeypatch,
    )

    summary = json.loads((out_dir / "metrics.json").read_text())["summary"]
    assert set(summary) >= {"fidelity", "coverage", "privacy", "utility"}
    assert set(summary["coverage"]) == {"precision", "recall", "density", "coverage"}
    assert summary["rows_real"] == 80
    for name in ("pca.png", "distributions.png", "correlations.png"):
        assert (out_dir / "plots" / name).exists()

    html = report.read_text()
    referenced = html.split('src="')[1].split('"')[0]
    assert (report.parent / referenced).resolve().exists()


def test_cli_with_config_and_holdout(tmp_path, monkeypatch):
    real = tmp_path / "real.csv"
    synth = tmp_path / "synthetic.csv"
    holdout = tmp_path / "holdout.csv"
    _write_csv(real, 0)
    _write_csv(synth, 1)
    _write_csv(holdout, 2)
    out_dir = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("target: target\n" "task: classification\n" "utility_model: rf\n" "k: 4\n")
    _run(
        [
            "--config",
            str(cfg),
            "--real",
            str(real),
            "--synthetic",
            str(synth),
            "--holdout",
            str(holdout),
            "--out_dir",
            str(out_dir),
        ],
        monkeypatch,
    )
    summary = json.loads((out_dir / "metrics.json").read_text())["summary"]
    assert "mia_auc_holdout" in summary["privacy"]
    assert summary["utility"] is not None


def test_cli_missing_required(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(["--real", str(tmp_path / "r.csv")], monkeypatch)


def test_module_entrypoint_importable():
    import autocurator.cli as cli

    assert callable(cli.main)
