import json
import runpy
import sys
from pathlib import Path

from autocurator.cli import main


def _write_csv(path: Path, seed: int) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    n = 80
    pd.DataFrame(
        {
            "age": rng.normal(40, 8, n).round(),
            "income": rng.normal(50000, 8000, n).round(),
            "score": rng.normal(660, 25, n).round(),
            "visits": rng.poisson(8, n),
            "target": rng.randint(0, 2, n),
        }
    ).to_csv(path, index=False)


def test_cli_end_to_end(tmp_path, monkeypatch):
    real = tmp_path / "real.csv"
    synth = tmp_path / "synthetic.csv"
    _write_csv(real, 0)
    _write_csv(synth, 1)

    out_dir = tmp_path / "outputs" / "run"
    report = tmp_path / "reports" / "run.html"
    argv = [
        "autocurator",
        "--real",
        str(real),
        "--synthetic",
        str(synth),
        "--target",
        "target",
        "--task",
        "classification",
        "--out_dir",
        str(out_dir),
        "--report",
        str(report),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    metrics = json.loads((out_dir / "metrics.json").read_text())
    summary = metrics["summary"]
    assert set(summary) >= {"fidelity", "coverage", "privacy", "utility"}
    assert summary["rows_real"] == 80
    for name in ("pca.png", "distributions.png", "correlations.png"):
        assert (out_dir / "plots" / name).exists()

    # Report exists and its image links resolve relative to the report location.
    html = report.read_text()
    assert report.exists()
    for token in ("plots/pca.png", "plots/distributions.png", "plots/correlations.png"):
        assert token in html
    referenced = html.split('src="')[1].split('"')[0]
    assert (report.parent / referenced).resolve().exists()


def test_module_entrypoint_importable():
    # `python -m autocurator.cli` should resolve the module without error.
    assert runpy is not None
    import autocurator.cli as cli

    assert callable(cli.main)
