import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve_settings
from .loaders import load_csv, split_schema
from .metrics.coverage import prdc
from .metrics.fidelity import correlation_distance, per_feature_similarity
from .metrics.privacy import (
    membership_inference_auc,
    membership_inference_holdout_auc,
    nn_distance_stats,
)
from .metrics.utility import tstr_trts_scores
from .preprocess import to_feature_space, transform_feature_space
from .report import render_report
from .viz import corr_heatmaps, pca_scatter, per_feature_hist


def _rel_to_report(path: Path, report_dir: Path) -> str:
    """Path to an asset, expressed relative to the report file's directory.

    Keeps image links valid regardless of where the report is written (e.g. a
    report in ``reports/`` pointing at plots under ``outputs/runs/<run>/plots``).
    """
    return Path(os.path.relpath(path, report_dir)).as_posix()


def _parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Benchmark synthetic vs. real tabular data.")
    ap.add_argument("--config", default=None, help="YAML config with defaults for any option.")
    ap.add_argument("--real")
    ap.add_argument("--synthetic")
    ap.add_argument("--target")
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--out_dir")
    ap.add_argument("--report")
    ap.add_argument("--holdout", help="Optional real holdout CSV enabling a holdout-based MIA.")
    ap.add_argument("--k", type=int, help="Neighbourhood size for PRDC (default 5).")
    ap.add_argument("--utility_model", choices=["linear", "rf"], help="Utility estimator family.")
    args = ap.parse_args(argv)

    cli = {k: v for k, v in vars(args).items() if k != "config"}
    config = load_config(args.config) if args.config else {}
    settings = resolve_settings(config, cli)

    missing = [k for k in ("real", "synthetic", "out_dir") if not settings.get(k)]
    if missing:
        ap.error(f"missing required settings (via CLI or --config): {', '.join(missing)}")
    return settings


def main(argv: list[str] | None = None) -> None:
    s = _parse_args(argv)

    out_dir = Path(s["out_dir"])
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    report_path = Path(s["report"]) if s["report"] else (out_dir / "report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_r = load_csv(s["real"])
    df_s = load_csv(s["synthetic"])
    if set(df_r.columns) != set(df_s.columns):
        missing_cols = set(df_r.columns) - set(df_s.columns)
        if missing_cols:
            raise ValueError(
                f"Synthetic data is missing columns present in real data: {sorted(missing_cols)}"
            )
    df_s = df_s[df_r.columns]  # align column order

    # Fidelity
    per_feat = per_feature_similarity(df_r, df_s, bins=30)
    corr_diff = correlation_distance(df_r, df_s)

    # Coverage + Privacy: one shared feature space fit on real, applied to all sets.
    num_cols, cat_cols, _ = split_schema(df_r, target=s["target"])
    Xr, meta = to_feature_space(df_r, num_cols, cat_cols)
    Xs = transform_feature_space(df_s, meta)
    coverage = prdc(Xr, Xs, k=s["k"])
    privacy = nn_distance_stats(Xr, Xs) | membership_inference_auc(Xr, Xs)
    if s["holdout"]:
        df_h = load_csv(s["holdout"])[df_r.columns]
        Xh = transform_feature_space(df_h, meta)
        privacy |= membership_inference_holdout_auc(Xr, Xs, Xh)

    # Utility (optional)
    util = (
        tstr_trts_scores(df_r, df_s, target=s["target"], task=s["task"], model=s["utility_model"])
        if s["target"]
        else None
    )

    # Save plots
    pca_scatter(df_r, df_s, out_path=str(plots / "pca.png"))
    per_feature_hist(df_r, df_s, out_path=str(plots / "distributions.png"))
    corr_heatmaps(df_r, df_s, out_path=str(plots / "correlations.png"))

    # Aggregate metrics
    summary: dict[str, Any] = {
        "fidelity": {
            "per_feature_mean_jsd": (
                float(pd.Series([v["JSD"] for v in per_feat.values()]).mean()) if per_feat else None
            ),
            "per_feature_mean_ks": (
                float(pd.Series([v["KS"] for v in per_feat.values()]).mean()) if per_feat else None
            ),
            "per_feature_mean_wasserstein": (
                float(pd.Series([v["Wasserstein"] for v in per_feat.values()]).mean())
                if per_feat
                else None
            ),
            "correlation_distance": corr_diff,
        },
        "coverage": coverage,
        "privacy": privacy,
        "utility": util,
        "rows_real": int(len(df_r)),
        "rows_synthetic": int(len(df_s)),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": per_feat}, f, indent=2)

    report_dir = report_path.parent
    context = {
        "task": s["task"],
        "target": s["target"],
        "summary_json": json.dumps(summary, indent=2),
        "per_feature_json": json.dumps(per_feat, indent=2),
        "pca_path": _rel_to_report(plots / "pca.png", report_dir),
        "dist_path": _rel_to_report(plots / "distributions.png", report_dir),
        "corr_path": _rel_to_report(plots / "correlations.png", report_dir),
    }
    render_report(out_path=str(report_path), context=context)
    print(f"Saved metrics -> {out_dir/'metrics.json'}")
    print(f"Saved report  -> {report_path}")


if __name__ == "__main__":
    main()
