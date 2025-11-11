import argparse, json
from pathlib import Path
import pandas as pd

from .loaders import load_csv, split_schema
from .preprocess import to_feature_space
from .metrics.fidelity import per_feature_similarity, correlation_distance
from .metrics.utility import tstr_trts_scores
from .metrics.coverage import prdc_like
from .metrics.privacy import nn_distance_stats, membership_inference_auc
from .viz import pca_scatter, per_feature_hist, corr_heatmaps
from .report import render_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--task", choices=["classification","regression"], default="classification")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); plots = out_dir/"plots"; plots.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else (out_dir/"report.html")

    df_r = load_csv(args.real)
    df_s = load_csv(args.synthetic)
    if set(df_r.columns) != set(df_s.columns):
        df_s = df_s[df_r.columns]  # align columns if order differs

    # Fidelity
    per_feat = per_feature_similarity(df_r, df_s, bins=30)
    corr_diff = correlation_distance(df_r, df_s)

    # Coverage + Privacy need feature space
    num_cols, cat_cols, _ = split_schema(df_r, target=args.target)
    Xr, meta = to_feature_space(df_r, num_cols, cat_cols)
    Xs, _    = to_feature_space(df_s, num_cols, cat_cols)
    prdc = prdc_like(Xr, Xs, k=5)
    privacy = nn_distance_stats(Xr, Xs) | membership_inference_auc(Xr, Xs)

    # Utility (optional)
    util = tstr_trts_scores(df_r, df_s, target=args.target, task=args.task) if args.target else None

    # Save plots
    pca_scatter(df_r, df_s, out_path=str(plots/"pca.png"))
    per_feature_hist(df_r, df_s, out_path=str(plots/"distributions.png"))
    corr_heatmaps(df_r, df_s, out_path=str(plots/"correlations.png"))

    # Aggregate metrics
    summary = {
        "fidelity": {
            "per_feature_mean_jsd": float(pd.Series([v["JSD"] for v in per_feat.values()]).mean()) if per_feat else None,
            "per_feature_mean_ks": float(pd.Series([v["KS"] for v in per_feat.values()]).mean()) if per_feat else None,
            "per_feature_mean_wasserstein": float(pd.Series([v["Wasserstein"] for v in per_feat.values()]).mean()) if per_feat else None,
            "correlation_distance": corr_diff,
        },
        "coverage": prdc,
        "privacy": privacy,
        "utility": util,
        "rows_real": int(len(df_r)),
        "rows_synthetic": int(len(df_s)),
    }

    # Persist
    with open(out_dir/"metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": per_feat}, f, indent=2)

    # Report
    context = {
        "task": args.task,
        "target": args.target,
        "summary_json": json.dumps(summary, indent=2),
        "per_feature_json": json.dumps(per_feat, indent=2),
        "pca_path": f"{plots.name}/pca.png",
        "dist_path": f"{plots.name}/distributions.png",
        "corr_path": f"{plots.name}/correlations.png",
    }
    render_report(template_path="templates/report.html", out_path=str(report_path), context=context)
    print(f"Saved metrics -> {out_dir/'metrics.json'}")
    print(f"Saved report  -> {report_path}")

if __name__ == "__main__":
    main()
