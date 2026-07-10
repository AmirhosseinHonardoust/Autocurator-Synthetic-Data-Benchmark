"""Run every reference generator on every bundled dataset and print a Markdown
table of the resulting metrics. Used to generate BENCHMARKS.md.

Usage:
    python scripts/benchmark_generators.py > BENCHMARKS.md
"""

import numpy as np
import pandas as pd

from autocurator.datasets import AVAILABLE_DATASETS, load_dataset
from autocurator.generators import GENERATORS
from autocurator.metrics.coverage import prdc
from autocurator.metrics.fidelity import correlation_distance
from autocurator.metrics.privacy import membership_inference_holdout_auc
from autocurator.metrics.utility import tstr_trts_scores
from autocurator.preprocess import to_feature_space, transform_feature_space

TASKS = {"breast_cancer": "classification", "wine": "classification", "diabetes": "regression"}


def _spaces(members, synth, holdout):
    num = [c for c in members.columns if pd.api.types.is_numeric_dtype(members[c])]
    cat = [c for c in members.columns if c not in num]
    Xr, meta = to_feature_space(members, num, cat)
    return Xr, transform_feature_space(synth, meta), transform_feature_space(holdout, meta)


def _row(dataset, gen_name, members, holdout, task):
    synth = GENERATORS[gen_name](members, seed=1)
    Xr, Xs, Xh = _spaces(members, synth, holdout)
    cov = prdc(Xr, Xs, k=5)
    cd = correlation_distance(members, synth)
    mia = membership_inference_holdout_auc(Xr, Xs, Xh)["mia_auc_holdout"]
    util = tstr_trts_scores(members, synth, target="target", task=task)
    util_val = util["TSTR_AUC"] if task == "classification" else util["TSTR_R2"]
    return (
        f"| {dataset} | {gen_name} | {cov['precision']:.2f} | {cov['recall']:.2f} | "
        f"{cov['coverage']:.2f} | {cd:.3f} | {mia:.2f} | {util_val:.2f} |"
    )


def main() -> None:
    print("# Generator sensitivity benchmark\n")
    print(
        "Each real dataset is split into members (seen by the generator) and a "
        "holdout (unseen). Metrics should reward the high-quality `resample` "
        "generator and flag the row-copying `leaky` generator via the holdout "
        "MIA.\n"
    )
    print(
        "| Dataset | Generator | Precision | Recall | Coverage | CorrDist | HoldoutMIA | Utility |"
    )
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for dataset in AVAILABLE_DATASETS:
        df = load_dataset(dataset)
        task = TASKS[dataset]
        n_members = int(len(df) * 0.7)
        perm = np.random.RandomState(0).permutation(len(df))
        members = df.iloc[perm[:n_members]].reset_index(drop=True)
        holdout = df.iloc[perm[n_members:]].reset_index(drop=True)
        for gen_name in GENERATORS:
            print(_row(dataset, gen_name, members, holdout, task))
    print(
        "\nUtility is TSTR AUC for classification datasets and TSTR R² for "
        "regression (diabetes). Higher precision/recall/coverage/utility and "
        "lower CorrDist are better; HoldoutMIA near 0.5 means low leakage."
    )


if __name__ == "__main__":
    main()
