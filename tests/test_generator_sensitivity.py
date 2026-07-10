"""Sensitivity tests: metrics must respond to synthetic-data quality.

Using real bundled datasets and reference generators of known quality, we
assert the benchmark's metrics move in the expected direction (a high-quality
generator scores better on fidelity/coverage/utility; a row-copying generator
is flagged by the holdout membership-inference attack).
"""

import numpy as np
import pandas as pd
import pytest

from autocurator.datasets import load_dataset
from autocurator.generators import (
    independent_generator,
    leaky_generator,
    noise_generator,
    resample_generator,
)
from autocurator.metrics.coverage import prdc
from autocurator.metrics.fidelity import correlation_distance
from autocurator.metrics.privacy import membership_inference_holdout_auc
from autocurator.metrics.utility import tstr_trts_scores
from autocurator.preprocess import to_feature_space, transform_feature_space


def _split(df, n_members, seed=0):
    perm = np.random.RandomState(seed).permutation(len(df))
    members = df.iloc[perm[:n_members]].reset_index(drop=True)
    holdout = df.iloc[perm[n_members:]].reset_index(drop=True)
    return members, holdout


def _spaces(members, synth, holdout):
    num = [c for c in members.columns if pd.api.types.is_numeric_dtype(members[c])]
    cat = [c for c in members.columns if c not in num]
    Xr, meta = to_feature_space(members, num, cat)
    return Xr, transform_feature_space(synth, meta), transform_feature_space(holdout, meta)


def test_fidelity_and_coverage_track_quality():
    members, _ = _split(load_dataset("breast_cancer"), 400)
    good = resample_generator(members, seed=1)
    bad = noise_generator(members, seed=1)

    Xr, Xg, _ = _spaces(members, good, members)
    _, Xb, _ = _spaces(members, bad, members)

    cov_good = prdc(Xr, Xg, k=5)
    cov_bad = prdc(Xr, Xb, k=5)
    assert cov_good["precision"] > 0.8
    assert cov_bad["precision"] < 0.2
    assert cov_good["coverage"] > cov_bad["coverage"]

    # Destroying correlations should raise the correlation distance.
    shuffled = independent_generator(members, seed=1)
    assert correlation_distance(members, good) < correlation_distance(members, shuffled)


def test_holdout_mia_flags_leakage():
    members, holdout = _split(load_dataset("breast_cancer"), 400)
    leaky = leaky_generator(members, seed=1)
    private = independent_generator(members, seed=1)

    Xr, Xl, Xh = _spaces(members, leaky, holdout)
    _, Xp, _ = _spaces(members, private, holdout)

    mia_leaky = membership_inference_holdout_auc(Xr, Xl, Xh)["mia_auc_holdout"]
    mia_private = membership_inference_holdout_auc(Xr, Xp, Xh)["mia_auc_holdout"]

    assert mia_leaky > 0.65  # members clearly closer to synthetic than holdout
    assert mia_private < 0.6  # independent columns leak little
    assert mia_leaky > mia_private + 0.2


def test_utility_tracks_quality_classification():
    members, _ = _split(load_dataset("breast_cancer"), 400)
    good = tstr_trts_scores(members, resample_generator(members, seed=1), target="target")
    bad = tstr_trts_scores(members, independent_generator(members, seed=1), target="target")
    assert good["TSTR_AUC"] > bad["TSTR_AUC"]


@pytest.mark.parametrize("name,target,task", [("wine", "target", "classification")])
def test_multiclass_dataset_runs(name, target, task):
    members, _ = _split(load_dataset(name), 130)
    res = tstr_trts_scores(members, resample_generator(members, seed=1), target=target, task=task)
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}


def test_regression_dataset_utility_tracks_quality():
    members, _ = _split(load_dataset("diabetes"), 350)
    good = tstr_trts_scores(
        members, resample_generator(members, seed=1), target="target", task="regression"
    )
    bad = tstr_trts_scores(
        members, noise_generator(members, seed=1), target="target", task="regression"
    )
    assert good["TSTR_R2"] > bad["TSTR_R2"]
