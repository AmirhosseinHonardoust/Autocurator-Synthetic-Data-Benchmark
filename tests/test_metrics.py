import numpy as np
import pandas as pd

from autocurator.metrics.coverage import prdc
from autocurator.metrics.fidelity import correlation_distance, per_feature_similarity
from autocurator.metrics.privacy import (
    membership_inference_auc,
    membership_inference_holdout_auc,
    nn_distance_stats,
)
from autocurator.metrics.utility import tstr_trts_scores
from autocurator.preprocess import to_feature_space, transform_feature_space


def _feature_space(real, synth):
    num_cols = [c for c in real.columns if real[c].dtype.kind in "if"]
    cat_cols = [c for c in real.columns if c not in num_cols]
    Xr, meta = to_feature_space(real, num_cols, cat_cols)
    Xs = transform_feature_space(synth, meta)
    return Xr, Xs


def test_fidelity_keys(numeric_frames):
    real, synth = numeric_frames
    res = per_feature_similarity(real, synth)
    assert set(res) == set(real.columns)
    for stats in res.values():
        assert set(stats) == {"JSD", "KS", "Wasserstein"}
        assert 0.0 <= stats["KS"] <= 1.0
    assert correlation_distance(real, synth) is not None


def test_correlation_distance_single_feature():
    one = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert correlation_distance(one, one) is None


def test_fidelity_handles_nan_column():
    real = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 1.0, 2.0, 2.0]})
    synth = pd.DataFrame({"a": [1.5, 2.5, 3.0, np.nan], "b": [1.0, 1.0, 2.0, 3.0]})
    res = per_feature_similarity(real, synth)
    assert "a" in res and "b" in res


def test_prdc_range_and_keys(numeric_frames):
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    res = prdc(Xr, Xs, k=5)
    assert set(res) == {"precision", "recall", "density", "coverage"}
    for key in ("precision", "recall", "coverage"):
        assert 0.0 <= res[key] <= 1.0
    assert res["density"] >= 0.0


def test_prdc_small_sample():
    Xr = np.random.RandomState(0).randn(4, 3)
    Xs = np.random.RandomState(1).randn(4, 3)
    res = prdc(Xr, Xs, k=5)  # k > n must not raise
    assert set(res) == {"precision", "recall", "density", "coverage"}


def test_nn_distance_stats(numeric_frames):
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    res = nn_distance_stats(Xr, Xs)
    assert set(res) == {"syn_to_real_mean_nnd", "syn_to_real_min_nnd", "syn_to_real_1pct_nnd"}
    assert res["syn_to_real_min_nnd"] >= 0.0


def test_mia_not_degenerate(numeric_frames):
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    auc = membership_inference_auc(Xr, Xs)["mia_auc_distance"]
    assert 0.0 <= auc <= 1.0
    assert abs(auc - 0.5) < 0.25


def test_mia_holdout():
    rng = np.random.RandomState(0)
    members = rng.randn(100, 4)
    holdout = rng.randn(100, 4)
    synth = members + rng.randn(100, 4) * 0.01  # near-copies of members => leakage
    auc = membership_inference_holdout_auc(members, synth, holdout)["mia_auc_holdout"]
    assert 0.0 <= auc <= 1.0
    assert auc > 0.5


def test_utility_handles_categorical(mixed_frames):
    real, synth = mixed_frames
    res = tstr_trts_scores(real, synth, target="target", task="classification")
    assert res is not None
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}
    for v in res.values():
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_utility_rf_model(numeric_frames):
    real, synth = numeric_frames
    res = tstr_trts_scores(real, synth, target="target", model="rf")
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}


def test_utility_regression_task(numeric_frames):
    real, synth = numeric_frames
    res = tstr_trts_scores(real, synth, target="score", task="regression")
    assert set(res) == {"TSTR_R2", "TRTS_R2"}


def test_utility_regression_rf(numeric_frames):
    real, synth = numeric_frames
    res = tstr_trts_scores(real, synth, target="score", task="regression", model="rf")
    assert set(res) == {"TSTR_R2", "TRTS_R2"}


def test_utility_class_mismatch():
    real = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "target": [0, 1, 2, 0, 1, 2]})
    synth = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
    res = tstr_trts_scores(real, synth, target="target", task="classification")
    assert res is not None
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}


def test_utility_missing_target(numeric_frames):
    real, synth = numeric_frames
    assert tstr_trts_scores(real, synth, target="nope") is None
