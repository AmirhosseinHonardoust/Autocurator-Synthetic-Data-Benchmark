import numpy as np

from autocurator.metrics.coverage import prdc_like
from autocurator.metrics.fidelity import correlation_distance, per_feature_similarity
from autocurator.metrics.privacy import membership_inference_auc, nn_distance_stats
from autocurator.metrics.utility import tstr_trts_scores
from autocurator.preprocess import to_feature_space


def _feature_space(real, synth):
    num_cols = [c for c in real.columns if real[c].dtype.kind in "if"]
    cat_cols = [c for c in real.columns if c not in num_cols]
    Xr, _ = to_feature_space(real, num_cols, cat_cols)
    Xs, _ = to_feature_space(synth, num_cols, cat_cols)
    return Xr, Xs


def test_fidelity_keys(numeric_frames):
    real, synth = numeric_frames
    res = per_feature_similarity(real, synth)
    assert set(res) == set(real.columns)
    for stats in res.values():
        assert set(stats) == {"JSD", "KS", "Wasserstein"}
        assert 0.0 <= stats["KS"] <= 1.0
    assert correlation_distance(real, synth) is not None


def test_coverage_range(numeric_frames):
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    res = prdc_like(Xr, Xs, k=5)
    assert set(res) == {"precision_like", "recall_like"}
    for v in res.values():
        assert 0.0 <= v <= 1.0


def test_nn_distance_stats(numeric_frames):
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    res = nn_distance_stats(Xr, Xs)
    assert set(res) == {"syn_to_real_mean_nnd", "syn_to_real_min_nnd", "syn_to_real_1pct_nnd"}
    assert res["syn_to_real_min_nnd"] >= 0.0


def test_mia_not_degenerate(numeric_frames):
    """Regression test: real self-matches must be excluded.

    With real and synthetic drawn from the same distribution, the attack
    should have little advantage, so AUC stays near 0.5 -- the old
    implementation returned 1.0 for essentially any input.
    """
    real, synth = numeric_frames
    Xr, Xs = _feature_space(real, synth)
    auc = membership_inference_auc(Xr, Xs)["mia_auc_distance"]
    assert 0.0 <= auc <= 1.0
    assert abs(auc - 0.5) < 0.25


def test_utility_handles_categorical(mixed_frames):
    """Regression test: utility must not crash on non-numeric features."""
    real, synth = mixed_frames
    res = tstr_trts_scores(real, synth, target="target", task="classification")
    assert res is not None
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}
    for v in res.values():
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_utility_numeric_classification(numeric_frames):
    real, synth = numeric_frames
    res = tstr_trts_scores(real, synth, target="target", task="classification")
    assert set(res) == {"TSTR_AUC", "TRTS_AUC"}


def test_utility_regression_task(numeric_frames):
    real, synth = numeric_frames
    res = tstr_trts_scores(real, synth, target="score", task="regression")
    assert set(res) == {"TSTR_R2", "TRTS_R2"}


def test_utility_missing_target(numeric_frames):
    real, synth = numeric_frames
    assert tstr_trts_scores(real, synth, target="nope") is None
