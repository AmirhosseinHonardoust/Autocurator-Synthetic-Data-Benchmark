"""Validate our metric implementations against independent references.

* PRDC is checked against the published ``prdc`` package (Naeem et al., 2020).
* The holdout membership-inference AUC is checked against a brute-force
  distance computation using scipy + scikit-learn.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from autocurator.metrics.coverage import prdc
from autocurator.metrics.privacy import membership_inference_holdout_auc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prdc_matches_reference(seed):
    prdc_ref = pytest.importorskip("prdc")
    rng = np.random.RandomState(seed)
    Xr = rng.randn(200, 8)
    Xs = rng.randn(200, 8) + 0.3
    k = 5
    ref = prdc_ref.compute_prdc(real_features=Xr, fake_features=Xs, nearest_k=k)
    ours = prdc(Xr, Xs, k=k)
    for key in ("precision", "recall", "density", "coverage"):
        assert abs(float(ref[key]) - ours[key]) < 1e-9


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_holdout_mia_matches_bruteforce(seed):
    rng = np.random.RandomState(seed)
    members = rng.randn(150, 6)
    holdout = rng.randn(120, 6)
    synth = members[:100] + rng.randn(100, 6) * 0.2

    ours = membership_inference_holdout_auc(members, synth, holdout)["mia_auc_holdout"]

    d_member = cdist(members, synth).min(axis=1)
    d_holdout = cdist(holdout, synth).min(axis=1)
    y = np.concatenate([np.ones(len(d_member)), np.zeros(len(d_holdout))])
    score = -np.concatenate([d_member, d_holdout])
    ref = roc_auc_score(y, score)

    assert abs(ours - ref) < 1e-9
