import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def nn_distance_stats(X_real: np.ndarray, X_syn: np.ndarray) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    d, _ = nn.kneighbors(X_syn, n_neighbors=1)
    d = d[:, 0]
    return {
        "syn_to_real_mean_nnd": float(np.mean(d)),
        "syn_to_real_min_nnd": float(np.min(d)),
        "syn_to_real_1pct_nnd": float(np.percentile(d, 1)),
    }


def membership_inference_auc(X_real: np.ndarray, X_syn: np.ndarray) -> dict[str, float]:
    """Distance-based membership-inference proxy (no holdout required).

    An attacker scores each record by proximity to the real set. Real records
    must be compared against their nearest *other* real record (self-matches at
    distance 0 would otherwise make the attack trivially perfect). Synthetic
    records are compared against their nearest real record.

    The returned AUC measures how separable the two groups are. A value near
    0.5 means real and synthetic are indistinguishable by nearest-neighbour
    distance (low membership-inference risk); values far from 0.5 mean the sets
    are distinguishable (values below 0.5 can indicate memorisation, where
    synthetic points sit unusually close to real ones).
    """
    nn = NearestNeighbors(n_neighbors=2).fit(X_real)
    # For real points, skip the self-match (column 0) and use the nearest other.
    d_real, _ = nn.kneighbors(X_real, n_neighbors=2)
    d_real = d_real[:, 1]
    # For synthetic points there is no self-match; the nearest real is column 0.
    d_syn, _ = nn.kneighbors(X_syn, n_neighbors=1)
    d_syn = d_syn[:, 0]

    y = np.concatenate([np.ones(len(d_real)), np.zeros(len(d_syn))])
    score = -np.concatenate([d_real, d_syn])  # smaller distance => more member-like
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        auc = float("nan")
    return {"mia_auc_distance": float(auc)}


def membership_inference_holdout_auc(
    X_members: np.ndarray, X_syn: np.ndarray, X_holdout: np.ndarray
) -> dict[str, float]:
    """Proper distance-to-closest-record membership-inference attack.

    Given real records that the generator trained on (``members``) and disjoint
    real records it never saw (``holdout``/non-members), the attacker labels a
    candidate a member if it lies unusually close to the synthetic set. AUC of
    (is-member) vs. (-distance-to-nearest-synthetic) quantifies attack success:
    **0.5 means the generator leaks nothing**; values above 0.5 indicate members
    are closer to synthetic than non-members, i.e. training-data leakage.
    """
    nn = NearestNeighbors(n_neighbors=1).fit(X_syn)
    d_member, _ = nn.kneighbors(X_members, n_neighbors=1)
    d_holdout, _ = nn.kneighbors(X_holdout, n_neighbors=1)
    y = np.concatenate([np.ones(len(d_member)), np.zeros(len(d_holdout))])
    score = -np.concatenate([d_member[:, 0], d_holdout[:, 0]])
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        auc = float("nan")
    return {"mia_auc_holdout": float(auc)}
