import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def nn_distance_stats(X_real: np.ndarray, X_syn: np.ndarray):
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    d, _ = nn.kneighbors(X_syn, n_neighbors=1)
    d = d[:, 0]
    return {
        "syn_to_real_mean_nnd": float(np.mean(d)),
        "syn_to_real_min_nnd": float(np.min(d)),
        "syn_to_real_1pct_nnd": float(np.percentile(d, 1))
    }

def membership_inference_auc(X_real: np.ndarray, X_syn: np.ndarray):
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    d_real, _ = nn.kneighbors(X_real, n_neighbors=1)
    d_syn, _  = nn.kneighbors(X_syn,  n_neighbors=1)
    y = np.concatenate([np.ones(len(d_real)), np.zeros(len(d_syn))])
    score = -np.concatenate([d_real[:,0], d_syn[:,0]])  # smaller dist => more member-like
    try:
        auc = roc_auc_score(y, score)
    except Exception:
        auc = float("nan")
    return {"mia_auc_distance": float(auc)}
