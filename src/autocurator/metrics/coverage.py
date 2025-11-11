import numpy as np
from sklearn.neighbors import NearestNeighbors

def prdc_like(X_real: np.ndarray, X_syn: np.ndarray, k: int = 5):
    # kNN radius in real space
    nn_r = NearestNeighbors(n_neighbors=k+1).fit(X_real)
    d_r, _ = nn_r.kneighbors(X_real)             # [N_real, k+1]
    radii = d_r[:, -1]                           # distance to k-th neighbor in real

    # Precision: synthetic inside real manifold
    d_sr, _ = nn_r.kneighbors(X_syn, n_neighbors=1)
    precision = float((d_sr[:, 0] <= radii.mean()).mean())

    # Recall: real covered by synthetic
    nn_s = NearestNeighbors(n_neighbors=1).fit(X_syn)
    d_rs, _ = nn_s.kneighbors(X_real, n_neighbors=1)
    recall = float((d_rs[:, 0] <= radii.mean()).mean())

    return {"precision_like": precision, "recall_like": recall}
