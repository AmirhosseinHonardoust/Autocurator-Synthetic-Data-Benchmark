import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _kth_nn_radii(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour (self excluded)."""
    n_neighbors = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, -1]


def prdc(X_real: np.ndarray, X_syn: np.ndarray, k: int = 5) -> dict[str, float]:
    """Precision, Recall, Density and Coverage (Naeem et al., 2020).

    Uses per-point k-NN hyperspheres rather than a single global radius:

    * ``precision`` - fraction of synthetic points inside any real hypersphere.
    * ``recall``    - fraction of real points inside any synthetic hypersphere.
    * ``density``   - average number of real hyperspheres each synthetic point
      falls into, normalised by ``k`` (robust to outliers).
    * ``coverage``  - fraction of real points with at least one synthetic point
      inside their hypersphere.

    Distances are computed in synthetic-row chunks to bound memory on large
    inputs.
    """
    real_radii = _kth_nn_radii(X_real, k)
    syn_radii = _kth_nn_radii(X_syn, k)

    n_real = len(X_real)
    n_syn = len(X_syn)

    precision_hits = 0
    density_sum = 0.0
    real_covered = np.zeros(n_real, dtype=bool)
    real_in_syn = np.zeros(n_real, dtype=bool)

    chunk = 1024
    for start in range(0, n_syn, chunk):
        Xs = X_syn[start : start + chunk]
        d = pairwise_distances(Xs, X_real)  # [chunk, n_real]
        within_real = d <= real_radii[None, :]  # inside a real hypersphere
        precision_hits += int(np.any(within_real, axis=1).sum())
        density_sum += float(within_real.sum())
        real_covered |= np.any(within_real, axis=0)
        # recall: real point inside this synthetic point's hypersphere
        within_syn = d <= syn_radii[start : start + chunk][:, None]
        real_in_syn |= np.any(within_syn, axis=0)

    precision = precision_hits / n_syn if n_syn else 0.0
    recall = float(real_in_syn.mean()) if n_real else 0.0
    density = density_sum / (k * n_syn) if n_syn else 0.0
    coverage = float(real_covered.mean()) if n_real else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }
