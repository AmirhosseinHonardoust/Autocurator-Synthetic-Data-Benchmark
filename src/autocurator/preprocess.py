from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def to_feature_space(
    df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a scaler + one-hot encoder and return the encoded matrix and metadata."""
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Xn = scaler.fit_transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
    Xc = encoder.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
    X = np.concatenate([Xn, Xc], axis=1).astype(float)
    meta: dict[str, Any] = {
        "scaler": scaler,
        "encoder": encoder,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return X, meta


def transform_feature_space(df: pd.DataFrame, meta: dict[str, Any]) -> np.ndarray:
    """Apply an already-fitted transform (from :func:`to_feature_space`) to ``df``.

    Keeps real, synthetic and holdout data in one shared feature space so that
    cross-set distances (coverage, privacy) are computed consistently.
    """
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    Xn = meta["scaler"].transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
    Xc = meta["encoder"].transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
    return np.concatenate([Xn, Xc], axis=1).astype(float)
