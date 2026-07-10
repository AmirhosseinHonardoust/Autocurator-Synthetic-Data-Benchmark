"""Loaders for real, bundled reference datasets (no network required).

These wrap scikit-learn's built-in datasets so the benchmark can be exercised
on genuine, non-synthetic data of varying shape and task type.
"""

from collections.abc import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine


def _to_frame(loader: Callable[..., object], target_name: str = "target") -> pd.DataFrame:
    bunch = loader(as_frame=True)
    df = bunch.frame.copy()  # type: ignore[attr-defined]
    df = df.rename(columns={bunch.target.name: target_name})  # type: ignore[attr-defined]
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """Return a bundled real dataset as a DataFrame with a ``target`` column.

    * ``breast_cancer`` - 569 rows, 30 numeric features, binary target.
    * ``wine``          - 178 rows, 13 numeric features, 3-class target.
    * ``diabetes``      - 442 rows, 10 numeric features, continuous target.
    """
    loaders: dict[str, Callable[..., object]] = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "diabetes": load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(loaders)}.")
    return _to_frame(loaders[name])


AVAILABLE_DATASETS = ("breast_cancer", "wine", "diabetes")
