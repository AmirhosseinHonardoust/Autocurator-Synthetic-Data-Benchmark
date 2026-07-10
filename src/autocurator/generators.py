"""Reference synthetic-data generators of known, varying quality.

These are deliberately simple baselines whose expected metric behaviour is
understood, so they can be used to check that the benchmark's metrics respond
in the right direction:

* :func:`resample_generator` - high quality (preserves marginals and, via
  bootstrap, most joint structure).
* :func:`independent_generator` - preserves each marginal but destroys
  correlations (low fidelity/coverage, still private).
* :func:`noise_generator` - wrong distribution (poor fidelity and utility).
* :func:`leaky_generator` - copies real rows verbatim (a privacy leak that a
  holdout membership-inference attack should detect).

Only continuous (float) columns are jittered or replaced; discrete columns
(e.g. integer-coded class labels) are resampled as-is so a classification
target stays categorical.
"""

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _is_continuous(s: pd.Series) -> bool:
    return bool(pd.api.types.is_float_dtype(s))


def resample_generator(real: pd.DataFrame, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Bootstrap rows and jitter continuous columns slightly. High quality."""
    rng = _rng(seed)
    n = n or len(real)
    idx = rng.randint(0, len(real), size=n)
    out = real.iloc[idx].reset_index(drop=True).copy()
    for col in out.columns:
        if _is_continuous(out[col]):
            std = float(out[col].std(ddof=0)) or 1.0
            out[col] = out[col] + rng.normal(0, 0.05 * std, size=n)
    return out


def independent_generator(real: pd.DataFrame, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Sample each column independently, destroying cross-feature correlations."""
    rng = _rng(seed)
    n = n or len(real)
    data = {}
    for col in real.columns:
        values = real[col].to_numpy()
        data[col] = rng.choice(values, size=n, replace=True)
    return pd.DataFrame(data)


def noise_generator(real: pd.DataFrame, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Continuous columns replaced with unrelated Gaussian noise. Poor quality."""
    rng = _rng(seed)
    n = n or len(real)
    out = pd.DataFrame(index=range(n))
    for col in real.columns:
        if _is_continuous(real[col]):
            out[col] = rng.normal(0, 1, size=n)
        else:
            out[col] = rng.choice(real[col].to_numpy(), size=n, replace=True)
    return out.reset_index(drop=True)


def leaky_generator(
    real: pd.DataFrame, n: int | None = None, seed: int = 0, copy_fraction: float = 1.0
) -> pd.DataFrame:
    """Copy real rows verbatim (optionally a fraction). A deliberate privacy leak."""
    rng = _rng(seed)
    n = n or len(real)
    n_copy = int(round(copy_fraction * n))
    idx = rng.randint(0, len(real), size=n_copy)
    copied = real.iloc[idx].reset_index(drop=True).copy()
    if n_copy >= n:
        return copied.iloc[:n].reset_index(drop=True)
    filler = resample_generator(real, n=n - n_copy, seed=seed + 1)
    return pd.concat([copied, filler], ignore_index=True)


GENERATORS = {
    "resample": resample_generator,
    "independent": independent_generator,
    "noise": noise_generator,
    "leaky": leaky_generator,
}
