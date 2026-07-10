import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numeric_frames():
    rng = np.random.RandomState(0)
    n = 200
    real = pd.DataFrame(
        {
            "age": rng.normal(40, 8, n).round(),
            "income": rng.normal(50000, 8000, n).round(),
            "score": rng.normal(660, 25, n).round(),
            "visits": rng.poisson(8, n),
            "target": rng.randint(0, 2, n),
        }
    )
    # Synthetic drawn from the same generating process (a good synthesiser).
    synth = pd.DataFrame(
        {
            "age": rng.normal(40, 8, n).round(),
            "income": rng.normal(50000, 8000, n).round(),
            "score": rng.normal(660, 25, n).round(),
            "visits": rng.poisson(8, n),
            "target": rng.randint(0, 2, n),
        }
    )
    return real, synth


@pytest.fixture
def mixed_frames():
    """Frames that include a categorical feature (regression test for utility)."""
    rng = np.random.RandomState(1)
    n = 120
    cities = np.array(["A", "B", "C"])

    def make():
        return pd.DataFrame(
            {
                "age": rng.normal(40, 8, n).round(),
                "city": cities[rng.randint(0, 3, n)],
                "target": rng.randint(0, 2, n),
            }
        )

    return make(), make()
