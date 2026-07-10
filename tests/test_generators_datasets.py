import pandas as pd
import pytest

from autocurator.datasets import AVAILABLE_DATASETS, load_dataset
from autocurator.generators import (
    GENERATORS,
    independent_generator,
    leaky_generator,
    noise_generator,
    resample_generator,
)


@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_load_dataset_shape(name):
    df = load_dataset(name)
    assert "target" in df.columns
    assert len(df) > 0


def test_load_dataset_unknown():
    with pytest.raises(ValueError):
        load_dataset("does_not_exist")


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize("gen", list(GENERATORS.values()))
def test_generators_preserve_columns(gen, frame):
    out = gen(frame, seed=0)
    assert list(out.columns) == list(frame.columns)
    assert len(out) == len(frame)


def test_generators_custom_n(frame):
    assert len(resample_generator(frame, n=10, seed=0)) == 10
    assert len(independent_generator(frame, n=3, seed=0)) == 3
    assert len(noise_generator(frame, n=7, seed=0)) == 7


def test_leaky_partial_copy_exercises_filler(frame):
    out = leaky_generator(frame, n=6, seed=0, copy_fraction=0.5)
    assert list(out.columns) == list(frame.columns)
    assert len(out) == 6


def test_noise_generator_preserves_discrete_target(frame):
    out = noise_generator(frame, seed=0)
    # target is integer-coded; it must remain among the original classes.
    assert set(out["target"]).issubset(set(frame["target"]))
