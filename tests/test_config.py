import pytest

from autocurator.config import DEFAULTS, load_config, resolve_settings


def test_load_config_roundtrip(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("real: a.csv\nsynthetic: b.csv\nk: 7\n")
    data = load_config(str(cfg))
    assert data == {"real": "a.csv", "synthetic": "b.csv", "k": 7}


def test_load_config_empty(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("")
    assert load_config(str(cfg)) == {}


def test_load_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        load_config(str(cfg))


def test_load_config_rejects_non_mapping(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(str(cfg))


def test_resolve_precedence():
    config = {"k": 7, "task": "regression"}
    cli = {"k": 9, "task": None}  # CLI k wins; CLI task is unset so config wins
    merged = resolve_settings(config, cli)
    assert merged["k"] == 9
    assert merged["task"] == "regression"
    assert merged["utility_model"] == DEFAULTS["utility_model"]
