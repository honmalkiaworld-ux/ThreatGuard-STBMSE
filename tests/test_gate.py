"""Quality-gate tests: the gate must pass a genuinely useful model and fail useless or broken ones."""
import numpy as np
import pytest

from stbmse.gate import check_quality_gate, min_gain


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return (rng.random(600) < 0.3).astype(int)


def _noisy_signal(y, strength, seed):
    rng = np.random.default_rng(seed)
    return np.clip(0.3 + strength * (y - 0.3) + rng.normal(0, 0.15, len(y)), 0.001, 0.999)


def test_gate_passes_a_good_model(labels):
    pers = np.roll(labels, 1).astype(float)                       # weak persistence-like baseline
    res = check_quality_gate(labels, pers, {"good": _noisy_signal(labels, 0.9, 1)}, threshold=0.3)
    assert res.passed, res.summary()


def test_gate_fails_a_model_no_better_than_persistence(labels):
    pers = _noisy_signal(labels, 0.9, 2)                          # persistence here is as good as the model
    res = check_quality_gate(labels, pers, {"same": _noisy_signal(labels, 0.9, 2)}, threshold=0.3)
    assert not res.passed
    assert any("beats persistence" in c["name"] and not c["ok"] for c in res.checks)


def test_gate_fails_on_nan_scores(labels):
    bad = _noisy_signal(labels, 0.9, 3)
    bad[5] = np.nan
    res = check_quality_gate(labels, np.zeros(len(labels)), {"bad": bad}, threshold=0.3)
    assert not res.passed


def test_gate_fails_on_degenerate_threshold(labels):
    good = _noisy_signal(labels, 0.9, 4)
    res = check_quality_gate(labels, np.zeros(len(labels)), {"good": good}, threshold=0.02)
    assert not res.passed
    assert any("degenerate" in c["name"] and not c["ok"] for c in res.checks)


def test_gate_fails_when_validation_has_one_class():
    y = np.zeros(100, dtype=int)
    res = check_quality_gate(y, np.zeros(100), {"m": np.full(100, 0.1)}, threshold=0.3)
    assert not res.passed


def test_strictness_can_be_raised_to_break_the_gate(labels, monkeypatch):
    """The 'break the gate' exercise: an impossible bar must stop an otherwise good model."""
    pers = np.roll(labels, 1).astype(float)
    good = {"good": _noisy_signal(labels, 0.9, 1)}
    assert check_quality_gate(labels, pers, good, threshold=0.3).passed
    monkeypatch.setenv("STBMSE_MIN_GAIN", "0.99")
    assert min_gain() == 0.99
    assert not check_quality_gate(labels, pers, good, threshold=0.3).passed
