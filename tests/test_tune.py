"""Tuning study: test-safe by construction, deterministic, and it must not change the shipped model's defaults."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

from stbmse import config as C  # noqa: E402
from stbmse.modeling import DEFAULT_PARAMS, make_learners  # noqa: E402
from stbmse.tune import CV_YEARS, cv_score, tune_learner  # noqa: E402


@pytest.fixture(scope="module")
def tiny_prep():
    rng = np.random.default_rng(3)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = pd.Series((X.a + rng.normal(0, 1, n) > 0.4).astype(int))
    ty = np.sort(rng.integers(2015, 2025, n))             # forecast years 2015..2024
    return SimpleNamespace(X=X, y=y, target_year=ty)


def test_tuning_never_touches_the_test_period():
    assert max(CV_YEARS) <= C.VAL_END_YEAR < 2022


def test_default_settings_are_the_shipped_ones():
    """If this fails, the published numbers no longer describe the shipped model."""
    p = make_learners()
    assert p["logistic_regression"].steps[-1][1].C == 0.5
    rf = p["random_forest"]
    assert (rf.n_estimators, rf.min_samples_leaf) == (400, 5)
    xgb = p["xgboost"].get_params()
    assert (xgb["n_estimators"], xgb["max_depth"], xgb["learning_rate"], xgb["subsample"], xgb["colsample_bytree"]) == (300, 3, 0.05, 0.8, 0.8)
    assert DEFAULT_PARAMS["logistic_regression"] == {"C": 0.5}


def test_overrides_apply_only_to_the_named_learner():
    p = make_learners(params={"xgboost": {"max_depth": 5}})
    assert p["xgboost"].get_params()["max_depth"] == 5
    assert p["random_forest"].n_estimators == 400


def test_cv_score_is_a_pr_auc_in_range(tiny_prep):
    s = cv_score("logistic_regression", {}, tiny_prep)
    assert 0.0 < s <= 1.0


def test_tuning_is_deterministic_and_reports_a_like_for_like_baseline(tiny_prep):
    a = tune_learner("logistic_regression", tiny_prep, n_trials=4)
    b = tune_learner("logistic_regression", tiny_prep, n_trials=4)
    assert a["best_params"] == b["best_params"]
    assert a["cv_gain"] == pytest.approx(a["cv_pr_auc_tuned"] - a["cv_pr_auc_fixed"])
    assert a["cv_pr_auc_tuned"] >= a["cv_pr_auc_fixed"] - 0.05 and a["n_trials"] == 4
