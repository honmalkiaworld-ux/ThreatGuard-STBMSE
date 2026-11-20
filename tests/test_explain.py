"""SHAP explanations must be additive: contributions + base value reproduce each model's own output."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from stbmse.explain import additivity_errors, importance_table, shap_by_model  # noqa: E402
from stbmse.modeling import StackedEnsemble  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    n = 700
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = pd.Series((X.a + 0.5 * X.b + rng.normal(0, 1, n) > 0.5).astype(int))
    ens = StackedEnsemble().fit(X.iloc[:400], y.iloc[:400], X.iloc[400:550], y.iloc[400:550])
    return ens, X.iloc[550:].reset_index(drop=True)


def test_every_explainer_is_additive(fitted):
    ens, X = fitted
    values = shap_by_model(ens, X)
    errs = additivity_errors(ens, X, values)
    assert set(errs) == {"logistic_regression", "random_forest", "xgboost"}
    assert max(errs.values()) < 1e-3, errs


def test_shapes_match_the_data(fitted):
    ens, X = fitted
    for name, (sv, base, unit) in shap_by_model(ens, X).items():
        assert sv.shape == (len(X), X.shape[1]), name
        assert np.isfinite(sv).all() and np.isfinite(base) and unit in ("log-odds", "probability")


def test_the_true_driver_is_ranked_first(fitted):
    """The label was built from column 'a' (and a little 'b'): a sound explanation must find that."""
    ens, X = fitted
    tab = importance_table(shap_by_model(ens, X), list(ens.columns), ens.meta_weights)
    assert tab.iloc[0]["feature"] == "a"
    assert abs(tab["weighted_share"].sum() - 1.0) < 1e-6          # XGBoost's SHAP values are float32
    assert set(tab.feature.head(2)) == {"a", "b"}
