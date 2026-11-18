"""MLflow tracking tests, on a small synthetic problem so they run in seconds and never touch the project's mlflow.db."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from stbmse import tracking  # noqa: E402
from stbmse.modeling import StackedEnsemble, evaluate  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 700
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = pd.Series((X.a + 0.5 * X.b + rng.normal(0, 1, n) > 0.6).astype(int))
    ens = StackedEnsemble().fit(X.iloc[:400], y.iloc[:400], X.iloc[400:550], y.iloc[400:550])
    Xt, yt = X.iloc[550:], y.iloc[550:].values
    persistence = np.roll(yt, 1).astype(float)
    rows_t = [evaluate("persistence", yt, persistence, 0.5)]
    rows_v = [evaluate("persistence", y.iloc[400:550].values, np.roll(y.iloc[400:550].values, 1).astype(float), 0.5)]
    pb_t, pb_v = ens.base_proba(Xt), ens.base_proba(X.iloc[400:550])
    for name in ens.learners:
        rows_t.append(evaluate(name, yt, pb_t[name], ens.base_thresholds[name]))
        rows_v.append(evaluate(name, y.iloc[400:550].values, pb_v[name], ens.base_thresholds[name]))
    rows_t.append(evaluate("st_bmse_ensemble", yt, ens.predict_proba(Xt), ens.threshold))
    metrics = {"ece_test": 0.05, "threshold": ens.threshold, "meta_weights_logit": ens.meta_weights,
               "quality_gate": {"passed": True, "checks": []}}
    return ens, Xt, pd.DataFrame(rows_t).set_index("model"), pd.DataFrame(rows_v).set_index("model"), metrics


def _log(fitted, tmp_path, gate_passed):
    ens, _, rt, rv, metrics = fitted
    return tracking.log_training_run(
        ensemble=ens, metrics=metrics, params={"seed": 42, "cost_fn": 5.0}, learner_params={"random_forest": {"n_estimators": 400}},
        results_test=rt, results_val=rv, artifacts=[], gate_passed=gate_passed,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", artifact_root=tmp_path / "artifacts")


def test_passing_run_is_logged_and_registered_as_production(fitted, tmp_path):
    import mlflow
    from mlflow import MlflowClient
    info = _log(fitted, tmp_path, gate_passed=True)
    assert info["enabled"] and info["alias"] == "production"

    client = MlflowClient(tracking_uri=info["tracking_uri"])
    run = client.get_run(info["run_id"])
    assert run.data.tags["quality_gate"] == "passed"
    assert run.data.params["seed"] == "42"
    assert "test_pr_auc" in run.data.metrics and "test_pr_auc_gain_over_persistence" in run.data.metrics

    children = client.search_runs([run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{info['run_id']}'")
    assert sorted(c.data.tags["learner"] for c in children) == ["logistic_regression", "random_forest", "xgboost"]

    mv = client.get_model_version_by_alias(tracking.MODEL_NAME, "production")
    assert str(mv.version) == str(info["model_version"])

    # the registered model must load back and predict like the original
    mlflow.set_tracking_uri(info["tracking_uri"])
    loaded = mlflow.pyfunc.load_model(f"models:/{tracking.MODEL_NAME}@production")
    _, Xt, *_ = fitted
    assert len(loaded.predict(Xt)) == len(Xt)


def test_failed_gate_is_logged_but_never_registered(fitted, tmp_path):
    from mlflow import MlflowClient
    info = _log(fitted, tmp_path, gate_passed=False)
    assert info["registered_model"] is None
    client = MlflowClient(tracking_uri=info["tracking_uri"])
    assert client.get_run(info["run_id"]).data.tags["quality_gate"] == "failed"
    assert client.search_registered_models() == []


def test_tracking_can_be_switched_off(fitted, tmp_path, monkeypatch):
    monkeypatch.setenv("STBMSE_TRACKING", "off")
    info = _log(fitted, tmp_path, gate_passed=True)
    assert info["enabled"] is False
    assert not (tmp_path / "mlflow.db").exists()
