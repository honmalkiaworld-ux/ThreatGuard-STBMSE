"""MLflow experiment tracking and model registry.

Optional by design: it is skipped when `STBMSE_TRACKING=off` or when MLflow is not installed. The slim Docker image
installs only serving dependencies, so it trains without tracking; the full environment and CI track every run.

Logged per training run (parent run `st_bmse_ensemble`):
  params     split years, cost ratio, EB prior strength, seed, bootstrap size
  metrics    test and validation PR-AUC / ROC-AUC / recall / false-alarm rate / Brier, ECE, threshold, gate gain
  artifacts  results, calibration and importance tables, data-quality report, the gate verdict, the model
  children   one nested run per base learner (Logistic Regression, Random Forest, XGBoost) with its hyper-parameters
Registry: if (and only if) the quality gate passed, the model is registered as `stbmse-ensemble` and given the
alias `production`. A failed run is still logged, tagged `quality_gate=failed`, but never registered.

Browse it:  mlflow ui --backend-store-uri sqlite:///mlflow.db     (http://127.0.0.1:5000)
Store location can be overridden with the MLFLOW_TRACKING_URI environment variable.
"""
import os
from pathlib import Path

from . import config as C

EXPERIMENT = "stbmse-county-escalation"
MODEL_NAME = "stbmse-ensemble"
METRIC_COLS = ["pr_auc", "roc_auc", "recall", "false_alarm_rate", "precision", "brier", "expected_cost_per_row"]

# MLflow 3 stores sklearn models with `skops`, which refuses to load any type it has not been told to trust (unlike
# raw pickle, which would execute anything). These are the only four non-default types inside this project's model;
# each is code this project owns or a standard tree/booster node store. Do not widen this list to "everything".
TRUSTED_TYPES = [
    "stbmse.modeling.StackedEnsemble",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def disabled_by_env() -> bool:
    return os.environ.get("STBMSE_TRACKING", "on").strip().lower() in {"off", "0", "false", "no"}


def available() -> bool:
    if disabled_by_env():
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}")


def _log_row_metrics(prefix: str, row) -> None:
    import mlflow
    for col in METRIC_COLS:
        if col in row.index and row[col] == row[col]:          # skip NaN
            mlflow.log_metric(f"{prefix}_{col}", float(row[col]))


def log_training_run(*, ensemble, metrics: dict, params: dict, learner_params: dict, results_test, results_val,
                     artifacts: list, sample_X=None, gate_passed: bool, tracking_uri: str | None = None,
                     experiment: str = EXPERIMENT, register: bool = True, artifact_root: Path | None = None) -> dict:
    """Log one training run. Returns a small dict describing what was recorded (stored in metrics.json)."""
    if not available():
        why = "disabled by STBMSE_TRACKING" if disabled_by_env() else "mlflow is not installed"
        print(f"[tracking] skipped: {why}", flush=True)
        return {"enabled": False, "reason": why}

    import mlflow
    from mlflow import MlflowClient

    uri = tracking_uri or default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        # keep artifacts beside the project, not in whatever the current directory happens to be
        client.create_experiment(experiment, artifact_location=Path(artifact_root or (C.ROOT / "mlruns")).resolve().as_uri())
    mlflow.set_experiment(experiment)

    info = {"enabled": True, "tracking_uri": uri, "experiment": experiment}
    with mlflow.start_run(run_name="st_bmse_ensemble") as run:
        mlflow.log_params({k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in params.items()})
        mlflow.set_tags({
            "quality_gate": "passed" if gate_passed else "failed",
            "data": "kenya_conflict_monthly.csv",
            "data_provenance": "unverified (not a raw ACLED/KNBS export)",
            "task": "county-level next-month conflict escalation",
        })

        _log_row_metrics("test", results_test.loc["st_bmse_ensemble"])
        _log_row_metrics("test_persistence", results_test.loc["persistence"])
        mlflow.log_metric("test_pr_auc_gain_over_persistence",
                          float(results_test.loc["st_bmse_ensemble", "pr_auc"] - results_test.loc["persistence", "pr_auc"]))
        mlflow.log_metric("ece_test", float(metrics["ece_test"]))
        mlflow.log_metric("alert_threshold", float(metrics["threshold"]))
        for name, w in metrics["meta_weights_logit"].items():
            mlflow.log_metric(f"meta_weight_{name}", float(w))
        best_val = results_val.drop(index="persistence")["pr_auc"].max()
        mlflow.log_metric("val_best_base_pr_auc", float(best_val))
        mlflow.log_metric("val_best_base_gain_over_persistence", float(best_val - results_val.loc["persistence", "pr_auc"]))

        mlflow.log_dict(metrics["quality_gate"], "quality_gate.json")
        for p in artifacts:
            if Path(p).exists():
                mlflow.log_artifact(str(p), "reports")

        for name in ensemble.learners:                         # nested runs make the three learners comparable in the UI
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("learner", name)
                mlflow.log_params(learner_params.get(name, {}))
                mlflow.log_metric("alert_threshold", float(ensemble.base_thresholds[name]))
                _log_row_metrics("val", results_val.loc[name])
                _log_row_metrics("test", results_test.loc[name])

        model_info = mlflow.sklearn.log_model(ensemble, name="model", skops_trusted_types=TRUSTED_TYPES)
        info["run_id"] = run.info.run_id
        info["model_uri"] = model_info.model_uri

    if gate_passed and register:
        mv = mlflow.register_model(info["model_uri"], MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
        info.update({"registered_model": MODEL_NAME, "model_version": str(mv.version), "alias": "production"})
        print(f"[tracking] registered {MODEL_NAME} v{mv.version} with alias 'production'", flush=True)
    else:
        info["registered_model"] = None
        print("[tracking] run logged; model NOT registered" + ("" if gate_passed else " (quality gate failed)"), flush=True)
    print(f"[tracking] run {info['run_id']} in {uri}", flush=True)
    return info
