"""Stage 3: train the ST-BMSE ensemble, gate it, evaluate it on the untouched test period, track it and serialise it.

Order matters:
  1. fit base learners (train period) and the stacker + threshold (validation period)
  2. QUALITY GATE on validation scores only. The test period never influences whether a model ships.
  3. evaluate on the test period (reported, never used for decisions)
  4. log everything to MLflow (a failed run is logged too, for traceability, but is never registered)
  5. only if the gate passed: write the deployable artifact and register the model as `production`
"""
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score

from . import __version__
from . import config as C
from . import tracking
from .data import ingest
from .features import build_features, prepare_xy
from .gate import QualityGateError, check_quality_gate
from .modeling import (StackedEnsemble, bootstrap_ci, calibration_table, evaluate,
                       paired_difference_ci, permutation_importance)

LOGGED_PARAMS = {
    "logistic_regression": ["C", "max_iter"],
    "random_forest": ["n_estimators", "min_samples_leaf", "random_state"],
    "xgboost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state"],
}


def _learner_params(ens: StackedEnsemble) -> dict:
    out = {}
    for name, model in ens.learners.items():
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        p = est.get_params()
        out[name] = {k: p[k] for k in LOGGED_PARAMS[name] if k in p}
    return out


def _portable(metrics: dict) -> dict:
    """Copy of the metrics that is safe to embed in the artifact and to commit.

    Tracking details contain absolute local paths (the MLflow store location), which would leak the machine's folder
    layout and user name into a file that is shared. Run id, experiment and model version are kept; the paths are not.
    """
    m = json.loads(json.dumps(metrics))
    for key in ("tracking_uri", "model_uri"):
        m.get("tracking", {}).pop(key, None)
    return m


def train(raw_path=C.RAW_CSV, fast: bool = False) -> dict:
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df, quality = ingest(raw_path)
    feat = build_features(df)
    prep = prepare_xy(feat)
    X, y, meta = prep.X, prep.y, prep.meta
    tr, va, te = prep.train, prep.val, prep.test

    # reproducible, inspectable modelling table (features known at t, label = escalation at t+1)
    out = pd.concat([meta, X, y.rename("target_next")], axis=1)
    out["split"] = np.where(tr, "train", np.where(va, "validation", "test"))
    out.to_csv(C.PROCESSED_DIR / "modelling_table.csv", index=False)

    # 1. fit
    ens = StackedEnsemble().fit(X[tr], y[tr], X[va], y[va])

    # 2. quality gate (validation only)
    yv = y[va].values
    pv_base = ens.base_proba(X[va])                     # base learners were fitted on TRAIN, so these are out-of-sample
    gate = check_quality_gate(yv, prep.persistence[va], pv_base, ens.threshold)
    print("\nQuality gate (validation data only):\n" + gate.summary() + "\n", flush=True)

    val_rows = [evaluate("persistence", yv, prep.persistence[va], 0.5)]
    val_rows += [evaluate(n, yv, pv_base[n], ens.base_thresholds[n]) for n in ens.learners]
    val_results = pd.DataFrame(val_rows).set_index("model")
    val_results.round(4).to_csv(C.REPORT_DIR / "results_validation_base_learners.csv")

    # 3. test-period evaluation
    yt = y[te].values
    p_base = ens.base_proba(X[te])
    p_ens = ens.predict_proba(X[te])
    scores = {"persistence": prep.persistence[te], **p_base, "st_bmse_ensemble": p_ens}
    rows = [evaluate("persistence", yt, scores["persistence"], 0.5)]
    for n in ens.learners:
        rows.append(evaluate(n, yt, p_base[n], ens.base_thresholds[n]))
    rows.append(evaluate("st_bmse_ensemble", yt, p_ens, ens.threshold))
    results = pd.DataFrame(rows).set_index("model")

    groups = meta.loc[te, "County"].values
    n_boot = 100 if fast else C.N_BOOTSTRAP
    ci_pr = bootstrap_ci(yt, scores, groups, average_precision_score, n_boot)
    ci_roc = bootstrap_ci(yt, scores, groups, roc_auc_score, n_boot)
    for k in results.index:
        results.loc[k, "pr_auc_ci_low"], results.loc[k, "pr_auc_ci_high"] = ci_pr[k]
        results.loc[k, "roc_auc_ci_low"], results.loc[k, "roc_auc_ci_high"] = ci_roc[k]
    results.round(4).to_csv(C.REPORT_DIR / "results_test.csv")

    paired = {
        "ensemble_minus_persistence_pr_auc": paired_difference_ci(yt, p_ens, scores["persistence"], groups, average_precision_score, n_boot),
        "ensemble_minus_random_forest_pr_auc": paired_difference_ci(yt, p_ens, p_base["random_forest"], groups, average_precision_score, n_boot),
        "ensemble_minus_logistic_pr_auc": paired_difference_ci(yt, p_ens, p_base["logistic_regression"], groups, average_precision_score, n_boot),
    }

    cal, ece = calibration_table(yt, p_ens)
    cal.round(4).to_csv(C.REPORT_DIR / "calibration_test.csv", index=False)

    imp = permutation_importance(ens, X[va], yv, n_repeats=2 if fast else 5)
    imp.round(5).to_csv(C.REPORT_DIR / "feature_importance_validation.csv", index=False)

    metrics = {
        "version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())},
        "n_features": int(X.shape[1]),
        "escalation_rate": {"train": float(y[tr].mean()), "validation": float(y[va].mean()), "test": float(y[te].mean())},
        "threshold": ens.threshold, "cost_ratio_fn_to_fp": C.FN_COST / C.FP_COST,
        "meta_weights_logit": ens.meta_weights,
        "quality_gate": {"passed": gate.passed, "checks": gate.checks},
        "validation_results_base_learners": json.loads(val_results.round(4).to_json(orient="index")),
        "test_results": json.loads(results.round(4).to_json(orient="index")),
        "paired_pr_auc_differences": paired, "ece_test": ece,
        "top_features_validation": imp.head(10)["feature"].tolist(),
        "data_quality": {k: quality[k] for k in ("rows", "counties", "period", "panel_cells_present", "warnings")},
    }

    # 4. experiment tracking (MLflow)
    params = {
        "seed": C.SEED, "train_end_year": C.TRAIN_END_YEAR, "val_end_year": C.VAL_END_YEAR,
        "cost_fn": C.FN_COST, "cost_fp": C.FP_COST, "k_prior": C.K_PRIOR, "n_features": int(X.shape[1]),
        "n_bootstrap": n_boot, "raw_data": str(raw_path),
    }
    artifacts = [C.REPORT_DIR / f for f in ("results_test.csv", "results_validation_base_learners.csv",
                                            "calibration_test.csv", "feature_importance_validation.csv", "data_quality.json")]
    metrics["tracking"] = tracking.log_training_run(
        ensemble=ens, metrics=metrics, params=params, learner_params=_learner_params(ens),
        results_test=results, results_val=val_results, artifacts=artifacts,
        sample_X=X[va].head(5), gate_passed=gate.passed,
    )
    portable = _portable(metrics)
    C.METRICS_PATH.write_text(json.dumps(portable, indent=2), encoding="utf-8")

    # 5. ship only if the gate passed
    if not gate.passed:
        raise QualityGateError("Quality gate FAILED; no model artifact was written and nothing was registered.\n" + gate.summary())

    joblib.dump({
        "ensemble": ens, "feature_columns": list(X.columns), "metrics": portable,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }, C.MODEL_PATH)
    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps(m["test_results"]["st_bmse_ensemble"], indent=2))
