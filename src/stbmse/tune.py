"""Hyper-parameter tuning study (Optuna), time-aware and test-safe.

Question answered: does searching the hyper-parameters improve on the fixed settings the model ships with?

Protocol (all of it is blind to the 2022-2024 test period):
  * Each base learner is tuned separately with Optuna (TPE sampler, fixed seed).
  * The objective is the MEAN PR-AUC over four expanding-window folds: for forecast year Y in 2018..2021,
    fit on forecasts of years < Y and score year Y. Rows from the future never train a model that is scored on the past.
  * The fixed settings are scored on the very same folds, so the comparison is like for like.
  * Only afterwards are the tuned and fixed ensembles evaluated, once, on the untouched test period, with county-block
    bootstrap intervals and a paired difference.

The study does NOT replace the shipped model. Adopting tuned settings would change every published number, so that is a
decision for the author; `reports/tuned_params.json` holds the settings if wanted.

    python -m stbmse.tune --trials 25
"""
import argparse
import json
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .data import ingest
from .features import build_features, prepare_xy
from .modeling import (DEFAULT_PARAMS, StackedEnsemble, bootstrap_ci, evaluate, make_learners,
                       paired_difference_ci)
from .rolling import rolling_origin

CV_YEARS = (2018, 2019, 2020, 2021)       # forecast years used as validation folds; all strictly before the test period
assert max(CV_YEARS) <= C.VAL_END_YEAR


def search_space(name: str, trial: "optuna.Trial") -> dict:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 12, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5, 1.0]),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(name)


def cv_score(name: str, params: dict, prep, years=CV_YEARS) -> float:
    """Mean PR-AUC over expanding-window folds (train on years < Y, score year Y)."""
    X, y, ty = prep.X, prep.y, prep.target_year
    scores = []
    for Y in years:
        tr, va = ty < Y, ty == Y
        model = make_learners(C.SEED, {name: params})[name]
        model.fit(X[tr], y[tr])
        scores.append(average_precision_score(y[va], model.predict_proba(X[va])[:, 1]))
    return float(np.mean(scores))


def tune_learner(name: str, prep, n_trials: int) -> dict:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(lambda t: cv_score(name, search_space(name, t), prep), n_trials=n_trials)
    fixed = cv_score(name, {}, prep)
    return {"best_params": study.best_params, "cv_pr_auc_tuned": float(study.best_value),
            "cv_pr_auc_fixed": fixed, "cv_gain": float(study.best_value - fixed), "n_trials": n_trials,
            "fixed_params": DEFAULT_PARAMS[name]}


def main(n_trials: int = 25) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")
    df, _ = ingest(write_report=False)
    prep = prepare_xy(build_features(df))
    X, y, tr, va, te = prep.X, prep.y, prep.train, prep.val, prep.test

    learners = {n: tune_learner(n, prep, n_trials) for n in DEFAULT_PARAMS}
    tuned = {n: v["best_params"] for n, v in learners.items()}
    (C.REPORT_DIR / "tuned_params.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")

    # one evaluation on the untouched test period: fixed vs tuned ensemble, same protocol as the main results
    yt, groups = y[te].values, prep.meta.loc[te, "County"].values
    fixed_e = StackedEnsemble().fit(X[tr], y[tr], X[va], y[va])
    tuned_e = StackedEnsemble(learner_params=tuned).fit(X[tr], y[tr], X[va], y[va])
    p_f, p_t, p_0 = fixed_e.predict_proba(X[te]), tuned_e.predict_proba(X[te]), prep.persistence[te]
    scores = {"fixed": p_f, "tuned": p_t, "persistence": p_0}
    ci_pr = bootstrap_ci(yt, scores, groups, average_precision_score, C.N_BOOTSTRAP)
    ci_roc = bootstrap_ci(yt, scores, groups, roc_auc_score, C.N_BOOTSTRAP)
    test = {}
    for k, (p, thr) in {"fixed": (p_f, fixed_e.threshold), "tuned": (p_t, tuned_e.threshold)}.items():
        m = evaluate(k, yt, p, thr)
        test[k] = {"pr_auc": m["pr_auc"], "pr_auc_ci95": ci_pr[k], "roc_auc": m["roc_auc"], "roc_auc_ci95": ci_roc[k],
                   "recall": m["recall"], "false_alarm_rate": m["false_alarm_rate"], "brier": m["brier"], "threshold": thr}
    paired = {
        "tuned_minus_fixed_pr_auc": paired_difference_ci(yt, p_t, p_f, groups, average_precision_score, C.N_BOOTSTRAP),
        "tuned_minus_persistence_pr_auc": paired_difference_ci(yt, p_t, p_0, groups, average_precision_score, C.N_BOOTSTRAP),
        "fixed_minus_persistence_pr_auc": paired_difference_ci(yt, p_f, p_0, groups, average_precision_score, C.N_BOOTSTRAP),
    }

    # per-year check on test years only (the tuned settings never saw 2022 or later)
    per_year = rolling_origin(test_years=(2022, 2023, 2024), learner_params=tuned, write=False)
    base_year = rolling_origin(test_years=(2022, 2023, 2024), write=False)
    per = pd.DataFrame({"test_year": per_year.test_year, "fixed_pr_auc": base_year.pr_auc, "tuned_pr_auc": per_year.pr_auc})
    per["tuned_minus_fixed"] = per.tuned_pr_auc - per.fixed_pr_auc
    per.round(4).to_csv(C.REPORT_DIR / "tuning_per_year.csv", index=False)

    d = paired["tuned_minus_fixed_pr_auc"]
    verdict = ("tuned settings improve test PR-AUC (paired interval excludes 0)" if d["ci95"][0] > 0 else
               "tuned settings are worse on test (paired interval excludes 0)" if d["ci95"][1] < 0 else
               "no evidence that tuning changes test PR-AUC (paired interval includes 0)")
    result = {
        "protocol": {"cv_forecast_years": list(CV_YEARS), "objective": "mean PR-AUC over expanding-window folds",
                     "sampler": "TPE, seed %d" % C.SEED, "trials_per_learner": n_trials,
                     "test_period_used_for_tuning": False},
        "learners": learners, "test_comparison": test, "paired_pr_auc": paired, "verdict": verdict,
        "shipped_model": "unchanged (fixed settings); tuned settings are in reports/tuned_params.json if the author adopts them",
    }
    (C.REPORT_DIR / "tuning_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    r = main(ap.parse_args().trials)
    for n, v in r["learners"].items():
        print(f"{n:20s} CV PR-AUC fixed {v['cv_pr_auc_fixed']:.4f} -> tuned {v['cv_pr_auc_tuned']:.4f} (gain {v['cv_gain']:+.4f})")
    print("test PR-AUC fixed", round(r["test_comparison"]["fixed"]["pr_auc"], 4), "tuned", round(r["test_comparison"]["tuned"]["pr_auc"], 4))
    d = r["paired_pr_auc"]["tuned_minus_fixed_pr_auc"]
    print(f"paired tuned-fixed PR-AUC: {d['point']:+.4f}  95% CI [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")
    print("verdict:", r["verdict"])
