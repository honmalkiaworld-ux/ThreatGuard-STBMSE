"""Rolling-origin (expanding-window) evaluation.

The main evaluation uses ONE split (train <= 2019, validate 2020-2021, test 2022-2024). A single split can be lucky or
unlucky, and here the base rate of escalation shifts a lot over time. This module repeats the full procedure with the
cut moved forward one year at a time, so the headline claim ("the ensemble beats persistence") can be checked year by year.

For each test year Y (default 2020..2024), everything is refitted from scratch:
    base learners        fitted on forecasts of years   <= Y-3
    stacker + threshold  fitted on forecasts of years   Y-2 .. Y-1
    test                 forecasts of year Y only
So no window ever sees data from its own test year or later (checked by tests/test_rolling.py).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C
from .data import ingest
from .features import build_features, prepare_xy
from .modeling import StackedEnsemble, evaluate, paired_difference_ci

TEST_YEARS = (2020, 2021, 2022, 2023, 2024)
VAL_YEARS = 2


def window_masks(target_year, test_year: int, val_years: int = VAL_YEARS):
    """Boolean masks (train, validation, test) over forecast-year labels for one rolling window."""
    ty = np.asarray(target_year)
    last_train = test_year - 1 - val_years
    train = ty <= last_train
    val = (ty > last_train) & (ty <= test_year - 1)
    test = ty == test_year
    return train, val, test


def rolling_origin(test_years=TEST_YEARS, val_years: int = VAL_YEARS, n_boot: int = 200,
                   learner_params: dict | None = None, write: bool = True, prep=None) -> pd.DataFrame:
    if prep is None:
        df, _ = ingest(write_report=False)
        prep = prepare_xy(build_features(df))
    X, y = prep.X, prep.y
    rows = []
    for Y in test_years:
        tr, va, te = window_masks(prep.target_year, Y, val_years)
        ens = StackedEnsemble(learner_params=learner_params).fit(X[tr], y[tr], X[va], y[va])
        yt = y[te].values
        p = ens.predict_proba(X[te])
        pers = prep.persistence[te]
        m = evaluate("ensemble", yt, p, ens.threshold)
        gain = paired_difference_ci(yt, p, pers, prep.meta.loc[te, "County"].values, average_precision_score, n_boot)
        rows.append({
            "test_year": Y, "base_learners_trained_through": Y - 1 - val_years,
            "stacker_validation_years": f"{Y - val_years}-{Y - 1}", "n_test": int(te.sum()),
            "base_rate": float(yt.mean()), "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
            "persistence_pr_auc": float(average_precision_score(yt, pers)),
            "pr_auc_gain_over_persistence": gain["point"], "gain_ci_low": gain["ci95"][0], "gain_ci_high": gain["ci95"][1],
            "recall": m["recall"], "false_alarm_rate": m["false_alarm_rate"], "threshold": m["threshold"],
        })
    out = pd.DataFrame(rows)
    if write:
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out.round(4).to_csv(C.REPORT_DIR / "rolling_origin.csv", index=False)
    return out


def summarise(out: pd.DataFrame) -> dict:
    clear = int((out["gain_ci_low"] > 0).sum())
    return {"windows": int(len(out)), "windows_where_gain_ci_excludes_zero": clear,
            "mean_pr_auc_gain": float(out["pr_auc_gain_over_persistence"].mean()),
            "min_pr_auc_gain": float(out["pr_auc_gain_over_persistence"].min()),
            "max_pr_auc_gain": float(out["pr_auc_gain_over_persistence"].max())}


if __name__ == "__main__":
    r = rolling_origin()
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(summarise(r))
