"""Stage 4: leakage audit and drift monitoring.

Leakage audit: shows how much of a "good" score comes from same-month information that would not exist when
forecasting, and whether any single column (Sentiments) is doing the work.
Drift monitoring: Population Stability Index (PSI) of the inputs and of the model score between periods, plus the
shift in the escalation rate. PSI > 0.25 is conventionally treated as a major shift.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .data import ingest
from .features import NUMERIC_BASE, CATEGORICAL, build_features, prepare_xy
from .modeling import StackedEnsemble, evaluate

HISTORY_ONLY = ["y_lag0", "y_lag1", "y_lag2", "y_roll3", "y_roll6", "y_roll12", "cum_hist",
                "eb_markov_next", "eb_p_stay", "eb_p_enter"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def leakage_audit() -> pd.DataFrame:
    df, _ = ingest(write_report=False)
    feat = build_features(df)
    prep = prepare_xy(feat)
    X, y, tr, va, te = prep.X, prep.y, prep.train, prep.val, prep.test
    rows = []

    # (1) NOWCAST: current-month indicators -> SAME-month label. Not a forecast; shows what leakage looks like.
    contemporaneous = [c for c in X.columns
                       if c in NUMERIC_BASE + ["Historical_Conflict_Presence", "tension_num"]
                       or any(c.startswith(cat + "_") for cat in CATEGORICAL)]
    now_label = prep.persistence.astype(int)          # persistence score == this month's own label
    now_tr, now_te = tr | va, te
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=C.SEED).fit(
        X.loc[now_tr, contemporaneous], now_label[now_tr])
    p = rf.predict_proba(X.loc[now_te, contemporaneous])[:, 1]
    rows.append({"setting": "NOWCAST (not a forecast): this month's indicators -> this month's label",
                 "pr_auc": average_precision_score(now_label[now_te], p), "roc_auc": roc_auc_score(now_label[now_te], p)})

    # (2) same indicators, but forecasting NEXT month
    rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=C.SEED).fit(X.loc[tr | va, contemporaneous], y[tr | va])
    p = rf.predict_proba(X.loc[te, contemporaneous])[:, 1]
    rows.append({"setting": "FORECAST with this month's indicators only (no history)",
                 "pr_auc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p)})

    # (3) the full ensemble and ablations (all forecast next month, thresholds from validation)
    def run(cols, label):
        e = StackedEnsemble().fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
        m = evaluate(label, y[te], e.predict_proba(X.loc[te, cols]), e.threshold)
        rows.append({"setting": label, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"]})

    run(list(X.columns), "FORECAST ST-BMSE ensemble, all features")
    run([c for c in X.columns if not c.startswith("Sentiments_")], "FORECAST ensemble WITHOUT Sentiments")
    run([c for c in X.columns if c in HISTORY_ONLY], "FORECAST ensemble, escalation history only")
    out = pd.DataFrame(rows).round(4)
    rates = pd.DataFrame({
        "Sentiments": ["Positive", "Neutral", "Negative"],
        "same_month_escalation_rate": [float(df.loc[df.Sentiments == s, C.TARGET].mean()) for s in ("Positive", "Neutral", "Negative")],
    }).round(4)
    out.to_csv(C.REPORT_DIR / "leakage_audit.csv", index=False)
    rates.to_csv(C.REPORT_DIR / "sentiments_vs_same_month_label.csv", index=False)
    return out


def drift_report() -> dict:
    df, _ = ingest(write_report=False)
    feat = build_features(df)
    prep = prepare_xy(feat)
    X, y = prep.X, prep.y
    cols = [c for c in X.columns if X[c].nunique() > 2][:]            # continuous features only
    rows = []
    for c in cols:
        rows.append({"feature": c, "psi_train_vs_test": psi(X.loc[prep.train, c].values, X.loc[prep.test, c].values)})
    table = pd.DataFrame(rows).sort_values("psi_train_vs_test", ascending=False).round(4)
    table.to_csv(C.REPORT_DIR / "drift_psi_features.csv", index=False)
    report = {
        "escalation_rate_train": float(y[prep.train].mean()), "escalation_rate_validation": float(y[prep.val].mean()),
        "escalation_rate_test": float(y[prep.test].mean()),
        "features_psi_gt_0_25": table[table.psi_train_vs_test > 0.25]["feature"].tolist(),
        "features_psi_0_10_to_0_25": table[(table.psi_train_vs_test > 0.10) & (table.psi_train_vs_test <= 0.25)]["feature"].tolist(),
        "max_feature_psi": float(table.psi_train_vs_test.max()),
        "interpretation": "The base rate of escalation more than doubles between training and test periods (prior shift).",
    }
    (C.REPORT_DIR / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def subgroup_report() -> pd.DataFrame:
    """Test-period performance of the deployed ensemble per region (fairness / uneven-coverage check)."""
    import joblib
    art = joblib.load(C.MODEL_PATH)
    ens = art["ensemble"]
    df, _ = ingest(write_report=False)
    prep = prepare_xy(build_features(df))
    region_of = df.drop_duplicates("County").set_index("County")["Region"]
    te = prep.test
    meta = prep.meta[te].reset_index(drop=True)
    y = prep.y[te].reset_index(drop=True).values
    p = ens.predict_proba(prep.X[te])
    reg = meta["County"].map(region_of).values
    rows = []
    for r in sorted(set(reg)):
        m = reg == r
        pred = p[m] >= ens.threshold
        pos, neg = (y[m] == 1), (y[m] == 0)
        rows.append({
            "region": r, "counties": int(meta.loc[m, "County"].nunique()), "n": int(m.sum()),
            "observed_escalation_rate": float(y[m].mean()), "mean_predicted_probability": float(p[m].mean()),
            "recall": float(pred[pos].mean()) if pos.any() else float("nan"),
            "false_alarm_rate": float(pred[neg].mean()) if neg.any() else float("nan"),
            "pr_auc": float(average_precision_score(y[m], p[m])) if 0 < y[m].sum() < m.sum() else float("nan"),
            "roc_auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].sum() < m.sum() else float("nan"),
        })
    out = pd.DataFrame(rows).round(3)
    out.to_csv(C.REPORT_DIR / "subgroup_by_region.csv", index=False)
    return out


if __name__ == "__main__":
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(leakage_audit().to_string(index=False))
    print(json.dumps(drift_report(), indent=2))
    print(subgroup_report().to_string(index=False))
