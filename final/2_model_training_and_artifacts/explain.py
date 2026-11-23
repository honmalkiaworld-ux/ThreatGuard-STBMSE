"""SHAP explanations of the shipped model.

The ensemble blends three learners through a logistic stacker, so there is no single closed-form SHAP for the blend.
Instead each base learner is explained EXACTLY and the attributions are combined with the stacker's weights:

  logistic regression   exact linear SHAP in log-odds: contribution_j = coef_j * z_j  (z = standardised feature, 0 = training mean)
  random forest         TreeSHAP (shap.TreeExplainer), in probability units
  XGBoost               TreeSHAP (shap.TreeExplainer), in log-odds

Because the three are in different units, each model's mean |SHAP| is turned into a SHARE of that model's total, and the
ensemble figure is the share averaged with the stacker's absolute weights. This is an approximation of the blend's
attribution, stated as such. Every explainer is checked for additivity (contributions + base value reproduce the model's
own output); the worst error is recorded in the summary.

Computed on the VALIDATION period, like the permutation importance, so the test set is not used for any analysis choice.

    python -m stbmse.explain
"""
import json

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import ingest
from .features import build_features, prepare_xy


def available() -> bool:
    """SHAP is an optional dependency (requirements.txt, not requirements-api.txt); the slim Docker image lacks it."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


def _tree_values(model, X):
    import shap                                   # imported here so that importing this module never needs it
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    base = expl.expected_value
    if isinstance(sv, list):                      # older shap: one array per class
        sv, base = sv[1], base[1]
    elif getattr(sv, "ndim", 2) == 3:             # newer shap: (rows, features, classes)
        sv, base = sv[:, :, 1], np.asarray(base)[1]
    return np.asarray(sv), float(np.asarray(base).reshape(-1)[0] if np.ndim(base) else base)


def shap_by_model(ens, X: pd.DataFrame) -> dict:
    """{learner: (values[n, p], base_value, unit)} for the three base learners. Features are ens.columns."""
    X = X.reindex(columns=ens.columns, fill_value=0.0)
    out = {}
    lr = ens.learners["logistic_regression"]
    scaler, clf = lr.steps[0][1], lr.steps[-1][1]
    z = scaler.transform(X)
    out["logistic_regression"] = (z * clf.coef_[0], float(clf.intercept_[0]), "log-odds")
    sv, base = _tree_values(ens.learners["random_forest"], X)
    out["random_forest"] = (sv, base, "probability")
    sv, base = _tree_values(ens.learners["xgboost"], X)
    out["xgboost"] = (sv, base, "log-odds")
    return out


def additivity_errors(ens, X: pd.DataFrame, values: dict) -> dict:
    """Max |sum(contributions) + base - model output| per learner (should be ~0)."""
    X = X.reindex(columns=ens.columns, fill_value=0.0)
    errs = {}
    lr = ens.learners["logistic_regression"]
    errs["logistic_regression"] = float(np.abs(values["logistic_regression"][0].sum(1) + values["logistic_regression"][1]
                                               - lr.decision_function(X)).max())
    sv, base, _ = values["random_forest"]
    errs["random_forest"] = float(np.abs(sv.sum(1) + base - ens.learners["random_forest"].predict_proba(X)[:, 1]).max())
    sv, base, _ = values["xgboost"]
    margin = ens.learners["xgboost"].predict(X, output_margin=True)
    errs["xgboost"] = float(np.abs(sv.sum(1) + base - margin).max())
    return errs


def importance_table(values: dict, columns: list, meta_weights: dict) -> pd.DataFrame:
    w = pd.Series({k: abs(v) for k, v in meta_weights.items()})
    w = w / w.sum()
    cols = {}
    for name, (sv, _, _) in values.items():
        m = np.abs(sv).mean(0)
        cols[f"share_{name}"] = m / m.sum()
    t = pd.DataFrame(cols, index=columns)
    t["weighted_share"] = sum(t[f"share_{n}"] * w[n] for n in values)
    return t.sort_values("weighted_share", ascending=False).rename_axis("feature").reset_index()


def main() -> dict:
    art = joblib.load(C.MODEL_PATH)
    ens = art["ensemble"]
    df, _ = ingest(write_report=False)
    prep = prepare_xy(build_features(df))
    Xv = prep.X[prep.val].reset_index(drop=True)

    values = shap_by_model(ens, Xv)
    errs = additivity_errors(ens, Xv, values)
    tab = importance_table(values, list(ens.columns), ens.meta_weights)
    tab.round(5).to_csv(C.REPORT_DIR / "shap_importance.csv", index=False)

    # does SHAP agree with the permutation importance already reported? (rank correlation over all features)
    perm_path = C.REPORT_DIR / "feature_importance_validation.csv"
    rho = None
    if perm_path.exists():
        perm = pd.read_csv(perm_path).set_index("feature")["pr_auc_drop"]
        j = tab.set_index("feature")["weighted_share"].to_frame().join(perm, how="inner")
        rho = float(j["weighted_share"].rank().corr(j["pr_auc_drop"].rank()))

    summary = {
        "rows_explained": int(len(Xv)), "period": "validation", "units": {k: v[2] for k, v in values.items()},
        "max_additivity_error": errs, "additivity_ok": bool(max(errs.values()) < 1e-3),
        "stacker_weights_abs_normalised": {k: round(abs(v) / sum(abs(x) for x in ens.meta_weights.values()), 4)
                                           for k, v in ens.meta_weights.items()},
        "top_12_weighted_share": tab.head(12)[["feature", "weighted_share"]].round(4).to_dict("records"),
        "spearman_vs_permutation_importance": rho,
        "note": "Approximation of the blend: per-model shares averaged with the stacker's absolute weights.",
    }
    (C.REPORT_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = main()
    print("rows explained:", s["rows_explained"], "| additivity ok:", s["additivity_ok"], s["max_additivity_error"])
    print("agreement with permutation importance (Spearman):", s["spearman_vs_permutation_importance"])
    for r in s["top_12_weighted_share"]:
        print(f"  {r['feature']:38s} {r['weighted_share']:.3f}")
