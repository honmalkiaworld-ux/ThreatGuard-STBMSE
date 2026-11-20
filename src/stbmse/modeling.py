"""Stage 3 helpers: the stacked ensemble, the cost-sensitive threshold and honest evaluation tools."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def best_threshold(y, p, fn_cost: float = C.FN_COST, fp_cost: float = C.FP_COST) -> float:
    """Threshold minimising expected cost on the supplied (validation) data."""
    y = np.asarray(y)
    grid = np.linspace(0.02, 0.98, 97)
    costs = [fn_cost * ((p < t) & (y == 1)).sum() + fp_cost * ((p >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


# The fixed hyper-parameters of the shipped model. `tune.py` searches around these; they are not changed here.
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 0.5},
    "random_forest": {"n_estimators": 400, "min_samples_leaf": 5},
    "xgboost": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
}


def make_learners(seed: int = C.SEED, params: dict | None = None) -> dict:
    """Build the three base learners. `params` overrides DEFAULT_PARAMS per learner (used by the tuning study)."""
    p = {k: {**v, **((params or {}).get(k, {}))} for k, v in DEFAULT_PARAMS.items()}
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed,
                                                                                   **p["logistic_regression"])),
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=seed, **p["random_forest"]),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=4, **p["xgboost"]),
    }


class StackedEnsemble:
    """Logistic Regression + Random Forest + XGBoost, blended by a calibrated logistic meta-learner.

    Base learners are fitted on the training period; the meta-learner (which is also the probability
    calibrator) and the decision threshold are fitted on the later validation period, so neither sees test data.
    """

    def __init__(self, seed: int = C.SEED, fn_cost: float = C.FN_COST, fp_cost: float = C.FP_COST,
                 learner_params: dict | None = None):
        self.seed, self.fn_cost, self.fp_cost = seed, fn_cost, fp_cost
        self.learners = make_learners(seed, learner_params)
        self.meta = LogisticRegression(C=1.0, max_iter=2000)
        self.columns: list[str] = []
        self.threshold: float = 0.5
        self.base_thresholds: dict[str, float] = {}

    def fit(self, X_train, y_train, X_val, y_val):
        self.columns = list(X_train.columns)
        for m in self.learners.values():
            m.fit(X_train, y_train)
        pv = self.base_proba(X_val)
        Z = np.column_stack([_logit(pv[n]) for n in self.learners])
        self.meta.fit(Z, y_val)
        self.threshold = best_threshold(y_val, self.meta.predict_proba(Z)[:, 1], self.fn_cost, self.fp_cost)
        self.base_thresholds = {n: best_threshold(y_val, pv[n], self.fn_cost, self.fp_cost) for n in self.learners}
        return self

    def base_proba(self, X) -> dict:
        X = X.reindex(columns=self.columns, fill_value=0.0)
        return {n: m.predict_proba(X)[:, 1] for n, m in self.learners.items()}

    def predict_proba(self, X) -> np.ndarray:
        pb = self.base_proba(X)
        Z = np.column_stack([_logit(pb[n]) for n in self.learners])
        return self.meta.predict_proba(Z)[:, 1]

    def predict(self, X) -> np.ndarray:
        """Alert flag (0/1) at the cost-tuned threshold. Lets the ensemble be loaded back through MLflow's pyfunc."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    @property
    def meta_weights(self) -> dict:
        return {n: float(w) for n, w in zip(self.learners, self.meta.coef_[0])}


# ---------------------------------------------------------------------------------- evaluation
def evaluate(name, y, p, threshold, fn_cost: float = C.FN_COST, fp_cost: float = C.FP_COST) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "recall": float(recall_score(y, pred)), "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "threshold": float(threshold),
        "expected_cost_per_row": float((fn_cost * fn + fp_cost * fp) / len(y)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _cluster_indices(groups, rng, n_boot):
    groups = np.asarray(groups)
    idx_by = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys = list(idx_by)
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        yield np.concatenate([idx_by[keys[i]] for i in pick])


def bootstrap_ci(y, scores: dict, groups, metric=average_precision_score, n_boot: int = C.N_BOOTSTRAP,
                 seed: int = C.SEED) -> dict:
    """County-block bootstrap (resamples whole counties, so autocorrelated rows are kept together)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in scores}
    for idx in _cluster_indices(groups, rng, n_boot):
        if y[idx].min() == y[idx].max():
            continue
        for k, s in scores.items():
            draws[k].append(metric(y[idx], np.asarray(s)[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def paired_difference_ci(y, s_a, s_b, groups, metric=average_precision_score, n_boot: int = C.N_BOOTSTRAP,
                         seed: int = C.SEED) -> dict:
    """CI for metric(a) - metric(b) on the same county-block resamples. If the CI excludes 0 the gap is not noise.

    `point` is the plain difference on the full sample (the number to report next to the interval); `mean` is the average
    over the bootstrap resamples, which differs slightly because resampling counties changes the sample's composition.
    """
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in _cluster_indices(groups, rng, n_boot):
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx]))
    return {"point": float(metric(y, s_a) - metric(y, s_b)), "mean": float(np.mean(diffs)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def calibration_table(y, p, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Reliability table and expected calibration error (ECE)."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return pd.DataFrame(rows), float(ece)


def permutation_importance(ensemble: StackedEnsemble, X, y, n_repeats: int = 5, seed: int = C.SEED) -> pd.DataFrame:
    """Drop in PR-AUC when one feature is shuffled. Computed on VALIDATION data so the test set stays untouched."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, ensemble.predict_proba(X))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - average_precision_score(y, ensemble.predict_proba(Xp)))
        rows.append({"feature": col, "pr_auc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)
