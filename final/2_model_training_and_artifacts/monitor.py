"""Stage 4d: Evidently data-drift report (course Week 13: "detect degradation; data and concept drift").

Question answered: has the data the model is being applied to moved away from the data it learned from?
  reference = rows whose forecast month falls in the TRAINING period (<= 2019)
  current   = rows whose forecast month falls in the TEST period (2022-2024)
Covered: input features, the model's predicted probability (prediction drift) and the next-month label (target drift).

Evidently's default methods are chosen from the sample size and column type (Wasserstein distance for numbers,
Jensen-Shannon distance for categories, threshold 0.1). A column is "drifted" when its Evidently test fails.
This complements the hand-built PSI check in audit.py; the two should broadly agree.

Evidently is an optional dependency: pip install evidently (it is in requirements.txt, not in requirements-api.txt).
Run on its own:  python -m stbmse.monitor
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import ingest
from .features import CATEGORICAL, build_features, numeric_feature_names, prepare_xy

# numeric-coded columns that are really binary / ordinal states: treated as categorical so the right test is used
CATEGORICAL_CODED = ["Historical_Conflict_Presence", "tension_num", "y_lag0", "y_lag1", "y_lag2"]
HTML_PATH = C.REPORT_DIR / "evidently_drift_report.html"
JSON_PATH = C.REPORT_DIR / "evidently_drift_summary.json"


def available() -> bool:
    try:
        import evidently  # noqa: F401
    except ImportError:
        return False
    return True


def monitoring_frame(with_predictions: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """One row per (county, origin month): features known at t, optional model score, and the next-month label."""
    df, _ = ingest(write_report=False)
    feat = build_features(df)
    prep = prepare_xy(feat)
    raw = feat[["County", "t"] + CATEGORICAL].copy()
    frame = prep.meta.merge(raw, on=["County", "t"], how="left").reset_index(drop=True)
    num = prep.X[[c for c in numeric_feature_names()]].reset_index(drop=True)
    frame = pd.concat([frame[["County", "t"] + CATEGORICAL], num], axis=1)
    frame["target_next"] = prep.y.reset_index(drop=True).astype(int)
    if with_predictions and C.MODEL_PATH.exists():
        import joblib
        ens = joblib.load(C.MODEL_PATH)["ensemble"]
        frame["prediction"] = ens.predict_proba(prep.X)
    return frame, prep.train, prep.val, prep.test


def _parse_tests(snapshot_dict: dict) -> list[dict]:
    cols = []
    for t in snapshot_dict.get("tests", []):
        m = re.match(r"Value Drift for column (.+)", t.get("name", ""))
        if not m:
            continue
        d = t.get("description", "")
        num = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"                  # a number, never the sentence's trailing full stop
        score = re.search(rf"Drift score is {num}", d)
        method = re.search(rf"method is (.+?)\. The drift threshold is {num}", d)
        cols.append({
            "column": m.group(1), "drifted": str(t.get("status", "")).endswith("FAIL"),
            "score": float(score.group(1)) if score else None,
            "method": method.group(1) if method else None,
            "threshold": float(method.group(2)) if method else None,
        })
    return cols


def drift_report(reference: str = "train", current: str = "test", html_path: Path | None = HTML_PATH,
                 json_path: Path | None = JSON_PATH, drift_share: float = 0.5) -> dict:
    """Run Evidently's DataDriftPreset between two periods. Writes an HTML report and a JSON summary."""
    if not available():
        raise ImportError("Evidently is not installed. Run: pip install evidently")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    frame, tr, va, te = monitoring_frame()
    masks = {"train": tr, "validation": va, "test": te}
    if reference not in masks or current not in masks or reference == current:
        raise ValueError("reference/current must be two different periods from: train, validation, test")
    body = frame.drop(columns=["County", "t"])
    cat = CATEGORICAL + CATEGORICAL_CODED + ["target_next"]
    num = [c for c in body.columns if c not in cat]
    dd = DataDefinition(numerical_columns=num, categorical_columns=[c for c in cat if c in body.columns])
    ref_ds = Dataset.from_pandas(body[masks[reference]].reset_index(drop=True), data_definition=dd)
    cur_ds = Dataset.from_pandas(body[masks[current]].reset_index(drop=True), data_definition=dd)

    snapshot = Report([DataDriftPreset(drift_share=drift_share)], include_tests=True).run(cur_ds, ref_ds)
    d = snapshot.dict()
    cols = _parse_tests(d)
    drifted = [c for c in cols if c["drifted"]]
    special = {c["column"]: c for c in cols if c["column"] in ("prediction", "target_next")}
    summary = {
        "reference_period": reference, "current_period": current,
        "n_reference_rows": int(masks[reference].sum()), "n_current_rows": int(masks[current].sum()),
        "columns_tested": len(cols), "columns_drifted": len(drifted),
        "share_drifted": round(len(drifted) / max(len(cols), 1), 4),
        "dataset_drift_threshold_share": drift_share,
        "dataset_drift_detected": bool(len(drifted) / max(len(cols), 1) >= drift_share),
        "prediction_drift": special.get("prediction"), "target_drift": special.get("target_next"),
        "drifted_columns": sorted(drifted, key=lambda c: -(c["score"] or 0)),
        "stable_columns": sorted(c["column"] for c in cols if not c["drifted"]),
        "note": ("Evidently flags columns whose distribution moved between the two periods. It says nothing about "
                 "whether the model's accuracy fell; that needs labels for the new period (here they exist, see results_test.csv)."),
    }
    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = drift_report()
    print(f"{s['columns_drifted']} of {s['columns_tested']} columns drifted ({s['share_drifted']:.0%}); "
          f"dataset drift detected: {s['dataset_drift_detected']}")
    print("prediction drift:", s["prediction_drift"])
    print("target drift:", s["target_drift"])
    print(f"HTML report: {HTML_PATH}")
