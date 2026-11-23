"""Stage 1: ingest the raw county-month CSV, validate it, and write a data-quality report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataError(ValueError):
    """Raised when the raw data violates a hard schema/integrity rule."""


def load_raw(path: Path = C.RAW_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Return a quality report. Raises DataError on hard failures; soft issues go in report['warnings']."""
    errors, warnings = [], []
    missing_cols = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["County"] = df["County"].astype(str).str.strip()

    n_dups = int(df.duplicated(["County", "Year", "Month"]).sum())
    if n_dups:
        errors.append(f"{n_dups} duplicated (County, Year, Month) keys")
    if not df["Month"].between(1, 12).all():
        errors.append("Month outside 1-12")
    for col in (C.TARGET, "Historical_Conflict_Presence"):
        if not df[col].isin([0, 1]).all():
            errors.append(f"{col} contains values other than 0/1")
    if (df["Security_Force_Response_Time"] <= 0).any():
        errors.append("Non-positive Security_Force_Response_Time")
    if not df["Social_Unrest_Score"].between(0, 100).all():
        warnings.append("Social_Unrest_Score outside the 0-100 range")
    bad_tension = sorted(set(df["Diplomatic_Tension_Level"].dropna()) - set(C.ALLOWED_TENSION))
    bad_sent = sorted(set(df["Sentiments"].dropna()) - set(C.ALLOWED_SENTIMENTS))
    if bad_tension:
        errors.append(f"Unknown Diplomatic_Tension_Level values: {bad_tension}")
    if bad_sent:
        errors.append(f"Unknown Sentiments values: {bad_sent}")

    miss = df[C.REQUIRED_COLUMNS].isna().sum()
    miss = {k: int(v) for k, v in miss.items() if v}
    if miss:
        warnings.append(f"Missing values: {miss}")

    n_counties = int(df["County"].nunique())
    n_months = int(df.groupby(["Year", "Month"]).ngroups)
    expected = n_counties * n_months
    present = int(df.drop_duplicates(["County", "Year", "Month"]).shape[0])
    if present < expected:
        warnings.append(f"Panel incomplete: {present} of {expected} county-month cells present")

    by_year = df.groupby("Year")[C.TARGET].mean().round(4)
    report = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "counties": n_counties, "year_months": n_months,
        "period": f"{int(df.Year.min())}-{int(df.Month[df.Year == df.Year.min()].min()):02d} to "
                  f"{int(df.Year.max())}-{int(df.Month[df.Year == df.Year.max()].max()):02d}",
        "panel_cells_expected": expected, "panel_cells_present": present, "duplicate_keys": n_dups,
        "missing_values": miss,
        "escalation_rate_overall": round(float(df[C.TARGET].mean()), 4),
        "escalation_rate_by_year": {int(k): float(v) for k, v in by_year.items()},
        "errors": errors, "warnings": warnings,
        "provenance_note": ("Synthetic (simulated) county-month data built to the proposal's design; not observed "
                            "ACLED events or KNBS statistics. Real data aggregated to this schema can replace it."),
    }
    if errors:
        raise DataError("; ".join(errors))
    return report


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes, ordering and the integer month index `t` used by every later stage."""
    df = df.copy()
    df["County"] = df["County"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.strip()
    df = df.sort_values(["County", "Year", "Month"]).reset_index(drop=True)
    y0 = int(df["Year"].min())
    df["t"] = (df["Year"] - y0) * 12 + (df["Month"] - 1)
    df["tension_num"] = df["Diplomatic_Tension_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(float)
    return df


def ingest(path: Path = C.RAW_CSV, write_report: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = load_raw(path)
    report = validate(raw)
    df = preprocess(raw)
    if write_report:
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (C.REPORT_DIR / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return df, report
