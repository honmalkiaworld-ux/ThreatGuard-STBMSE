"""Stage 5: score a county-month with the trained artifact. Used by the CLI and by the API."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import ingest
from .features import build_features, design_matrix


class ScoringError(ValueError):
    """The request cannot be scored (unknown county, month outside data, not enough history)."""


def _next_month(year: int, month: int) -> str:
    return f"{year + 1}-01" if month == 12 else f"{year}-{month + 1:02d}"


class Scorer:
    """Loads the serialised ensemble and the county-month history, and forecasts next month's escalation.

    Features are rebuilt from the raw history with the same code used in training, so training and serving
    cannot drift apart. To score newer months, append rows to the raw CSV (same schema) and restart.
    """

    def __init__(self, model_path: Path = C.MODEL_PATH, raw_path: Path = C.RAW_CSV):
        art = joblib.load(model_path)
        self.ensemble = art["ensemble"]
        self.columns: list[str] = art["feature_columns"]
        self.metrics: dict = art["metrics"]
        df, _ = ingest(raw_path, write_report=False)
        self.feat = build_features(df)
        self.X = design_matrix(self.feat, self.columns)
        self._key = {(c.lower(), int(y), int(m)): i for i, (c, y, m) in
                     enumerate(zip(self.feat.County, self.feat.Year, self.feat.Month))}
        self.counties = sorted(self.feat.County.unique())
        last = self.feat.sort_values("t").iloc[-1]
        self.data_through = f"{int(last.Year)}-{int(last.Month):02d}"

    def _score_rows(self, idx: list[int], threshold: float | None) -> list[dict]:
        thr = self.ensemble.threshold if threshold is None else float(threshold)
        Xs = self.X.iloc[idx]
        base = self.ensemble.base_proba(Xs)
        p = self.ensemble.predict_proba(Xs)
        out = []
        for j, i in enumerate(idx):
            r = self.feat.iloc[i]
            out.append({
                "county": r.County, "region": r.Region,
                "as_of": f"{int(r.Year)}-{int(r.Month):02d}",
                "forecast_for": _next_month(int(r.Year), int(r.Month)),
                "escalation_probability": round(float(p[j]), 4),
                "alert": bool(p[j] >= thr), "threshold": round(thr, 4),
                "base_model_probabilities": {k: round(float(v[j]), 4) for k, v in base.items()},
                "context": {
                    "escalating_this_month": bool(r.y_lag0),
                    "escalation_share_last_3_months": round(float(r.y_roll3), 3),
                    "escalation_share_last_12_months": round(float(r.y_roll12), 3),
                    "region_peers_escalating_this_month": round(float(r.spill_region), 3) if pd.notna(r.spill_region) else None,
                    "markov_p_escalate_next": round(float(r.eb_markov_next), 3),
                },
            })
        return out

    def _index(self, county: str, year: int, month: int) -> int:
        i = self._key.get((county.strip().lower(), int(year), int(month)))
        if i is None:
            if county.strip().lower() not in {c.lower() for c in self.counties}:
                raise ScoringError(f"Unknown county '{county}'")
            raise ScoringError(f"No data for {county} in {year}-{int(month):02d}; data runs through {self.data_through}")
        if pd.isna(self.feat.at[i, "y_lag2"]):
            raise ScoringError("Not enough history: at least 3 months of data are needed before the as-of month")
        return i

    def score(self, county: str, year: int, month: int, threshold: float | None = None) -> dict:
        return self._score_rows([self._index(county, year, month)], threshold)[0]

    def score_all(self, year: int, month: int, threshold: float | None = None, top_k: int | None = None) -> list[dict]:
        idx = [self._index(c, year, month) for c in self.counties]
        res = sorted(self._score_rows(idx, threshold), key=lambda r: r["escalation_probability"], reverse=True)
        return res[:top_k] if top_k else res


def main():
    ap = argparse.ArgumentParser(description="Forecast next-month escalation for a county (or the top-k counties).")
    ap.add_argument("--county"); ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True); ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()
    s = Scorer()
    res = s.score(a.county, a.year, a.month) if a.county else s.score_all(a.year, a.month, top_k=a.top_k)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
