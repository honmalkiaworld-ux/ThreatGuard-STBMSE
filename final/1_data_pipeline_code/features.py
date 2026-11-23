"""Stage 2: leakage-safe feature engineering.

Every feature of row (county, t) is computed ONLY from data observed at or before month t.
The label is the NEXT month's escalation, y[t+1], and is attached separately in `prepare_xy`.
tests/test_features.py enforces this by corrupting the future and checking that features do not change.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

NUMERIC_BASE = ["Social_Unrest_Score", "Economic_Volatility_Index", "Security_Force_Response_Time"]
CATEGORICAL = ["Region", "Conflict_Type", "Sentiments"]


def build_features(df: pd.DataFrame, k_prior: float = C.K_PRIOR) -> pd.DataFrame:
    """Add causal features to a preprocessed county-month frame (see data.preprocess)."""
    y = C.TARGET
    df = df.sort_values(["County", "t"]).reset_index(drop=True).copy()
    g = df.groupby("County")

    # temporal memory of the county
    df["y_lag0"] = df[y]
    df["y_lag1"] = g[y].shift(1)
    df["y_lag2"] = g[y].shift(2)
    for w in (3, 6, 12):
        df[f"y_roll{w}"] = g[y].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    df["cum_hist"] = g[y].transform(lambda s: s.expanding().mean())
    for c in NUMERIC_BASE:
        df[f"{c}_lag1"] = g[c].shift(1)
        df[f"{c}_roll3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # spatial spill-over proxy
    # No county adjacency file is available, so "neighbourhood" = the other counties of the same region.
    reg_sum = df.groupby(["Region", "t"])[y].transform("sum")
    reg_n = df.groupby(["Region", "t"])[y].transform("count")
    df["spill_region"] = (reg_sum - df[y]) / (reg_n - 1).replace(0, np.nan)
    df["spill_region_lag1"] = df.groupby("County")["spill_region"].shift(1)
    df["spill_national"] = df.groupby("t")[y].transform("mean")
    df["spill_national_lag1"] = df.groupby("County")["spill_national"].shift(1)

    # empirical-Bayes Markov transition probabilities
    # P(escalate next | current state), shrunk toward the pooled rate; counts include only transitions seen by t.
    df["prev"] = g[y].shift(1)
    tr = df.dropna(subset=["prev"])[["County", "t", "prev", y]].copy()
    tr["from"] = tr["prev"].astype(int)
    count_cols = []
    for s in (0, 1):
        tr[f"n{s}"] = (tr["from"] == s).astype(int)
        tr[f"n{s}_to1"] = ((tr["from"] == s) & (tr[y] == 1)).astype(int)
        count_cols += [f"n{s}", f"n{s}_to1"]
    cum_county = tr.groupby("County")[count_cols].cumsum().add_suffix("_c")
    cum_pool = tr.groupby("t")[count_cols].sum().cumsum()
    tr = tr.join(cum_county)
    for c in count_cols:
        tr[c + "_g"] = tr["t"].map(cum_pool[c])
    keep = ["County", "t"] + [c + "_c" for c in count_cols] + [c + "_g" for c in count_cols]
    df = df.merge(tr[keep], on=["County", "t"], how="left")
    for s in (0, 1):
        den_g = df[f"n{s}_g"]
        pooled = np.where(den_g > 0, df[f"n{s}_to1_g"] / den_g.where(den_g > 0, 1), 0.5)  # 0.5 = neutral prior before any data
        num_c = df[f"n{s}_to1_c"].fillna(0)
        den_c = df[f"n{s}_c"].fillna(0)
        df[f"eb_p_from{s}"] = (num_c + k_prior * pooled) / (den_c + k_prior)
    df["eb_markov_next"] = np.where(df[y] == 1, df["eb_p_from1"], df["eb_p_from0"])
    df["eb_p_stay"] = df["eb_p_from1"]
    df["eb_p_enter"] = df["eb_p_from0"]

    # calendar
    df["m_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["m_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df.drop(columns=["prev"] + [c for c in df.columns if c.startswith("n0") or c.startswith("n1")])


def numeric_feature_names() -> list[str]:
    lagged = [f"{c}_{s}" for c in NUMERIC_BASE for s in ("lag1", "roll3")]
    return (NUMERIC_BASE + ["Historical_Conflict_Presence", "tension_num",
                            "y_lag0", "y_lag1", "y_lag2", "y_roll3", "y_roll6", "y_roll12", "cum_hist"]
            + lagged
            + ["spill_region", "spill_region_lag1", "spill_national", "spill_national_lag1",
               "eb_markov_next", "eb_p_stay", "eb_p_enter", "m_sin", "m_cos"])


def design_matrix(feat: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric features + one-hot categoricals. Pass `columns` (from the trained artifact) to align exactly."""
    X = pd.concat([feat[numeric_feature_names()], pd.get_dummies(feat[CATEGORICAL], dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


@dataclass
class Prepared:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame         # County, Year, Month, t of the ORIGIN month (features known at t)
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    persistence: np.ndarray    # score of the naive "next month = this month" baseline
    target_year: np.ndarray    # calendar year of the month being forecast (t+1), used to cut rolling windows


def prepare_xy(feat: pd.DataFrame) -> Prepared:
    """Attach the next-month label, drop rows without enough history, and make the temporal split."""
    feat = feat.sort_values(["County", "t"]).reset_index(drop=True)
    target_next = feat.groupby("County")[C.TARGET].shift(-1)
    keep = target_next.notna() & feat["y_lag2"].notna()
    d = feat[keep].reset_index(drop=True)
    X = design_matrix(feat)[keep.values].reset_index(drop=True)
    y = target_next[keep].astype(int).reset_index(drop=True)
    y0 = int(feat["Year"].min())
    tgt_year = y0 + (d["t"] + 1) // 12                  # calendar year of the month being forecast
    return Prepared(
        X=X, y=y, meta=d[["County", "Year", "Month", "t"]],
        train=(tgt_year <= C.TRAIN_END_YEAR).values,
        val=((tgt_year > C.TRAIN_END_YEAR) & (tgt_year <= C.VAL_END_YEAR)).values,
        test=(tgt_year > C.VAL_END_YEAR).values,
        persistence=d["y_lag0"].values.astype(float),
        target_year=tgt_year.values.astype(int),
    )
