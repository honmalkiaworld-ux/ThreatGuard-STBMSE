"""Leakage and data-integrity tests for the feature pipeline."""
import numpy as np
import pandas as pd
import pytest

from stbmse import config as C
from stbmse.data import DataError, ingest, load_raw, preprocess, validate
from stbmse.features import build_features, prepare_xy


@pytest.fixture(scope="module")
def df():
    d, _ = ingest(write_report=False)
    return d


def test_panel_is_complete(df):
    assert df.groupby("County").size().nunique() == 1
    assert df.County.nunique() == 47


def test_features_are_causal(df):
    """Corrupting every observation after month T0 must not change any feature at or before T0."""
    T0 = 60
    base = build_features(df)
    rng = np.random.default_rng(0)
    bad = df.copy()
    future = bad.t > T0
    bad.loc[future, C.TARGET] = rng.integers(0, 2, future.sum())
    for c in ("Social_Unrest_Score", "Economic_Volatility_Index", "Security_Force_Response_Time"):
        bad.loc[future, c] = rng.uniform(0, 100, future.sum())
    corrupted = build_features(bad)
    feats = [c for c in base.columns if c not in ("Sub_County",) and base[c].dtype != object]
    a = base[base.t <= T0].sort_values(["County", "t"]).reset_index(drop=True)[feats]
    b = corrupted[corrupted.t <= T0].sort_values(["County", "t"]).reset_index(drop=True)[feats]
    pd.testing.assert_frame_equal(a, b)


def test_label_is_next_month(df):
    prep = prepare_xy(build_features(df))
    d = df.set_index(["County", "t"])[C.TARGET]
    sample = prep.meta.sample(200, random_state=1)
    expected = [d[(r.County, r.t + 1)] for r in sample.itertuples()]
    assert list(prep.y.loc[sample.index]) == expected


def test_splits_are_disjoint_and_ordered(df):
    prep = prepare_xy(build_features(df))
    assert not (prep.train & prep.val).any() and not (prep.train & prep.test).any() and not (prep.val & prep.test).any()
    assert (prep.train | prep.val | prep.test).all()
    assert prep.meta.t[prep.train].max() < prep.meta.t[prep.val].min() <= prep.meta.t[prep.val].max() < prep.meta.t[prep.test].min()


def test_validation_rejects_bad_data():
    raw = load_raw()
    with pytest.raises(DataError):
        validate(raw.drop(columns=["County"]))
    dup = pd.concat([raw, raw.iloc[[0]]])
    with pytest.raises(DataError):
        validate(dup)
    bad = raw.copy()
    bad.loc[0, "Month"] = 13
    with pytest.raises(DataError):
        validate(bad)
