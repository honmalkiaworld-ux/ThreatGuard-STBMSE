"""Rolling-origin windows must never let a model see its own test year or anything later."""
import numpy as np
import pytest

from stbmse.rolling import TEST_YEARS, VAL_YEARS, window_masks, summarise


@pytest.fixture
def years():
    return np.repeat(np.arange(2015, 2025), 100)          # 100 forecasts per year, 2015..2024


@pytest.mark.parametrize("test_year", TEST_YEARS)
def test_windows_are_disjoint_and_ordered(years, test_year):
    tr, va, te = window_masks(years, test_year)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert years[tr].max() < years[va].min() <= years[va].max() < years[te].min()
    assert set(years[te]) == {test_year}                  # tests only the target year


@pytest.mark.parametrize("test_year", TEST_YEARS)
def test_nothing_from_the_test_year_or_later_is_used_for_fitting(years, test_year):
    tr, va, te = window_masks(years, test_year)
    assert years[tr | va].max() == test_year - 1


def test_validation_span_matches_the_setting(years):
    tr, va, te = window_masks(years, 2023, val_years=VAL_YEARS)
    assert sorted(set(years[va])) == [2021, 2022] and years[tr].max() == 2020


def test_main_protocol_is_one_of_the_windows(years):
    """The 2022 window equals the shipped split (train <= 2019, validate 2020-2021), tested on 2022 only."""
    tr, va, te = window_masks(years, 2022)
    assert years[tr].max() == 2019 and sorted(set(years[va])) == [2020, 2021]


def test_paired_difference_reports_the_plain_difference_not_the_bootstrap_mean():
    """A table that shows 0.448 and 0.286 must not list the gain as the bootstrap mean (it differs slightly)."""
    from sklearn.metrics import average_precision_score as ap
    from stbmse.modeling import paired_difference_ci
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    a, b = y * 0.4 + rng.random(400) * 0.6, rng.random(400)
    groups = np.repeat(np.arange(20), 20)
    d = paired_difference_ci(y, a, b, groups, ap, n_boot=60)
    assert d["point"] == pytest.approx(ap(y, a) - ap(y, b))
    assert d["ci95"][0] <= d["point"] <= d["ci95"][1] and "mean" in d


def test_summarise_counts_clear_wins():
    import pandas as pd
    out = pd.DataFrame({"gain_ci_low": [0.05, -0.01, 0.02], "pr_auc_gain_over_persistence": [0.1, 0.02, 0.06]})
    s = summarise(out)
    assert s["windows"] == 3 and s["windows_where_gain_ci_excludes_zero"] == 2
    assert s["min_pr_auc_gain"] == 0.02 and s["max_pr_auc_gain"] == 0.1
