"""Evidently drift-report tests (skipped when Evidently is not installed)."""
import json

import pytest

pytest.importorskip("evidently")

from stbmse import monitor  # noqa: E402


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    d = tmp_path_factory.mktemp("drift")
    s = monitor.drift_report(html_path=d / "drift.html", json_path=d / "drift.json")
    s["_dir"] = d
    return s


def test_report_files_are_written(summary):
    html = summary["_dir"] / "drift.html"
    assert html.exists() and html.stat().st_size > 100_000
    saved = json.loads((summary["_dir"] / "drift.json").read_text())
    assert saved["columns_tested"] == summary["columns_tested"]


def test_summary_is_consistent(summary):
    assert summary["columns_tested"] >= 30
    assert summary["columns_drifted"] + len(summary["stable_columns"]) == summary["columns_tested"]
    assert 0 <= summary["share_drifted"] <= 1
    assert summary["reference_period"] == "train" and summary["current_period"] == "test"
    assert summary["n_reference_rows"] > 0 and summary["n_current_rows"] > 0


def test_known_prior_shift_is_detected(summary):
    """The escalation rate rises from about 20% to 43% between training and test: target drift must be flagged."""
    assert summary["target_drift"] is not None and summary["target_drift"]["drifted"] is True


def test_rejects_bad_period_arguments():
    with pytest.raises(ValueError):
        monitor.drift_report(reference="train", current="train", html_path=None, json_path=None)
    with pytest.raises(ValueError):
        monitor.drift_report(reference="train", current="future", html_path=None, json_path=None)
