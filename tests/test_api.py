"""API tests. Requires the trained artifact (run `python run_pipeline.py` first)."""
import pytest
from fastapi.testclient import TestClient

from stbmse import config as C

pytestmark = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="run `python run_pipeline.py` first")


@pytest.fixture(scope="module")
def client():
    from stbmse.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["counties"] == 47


def test_predict_ok(client):
    r = client.post("/predict", json={"county": "Nairobi", "year": 2024, "month": 12})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["escalation_probability"] <= 1
    assert body["forecast_for"] == "2025-01"
    assert isinstance(body["alert"], bool)


def test_predict_county_is_case_insensitive(client):
    a = client.post("/predict", json={"county": "nairobi", "year": 2023, "month": 6}).json()
    b = client.post("/predict", json={"county": "Nairobi", "year": 2023, "month": 6}).json()
    assert a["escalation_probability"] == b["escalation_probability"]


def test_threshold_override_changes_alert_only(client):
    lo = client.post("/predict", json={"county": "Kisumu", "year": 2024, "month": 12, "threshold": 0.01}).json()
    hi = client.post("/predict", json={"county": "Kisumu", "year": 2024, "month": 12, "threshold": 0.99}).json()
    assert lo["escalation_probability"] == hi["escalation_probability"]
    assert lo["alert"] is True and hi["alert"] is False


def test_unknown_county_404(client):
    assert client.post("/predict", json={"county": "Atlantis", "year": 2024, "month": 12}).status_code == 404


def test_out_of_range_month_422(client):
    assert client.post("/predict", json={"county": "Nairobi", "year": 2024, "month": 13}).status_code == 422


def test_beyond_data_404(client):
    assert client.post("/predict", json={"county": "Nairobi", "year": 2030, "month": 1}).status_code == 404


def test_batch_is_ranked(client):
    r = client.post("/predict/batch", json={"year": 2024, "month": 12, "top_k": 5})
    assert r.status_code == 200
    probs = [x["escalation_probability"] for x in r.json()["results"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and "test_results" in r.json()
