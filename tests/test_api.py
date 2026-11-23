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


def test_root_redirects_to_the_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui"


def test_ui_logo_is_served_from_the_same_origin(client):
    """The header logo must come from this server (no external request) and be a real PNG."""
    page = client.get("/ui").text
    assert 'src="/logo.png"' in page and 'alt="Strathmore University"' in page
    r = client.get("/logo.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n" and len(r.content) > 1000


def test_ui_page_is_served_and_calls_the_api(client):
    r = client.get("/ui")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "County escalation outlook" in page
    for endpoint in ("/periods", "/predict/batch"):          # the page is a real client of the JSON API
        assert endpoint in page


def test_ui_makes_no_external_requests(client):
    """Self-contained: no CDN scripts, fonts or trackers, so it works offline and inside the container."""
    import re
    page = client.get("/ui").text
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", page)
    assert "<script src=" not in page and "<link " not in page


def test_ui_never_writes_api_text_as_html(client):
    """API values must be inserted as text (textContent), never as HTML, so a hostile value cannot inject markup."""
    page = client.get("/ui").text
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


def test_periods_describe_the_scorable_range(client):
    j = client.get("/periods").json()
    assert j["first"] == "2015-03" and j["last"] == "2024-12" and 0 < j["default_threshold"] < 1
    assert client.post("/predict", json={"county": "Nairobi", "year": 2015, "month": 3}).status_code == 200
    assert client.post("/predict", json={"county": "Nairobi", "year": 2015, "month": 2}).status_code == 404


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and "test_results" in r.json()
