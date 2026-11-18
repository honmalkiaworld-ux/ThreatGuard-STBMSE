"""Model service: FastAPI wrapper around the trained ST-BMSE artifact.

Run:  uvicorn stbmse.api:app --app-dir src --host 0.0.0.0 --port 8000      (docs at /docs)
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import __version__
from . import config as C
from .predict import Scorer, ScoringError

state: dict = {}
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not C.MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {C.MODEL_PATH}. Run `python run_pipeline.py` first.")
    state["scorer"] = Scorer()
    yield
    state.clear()


app = FastAPI(
    title="ThreatGuard ST-BMSE - county escalation forecast",
    version=__version__, lifespan=lifespan,
    description=("Forecasts the probability that a Kenyan county escalates NEXT month. Decision support for human "
                 "early-warning practitioners; not an automated decision system."),
)


class PredictRequest(BaseModel):
    county: str = Field(..., examples=["Nairobi"])
    year: int = Field(..., ge=2000, le=2100, examples=[2024])
    month: int = Field(..., ge=1, le=12, examples=[12])
    threshold: float | None = Field(None, gt=0, lt=1, description="Override the cost-tuned alert threshold.")


class BatchRequest(BaseModel):
    year: int = Field(..., ge=2000, le=2100, examples=[2024])
    month: int = Field(..., ge=1, le=12, examples=[12])
    top_k: int | None = Field(None, ge=1, le=47, description="Return only the k highest-risk counties.")
    threshold: float | None = Field(None, gt=0, lt=1)


def _scorer() -> Scorer:
    return state["scorer"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui():
    """The analyst view: a single static page that calls this API's JSON endpoints from the browser."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/periods")
def periods():
    """Months that can be scored, and the model's tuned alert threshold (used by the UI's controls)."""
    s = _scorer()
    return {"first": s.first_valid, "last": s.data_through, "default_threshold": s.ensemble.threshold}


@app.get("/health")
def health():
    s = _scorer()
    return {"status": "ok", "model_version": __version__, "data_through": s.data_through, "counties": len(s.counties)}


@app.get("/model-info")
def model_info():
    m = _scorer().metrics
    keep = ("version", "trained_at", "n_rows", "n_features", "escalation_rate", "threshold",
            "cost_ratio_fn_to_fp", "meta_weights_logit", "test_results", "ece_test")
    return {k: m[k] for k in keep if k in m}


@app.get("/counties")
def counties():
    return {"counties": _scorer().counties}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _scorer().score(req.county, req.year, req.month, req.threshold)
    except ScoringError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    try:
        res = _scorer().score_all(req.year, req.month, req.threshold, req.top_k)
    except ScoringError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"as_of": f"{req.year}-{req.month:02d}", "n": len(res), "alerts": sum(r["alert"] for r in res), "results": res}
