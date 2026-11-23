"""Central configuration. Every path and modelling constant lives here so the pipeline is reproducible."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("STBMSE_ROOT", Path(__file__).resolve().parents[2]))
RAW_CSV = Path(os.environ.get("STBMSE_DATA", ROOT / "data" / "raw" / "kenya_conflict_monthly.csv"))
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

SEED = 42
TARGET = "Conflict_Escalation"

# Temporal split, defined on the year of the month being FORECAST (t+1). No shuffling anywhere.
TRAIN_END_YEAR = 2019      # base learners are fitted on targets up to 2019
VAL_END_YEAR = 2021        # 2020-2021: meta-learner, calibration and threshold; 2022-2024 is the untouched test set

# Cost-sensitive decision rule: a missed escalation costs FN_COST, a false alarm costs FP_COST.
# This ratio is an ASSUMPTION (not elicited from practitioners). Change it here to re-tune the alert threshold.
FN_COST = 5.0
FP_COST = 1.0

# Empirical-Bayes prior strength (pseudo-observations) for the Markov transition probabilities.
K_PRIOR = 10.0

N_BOOTSTRAP = 500          # county-block bootstrap replicates for confidence intervals

REQUIRED_COLUMNS = [
    "Year", "Month", "County", "Sub_County", "Region", "Conflict_Type", "Sentiments",
    "Social_Unrest_Score", "Economic_Volatility_Index", "Diplomatic_Tension_Level",
    "Security_Force_Response_Time", "Historical_Conflict_Presence", "Conflict_Escalation",
]
ALLOWED_TENSION = ["Low", "Medium", "High"]
ALLOWED_SENTIMENTS = ["Positive", "Neutral", "Negative"]
