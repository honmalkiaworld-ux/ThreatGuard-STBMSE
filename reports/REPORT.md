# ST-BMSE: forecasting county-level conflict escalation in Kenya

DSA 8401 Applied Machine Learning, capstone report. Student: Esther Muthui (223923).
All numbers below are produced by `python run_pipeline.py` and are stored in `reports/` and `artifacts/metrics.json`.

## 1. Problem framing

**Task.** For each of Kenya's 47 counties, estimate the probability that the county **escalates in the next month**,
using only information known at the end of the current month. Output: a calibrated probability plus an alert flag.

**Why it matters.** Early-warning practice relies largely on qualitative field reports, which are slow to compile and
hard to compare across counties. A ranked, calibrated county list lets analysts decide where to look first. The system is
decision support for a human reviewer, not an automated decision-maker.

**Why a missed escalation is not the same as a false alarm.** A missed escalation can cost lives; a false alarm costs
analyst time. The alert threshold therefore minimises expected cost with a 5:1 missed-to-false-alarm ratio. That ratio
is an **assumption** of this project, not an elicited value.

## 2. Data and the complications built into the evaluation

**Data.** `kenya_conflict_monthly.csv`: 5,640 rows = 47 counties x 120 months (2015-01 to 2024-12), 13 columns, no missing
values, no duplicate keys (`reports/data_quality.json`). Label `Conflict_Escalation` is 0/1; overall rate 27.7%.

**Provenance caveat.** The proposal names ACLED and KNBS as sources. This file is not a raw export of either: ACLED has no
`Sentiments`, `Social_Unrest_Score` or `Security_Force_Response_Time` fields, each county has exactly one sub-county
("X Central"), and the file is unusually clean. Its origin could not be verified, so results demonstrate the method, not
real-world conflict dynamics.

**Complications that make the evaluation realistic (the capstone asks for these explicitly):**

1. **Strict temporal protocol.** Train on targets up to 2019, tune the meta-learner and threshold on 2020-2021, report once on 2022-2024. No shuffling, no random k-fold.
2. **Prior shift.** The escalation rate is 20.3% in training, 23.7% in validation and **42.9% in the test period**. A model tuned on the past is evaluated in a different regime, as it would be in deployment.
3. **A leakage trap in the columns.** Several features describe the same month as the label (section 6.1). A naive random split would report an excellent but meaningless score.
4. **No noise was injected.** The file is already complete; I did not fabricate missingness or label noise, so the complications above are structural, not simulated.

Note: the proposal expected escalation to be "relatively infrequent". At 28% overall and 43% in the test period it is not, so class imbalance is a minor issue here and the proposal's imbalance argument does not apply to this file.

## 3. Pipeline (scripted, reproducible)

`ingest + validate` -> `causal features` -> `temporal split` -> `fit base learners` -> `fit calibrated meta-learner and threshold` -> `evaluate` -> `audit` -> `serialise` -> `serve`.

- **Validation** (`data.py`) fails hard on missing columns, duplicate keys, impossible values and unknown categories.
- **Features** (43) use data up to month *t* only: escalation lags and 3/6/12-month rolling shares, rolling indicator means, same-region and national escalation shares (spill-over proxy), empirical-Bayes Markov transition probabilities, calendar terms and one-hot categories.
- **Empirical-Bayes Markov feature.** For each county, P(escalate next | currently escalating / not) is estimated from the transitions observed up to *t*, shrunk toward the pooled rate with a prior worth 10 pseudo-observations.
- **Leakage guard.** `tests/test_features.py` corrupts every observation after a cut-off month and asserts that no feature at or before the cut-off changes. I confirmed the test is sensitive by planting two deliberately leaky features; it flagged both and left the real features alone.

## 4. Technique choices, and what I left out

**Used, and why**

| Technique | Why it fits |
|---|---|
| Logistic regression | Interpretable baseline; the proposal asks for one. Gets the largest weight in the blend (0.48 on the logit scale) |
| Random forest | Captures non-linear interactions and region effects at low tuning cost |
| XGBoost | Strong on structured tabular data; handles feature interactions |
| Calibrated logistic stacker | Combines the three on the validation period and doubles as a Platt-style calibrator, so outputs are usable probabilities |
| Cost-sensitive threshold | Decision rule reflects unequal error costs |
| County-block bootstrap | Rows within a county are autocorrelated, so I resample whole counties to avoid over-confident intervals |
| Permutation importance (validation) | Model-agnostic importance without touching the test set |
| PSI drift check | Quantifies the shift between periods |
| MLflow tracking and model registry | Every run's parameters, validation and test metrics, tables and model are recorded; the three base learners appear as nested runs for side-by-side comparison. A passing model is registered as `stbmse-ensemble` with alias `production` |
| Quality gate (validation data only) | The best base learner must beat the persistence baseline by at least 0.05 PR-AUC on validation, scores must be finite, and the alert threshold must not be degenerate. A failed gate exits non-zero, writes no model and registers nothing. It uses validation only, so the test period stays untouched by any decision |
| GitHub Actions CI/CD | CI on every push: train with the gate, audits, drift report, tests, Docker build, run the container and call it. CD (push to the container registry) only from the default branch and only if CI passed |
| Evidently AI drift report | Course-standard drift detection on inputs, predictions and target between the training and test periods (section 6.2) |

**Considered and not used**

| Technique | Reason |
|---|---|
| LSTM / GRU / Transformer | 47 series of 120 points (about 5,600 rows). Too little data for sequence models to beat engineered lags and boosted trees; interpretability would also suffer |
| SVM | Needs scaling and a separate probability-calibration step, with no expected gain over the three learners already used |
| LightGBM | Redundant with XGBoost on a dataset this small |
| K-Means / DBSCAN / PCA / t-SNE | The task is supervised and a label exists. Clustering counties could be useful for describing regions but does not improve a one-month-ahead forecast |
| LLM fine-tuning, RAG, agentic loops | There is no text data in the file. Adding an LLM would be a gimmick |
| SHAP | Not installed in the environment; permutation importance was used instead. SHAP would be the natural next step for per-prediction explanations |
| Hyperparameter search | Not done. Hyperparameters were fixed, not tuned (no Optuna or grid search). This is a limitation, not a considered exclusion |

## 5. Results (test period 2022-2024, n = 1,692 county-months, 95% county-block bootstrap CIs)

| Model | PR-AUC | ROC-AUC | Recall | False-alarm rate | Brier |
|---|---|---|---|---|---|
| Persistence (next month = this month) | 0.498 (0.433-0.559) | 0.609 (0.571-0.644) | 0.55 | 0.34 | 0.383 |
| Logistic regression | 0.594 (0.507-0.679) | 0.672 (0.623-0.715) | 0.97 | 0.89 | 0.227 |
| Random forest | **0.619** (0.533-0.692) | **0.700** (0.644-0.743) | 0.93 | 0.75 | 0.221 |
| XGBoost | 0.587 (0.512-0.660) | 0.678 (0.623-0.719) | 0.89 | 0.72 | 0.231 |
| **ST-BMSE ensemble** | 0.608 (0.520-0.689) | 0.691 (0.639-0.734) | 0.94 | 0.81 | 0.221 |

Paired differences in PR-AUC (same resampled counties):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| Ensemble minus persistence | +0.109 | 0.062 to 0.161 | Real improvement |
| Ensemble minus random forest | -0.011 | -0.031 to +0.016 | No evidence the ensemble beats a single random forest |
| Ensemble minus logistic regression | +0.013 | 0.002 to 0.026 | Small gain, borderline |

**Interpretation, stated plainly**

- The model forecasts better than persistence, and the gap is outside bootstrap noise. That is the baseline the proposal names.
- Stacking did not help over the best single learner. The proposal's central claim, that the ensemble is better because it combines complementary learners, is **not supported** on this data.
- ROC-AUC of about 0.69 means moderate discrimination, not strong. This is a modest forecasting signal.
- Calibration is acceptable: expected calibration error 0.049. Probabilities are slightly too low in the lower bins because the test base rate is higher than the training one (`calibration_test.csv`).
- **The tuned alert is operationally poor.** The threshold (0.18) was chosen on 2020-2021, when 24% of months escalated. In the 43% test regime it flags most counties: recall 0.94 but a false-alarm rate of 0.81. On the latest month (2024-12), **46 of 47 counties alert** at 0.18 and 8 of 47 at 0.50. A practitioner would drown in alerts. The API therefore exposes a `threshold` override, and the right threshold should be set with practitioners, not by my assumed cost ratio.
- Most important inputs on the validation period (permutation importance): 3-month mean social-unrest score, 6- and 12-month escalation shares, cumulative county history, the Markov "enter escalation" probability, and the North Eastern region indicator.

## 6. Audits

### 6.1 Leakage audit (`leakage_audit.csv`)

| Setting | PR-AUC | ROC-AUC |
|---|---|---|
| Nowcast: this month's indicators -> **this month's** label | 0.975 | 0.982 |
| Forecast with this month's indicators only (no history) | 0.560 | 0.644 |
| Forecast, full ensemble | 0.608 | 0.691 |
| Forecast, ensemble without `Sentiments` | 0.608 | 0.691 |
| Forecast, escalation history only | **0.623** | 0.693 |

- Same-month indicators almost determine the same-month label. `Sentiments` alone: 0% escalation for Positive and Neutral rows, 83% for Negative rows. A random split on this file would report a near-perfect model.
- For the honest next-month task the indicator columns add **nothing measurable** beyond escalation history: the history-only ensemble scores as well as, or slightly higher than, the full one. The socio-economic indicators are therefore not demonstrated to help forecasting.

### 6.2 Drift (`evidently_drift_report.html`, `evidently_drift_summary.json`, `drift_psi_features.csv`)

Escalation rate rises 20% -> 24% -> 43% across train, validation and test. Two independent checks agree on the direction.

**Evidently AI** compares the training period (2,679 rows) with the test period (1,692 rows) using its default tests (normed Wasserstein distance for numbers and Jensen-Shannon distance for categories, threshold 0.1):

- **27 of 32 columns (84%) drifted**; dataset-level drift is detected.
- **Prediction drift:** the model's own predicted probability drifted strongly (score 1.31 against a 0.1 threshold), and **target drift** was flagged too (0.17).
- Largest input shifts: national and regional spill-over shares (2.65, 1.08), the social-unrest score and its lags (1.2 to 1.7).
- **Stable:** `Region`, `Security_Force_Response_Time` and its lag, and the calendar terms. The calendar terms act as a built-in sanity check: month-of-year should not drift, and it does not.

The hand-built **PSI** check (`audit.py`) found 18 continuous features above 0.25, the usual "major shift" line. PSI values on rate-like and discrete features are inflated by coarse bins, so read them as a direction, not a magnitude.

Evidently reports that the inputs moved, not that the model got worse; accuracy on the test period is the measure of that (section 5). Practical implication: the model must be retrained and the threshold re-tuned on recent data.

### 6.3 Per-region performance (`subgroup_by_region.csv`)

At the tuned threshold the false-alarm burden is uneven. Northern and North Eastern regions have a false-alarm rate of 1.00 (every quiet month flagged), Coastal 0.94, Nyanza 0.90, versus 0.64-0.65 for Eastern and Western. PR-AUC ranges from 0.47 (Coastal) to 0.76 (Central). Regions have only 3 to 11 counties, so these differences are indicative, not conclusive. Equal thresholds do not give equal treatment.

## 7. Gap between the proposal and what was delivered

| Proposal | Delivered |
|---|---|
| ACLED events + KNBS indicators | A pre-compiled county-month file of unverified origin; no event counts, fatalities or KNBS series |
| Spatial spill-over from neighbouring counties | Same-region share as a proxy (no adjacency data) |
| Data-quality challenges (missing events, reporting changes, geocoding errors) | Not present in this file, so not addressed |
| Class imbalance because escalation is rare | Escalation is common here (28% overall, 43% in test) |
| Stacked ensemble is expected to outperform | Not shown; ties a single random forest |
| Human-in-the-loop prototype in the ThreatGuard AI dashboard | A prediction API with context fields for an analyst; not integrated with the dashboard |

To use real data: obtain ACLED and KNBS, aggregate to county-month, define "escalation" explicitly (the proposal does not), and map the result to the schema in `config.REQUIRED_COLUMNS` or extend `data.py`.

## 8. Deployment and real-world relevance

FastAPI service (`/predict`, `/predict/batch`, `/health`, `/model-info`) loaded from a serialised artifact. Features are
rebuilt at serving time with the same code as training, so training and serving cannot diverge. The API returns the
probability, the base-model probabilities and context an analyst needs (recent escalation share, region peers, Markov
probability). Tested with pytest (27 tests passing) and with live HTTP calls.

**Experiment tracking and the quality gate.** Every training run is logged to MLflow (parameters, validation and test metrics, tables, the model, and one nested run per base learner). The gate runs on **validation data only** and a failed gate exits with code 1, writes no model artifact and registers nothing; the failed run is still logged and tagged `quality_gate=failed`. This was exercised both ways: the real data passes (best base learner beats persistence by +0.179 PR-AUC on validation against a required +0.05), and demanding an impossible +0.90 stopped the pipeline with exit code 1, left the model file untouched and left the registry holding only the passing version. The registered model loads back from `models:/stbmse-ensemble@production`.

**CI/CD.** `.github/workflows/ci.yml` runs on every push and pull request: install, train with the gate, audits and Evidently report, pytest, Docker build, then run the container and call `/health`, `/predict` and an invalid request (which must return 422). Publishing to the GitHub Container Registry happens only on the default branch and only if CI passed. **The workflow ran green on GitHub on its first push** (run 44397520310): the `ci` job and the `cd` job (build and push to the container registry) both succeeded. Not verified: I could not list the published image independently (the available token lacks the `read:packages` scope), the workflow file was not checked with a dedicated linter such as `actionlint`, and the gate's blocking behaviour was proven locally rather than on GitHub.

**Container verification:** the Docker image was rebuilt from the slimmed serving requirements (no MLflow, Evidently or test tools in the image) and run exactly as the CI job does (Docker Desktop, Windows 11). The quality gate runs during `docker build`, so a failing gate would fail the build. The container passed its health check, runs as a non-root user, and served correct predictions and a 422 for invalid input over HTTP. The image is 1.3 GB; part of that (about 200 MB) is a GPU-communication library that XGBoost's Linux wheel pulls in and this CPU-only service does not use. The model trained inside the Linux image differs slightly from the Windows-trained one used for the tables above (XGBoost is not bit-identical across platforms): ensemble PR-AUC 0.609 vs 0.608, but tuned threshold 0.16 vs 0.18 and false-alarm rate 0.89 vs 0.81. Point rankings and conclusions are unchanged; alert-level figures are not exactly reproducible across operating systems.

## 9. Limitations

- Data origin unverified; results are a method demonstration.
- One train/validation/test split; no rolling-origin evaluation.
- Hyperparameters were fixed and not tuned.
- The CI/CD workflow ran once on GitHub; the published image was not independently inspected, the file was not linted, and the gate's blocking behaviour was proven locally, not on GitHub.
- The quality-gate margin (+0.05 PR-AUC over persistence) is a judgement, not derived from a requirement.
- The 5:1 cost ratio is assumed.
- The alert threshold does not transfer across regimes; recalibrate on recent data.
- Spill-over is a regional proxy, not geographic adjacency.
- No causal claim: importance scores show predictive association only.

## 10. Reproduce

```
python -m pip install -r requirements.txt
python run_pipeline.py                      # gate + MLflow + Evidently included; exits 1 if the gate fails
python -m pytest -q
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse runs and the model registry
```

Open `reports/evidently_drift_report.html` in a browser for the Evidently report.
