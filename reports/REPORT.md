# ThreatGuard ST-BMSE: county-level conflict escalation forecasting

DSA 8401 Applied Machine Learning, capstone report. Student: Esther Muthui (223932).
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

The threshold is chosen on validation data by minimising

> threshold* = the threshold that minimises [ cost of a false negative × number of false negatives at that threshold + cost of a false positive × number of false positives at that threshold ], with cost of a false negative : cost of a false positive = 5 : 1

In words:

- **Threshold:** a county-month raises an alert when its predicted probability is at least the threshold.
- **False negatives:** real escalations that raised no alert (missed escalations).
- **Cost of a false negative:** the cost assigned to one missed escalation.
- **False positives:** alerts raised where no escalation followed (false alarms).
- **Cost of a false positive:** the cost assigned to one false alarm.
- **5 : 1:** one missed escalation is treated as five times as costly as one false alarm. This is an assumption, not an elicited value.

## 2. Data and the complications built into the evaluation

**Data.** `kenya_conflict_monthly.csv`: 5,640 rows = 47 counties x 120 months (2015-01 to 2024-12), 13 columns, no missing
values, no duplicate keys (`reports/data_quality.json`). Label `Conflict_Escalation` is 0/1; overall rate 27.7%.

**The data are synthetic.** They are simulated county-month records, not observed Armed Conflict Location and Event Data Project events or Kenya National Bureau of Statistics statistics. They
follow the proposal's design: 47 counties, monthly, January 2015 to December 2024, with 2022-2024 held out as the test
period, and they carry the indicators the proposal's approach needs (conflict type, social unrest, economic volatility,
diplomatic tension, security response time, sentiment) plus a binary next-step escalation label. The proposal's plan to
use Armed Conflict Location and Event Data Project events and Kenya National Bureau of Statistics indicators is the **next stage**: the pipeline validates and consumes a fixed county-month
schema, so real data aggregated to that schema can replace the synthetic file.

Because the data are synthetic, the results show how the method behaves on data with this structure. They are not
evidence about real conflict in Kenya, and the proposal's data-quality plan (missing events, reporting changes,
geocoding errors) is not exercised, since synthetic data are complete by construction.

**Complications that make the evaluation realistic (the capstone asks for these explicitly):**

1. **Strict temporal protocol.** Train on targets up to 2019, tune the meta-learner and threshold on 2020-2021, report once on 2022-2024. No shuffling, no random k-fold.
2. **Prior shift.** The escalation rate is 20.3% in training, 23.7% in validation and **42.9% in the test period**. A model tuned on the past is evaluated in a different regime, as it would be in deployment.
3. **A leakage trap in the columns.** Several features describe the same month as the label (section 6.1). A naive random split would report an excellent but meaningless score.
4. **No extra noise was injected.** The synthetic data are complete by construction and I added no missingness or label noise, so the complications above come from the evaluation design and the data's own shift, not from injected problems.

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
| Shapley additive explanations (exact per learner) | Course-standard attribution: exact linear Shapley additive explanations for logistic regression, tree Shapley additive explanations for the forest and XGBoost, checked for additivity and combined with the stacker's weights (section 6.4) |
| Rolling-origin evaluation | Repeats the whole procedure with the cut moved forward one year at a time (5 windows, 2020-2024), so the headline claim is not resting on one lucky or unlucky split (section 5.1) |
| Optuna tuning study | Searches the hyper-parameters of each learner with time-ordered folds that never touch the test period, and compares the result with the fixed settings like for like (section 5.2) |
| Population stability index drift check | Quantifies the shift between periods |
| MLflow tracking and model registry | Every run's parameters, validation and test metrics, tables and model are recorded; the three base learners appear as nested runs for side-by-side comparison. A passing model is registered as `stbmse-ensemble` with alias `production` |
| Quality gate (validation data only) | The best base learner must beat the persistence baseline by at least 0.05 precision-recall area under the curve on validation, scores must be finite, and the alert threshold must not be degenerate. A failed gate exits non-zero, writes no model and registers nothing. It uses validation only, so the test period stays untouched by any decision |
| GitHub Actions continuous integration and continuous delivery | Continuous integration on every push: train with the gate, audits, drift report, tests, Docker build, run the container and call it. CD (push to the container registry) only from the default branch and only if continuous integration passed |
| Evidently AI drift report | Course-standard drift detection on inputs, predictions and target between the training and test periods (section 6.2) |

**Considered and not used**

| Technique | Reason |
|---|---|
| Long short-term memory / gated recurrent unit / Transformer | 47 series of 120 points (about 5,600 rows). Too little data for sequence models to beat engineered lags and boosted trees; interpretability would also suffer |
| Support vector machine | Needs scaling and a separate probability-calibration step, with no expected gain over the three learners already used |
| LightGBM | Redundant with XGBoost on a dataset this small |
| K-Means / DBSCAN / principal component analysis / t-distributed stochastic neighbour embedding | The task is supervised and a label exists. Clustering counties could be useful for describing regions but does not improve a one-month-ahead forecast |
| Large language model fine-tuning, retrieval-augmented generation, agentic loops | There is no text data in the file. Adding an large language model would be a gimmick |

## 5. Results (test period 2022-2024, n = 1,692 county-months, 95% county-block bootstrap confidence intervals)

| Model | Precision-recall area under the curve | Receiver operating characteristic area under the curve | Recall | False-alarm rate | Brier |
|---|---|---|---|---|---|
| Persistence (next month = this month) | 0.498 (0.433-0.559) | 0.609 (0.571-0.644) | 0.55 | 0.34 | 0.383 |
| Logistic regression | 0.594 (0.507-0.679) | 0.672 (0.623-0.715) | 0.97 | 0.89 | 0.227 |
| Random forest | **0.619** (0.533-0.692) | **0.700** (0.644-0.743) | 0.93 | 0.75 | 0.221 |
| XGBoost | 0.587 (0.512-0.660) | 0.678 (0.623-0.719) | 0.89 | 0.72 | 0.231 |
| **Spatio-Temporal Bayesian-Markov Stacked Ensemble ensemble** | 0.608 (0.520-0.689) | 0.691 (0.639-0.734) | 0.94 | 0.81 | 0.221 |

Paired differences in precision-recall area under the curve (same resampled counties):

| Comparison | Difference | 95% confidence interval | Reading |
|---|---|---|---|
| Ensemble minus persistence | +0.109 | 0.062 to 0.161 | Real improvement |
| Ensemble minus random forest | -0.011 | -0.031 to +0.016 | No evidence the ensemble beats a single random forest |
| Ensemble minus logistic regression | +0.013 | 0.002 to 0.026 | Small gain, borderline |

**Interpretation, stated plainly**

- The model forecasts better than persistence, and the gap is outside bootstrap noise. That is the baseline the proposal names.
- Stacking did not help over the best single learner. The proposal's central claim, that the ensemble is better because it combines complementary learners, is **not supported** on this data.
- Receiver operating characteristic area under the curve of about 0.69 means moderate discrimination, not strong. This is a modest forecasting signal.
- Calibration is acceptable: expected calibration error 0.049. Probabilities are slightly too low in the lower bins because the test base rate is higher than the training one (`calibration_test.csv`).
- **The tuned alert is operationally poor.** The threshold (0.18) was chosen on 2020-2021, when 24% of months escalated. In the 43% test regime it flags most counties: recall 0.94 but a false-alarm rate of 0.81. On the latest month (2024-12), **46 of 47 counties alert** at 0.18 and 8 of 47 at 0.50. A practitioner would drown in alerts. The application programming interface therefore exposes a `threshold` override, and the right threshold should be set with practitioners, not by my assumed cost ratio.
- Most important inputs on the validation period (permutation importance): 3-month mean social-unrest score, 6- and 12-month escalation shares, cumulative county history, the Markov "enter escalation" probability, and the North Eastern region indicator. Shapley additive explanations (section 6.4) agrees on the top feature and only moderately on the rest.

### 5.1 Rolling-origin evaluation (`rolling_origin.csv`)

The single split above could be lucky or unlucky, and the escalation rate shifts a lot over time. So the whole procedure was repeated five times with the cut moved forward one year at a time. For test year *Y*, the base learners are fitted on forecasts up to *Y*-3, the stacker and threshold on *Y*-2 to *Y*-1, and only year *Y* is scored. Nothing from the test year or later is ever used, and a test enforces this. The 2022 window is exactly the shipped split, scored on 2022 only.

| Test year | Base rate | Ensemble precision-recall area under the curve | Persistence precision-recall area under the curve | Gain (95% confidence interval) | Recall | False-alarm rate |
|---|---|---|---|---|---|---|
| 2020 | 0.238 | 0.448 | 0.286 | +0.161 (0.070 to 0.243) | 0.92 | 0.77 |
| 2021 | 0.236 | 0.462 | 0.281 | +0.180 (0.076 to 0.261) | 0.89 | 0.67 |
| 2022 | 0.351 | 0.528 | 0.413 | +0.115 (0.056 to 0.169) | 0.86 | 0.66 |
| 2023 | 0.472 | 0.617 | 0.542 | +0.075 (0.012 to 0.151) | 0.98 | 0.90 |
| 2024 | 0.465 | 0.671 | 0.524 | +0.147 (0.086 to 0.201) | 1.00 | 0.96 |

The gain is the plain difference between the two precision-recall area under the curve columns; the interval is a county-block bootstrap percentile interval around it.

- **The headline claim holds in every window.** The ensemble beats persistence each year, the paired interval excludes zero all five times, and the mean gain is +0.136 (range +0.075 to +0.180). The earlier finding was not an accident of one split.
- **Precision-recall area under the curve rises over time mainly because the base rate rises** (0.24 to 0.47), and persistence rises with it. The gain over persistence is the fair comparison, not the raw level.
- **The alert problem is not a one-year accident.** The false-alarm rate is 0.66 to 0.96 in every window, and in 2024 the model flags every escalating month and almost every quiet one (recall 1.00, false-alarm rate 0.96).
- The ensemble is not compared with a single random forest here, so this does not revisit that finding.

### 5.2 Hyperparameter tuning study (`tuning_results.json`, `tuned_params.json`, `tuning_per_year.csv`)

The shipped settings were fixed by hand. This study asks whether a search improves on them. Each learner was tuned separately with Optuna (tree-structured Parzen estimator sampler, fixed seed, 25 trials). The objective was the mean precision-recall area under the curve over four expanding-window folds (forecast years 2018 to 2021, each fitted on earlier years only), so **the test period was never used for tuning**. The fixed settings were scored on the same folds. Only afterwards were the tuned and fixed ensembles evaluated once on 2022-2024.

| Learner | Cross-validated precision-recall area under the curve, fixed | Cross-validated precision-recall area under the curve, tuned | Gain |
|---|---|---|---|
| Logistic regression | 0.417 | 0.425 | +0.008 |
| Random forest | 0.422 | 0.435 | +0.014 |
| XGBoost | 0.406 | 0.426 | +0.019 |

| Test period 2022-2024 | Precision-recall area under the curve (95% confidence interval) | Receiver operating characteristic area under the curve | Recall | False-alarm rate | Threshold |
|---|---|---|---|---|---|
| Fixed (shipped) | 0.608 (0.520 to 0.689) | 0.691 | 0.94 | 0.81 | 0.18 |
| Tuned | 0.641 (0.544 to 0.717) | 0.710 | 0.97 | 0.86 | 0.15 |

Paired difference (tuned minus fixed) in test precision-recall area under the curve: **+0.033 (95% confidence interval 0.010 to 0.051)**, which excludes zero. By year it is +0.031 (2022), +0.056 (2023) and +0.003 (2024).

- **Tuning helps, modestly.** The cross-validated gains are small (+0.008 to +0.019). The gain on the test period is larger than those, and it is not uniform: it is near zero in 2024. One tuning run with 25 trials per learner cannot say more than that.
- **The tuned settings are much more regularised:** a far stronger penalty for logistic regression (C = 0.0013 against 0.5), shallower and more constrained trees (forest depth 4 with at least 26 samples per leaf), and a slower XGBoost with fewer trees. That points to a weak signal in which the hand-set models overfit a little.
- **It makes the alert problem slightly worse:** the false-alarm rate rises from 0.81 to 0.86 and the tuned threshold falls to 0.15.
- **The shipped model is unchanged.** Adopting the tuned settings would change every published number, so it is left as a decision for the author. The tuned settings are in `reports/tuned_params.json`, and `python run_pipeline.py --tune` reruns the study.

## 6. Audits

### 6.1 Leakage audit (`leakage_audit.csv`)

| Setting | Precision-recall area under the curve | Receiver operating characteristic area under the curve |
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

The hand-built **population stability index** check (`audit.py`) found 18 continuous features above 0.25, the usual "major shift" line. Population stability index values on rate-like and discrete features are inflated by coarse bins, so read them as a direction, not a magnitude.

Evidently reports that the inputs moved, not that the model got worse; accuracy on the test period is the measure of that (section 5). Practical implication: the model must be retrained and the threshold re-tuned on recent data.

### 6.3 Per-region performance (`subgroup_by_region.csv`)

At the tuned threshold the false-alarm burden is uneven. Northern and North Eastern regions have a false-alarm rate of 1.00 (every quiet month flagged), Coastal 0.94, Nyanza 0.90, versus 0.64-0.65 for Eastern and Western. Precision-recall area under the curve ranges from 0.47 (Coastal) to 0.76 (Central). Regions have only 3 to 11 counties, so these differences are indicative, not conclusive. Equal thresholds do not give equal treatment.

### 6.4 Shapley additive explanations (`shap_importance.csv`, `shap_summary.json`)

The ensemble blends three learners through a logistic stacker, so there is no closed-form Shapley additive explanations for the blend. Each learner is therefore explained exactly (exact linear Shapley additive explanations for logistic regression, tree Shapley additive explanations for the forest and XGBoost), on the 1,128 validation rows, and the three are combined. Because the units differ, each model's mean |Shapley additive explanations| is turned into a share of its own total and the shares are averaged with the stacker's absolute weights (logistic regression 49%, forest 32%, XGBoost 19%). This approximates the blend's attribution and is not an exact Shapley additive explanations of it. Every explainer was checked for additivity: the contributions plus the base value reproduce each model's own output to within 3e-6.

| Group of features | Share of attribution |
|---|---|
| Indicators (unrest, economy, tension, response time) | 31.2% |
| Escalation history | 25.8% |
| Empirical-Bayes Markov | 17.3% |
| Categorical (region, conflict type, sentiment) | 14.3% |
| Spill-over (region and national) | 9.3% |
| Calendar | 2.2% |

The single largest feature is the 3-month mean social-unrest score (9.6%), followed by cumulative county history and the two Markov transition probabilities (6.5% each).

- **The models lean on different things.** Logistic regression relies on the unrest score and the rolling escalation shares; the forest and XGBoost rely mostly on the Markov transition features.
- **Shapley additive explanations and permutation importance agree on the top feature but only moderately overall** (Spearman rank correlation 0.49 against the full permutation run; 0.41 against the faster run with fewer repeats that continuous integration uses). Correlated features share credit differently under the two methods, so neither ranking should be read as a precise ordering.
- **Attribution is not added predictive value.** The indicators carry 31% of the attribution, yet the leakage audit (section 6.1) found the history-only ensemble forecasts as well as the full one. Both are true: the indicators correlate with past escalation, so the models use them, but they do not add accuracy beyond it. Shapley additive explanations describes how the fitted model computes its output, not what causes escalation.

## 7. Gap between the proposal and what was delivered

| Proposal | Delivered |
|---|---|
| Armed Conflict Location and Event Data Project events + Kenya National Bureau of Statistics indicators | Synthetic county-month data built to the proposal's design (no real event counts, fatalities or Kenya National Bureau of Statistics series); real Armed Conflict Location and Event Data Project/Kenya National Bureau of Statistics data is the planned next stage |
| Spatial spill-over from neighbouring counties | Same-region share as a proxy (no adjacency data) |
| Data-quality challenges (missing events, reporting changes, geocoding errors) | Not present: synthetic data are complete by construction, so the validation stage is built and tested but real-world messiness is not exercised |
| Class imbalance because escalation is rare | Escalation is common here (28% overall, 43% in test) |
| Stacked ensemble is expected to outperform | Not shown; ties a single random forest |
| Human-in-the-loop prototype in the ThreatGuard AI dashboard | A prediction application programming interface with context fields for an analyst; not integrated with the dashboard |

To use real data: obtain Armed Conflict Location and Event Data Project and Kenya National Bureau of Statistics, aggregate to county-month, define "escalation" explicitly (the proposal does not), and map the result to the schema in `config.REQUIRED_COLUMNS` or extend `data.py`.

## 8. Deployment and real-world relevance

FastAPI service (`/predict`, `/predict/batch`, `/health`, `/model-info`) loaded from a serialised artifact. Features are
rebuilt at serving time with the same code as training, so training and serving cannot diverge. The application programming interface returns the
probability, the base-model probabilities and context an analyst needs (recent escalation share, region peers, Markov
probability). Tested with pytest (57 tests passing) and with live hypertext transfer protocol status calls.

**Frontend.** The application programming interface serves a single-page analyst view at `/ui` (`ui_screenshot.png` in `latex/figures/`): all 47 counties ranked by predicted probability, an as-of month selector, an alert-threshold slider, region and alerts-only filters, and a detail panel that shows what each of the three models says plus context an analyst can check. When the threshold makes nearly every county alert (as the tuned 0.18 does) the page says so and advises raising it. It is a real client of the JavaScript Object Notation endpoints (`/periods`, `/predict/batch`), makes no external requests, inserts application programming interface values as text only, and these properties are covered by tests. It was checked in headless Edge at desktop and phone widths in light and dark mode, and served from the Docker container; it has not been tested in Firefox or Safari, or with a screen reader.

**Experiment tracking and the quality gate.** Every training run is logged to MLflow (parameters, validation and test metrics, tables, the model, and one nested run per base learner). The gate runs on **validation data only** and a failed gate exits with code 1, writes no model artifact and registers nothing; the failed run is still logged and tagged `quality_gate=failed`. This was exercised both ways: the real data passes (best base learner beats persistence by +0.179 precision-recall area under the curve on validation against a required +0.05), and demanding an impossible +0.90 stopped the pipeline with exit code 1, left the model file untouched and left the registry holding only the passing version. The registered model loads back from `models:/stbmse-ensemble@production`.

**Continuous integration and continuous delivery.** `.github/workflows/ci.yml` runs on every push and pull request: install, train with the gate, audits and Evidently report, pytest, Docker build, then run the container and call `/health`, `/predict` and an invalid request (which must return 422). Publishing to the GitHub Container Registry happens only on the default branch and only if continuous integration passed. **The workflow ran green on GitHub on its first push** (run 44397520310): the `ci` job and the `cd` job (build and push to the container registry) both succeeded. Not verified: I could not list the published image independently (the available token lacks the `read:packages` scope), the workflow file was not checked with a dedicated linter such as `actionlint`, and the gate's blocking behaviour was proven locally rather than on GitHub.

**Container verification:** the Docker image was rebuilt from the slimmed serving requirements (no MLflow, Evidently or test tools in the image) and run exactly as the continuous integration job does (Docker Desktop, Windows 11). The quality gate runs during `docker build`, so a failing gate would fail the build. The container passed its health check, runs as a non-root user, and served correct predictions and a 422 for invalid input over hypertext transfer protocol status. The image is 1.3 gigabytes; part of that (about 200 megabytes) is a graphics processing unit-communication library that XGBoost's Linux wheel pulls in and this central processing unit-only service does not use. The model trained inside the Linux image differs slightly from the Windows-trained one used for the tables above (XGBoost is not bit-identical across platforms): ensemble precision-recall area under the curve 0.609 versus 0.608, but tuned threshold 0.16 versus 0.18 and false-alarm rate 0.89 versus 0.81. Point rankings and conclusions are unchanged; alert-level figures are not exactly reproducible across operating systems.

## 9. Limitations

- The data are synthetic: results demonstrate the method and say nothing about real conflict in Kenya.
- Synthetic data are clean and complete by construction, so raw-data problems (missing or duplicated events, geocoding errors) are not exercised; the evaluation's realism comes from the temporal protocol and the data's own prior shift.
- The shipped model uses hand-set hyper-parameters. A tuning study found a modest, uneven test-period gain (+0.033 precision-recall area under the curve) but the tuned settings were not adopted; the study used one search of 25 trials per learner.
- The rolling-origin evaluation has only five windows, and the 2022-2024 windows overlap the main test period.
- Shapley additive explanations for the blended model is an approximation (per-model shares averaged with the stacker's weights), and sequence models were argued out, not tested.
- The continuous integration and continuous delivery workflow ran once on GitHub; the published image was not independently inspected, the file was not linted, and the gate's blocking behaviour was proven locally, not on GitHub.
- The quality-gate margin (+0.05 precision-recall area under the curve over persistence) is a judgement, not derived from a requirement.
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

## Short forms and names explained

- **Spatio-Temporal Bayesian-Markov Stacked Ensemble:** the method of this project. Its code package is named `stbmse`, which is just the initials of these words and appears only in file, folder and command names.
- **DSA 8401:** the course code of Applied Machine Learning at Strathmore University (Master of Science in Data Science and Analytics).
- **Threshold:** the probability above which a county-month raises an alert.
- **False negative / false positive:** a missed escalation / a false alarm (listed in this order everywhere).
- **XGBoost:** a gradient-boosted decision-tree library, one of the three base learners.
- **MLflow:** an open-source tool that records experiment runs and registers models.
- **Optuna:** an open-source library for hyperparameter search.
- **Evidently:** an open-source library that produces data and prediction drift reports.
- **FastAPI:** a Python web framework used for the prediction service.
- **Docker:** software that packages the service and its dependencies into a container image.
- **GitHub Actions:** GitHub's automatic workflow runner, used here for continuous integration and continuous delivery.
- **Shapley additive explanations:** a method that splits each prediction into the contribution of each input feature.
- **Persistence baseline:** the simple forecast that next month's escalation equals this month's.
