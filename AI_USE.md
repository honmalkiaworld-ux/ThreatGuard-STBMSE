# Artificial intelligence Use Disclosure

Student: Esther Muthui (223932). Lecturers: Prof. Javier Serrano and Emmanuel Bundi.

This file discloses artificial intelligence assistance in this repository, as the DSA 8401 capstone checklist requires.
**Check it against the course's artificial intelligence-use policy template and edit it before you submit.** The course policy text was not
available when this was drafted, so the structure below is a reasonable default, not the official format.

## Tool

Claude (Anthropic), used through Claude Code in VS Code during November 2026.

## What the artificial intelligence did

| Area | Artificial intelligence contribution |
|---|---|
| Analysis of the data and proposal | Profiled `kenya_conflict_monthly.csv`, compared it with the proposal's stated requirements, and flagged leakage concerns. The student states that the dataset is synthetic, and the documents describe it that way |
| Code | Drafted and ran all Python under `src/`, `tests/` and `run_pipeline.py`, the `Dockerfile`, `docker-compose.yml`, the requirements files and the GitHub Actions workflow `.github/workflows/ci.yml`, modelled on the lab repository's `fraud_mlops` and `practicals-mlops` examples |
| Frontend | Designed and wrote the single-page browser user interface (`src/stbmse/static/index.html`), the `/ui` and `/periods` routes and their tests; checked the page in headless Edge at several widths and in dark mode |
| Modelling depth | Wrote the rolling-origin evaluation, the Optuna tuning study and the Shapley additive explanations module with their tests; ran them and wrote up the results, including the findings that cut against the proposal. Installed Optuna and Shapley additive explanations in the separate virtual environment. The tuning study was kept from changing the shipped model, so adopting its settings stays the student's decision |
| Machine learning operations additions | Wrote the validation-only quality gate, the MLflow tracking and registry module and the Evidently drift module; installed Docker Desktop, MLflow and Evidently on the student's machine (the Python packages in a separate virtual environment outside OneDrive) |
| Verification | Ran the pipeline in a clean environment, the 57 tests, a mutation check showing the leakage test catches planted leaks, a test that reproduces and guards against a continuous integration failure the artificial intelligence introduced (an optional package imported at start-up, which crashed the slim Docker build), a deliberately impossible gate (exit code 1, no model written, nothing registered), the continuous integration container steps locally, and live hypertext transfer protocol status calls against the application programming interface |
| Documents | Drafted `README.md`, this file and `reports/REPORT.md` from the actual outputs |
| Research | Searched the web for Armed Conflict Location and Event Data Project and Kenya National Bureau of Statistics access rules; those findings are in the chat history, not in the repo |

## What the artificial intelligence did not do, or could not do

- It did not obtain Armed Conflict Location and Event Data Project or Kenya National Bureau of Statistics data, and it did not create or log in to any account on the student's behalf.
- A dedicated linter (`actionlint`) was blocked by the sandbox, so the workflow file was only structurally validated before its first run. The workflow then ran successfully on GitHub after the repository was created and pushed (both the `ci` and `cd` jobs).
- The published container image was not independently inspected (the available token lacks the `read:packages` scope), and the quality gate's blocking behaviour was proven locally rather than on GitHub.
- It did not generate `kenya_conflict_monthly.csv` and does not know how it was generated. It is described as synthetic on the student's statement.
- It did not choose the research question, the proposal, the 5:1 cost ratio or the 0.05 precision-recall area under the curve gate margin. The proposal is the student's; the cost ratio and gate margin are assumptions written into `config.py` and `gate.py`.

## Student's declaration

I used artificial intelligence extensively in this capstone, and I am declaring that the work was produced almost entirely by artificial intelligence. Claude (Anthropic), used through Claude Code in VS Code in November 2026, wrote and ran all of the code in this repository (the data pipeline, model training and evaluation, hyperparameter tuning, Shapley additive explanations, the application programming interface, the browser interface, the Docker and continuous integration and continuous delivery configuration, and the tests). It also carried out the analysis and the web research on data sources, and drafted the README, the report and the slides from the results it produced. The synthetic dataset was generated with artificial intelligence assistance, using an artificial intelligence tool. My own role was to set the task, give instructions, decide the scope and what to publish (including correcting my student number and requiring that the data be described as synthetic), and submit the work. I did not independently write the code or collect real data, and the results demonstrate the method on synthetic data and say nothing about real conflict in Kenya.

## Names explained

- **Spatio-Temporal Bayesian-Markov Stacked Ensemble:** the method of this project. Its code package is named `stbmse`, which is just the initials of these words and appears only in file, folder and command names.
- **DSA 8401:** the course code of Applied Machine Learning at Strathmore University (Master of Science in Data Science and Analytics).
