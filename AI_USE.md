# AI Use Disclosure

This file discloses AI assistance in this repository, as the DSA 8401 capstone checklist requires.
**Check it against the course's AI-use policy template and edit it before you submit.** The course policy text was not
available when this was drafted, so the structure below is a reasonable default, not the official format.

## Tool

Claude (Anthropic), used through Claude Code in VS Code during November 2026.

## What the AI did

| Area | AI contribution |
|---|---|
| Analysis of the data and proposal | Profiled `kenya_conflict_monthly.csv`, compared it with the proposal's stated requirements, and flagged leakage concerns. The student states that the dataset is synthetic, and the documents describe it that way |
| Code | Drafted and ran all Python under `src/`, `tests/` and `run_pipeline.py`, the `Dockerfile`, `docker-compose.yml`, the requirements files and the GitHub Actions workflow `.github/workflows/ci.yml`, modelled on the lab repository's `fraud_mlops` and `practicals-mlops` examples |
| Frontend | Designed and wrote the single-page browser UI (`src/stbmse/static/index.html`), the `/ui` and `/periods` routes and their tests; checked the page in headless Edge at several widths and in dark mode |
| Modelling depth | Wrote the rolling-origin evaluation, the Optuna tuning study and the SHAP module with their tests; ran them and wrote up the results, including the findings that cut against the proposal. Installed Optuna and SHAP in the separate virtual environment. The tuning study was kept from changing the shipped model, so adopting its settings stays the student's decision |
| MLOps additions | Wrote the validation-only quality gate, the MLflow tracking and registry module and the Evidently drift module; installed Docker Desktop, MLflow and Evidently on the student's machine (the Python packages in a separate virtual environment outside OneDrive) |
| Verification | Ran the pipeline in a clean environment, the 56 tests, a mutation check showing the leakage test catches planted leaks, a test that reproduces and guards against a CI failure the AI introduced (an optional package imported at start-up, which crashed the slim Docker build), a deliberately impossible gate (exit code 1, no model written, nothing registered), the CI container steps locally, and live HTTP calls against the API |
| Documents | Drafted `README.md`, this file and `reports/REPORT.md` from the actual outputs |
| Research | Searched the web for ACLED and KNBS access rules; those findings are in the chat history, not in the repo |

## What the AI did not do, or could not do

- It did not obtain ACLED or KNBS data, and it did not create or log in to any account on the student's behalf.
- A dedicated linter (`actionlint`) was blocked by the sandbox, so the workflow file was only structurally validated before its first run. The workflow then ran successfully on GitHub after the repository was created and pushed (both the `ci` and `cd` jobs).
- The published container image was not independently inspected (the available token lacks the `read:packages` scope), and the quality gate's blocking behaviour was proven locally rather than on GitHub.
- It did not generate `kenya_conflict_monthly.csv` and does not know how it was generated. It is described as synthetic on the student's statement.
- It did not choose the research question, the proposal, the 5:1 cost ratio or the 0.05 PR-AUC gate margin. The proposal is the student's; the cost ratio and gate margin are assumptions written into `config.py` and `gate.py`.

## Student's declaration (to complete)

- [ ] I have read and understood all code in this repository and can explain every modelling decision.
- [ ] I re-ran `python run_pipeline.py` and `python -m pytest` myself and obtained the reported results.
- [ ] I have checked the claims in `reports/REPORT.md` against the generated files in `reports/`.
- [ ] I state how the synthetic dataset was produced (who generated it, with what method or tool): ______________________
- [ ] Anything I changed from the AI's draft: ______________________
