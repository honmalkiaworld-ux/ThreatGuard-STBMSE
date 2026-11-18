# AI Use Disclosure

This file discloses AI assistance in this repository, as the DSA 8401 capstone checklist requires.
**Check it against the course's AI-use policy template and edit it before you submit.** The course policy text was not
available when this was drafted, so the structure below is a reasonable default, not the official format.

## Tool

Claude (Anthropic), used through Claude Code in VS Code during November 2026.

## What the AI did

| Area | AI contribution |
|---|---|
| Analysis of the data and proposal | Profiled `kenya_conflict_monthly.csv`, compared it with the proposal's stated requirements, and flagged leakage and provenance concerns |
| Code | Drafted and ran all Python under `src/`, `tests/` and `run_pipeline.py`, the `Dockerfile`, `docker-compose.yml`, the requirements files and the GitHub Actions workflow `.github/workflows/ci.yml`, modelled on the lab repository's `fraud_mlops` and `practicals-mlops` examples |
| MLOps additions | Wrote the validation-only quality gate, the MLflow tracking and registry module and the Evidently drift module; installed Docker Desktop, MLflow and Evidently on the student's machine (the Python packages in a separate virtual environment outside OneDrive) |
| Verification | Ran the pipeline in a clean environment, the 27 tests, a mutation check showing the leakage test catches planted leaks, a deliberately impossible gate (exit code 1, no model written, nothing registered), the CI container steps locally, and live HTTP calls against the API |
| Documents | Drafted `README.md`, this file and `reports/REPORT.md` from the actual outputs |
| Research | Searched the web for ACLED and KNBS access rules; those findings are in the chat history, not in the repo |

## What the AI did not do, or could not do

- It did not obtain ACLED or KNBS data, and it did not create or log in to any account on the student's behalf.
- It could not run the GitHub Actions workflow (it needs a GitHub repository), and a dedicated linter (`actionlint`) was blocked by the sandbox, so the workflow file has only been structurally validated. Its steps were reproduced locally instead.
- The container registry push (the CD job) has never been exercised.
- It could not verify where `kenya_conflict_monthly.csv` came from.
- It did not choose the research question, the proposal, the 5:1 cost ratio or the 0.05 PR-AUC gate margin. The proposal is the student's; the cost ratio and gate margin are assumptions written into `config.py` and `gate.py`.

## Student's declaration (to complete)

- [ ] I have read and understood all code in this repository and can explain every modelling decision.
- [ ] I re-ran `python run_pipeline.py` and `python -m pytest` myself and obtained the reported results.
- [ ] I have checked the claims in `reports/REPORT.md` against the generated files in `reports/`.
- [ ] I verified (or I state here that I could not verify) the origin of the dataset: ______________________
- [ ] Anything I changed from the AI's draft: ______________________
