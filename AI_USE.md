# Artificial intelligence use disclosure

**Student:** Esther Muthui (223932) | **Course:** DSA 8401 Applied Machine Learning, Strathmore University
**Lecturers:** Prof. Javier Serrano and Emmanuel Bundi | **Live service:** https://threatguard-stbmse.onrender.com

## In short

- **Tool:** Claude (Anthropic), used through Claude Code in VS Code in November 2026.
- **How much:** extensively. The code, tests, configuration, analysis and written documents were produced almost entirely by the artificial intelligence, from my instructions.
- **What I did:** set the task, gave instructions, decided the scope and what to publish, corrected details (my student number, describing the data as synthetic), published the repository and deployed the live service, and submitted the work.
- **What this means for the results:** they demonstrate a method on **synthetic** data. They say nothing about real conflict in Kenya.
- **Where the credit is recorded:** commits made with the artificial intelligence carry a `Co-Authored-By: Claude` line, so the history matches this file.

This file is the disclosure the DSA 8401 capstone checklist asks for. The course's own disclosure template was not available when it was drafted, so the structure is a reasonable default, not the official format.

## My declaration

I used artificial intelligence extensively in this capstone, and I am declaring that the work was produced almost entirely by artificial intelligence. Claude (Anthropic), used through Claude Code in VS Code in November 2026, wrote and ran all of the code in this repository (the data pipeline, model training and evaluation, hyperparameter tuning, Shapley additive explanations, the application programming interface, the browser interface, the Docker and continuous integration and continuous delivery configuration, and the tests). It also carried out the analysis and the web research on data sources, and drafted the README, the report and the slides from the results it produced. The synthetic dataset was generated with artificial intelligence assistance, using an artificial intelligence tool. My own role was to set the task, give instructions, decide the scope and what to publish (including correcting my student number and requiring that the data be described as synthetic), and submit the work. I did not independently write the code or collect real data, and the results demonstrate the method on synthetic data and say nothing about real conflict in Kenya.

## What the artificial intelligence did

| Area | Contribution |
|---|---|
| Analysis of the data and proposal | Profiled `kenya_conflict_monthly.csv`, compared it with the proposal's stated requirements, and flagged leakage concerns. The student states that the dataset is synthetic, and the documents describe it that way |
| Code | Drafted and ran all Python under `src/`, `tests/` and `run_pipeline.py`, the `Dockerfile`, `docker-compose.yml`, the requirements files and the GitHub Actions workflow `.github/workflows/ci.yml`, modelled on the lab repository's `fraud_mlops` and `practicals-mlops` examples |
| Frontend | Designed and wrote the single-page browser user interface (`src/stbmse/static/index.html`), the `/ui` and `/periods` routes and their tests; checked the page in headless Edge at several widths and in dark mode |
| Modelling depth | Wrote the rolling-origin evaluation, the Optuna tuning study and the Shapley additive explanations module with their tests; ran them and wrote up the results, including the findings that cut against the proposal. The tuning study was kept from changing the shipped model, so adopting its settings stays the student's decision |
| Machine learning operations | Wrote the validation-only quality gate, the MLflow tracking and registry module and the Evidently drift module; installed Docker Desktop, MLflow and Evidently on the student's machine (the Python packages in a separate virtual environment outside OneDrive) |
| Verification | Ran the pipeline in a clean environment and the automated tests; a mutation check showing the leakage test catches planted leaks; a test that guards against a continuous integration failure the artificial intelligence itself introduced (an optional package imported at start-up, which crashed the slim Docker build); a deliberately impossible gate (exit code 1, no model written, nothing registered); the continuous integration container steps locally; and live hypertext transfer protocol status calls against the application programming interface, including the live Render service |
| Documents and slides | Drafted `README.md`, this file, `reports/REPORT.md` (and its Word version), the LaTeX report, the slide deck and the three-minute presentation `esthersides.pdf` from the actual outputs; also the `render.yaml` deployment file |
| Research | Searched the web for Armed Conflict Location and Event Data Project and Kenya National Bureau of Statistics access rules; those findings are in the chat history, not in the repository |

## What the artificial intelligence did not do

- It did not obtain Armed Conflict Location and Event Data Project or Kenya National Bureau of Statistics data, and it did not create or log in to any account on my behalf. I created the Render deployment myself in my own account.
- It did not generate `kenya_conflict_monthly.csv` and does not know how it was generated. The data are described as synthetic on my statement.
- It did not choose the research question or the proposal. The 5:1 cost ratio and the 0.05 precision-recall area under the curve gate margin are assumptions written into `config.py` and `gate.py`, not values taken from the data or from practitioners.

## Known limits of the checks

- A dedicated workflow linter (`actionlint`) was blocked by the sandbox, so the workflow file was only structurally validated before its first run. It then ran successfully on GitHub (both the `ci` and `cd` jobs).
- The published container image was not independently inspected (the available token lacks the `read:packages` permission), and the quality gate's blocking behaviour was proven locally rather than on GitHub.

## Names explained

- **Spatio-Temporal Bayesian-Markov Stacked Ensemble:** the method of this project. Its code package is named `stbmse`, the initials of these words; it appears only in file, folder and command names.
- **DSA 8401:** the course code of Applied Machine Learning at Strathmore University (Master of Science in Data Science and Analytics).
