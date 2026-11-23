FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# XGBoost needs the OpenMP runtime, which is not in the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Serving-only dependencies: MLflow, Evidently and the test tools are left out to keep the image small.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY data ./data
COPY run_pipeline.py .

# Train inside the build so the image is reproducible from the raw CSV alone (no artifact needs to be checked in).
# The quality gate runs here too: if it fails, `docker build` fails and no image exists to be published.
# --fast keeps the build short; --skip-audit and --no-tracking skip the parts that need MLflow / Evidently (CI runs those).
RUN python run_pipeline.py --fast --skip-audit --no-tracking

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "stbmse.api:app", "--host", "0.0.0.0", "--port", "8000"]
