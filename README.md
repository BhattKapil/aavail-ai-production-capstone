# AAVAIL Revenue Forecasting — AI in Production Capstone

A production-style implementation for the IBM AI Enterprise Workflow / AI in Production capstone. The course capstone asks learners to take a time-series revenue solution through data investigation, model comparison, deployment, testing and post-production monitoring. The official project guidance specifies an API that accepts a country and date, automated ingestion, Docker, unit tests and performance monitoring.

## Project layout

- `src/ingest.py` — automated JSON ingestion, schema normalization and daily aggregation.
- `src/model.py` — lag/rolling features, Random Forest and Extra Trees comparison, recursive 30-day forecasting.
- `src/api.py` — Flask `/train`, `/predict`, `/logs`, and `/health` endpoints.
- `src/logging_utils.py` — JSONL prediction logging.
- `src/monitor.py` — MAE, RMSE and Wasserstein monitoring.
- `tests/` — API, model, ingestion and logging tests.
- `scripts/` — ingestion, training, EDA, comparison and monitoring automation.
- `reports/` — generated EDA/model/monitoring outputs.
- `data/input/` — AAVAIL-shaped JSON transaction sources with intentionally varied field names.
- `Dockerfile` — reproducible container build.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/ingest.py
python scripts/train.py
python scripts/eda.py
python scripts/compare.py
python scripts/monitor.py
python run_tests.py
python -m src.api
```

API runs at `http://localhost:8080`.

Examples:

```bash
curl -X POST "http://localhost:8080/predict?date=2025-02-01&country=Australia&duration=30"
curl -X POST "http://localhost:8080/predict?date=2025-02-01&duration=30"
curl "http://localhost:8080/logs"
```

The second prediction call omits `country` and returns the combined forecast across all countries.

## Docker

```bash
docker build -t aavail-ai-production .
docker run --rm -p 8080:8080 aavail-ai-production
```

or:

```bash
docker compose up --build
```

## Unit tests

Run every unit test with one command:

```bash
python run_tests.py
```

Tests are deliberately isolated. The logging test uses a temporary directory and never writes to production `logs/`. API tests use Flask's test client.

## Model comparison

Two tree-based forecasting models are evaluated against a 7-day-mean baseline on the final validation window:

- Random Forest Regressor
- Extra Trees Regressor
- 7-day mean baseline

The model family with the lowest mean validation MAE is selected and retrained per country on the full available series.

## Monitoring

`scripts/monitor.py` creates `reports/post_production_metrics.csv` using held-out predictions and calculates:

- MAE
- RMSE
- Wasserstein distance

This provides a simple post-production drift/performance signal.

## Important data note

The repository includes a **deterministic demo dataset shaped like the course's AAVAIL transaction data** so the project is immediately runnable. For an actual Coursera submission, replace `data/input/*.json` with the transaction JSON files supplied by the course environment if your course provides them. Do not claim the included demo records are the original course dataset.

## Peer-review checklist

| Criterion | Implementation |
|---|---|
| API unit tests | `tests/test_api.py` |
| Model unit tests | `tests/test_model.py` |
| Logging unit tests | `tests/test_logging.py` |
| One test command | `run_tests.py` |
| Performance monitoring | `src/monitor.py`, `scripts/monitor.py` |
| Test/production isolation | `tmp_path` in tests |
| Country + all-country API | `/predict?...&country=Australia` and no country |
| Automated ingestion | `src/ingest.py`, `scripts/ingest.py` |
| Multiple models | Random Forest + Extra Trees + baseline |
| EDA visualizations | `scripts/eda.py` |
| Docker | `Dockerfile` |
| Model vs baseline visualization | `scripts/compare.py` |

## Course alignment

The official course page describes the capstone as a three-part project covering data investigation, model building/selection, and model production, followed by peer review. The official capstone repository states that Part 3 includes an API with train/predict/log endpoints, Docker packaging, unit testing, post-production analysis and comparison to known values.
