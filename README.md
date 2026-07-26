# Home-Credit-Default-Risk-Prediction

Production-grade MLOps pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition. Predicts whether an applicant will default on a loan (`TARGET=1`).

## Stack
- **Package manager**: [`uv`](https://github.com/astral-sh/uv), Python 3.11
- **Configs**: [Hydra](https://hydra.cc/) (hierarchical YAML + composition)
- **Data versioning**: [DVC](https://dvc.org/), remote = DagsHub storage
- **Experiment tracking**: [MLflow](https://mlflow.org/) on DagsHub managed MLflow
- **Model registry**: MLflow Model Registry — single PyFunc ensemble artifact
- **API**: [FastAPI](https://fastapi.tiangolo.com/) — `/predict`, `/predict_batch`, `/explain`
- **CI/CD**: ruff + mypy + pytest + coverage gate + pre-commit + Docker + GitHub Actions
- **Docker**: API service only (multi-stage, slim runtime)

## Zero-leakage principle (ESSENTIAL)
Every transform (imputation, encoding, scaling, selection, target-encoding, blend weights, calibration, threshold) is fit on training folds only and applied to validation/test as a transformer. Nothing crosses the train/val/test boundary. See `PLAN.md` §0 for the full contract and the enforcing tests (`tests/test_no_leakage.py`, `tests/test_transformation_isolation.py`).

## Quickstart
```bash
# 1. Install deps (creates a uv-managed Python 3.11 venv)
uv sync

# 2. Set up environment
cp .env.example .env
# edit .env: MLflow tracking URI + DagsHub token + HC_DATA path

# 3. Pull raw data (Phase 8 wires DVC; for now place CSVs in data/raw/)
dvc pull            # or scripts/download_data.py

# 4. Train (Hydra entrypoint)
uv run python -m home_credit.train data=raw model=blend

# 5. Serve the API (loads latest Staging model from registry)
uv run uvicorn home_credit.api.app:app --reload --host 0.0.0.0 --port 8000

# 6. Reproduce the full DVC pipeline
dvc repro
```

## Project layout
```
configs/            Hydra configs (data, model, train, api)
data/               Raw (DVC) / interim / processed / models
notebooks/          EDA reference notebooks (kept as-is)
src/home_credit/    The package
  data/             Loaders + per-table featurizers (application, bureau, previous)
  features/         Target encoder + fold-safe selection
  models/           LGB, XGB, blender (nested CV), calibrator (held-out split)
  evaluate/         Metrics, drift monitor (NaN-aware), fairness audit
  explain/          SHAP (sigmoid → calibrator, averaged across folds)
  registry/         MLflow client + custom PyFunc ensemble artifact
  api/              FastAPI routes, schemas, middleware
tests/              smoke + leakage + transformation-isolation + regression
deployments/        api.Dockerfile (multi-stage)
docs/               ADRs, model card, data dictionary
```

## Development
```bash
make dev          # uv sync
make lint         # ruff check
make fmt          # ruff format
make type         # mypy src
make test         # pytest with coverage
make smoke        # pytest tests/test_smoke.py
make ci           # lint + type + test
```

Pre-commit: `pre-commit install` runs ruff, ruff-format, mypy, end-of-file-fixer, nbstripout on every commit.

## Bug-fix ledger
This refactor folds in critical fixes from the notebook review (see `PLAN.md` §12 for the full table). Highlights: hardcoded Kaggle paths removed (C1); calibrator fit/eval split disjoint (C2); SHAP raw-vs-probability mismatch fixed (C3); SHAP return-shape handled across versions (C4); transformer state isolation asserted (C5); per-fold feature selection (W1); nested-CV blend weights (W6); XGB native NaN (W7); NaN-aware drift monitor (W16); index-aligned fairness joins (W12); dynamic ensemble AUC instead of hardcoded (W14); tuned decision threshold instead of hardcoded `0.15` (W15).

## Status
Phase 0 (scaffolding & tooling) complete. See `PLAN.md` for the remaining phases (1–11).