# Home-Credit-Default-Risk-Prediction

Production MLOps pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition. Predicts loan default (`TARGET=1`).

## Stack
| Layer | Tool |
|---|---|
| Package / venv | `uv`, Python 3.11 |
| Configs | Hydra (hierarchical YAML) |
| Data versioning | DVC → DagsHub storage |
| Experiment tracking | MLflow (DagsHub managed) |
| Model registry | MLflow Model Registry (single PyFunc ensemble) |
| API | FastAPI (`/predict`, `/predict_batch`, `/explain`) |
| CI/CD | ruff + mypy + pytest + pre-commit + GitHub Actions |
| Docker | Multi-stage API image |

## Zero-leakage principle (ESSENTIAL)
All transforms fit on training folds only, applied to val/test as transformers. Nothing crosses the train/val/test boundary. Enforced by `tests/test_no_leakage.py` + `tests/test_transformation_isolation.py` in CI.

## Quickstart
```bash
uv sync                              # install (Python 3.11 venv)
cp .env.example .env                 # set MLflow URI + DagsHub token + HC_DATA
dvc pull                             # raw data (or place CSVs in data/raw/)
uv run python -m home_credit.train   # train → log → register
uv run uvicorn home_credit.api.app:app --reload  # serve API
dvc repro                            # full pipeline
```

## Layout
```
configs/         Hydra configs (data, model, train, api)
data/            raw (DVC) / interim / processed / models
notebooks/       EDA reference (kept as-is)
src/home_credit/ data | features | models | evaluate | explain | registry | api
tests/           smoke · leakage · transformation-isolation · regression
deployments/     api.Dockerfile
docs/            ADRs · model_card · data_dictionary
```

## Development
```bash
make ci      # lint + type + test (full gate)
make test    # pytest --cov
make serve   # uvicorn API
make repro   # dvc repro
```
`pre-commit install` → ruff, ruff-format, mypy, nbstripout on every commit.

## Bug-fix ledger
Refactor folds in notebook-review fixes: hardcoded Kaggle paths (C1), disjoint calibrator fit/eval (C2), SHAP raw-vs-probability (C3), SHAP return-shape (C4), transformer state isolation (C5), per-fold selection (W1), nested-CV blend weights (W6), native XGB NaN (W7), NaN-aware drift (W16), index-aligned fairness (W12), dynamic AUC (W14), tuned threshold (W15). Full table in `PLAN.md` §12.

## Status
Phase 0 (scaffolding) complete. Phases 1–11 in `PLAN.md`.