# Home-Credit-Default-Risk-Prediction

Production MLOps pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition — predicts loan default (`TARGET=1`).

## Stack
| Layer | Implementation |
|---|---|
| Package manager | `uv`, Python 3.11 |
| Configuration | Hydra (hierarchical YAML) |
| Data versioning | DVC → DagsHub storage |
| Experiment tracking | MLflow (DagsHub-managed) |
| Model registry | MLflow Model Registry (single PyFunc ensemble) |
| API | FastAPI — `/predict`, `/predict_batch`, `/explain` |
| CI/CD | ruff, mypy, pytest, pre-commit, GitHub Actions |
| Containerisation | Multi-stage Docker image (API only) |

## Getting started
```bash
uv sync                              # install (Python 3.11 venv)
# Configure tracking (one-time):
uv run python scripts/init_dvc.py    # sets DVC remote + MLflow URI + .env
dvc pull                             # fetch raw data from DagsHub
uv run python -m home_credit.train   # train → log to MLflow → register
uv run uvicorn home_credit.api.app:app --reload  # serve API
dvc repro                            # reproduce full pipeline
```

## Project layout
```
configs/           Hydra configs (data, model, train, api)
data/              raw (DVC-tracked) · interim · processed · models
src/home_credit/   data · features · models · evaluate · explain · registry · api
tests/             smoke · data · config · encoders · selection · models · calibration · api · explain · registry
scripts/           download_data.py · seed_dagshub.py · init_dvc.py
dvc.yaml           DVC pipeline (featurize → train)
notebooks/         EDA reference (kept as-is)
deployments/       api.Dockerfile
docs/              ADRs · model card · data dictionary
```

## Development
```bash
make dev       # uv sync
make lint      # ruff check
make fmt       # ruff format
make type      # mypy src
make test      # pytest --cov
make ci        # lint + type + test (full gate)
make serve     # uvicorn API server
make repro     # dvc repro
make dvc-dag   # show DVC pipeline graph
```
`pre-commit install` enables ruff, ruff-format, mypy, and nbstripout on every commit.

## Bug-fix ledger
Notebook review findings folded into this refactor (5 critical, 16 warnings). See `PLAN.md` §12 for the full table.
