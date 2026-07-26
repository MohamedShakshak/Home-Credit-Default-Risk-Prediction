# Home-Credit-Default-Risk-Prediction

Production MLOps pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition — predicts whether an applicant will default on a loan (`TARGET=1`).

## Stack
| Layer | Implementation |
|---|---|
| Package manager | `uv`, Python 3.11 |
| Configuration | Hydra (hierarchical YAML composition) |
| Data versioning | DVC, remote via DagsHub storage |
| Experiment tracking | MLflow (DagsHub-managed) |
| Model registry | MLflow Model Registry — single PyFunc ensemble |
| API | FastAPI — `/predict`, `/predict_batch`, `/explain` |
| CI/CD | ruff, mypy, pytest, pre-commit, GitHub Actions |
| Containerisation | Multi-stage Docker image (API only) |

## Getting started
```bash
uv sync                              # install dependencies
cp .env.example .env                 # configure MLflow URI, DagsHub token, data path
dvc pull                             # fetch raw data
uv run python -m home_credit.train   # train → log to MLflow → register model
uv run uvicorn home_credit.api.app:app --reload  # start API server
dvc repro                            # reproduce full pipeline
```

## Project layout
```
configs/           Hydra configs (data, model, train, api)
data/              raw (DVC) / interim / processed / models
notebooks/         EDA reference
src/home_credit/   data | features | models | evaluate | explain | registry | api
tests/             smoke · data · config
deployments/       api.Dockerfile
docs/              ADRs · model card · data dictionary
```

## Development
```bash
make ci      # lint, type-check, test (CI gate)
make test    # pytest with coverage
make serve   # uvicorn API server
make repro   # dvc repro
```
`pre-commit install` enables ruff, ruff-format, mypy, and nbstripout on every commit.