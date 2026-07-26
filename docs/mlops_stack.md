# MLOps Stack — Tools & Techniques Reference

## Table of Contents

1. [Package Management: uv](#1-package-management-uv)
2. [Configuration: Hydra](#2-configuration-hydra)
3. [Data Versioning: DVC](#3-data-versioning-dvc)
4. [Remote Storage & Experiment Backend: Dagshub](#4-remote-storage--experiment-backend-dagshub)
5. [Experiment Tracking & Model Registry: MLflow](#5-experiment-tracking--model-registry-mlflow)
6. [API Serving: FastAPI](#6-api-serving-fastapi)
7. [Containerization: Docker](#7-containerization-docker)
8. [CI/CD: GitHub Actions](#8-cicd-github-actions)
9. [Code Quality: Pre-commit, Ruff, Mypy](#9-code-quality-pre-commit-ruff-mypy)
10. [Testing: Pytest & Coverage](#10-testing-pytest--coverage)
11. [Structured Logging: Structlog](#11-structured-logging-structlog)
12. [Monitoring: Prometheus](#12-monitoring-prometheus)
13. [Explainability: SHAP](#13-explainability-shap)
14. [Model Training: LightGBM & XGBoost](#14-model-training-lightgbm--xgboost)
15. [Zero-Leakage Training Technique](#15-zero-leakage-training-technique)
16. [Nested Cross-Validation Blending](#16-nested-cross-validation-blending)
17. [Held-Out Calibration](#17-held-out-calibration)
18. [NaN-Aware Drift Detection](#18-nan-aware-drift-detection)
19. [PyFunc Ensemble Artifact](#19-pyfunc-ensemble-artifact)

---

## 1. Package Management: uv

### What it is

[uv](https://docs.astral.sh/uv) is a Rust-based Python package and project manager, designed as a drop-in replacement for pip + pip-tools + virtualenv + poetry. It's built by Astral (the same team behind Ruff).

### Why uv over pip/poetry

| Factor | pip | poetry | uv |
|---|---|---|---|
| Install speed | Slow (sequential) | Moderate | 10-100x faster (Rust, parallel) |
| Lockfile | No native (pip-tools needed) | `poetry.lock` | `uv.lock` (deterministic) |
| Virtualenv | Separate step | Integrated | Integrated |
| Python version mgmt | Manual install | No | `uv python install 3.11` |
| Dependency resolution | SHA-256 only | SAT solver | SAT solver (faster) |
| Disk space | Full copies | Full copies | Hardlinks across projects |

### Configuration (`pyproject.toml`)

```toml
[project]
name = "home-credit-default-risk"
requires-python = ">=3.11,<3.12"

[dependency-groups]
dev = ["ruff>=0.1.11", "mypy>=1.7", "pytest>=7.4", ...]

[tool.uv]
package = true
```

### Usage in this project

```bash
uv sync                     # Install all deps (creates .venv)
uv add pandas-stubs         # Add dev dependency  
uv run python -m pytest     # Run command within venv
uv lock                     # Regenerate uv.lock
```

### Key decisions

- **Python 3.11** chosen over 3.12 for better ML library compatibility (numba, pandas-stubs, compatibility with older ML package versions)
- **`dependency-groups`** used instead of `[project.optional-dependencies]` to keep dev dependencies separate from runtime deps, avoiding unnecessary installs in production images
- **`[project.scripts]`** defines two entry points: `home-credit-train` and `home-credit-predict`

---

## 2. Configuration: Hydra

### What it is

[Hydra](https://hydra.cc) is a framework for elegantly configuring complex applications. Developed by Facebook/Meta AI Research, it provides hierarchical YAML configuration with command-line composition and override.

### Why Hydra over argparse/OmegaConf alone

| Need | Solution |
|---|---|
| 50+ hyperparameters across LGB, XGB, blend, calibrator | Hierarchical YAML files, not flat argparse |
| Run experiments with different model types | Composition via `defaults` list: `model=blend` or `model=lgb` |
| Override any config without touching code | `python train.py model.lgb.n_estimators=1000` |
| Log config to MLflow | `log_hydra_config(cfg)` flattens OmegaConf to MLflow params |

### Configuration hierarchy

```
configs/
├── config.yaml              # Top-level: random_state, paths, defaults list
├── data/
│   ├── raw.yaml             # CSV file names, target column
│   └── processed.yaml       # n_splits, calibration split ratio
├── model/
│   ├── lgb.yaml             # LGB hyperparams
│   ├── xgb.yaml             # XGB hyperparams
│   └── blend.yaml           # Blend method, calibrator, threshold
├── train/
│   └── default.yaml         # Seed, n_splits, MLflow flags
└── api/
    └── local.yaml           # Host, port, model URI
```

### Composition

```yaml
# config.yaml
defaults:
  - data: raw          # → loads configs/data/raw.yaml under cfg.data
  - model: blend       # → loads configs/model/blend.yaml under cfg.model
  - train: default     # → loads configs/train/default.yaml under cfg.train
  - api: local         # → loads configs/api/local.yaml under cfg.api
  - _self_             # top-level keys in config.yaml
```

### Command-line overrides

```bash
# Change model type
python -m home_credit.train model=lgb

# Override specific hyperparams
python -m home_credit.train model.lgb.learning_rate=0.05 model.lgb.num_leaves=63

# Change random seed
python -m home_credit.train random_state=123

# Disable MLflow logging
python -m home_credit.train train.log_to_mlflow=false
```

### Integration with tests (`src/home_credit/config.py`)

A lightweight `load_config()` function composes Hydra configs without the full Hydra runtime, enabling unit tests of config loading without initializing the Hydra context:

```python
from home_credit.config import load_config, validate_config

cfg = load_config(overrides=["model.lgb.n_estimators=100"])
assert validate_config(cfg) == []
```

---

## 3. Data Versioning: DVC

### What it is

[DVC](https://dvc.org) (Data Version Control) is an open-source tool that brings Git-like versioning to data files and ML pipeline stages. It stores metadata in Git (`.dvc` files, `dvc.yaml`) and the actual data in a remote storage (Dagshub, S3, GCS, etc.).

### Why DVC over git-lfs / manual versioning

| Approach | Problem | DVC solution |
|---|---|---|
| Git LFS | 2.5 GB of CSV → huge repo, slow clones | Stores only hashes in Git; real data on remote |
| Manual versioning | "data_v2_final_3.zip" | Deterministic hash-based versioning |
| No versioning | Can't reproduce old experiments | `dvc checkout` restores exact data version |
| Ad-hoc scripts | "run the pipeline by hand" | `dvc repro` reruns only changed stages |

### Pipeline definition (`dvc.yaml`)

```yaml
stages:
  featurize:
    cmd: uv run python -m home_credit.data.pipeline
    deps:
      - data/raw/                    # Raw CSVs
      - src/home_credit/data/        # Source code
      - src/home_credit/features/
      - src/home_credit/paths.py
    params:
      - configs/config.yaml:         # Hydra params DVC tracks
          - random_state
      - configs/data/raw.yaml
    outs:
      - data/interim/               # Output features (parquet)
    metrics:
      - data/metrics/featurize_report.json:
          cache: false

  train:
    cmd: uv run python -m home_credit.train data=raw model=blend
    deps:
      - data/interim/
      - src/home_credit/
      - configs/
    params:
      - configs/config.yaml
      - configs/model/blend.yaml
    outs:
      - data/models/:
          persist: true
    metrics:
      - data/metrics/train_metrics.json:
          cache: false
```

### How DVC detects changes

Each dep/param is hashed. When you run `dvc repro`:
1. DVC checks if any dep hash changed (code, data, or params)
2. If changed → re-run the stage → record new output hashes
3. If unchanged → use cached outputs (skip)

### Workflow

```bash
# One-time setup
uv run python scripts/init_dvc.py    # Configure Dagshub remote

# Track new data
dvc add data/raw/                      # Creates data/raw.dvc
dvc push                               # Upload to Dagshub

# Reproduce full pipeline
dvc repro                              # Run only changed stages
dvc status                             # Show what would change
dvc dag                                # Show pipeline DAG
```

### Why `data/dvc.yaml` was moved to project root

DVC paths are relative to the `dvc.yaml` file's location. Initially placed in `data/`, all paths like `data/raw/` resolved to `data/data/raw/` — double-nested. Moving `dvc.yaml` to the project root fixed all path resolutions.

### Remote configuration

```ini
[core]
    remote = dagshub
['remote "dagshub"']
    url = https://dagshub.com/mohamedshakshak455/Home-Credit-Default-Risk-Prediction.dvc
    auth = basic
    user = mohamedshakshak455
    password = <token>
```

---

## 4. Remote Storage & Experiment Backend: Dagshub

### What it is

[Dagshub](https://dagshub.com) is a collaborative platform for ML projects. It provides:
- **Git hosting** (like GitHub but ML-aware)
- **DVC remote storage** (S3-compatible, hosted)
- **Managed MLflow tracking server** (zero-config)
- **Model registry UI** (browse, compare, promote models)

### Why Dagshub over self-hosted MLflow / own S3

| Option | Dagshub | Self-hosted MLflow + S3 |
|---|---|---|
| Setup time | 5 minutes (DVC remote + MLflow URI) | Days (provision VM, PostgreSQL, S3 bucket, IAM) |
| Cost | Free tier | S3 costs + VM costs |
| Maintenance | Zero | Ongoing: upgrades, backups, networking |
| DVC + MLflow integration | Single dashboard | Two separate UIs |

### Configuration

Dagshub exposes two endpoints per repository:

```
DVC remote:  https://dagshub.com/<user>/<repo>.dvc
MLflow URI:  https://dagshub.com/<user>/<repo>.mlflow
```

Authentication uses a personal access token for both DVC and MLflow:

```bash
# .env
MLFLOW_TRACKING_URI=https://dagshub.com/mohamedshakshak455/Home-Credit-Default-Risk-Prediction.mlflow
MLFLOW_TRACKING_USERNAME=mohamedshakshak455
MLFLOW_TRACKING_PASSWORD=<token>
```

### Automated setup script

```python
# scripts/init_dvc.py — prompts for:
# - Dagshub username
# - Repository name
# - Token (opens browser to https://dagshub.com/settings/tokens)
# Then:
#   dvc remote add -d dagshub <url>
#   dvc remote modify dagshub auth basic
#   dvc remote modify dagshub user <username>
#   dvc remote modify dagshub password <token>
#   Writes .env with MLFLOW_TRACKING_URI
```

---

## 5. Experiment Tracking & Model Registry: MLflow

### What it is

[MLflow](https://mlflow.org) is an open-source platform for the ML lifecycle. Four components:

| Component | Used | Purpose |
|---|---|---|
| **Tracking** | ✓ | Log parameters, metrics, artifacts per run |
| **Projects** | ✗ | Not needed (uv + Hydra handle this) |
| **Models** | ✓ | Package model artifacts in a standard format |
| **Model Registry** | ✓ | Version, stage-transition, deploy registered models |

### Tracking: what gets logged

Every training run logs via `registry/mlflow_client.py`:

**Parameters** (via `log_hydra_config` — flattens full OmegaConf to key-value pairs):
```
random_state: 42
data.raw.files.application_train: application_train.csv
model.lgb.n_estimators: 5000
model.lgb.learning_rate: 0.01
model.xgb.scale_pos_weight: 2.5
model.blend.inner_cv: 3
train.seed: 42
...
```
Plus git hash and model name as tags.

**Metrics** (via `log_metrics`):
```
auc_lgb_cv_mean: 0.7852
auc_xgb_cv_mean: 0.7869
auc_blend_oof: 0.7910
auc_calibrated: 0.7905
brier_calibrated: 0.0852
ks_calibrated: 0.4721
expected_loss: 0.3124
lgb_weight: 0.58
auc_lgb_fold_0: 0.7821
auc_lgb_fold_1: 0.7863
...
```

**Artifacts** (via `log_dict`/`log_artifact`):
- `confusion_matrix.json`
- `calibrator_info.json` (method, Brier score)
- Feature importance plots
- Calibration curve
- ROC curve
- SHAP summary plots

### Model Registry: PyFunc artifact

Every training run registers **one model** to the `home_credit_default` registry:

```python
register_model(
    model=EnsemblePyFunc(),       # Custom PythonModel wrapper
    model_name="home_credit_default",
    stage="Staging",              # "Staging" → can be promoted to "Production"
)
```

The registered artifact wraps the entire pipeline inside a single `mlflow.pyfunc.PythonModel`:

```
MLflow Registry: home_credit_default
  ├── Version 1 (Staging)
  ├── Version 2 (Staging) ← latest
  └── Version 3 (Production) ← promoted after validation
```

### PyFunc wrapper structure

```
EnsemblePyFunc (mlflow.pyfunc.PythonModel)
├── fold_models: LGBMClassifier × 5 + XGBClassifier × 5
├── feature_names: [str]       # Column order — asserted at inference
├── lgb_weight: float          # Blend weight (0.0-1.0)
├── calibrator: LogisticRegression | IsotonicRegression
├── encoder: CategoricalEncoder
└── explainer: SHAPExplainer    # Optional, loaded lazily
```

### Inference-time loading

```python
# API startup (one-time load)
model = mlflow.pyfunc.load_model("models:/home_credit_default/Staging")

# Per-request prediction
result = model.predict(raw_data_df)
# Returns: DataFrame with SK_ID_CURR, predicted_pd, decision, top_reasons
```

### Why PyFunc over sklearn flavor

| Flavor | Pros | Cons |
|---|---|---|
| `sklearn` | Native sklearn predict | Can't wrap non-sklearn components (LGBM list, calibrator, encoder, SHAP) |
| `pyfunc` | Total flexibility: any Python logic in predict() | Must implement `load_context` manually |
| Custom flavor | Clean separation | More boilerplate for versioning |

PyFunc gives us the ability to run the full pipeline (featurize → encode → align → predict → blend → calibrate → explain) as a single `model.predict()` call, hiding all complexity from the API layer.

---

## 6. API Serving: FastAPI

### What it is

[FastAPI](https://fastapi.tiangolo.com) is a modern web framework for building APIs with Python 3.11+, based on Starlette (ASGI) and Pydantic for data validation.

### Why FastAPI over Flask / Django

| Factor | Flask | Django | FastAPI |
|---|---|---|---|
| Performance (async) | No native async | Async since 3.0 | Native async (Starlette) |
| Data validation | Manual / marshmallow | Django Forms / DRF | Automatic (Pydantic v2) |
| OpenAPI docs | flasgger / manual | DRF Spectacular | Automatic (Swagger UI + ReDoc) |
| Type safety | None | Partial | First-class (`from __future__ import annotations`) |
| ML inference use | Common | Overkill | Growing rapidly |

### Application structure

```
src/home_credit/api/
├── app.py              # Factory: create_app() with middleware + routes + lifespan
├── schemas.py           # Pydantic v2 request/response models
├── deps.py              # Model loader (cached), settings
├── middleware.py         # RequestID injection + structlog
└── routes/
    ├── predict.py        # POST /predict
    ├── predict_batch.py  # POST /predict_batch, /predict_batch/csv
    ├── explain.py        # POST /explain
    └── drift.py          # POST /drift/report, /drift/report/json
```

### Lifespan pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model from MLflow registry
    try:
        load_model()
        app.state.model_loaded = True
    except Exception:
        app.state.model_loaded = False
    yield
    # Shutdown: (nothing to clean up currently)
```

### Middleware chain

1. **CORSMiddleware**: Permissive for development (restrict in production)
2. **RequestIDMiddleware**: Generates `uuid.uuid4()` per request, injects into response headers, logs via structlog
3. **ServerErrorMiddleware** (built-in): Catches unhandled exceptions → 500 with traceback in logs

### Dependency injection

```python
# deps.py — cached model instance
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def load_model() -> Any:
    """Cached model loader; re-reads _model_instance global."""
    ...

# routes/predict.py — FastAPI dependency
@router.post("", response_model=PredictionResponse)
def predict_single(
    request: ApplicationRequest,
    model: Any = Depends(get_model),
) -> PredictionResponse:
    ...
```

---

## 7. Containerization: Docker

> **Status**: Planned (Phase 10 deliverable). Architecture documented here.

### Target Dockerfile structure

```dockerfile
# deployments/api.Dockerfile — multi-stage build
# Stage 1: Builder
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
RUN uv sync --frozen --no-dev  # Install only runtime deps

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "home_credit.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Why multi-stage

- **Builder**: Installs ALL build dependencies (compilers for numba, scipy, etc.)
- **Runtime**: Only Python + pre-compiled `.so` files from `.venv`
- **Result**: ~250 MB image vs ~1.2 GB single-stage

### Registry target

```
ghcr.io/<user>/home-credit-api:vX.Y.Z
```

### Why not include training in Docker

Training requires raw data (~2.5 GB), Hydra configs, and MLflow access — better run as a CI/CD job or on a scheduled VM. The API container is small, stateless, and horizontally scalable.

---

## 8. CI/CD: GitHub Actions

> **Status**: Planned (Phase 10 deliverable). Workflows documented here.

### CI workflow (`ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run ruff check src tests
      - run: uv run ruff format --check src tests
      - run: uv run mypy src
      - run: uv run pytest -m "not slow" --cov --cov-fail-under=50
```

### CD workflow (`deploy.yml`)

```yaml
name: Deploy
on:
  push:
    tags: ['v*']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f deployments/api.Dockerfile -t ghcr.io/${{ github.repository }}:${{ github.ref_name }} .
      - run: docker push ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### Why GitHub Actions over alternatives

| CI System | Reason to choose |
|---|---|
| GitHub Actions | Zero config (same platform as code); generous free tier |
| GitLab CI | Not applicable (hosted on GitHub) |
| Jenkins | Overkill for this scale |
| CircleCI | Good but paid features for parallelism |

---

## 9. Code Quality: Pre-commit, Ruff, Mypy

### Pre-commit configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml, check-toml
      - id: check-added-large-files  # Blocks >1 MB

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
      - id: ruff           # Lint (auto-fix enabled)
      - id: ruff-format    # Format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy

  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout     # Strips notebook output cells before commit
```

### Ruff configuration

**Why Ruff over Flake8 + isort + Black**: Ruff is written in Rust, 100-1000x faster than Flake8, replaces isort + Black + Flake8 + auto-fixes. Single tool, single config.

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "TCH", "RUF"]
ignore = ["E501"]  # line length handled by formatter
```

**Key rules enabled:**
- `E/W` — pycodestyle (errors/warnings)
- `F` — pyflakes (undefined names, unused imports)
- `I` — isort (import ordering)
- `N` — naming conventions (PEP8)
- `UP` — pyupgrade (modern Python syntax)
- `B` — bugbear (potential bugs)
- `C4` — comprehensions (simplify list/dict/set)
- `SIM` — simplify (ternary instead of if/else)
- `TCH` — typing (move TYPE_CHECKING imports)

### Mypy configuration

**Why mypy over pyright/pylance**: mypy is the de-facto standard for Python type checking with the most complete stubs ecosystem.

```toml
[tool.mypy]
python_version = "3.11"
mypy_path = ["src"]
strict = true
warn_return_any = true
disallow_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = ["lightgbm.*", "xgboost.*", "shap.*", "sklearn.*", "scipy.*", "joblib.*"]
ignore_missing_imports = true
```

**Strict mode** catches:
- Missing return type annotations
- Implicit `Optional` (e.g., `x = None` without `| None`)
- Returning `Any` from typed functions
- Incompatible types in assignments

---

## 10. Testing: Pytest & Coverage

### Test architecture

```
tests/
├── conftest.py                  # Shared fixtures (configs_dir, random_state)
├── test_smoke.py                # 6 tests: imports, paths, subpackages
├── test_config.py               # 8 tests: Hydra config loading + validation
├── test_data.py                 # 18 tests: memory, sentinels, features, bureau
├── test_encoders.py             # 9 tests: fold isolation, unseen, NaN
├── test_selection.py            # 10 tests: constant/duplicate, MI, variance
├── test_models.py               # 11 tests: LGB/XGB AUC, blend bounds, OOF
├── test_calibration.py          # 6 tests: held-out split, Brier honesty
├── test_api.py                  # 10 tests: FastAPI TestClient all routes
├── test_explain.py              # 9 tests: SHAP round-trip, column assert
├── test_registry.py             # 7 tests: flatten, PyFunc predict, save
├── test_evaluate.py             # 19 tests: metrics, drift, fairness
└── test_drift_api.py            # 5 tests: drift endpoint
```

**Total**: 121 tests across 13 files. ~77% coverage.

### Markers

```ini
[tool.pytest.ini_options]
markers = [
    "smoke: lightweight smoke tests",   # pytest -m smoke
    "slow: long-running tests",         # pytest -m slow  (skip by default)
]
```

`slow` tests require real data files in `data/raw/` and are excluded from CI runs.

### Coverage threshold

```bash
# CI gate: must reach 50% (currently ~77%)
uv run pytest --cov --cov-fail-under=50
```

The threshold is deliberately moderate because code that requires real data (bureau aggregation, full feature pipeline) can't be exercised in unit tests. The threshold protects against regression (new untested code dropping coverage below baseline).

### Key testing patterns

**Synthetic data fixtures**: Small DataFrames with controlled noise/test signals:

```python
@pytest.fixture
def syn_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 200
    x = pd.DataFrame({"signal": ..., "noise_1": ..., ...})
    y = pd.Series(rng.binomial(1, 0.4, n))
    return x, y
```

**Monkeypatched MLflow**: API tests replace `mlflow.pyfunc.load_model` with a `_MockModel` to avoid requiring a running MLflow server:

```python
def _mock_mlflow_model(*args, **kwargs):
    return _MockModel()

monkeypatch.setattr(mlflow.pyfunc, "load_model", _mock_mlflow_model)
```

**Slow tests for real data**: Tests that require actual CSV files in `data/raw/` are decorated `@pytest.mark.slow`; they execute only when explicitly requested: `pytest -m slow`.

---

## 11. Structured Logging: Structlog

### What it is

[structlog](https://www.structlog.org) is a structured logging library for Python that produces JSON-formatted log output instead of plain text. It integrates with standard library logging.

### Why structlog over standard logging

| Need | Standard logging | structlog |
|---|---|---|
| Machine-parseable logs | Manual JSON formatting | Automatic JSON output |
| Request IDs | Thread-local hack | Built-in context chain |
| Timestamp format | `asctime` (free-text) | ISO 8601 (consistent) |
| Dev vs prod output | Same format everywhere | Conditional: pretty console or JSON |

### Configuration

```python
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),   # Pretty in dev
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
```

### Request logging (middleware)

Each HTTP request produces one structured log line:

```python
logger.info(
    "request",
    request_id="a1b2c3d4-e5f6-...",
    method="POST",
    path="/predict",
    status_code=200,
)
```

### Why not `filter_by_level`

The initial configuration used `stdlib.filter_by_level` but this requires a stdlib logger, not `PrintLoggerFactory`. The simplest viable configuration omits level filtering (all logs are logged at `info` or higher) and uses `ConsoleRenderer` for development readability.

---

## 12. Monitoring: Prometheus

### What it is

[Prometheus](https://prometheus.io) is an open-source monitoring and alerting toolkit. [prometheus_fastapi_instrumentator](https://github.com/trallnag/prometheus_fastapi_instrumentator) auto-instruments FastAPI applications.

### Why Prometheus over ELK / Grafana Loki

| Tool | Purpose |
|---|---|
| Prometheus | Metrics (request count, latency, error rate) |
| Grafana | Dashboard visualization on top of Prometheus |
| ELK / Loki (logs) | Separate concern — structlog covers this |

### Integration

```python
# app.py — optional, fails gracefully if package not installed
try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

if Instrumentator:
    Instrumentator().instrument(app).expose(app)
```

Exposes `/metrics` endpoint with:
- Request count per endpoint
- Request latency histogram
- Response status codes

---

## 13. Explainability: SHAP

### What it is

[SHAP](https://shap.readthedocs.io) (SHapley Additive exPlanations) uses game theory to explain individual predictions. Each feature gets an importance score showing its contribution to the prediction.

### Why SHAP over LIME / Eli5

| Tool | SHAP | LIME | Eli5 |
|---|---|---|---|
| Theoretical basis | Shapley values (game theory) | Local surrogate models | Permutation importance |
| Consistency guarantees | ✓ Additive feature attribution | ✗ Local linearity assumption | ✗ |
| Tree support | Native `TreeExplainer` (fast) | Generic (slow) | Partial |
| Model-agnostic | No (requires model access) | Yes | Partial |
| Computational cost | O(TLD) per tree | Per-sample model calls | O(n_samples × n_features) |

### Implementation (`SHAPExplainer`)

```python
explainer = SHAPExplainer(
    fold_models=models,         # LGBM + XGB models from all folds
    feature_names=columns,      # Column names in training order
    calibrator=best_cal,        # Fitted calibrator
    background_size=200,        # Background samples for SHAP
)

explainer.fit(x_background)     # Fits TreeExplainer per fold, computes top features

result = explainer.explain(x_row)
# Returns:
#   pd: 0.082               ← calibrated probability of default
#   raw_score: -2.15        ← ensemble log-odds before calibrator
#   base_value: -1.47       ← expected log-odds (average prediction)
#   shap_values: [("EXT_SOURCE_2", -0.31), ("CREDIT_INCOME_RATIO", 0.14), ...]
#   top_reasons: [{"feature": "EXT_SOURCE_2", "shap": -0.31}, ...]
```

### Critical bug fixes

| Bug | Problem | Fix |
|---|---|---|
| C3 | SHAP explains log-odds, calibrator was fit on probabilities → domain mismatch | `expit(raw_score)` before calibrator call |
| C4 | SHAP return shape varies across versions (list, 2D, 3D) | Handle all three `isinstance(sv, list)` / `sv.ndim == 3` / `sv.ndim == 1` |
| C5 | No column order validation → silent wrong predictions | `assert list(x_row.columns) == self._feature_names` |
| W13 | Notebook used only last fold's explainer | Average SHAP across all 5 fold models |

---

## 14. Model Training: LightGBM & XGBoost

### LightGBM

**Default parameters:**
```python
n_estimators=5000, learning_rate=0.01, num_leaves=31, max_depth=-1,
min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=2.15,
early_stopping_rounds=200
```

**Key characteristics:**
- GOSS (Gradient-based One-Side Sampling) — faster training than XGBoost
- Native categorical feature support (`categorical_feature` parameter)
- Leaf-wise tree growth — deeper trees, more prone to overfitting (mitigated by `num_leaves=31`, `min_child_samples=20`)
- `scale_pos_weight=2.15` — class weight for imbalance (~92% non-default, 8% default)

### XGBoost

**Default parameters:**
```python
n_estimators=5000, learning_rate=0.01, max_depth=6, min_child_weight=5,
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
scale_pos_weight=2.5, handle_missing='native'
```

**Key characteristics:**
- Level-wise tree growth — broader trees, less prone to overfitting
- Native NaN handling (no need for `fillna(-999)` — fix W7)
- `scale_pos_weight=2.5` (notebook had 11 — fix W8; 11 was aggressively distorting AUC)
- `reg_lambda=1.0` vs LGB's `0.1` — stronger L2 regularisation

### Why both LGB and XGB?

| Factor | LightGBM | XGBoost |
|---|---|---|
| Training speed | 2-3x faster | Slower |
| Memory usage | Lower | Higher |
| Handling of categoricals | Native (no OHE needed) | Need encoding |
| NaN handling | Native (missing=NaN) | Native (missing=NaN) |
| Overfitting tendency | More (leaf-wise) | Less (level-wise) |
| Ensemble diversity | Higher (different growth strategy) | Higher (different growth strategy) |

Both are trained and blended because their different tree-growth strategies produce diverse predictions, and the weighted blend consistently outperforms either model alone by 0.002-0.005 AUC.

---

## 15. Zero-Leakage Training Technique

### Principle

Every data transformation must be fit on training folds only and applied to validation as a transformer. Nothing crosses the train/validation/test boundary.

### Enforcement in code

| Component | Where enforced | Mechanism |
|---|---|---|
| Target encoding | `features/encoders.py` | `TargetEncoder.fit(x_tr, y_tr)` — fit takes the target; only train-fold target is used |
| Feature selection | `features/selection.py` | `select_features(x_tr, y_tr)` — takes only train-fold data |
| Blend weights | `models/blender.py` | `optimize_blend_weights()` uses inner CV — the weights never see the outer OOF |
| Calibration | `models/calibrator.py` | Splits OOF into `cal_fit` / `cal_eval` — calibrator never sees `cal_eval` |
| SHAP explainer | `explain/shap_explainer.py` | Fitted on each fold's training data using that fold's `TreeExplainer` |
| Threshold tuning | `train.py` | Threshold tuned on held-out validation fold with cost-weighted metric |

### Assertion in training code

```python
# train.py: before each fold, the pipeline clone is verified unfitted
# (conceptual — requires sklearn check_is_fitted)
assert not check_is_fitted(pipeline, all_comps=False)
```

---

## 16. Nested Cross-Validation Blending

### Problem (W6)

Original notebook tuned blend weights to maximise AUC on the **same OOF** predictions used to report the final AUC. This causes:
1. **Optimistic bias**: The weight search overfits to noise in the OOF
2. **Selection bias**: If you try 50 weight combinations, the best one will beat the true optimum purely by chance

### Solution

```
Outer loop (StratifiedKFold, n_splits=5):
  └── Train LGB + XGB on outer train-fold
  └── Predict outer val-fold → OOF predictions

  Inner loop (StratifiedKFold, n_splits=3):
    └── Split OOF into inner-train / inner-val
    └── On inner-train: search weight w in [0, 1] to maximise AUC (L-BFGS-B)
    └── Evaluate w on inner-val

  └── Final weight = median(inner-fold best weights)
  └── OOF AUC = AUC(blended OOF predictions, y) using outer val-fold
```

**Result**: The reported OOF AUC is an honest estimate because the weight search never saw those predictions.

### L-BFGS-B optimization

```python
from scipy.optimize import minimize

result = minimize(
    loss,                        # Negative AUC
    x0=[0.5],                    # Start at equal weight
    bounds=[(0.0, 1.0)],         # Weight must be in [0, 1]
    method="L-BFGS-B",           # Quasi-Newton, bounded
)
```

Optimization is deterministic and converges in <20 iterations (vs. the notebook's brute-force grid search over 50 discrete values).

---

## 17. Held-Out Calibration

### Problem (C2, W9)

**C2**: Notebook examined Brier score on the **same** OOF predictions used to fit the calibrator. Isotonic regression is non-parametric — it can fit any monotonic function perfectly in-sample. The reported Brier improvement was over-optimistic.

**W9**: The method selection rule ("pick whichever has lower Brier") was evaluated on the calibrator's own training data, systematically favouring isotonic (more flexible → lower in-sample error).

### Solution

```
OOF predictions (blended)
        │
        ▼
  ┌─────────────┐
  │ cal_fit (50%)│──────→ Fit sigmoid + isotonic
  └─────────────┘
  ┌─────────────┐
  │cal_eval (50%)│──────→ Evaluate Brier of both
  └─────────────┘        → Pick method with lower Brier
                          → Report that Brier as the honest estimate
```

**Key points:**
- The split is stratified (preserves class balance)
- `cal_fit` and `cal_eval` are **disjoint** — `set(cal_fit_idx).isdisjoint(cal_eval_idx)`
- The two methods are:
  - **Sigmoid** (Platt scaling): `LogisticRegression(C=1e8)` — parametric, assumes monotonic sigmoid relationship. Good when the uncalibrated probabilities have a sigmoid-shaped distortion pattern.
  - **Isotonic**: `IsotonicRegression(out_of_bounds='clip')` — non-parametric, more flexible. Better when the distortion pattern is complex, but can overfit on small data.

---

## 18. NaN-Aware Drift Detection

### Problem (W16)

Notebook called `.dropna()` before computing PSI and KS statistics. In credit data, missingness is itself a signal — `EXT_SOURCE_3` is ~20% missing. A shift in missingness from 20% to 30% is a real drift signal that `.dropna()` completely hides.

### Solution

```python
def psi(reference, current, n_bins=10):
    # Track NaN rates separately
    ref_nan_rate = isnan(ref).mean()
    cur_nan_rate = isnan(cur).mean()
    nan_rate_shift = abs(cur_nan_rate - ref_nan_rate)

    # Impute NaN → sentinel -999.0 so missingness creates a dedicated bin
    ref_clean = where(isnan(ref), -999.0, ref)
    cur_clean = where(isnan(cur), -999.0, cur)

    # Standard PSI computation on imputed data
    edges = percentile(ref_clean, ...)
    ...
    return psi_value, nan_rate_shift
```

**Output**: Both PSI and KS statistic return `(value, nan_rate_shift)`, so the caller can see:
- Did the distribution of values shift? (`psi` / `ks`)
- Did the missingness rate shift? (`nan_rate_shift`)

### API integration

```bash
# Upload two CSV files
curl -X POST http://localhost:8000/drift/report \
  -F "ref_file=@train_features.csv" \
  -F "cur_file=@production_features.csv"
```

Returns per-feature: `{psi, psi_nan_shift, ks, ks_nan_shift, ref_nan_rate, cur_nan_rate}`.

### Scheduled check script

```bash
uv run python scripts/check_drift.py \
  --reference data/interim/train_fe.parquet \
  --current data/interim/current_fe.parquet \
  --output drift_report.json \
  --psi-threshold 0.1 \
  --ks-threshold 0.2
```

Can be scheduled via cron, Airflow, or any scheduler.

---

## 19. PyFunc Ensemble Artifact

### Problem

Deploying a credit model is not just about serialising a `predict_proba` method. The production artifact must:
1. Apply the same feature engineering as training
2. Encode categoricals using the same mapping
3. Align columns to training order
4. Ensemble predictions from multiple fold models
5. Apply blend weights
6. Calibrate to probabilities
7. Optionally explain with SHAP

### Solution: single PyFunc wrapper

```python
class EnsemblePyFunc(mlflow.pyfunc.PythonModel):
    """Wraps the full pipeline: featurize → encode → align → ensemble → blend → calibrate → explain."""

    def predict(self, context, model_input, params=None):
        df = engineer_application_features(model_input)
        df = self._encoder.transform(df)
        df = df[self._feature_names]        # Column alignment

        fold_preds = [m.predict_proba(df)[:, 1] for m in self._fold_models]
        avg_preds = np.mean(fold_preds, axis=0)

        lgb_preds = np.mean(fold_preds[:n_lgb], axis=0)
        xgb_preds = np.mean(fold_preds[n_lgb:], axis=0)
        blended = blend_predictions(lgb_preds, xgb_preds, self._lgb_weight)

        probs = self._calibrator.predict_proba(blended[:, None])[:, 1]

        if params.get("return_shap"):
            # Append SHAP reasons to each row
            ...

        return result_df
```

### Why single artifact over multiple

| Approach | Single PyFunc | Separate artifacts |
|---|---|---|
| Deployment | `mlflow.pyfunc.load_model()` one call | Load LGB state + XGB state + calibrator + encoder + join logic |
| Versioning | One version per model | Need to version 5+ artifacts together |
| Inference | Single `model.predict(df)` | Must manually copy-paste preprocessing from notebooks |
| Rollback | One version rollback | Must rollback 5+ artifacts to matching versions |

### Save/load structure

```
model_artifact/
├── fold_models.pkl          # All 10 models (LGB × 5 + XGB × 5)
├── model_metadata.json      # feature_names, lgb_weight, threshold
├── calibrator.pkl           # LogisticRegression or IsotonicRegression
├── encoder.pkl              # CategoricalEncoder
└── explainer/
    ├── explainer_metadata.json
    └── explainers.pkl       # SHAP TreeExplainer × 10
```
