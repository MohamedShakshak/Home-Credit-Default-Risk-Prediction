.PHONY: help dev lint fmt type test test-fast smoke ci train predict serve repro dvc-push

help:
	@echo "Home Credit Default Risk — make targets"
	@echo "  dev          install dev deps via uv"
	@echo "  lint         ruff check"
	@echo "  fmt          ruff format"
	@echo "  type         mypy src"
	@echo "  test         pytest with coverage"
	@echo "  test-fast    pytest minus slow tests"
	@echo "  smoke        smoke tests only"
	@echo "  ci           full CI gate (lint+type+test)"
	@echo "  train        Hydra training entrypoint"
	@echo "  predict      score a sample CSV from registry"
	@echo "  serve        uvicorn API server"
	@echo "  repro        dvc repro full pipeline"
	@echo "  dvc-push     dvc push to DagsHub remote"

dev:
	uv sync

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

type:
	uv run mypy src

test:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest -m "slow"

test-all:
	uv run pytest

smoke:
	uv run pytest tests/test_smoke.py

ci: lint type test
	@echo "CI gate passed."

train:
	uv run python -m home_credit.train

predict:
	uv run python -m home_credit.predict

serve:
	uv run uvicorn home_credit.api.app:app --reload --host 0.0.0.0 --port 8000

repro:
	dvc repro

dvc-push:
	dvc push

dvc-pull:
	dvc pull

dvc-add:
	dvc add data/raw/

dvc-status:
	dvc status

init-dvc:
	uv run python scripts/init_dvc.py

dvc-dag:
	dvc dag