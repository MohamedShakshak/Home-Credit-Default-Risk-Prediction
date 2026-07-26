"""Dependency injection: MLflow model loader, cached global model instance."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlflow.pyfunc
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_model_uri: str = os.environ.get("MODEL_URI", "models:/home_credit_default/Staging")
    model_stage: str = os.environ.get("MODEL_STAGE", "Staging")
    log_dir: str = os.environ.get("API_LOG_DIR", "api/logs")

    model_config = {"env_prefix": "", "case_sensitive": False}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


_model_instance: Any = None


def load_model() -> Any:
    """Load the model from MLflow registry (cached after first call)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    settings = get_settings()
    _model_instance = mlflow.pyfunc.load_model(settings.mlflow_model_uri)
    return _model_instance


def get_model() -> Any:
    """FastAPI dependency: yield the cached model."""
    return load_model()


def reset_model() -> None:
    """Force reload on next call (useful for tests)."""
    global _model_instance
    _model_instance = None
