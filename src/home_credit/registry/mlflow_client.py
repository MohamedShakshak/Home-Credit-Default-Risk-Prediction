"""MLflow client: log params, metrics, artifacts; register model; transition stage.

Supports both DagsHub-managed MLflow (remote) and local file-based tracking.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
from omegaconf import OmegaConf

_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "home_credit_mlflow_artifacts"


def _get_tracking_uri() -> str | None:
    return os.environ.get("MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URL")


def setup_tracking() -> None:
    """Configure MLflow tracking URI from environment (no-op if already set)."""
    uri = _get_tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def start_run(
    experiment_name: str = "home_credit_default",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Start a new MLflow run. Returns the run ID."""
    setup_tracking()
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = cast("str", mlflow.create_experiment(experiment_name) if exp is None else exp.experiment_id)

    run = mlflow.start_run(experiment_id=exp_id, run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return cast("str", run.info.run_id)


def end_run() -> None:
    mlflow.end_run()


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str | Path) -> None:
    mlflow.log_artifact(str(local_path))


def log_dict(name: str, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fpath = f.name
    mlflow.log_artifact(fpath, artifact_path=name.replace(".json", ""))
    os.unlink(fpath)


def log_hydra_config(cfg: Any) -> None:
    """Flatten an OmegaConf config to params and log."""
    container = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    flat = _flatten(container)
    mlflow.log_params(flat)


def _flatten(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, str]:
    """Recursively flatten a nested dict to MLflow-friendly param names."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            nested = _flatten(v, new_key, sep=sep)
            items.extend(nested.items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, str(v)))
    return dict(items)


def register_model(
    model: Any,
    model_name: str,
    artifact_path: str = "ensemble_model",
    stage: str = "Staging",
    input_example: Any = None,
    signature: Any = None,
) -> str | None:
    """Register a model to the MLflow Model Registry.

    Parameters
    ----------
    model : Any
        MLflow-compatible model object (e.g. ``mlflow.pyfunc.PythonModel``).
    model_name : str
        Registered model name (e.g. ``"home_credit_default"``).
    artifact_path : str
        Path inside the run's artifact URI.
    stage : str
        Target stage (``"Staging"``, ``"Production"``, ``"Archived"``, or ``"None"``).
    input_example, signature : optional
        Passed through to ``mlflow.pyfunc.log_model``.

    Returns
    -------
    model_version : str or None
        The registered model version string, or None if not available.
    """
    setup_tracking()

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        artifacts={},
        input_example=input_example,
        signature=signature,
    )

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "no-active-run"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)

    if stage and stage.lower() != "none":
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
        )

    return str(result.version)


def running_in_dagshub() -> bool:
    """Check if we are running inside a DagsHub notebook / CI."""
    uri = _get_tracking_uri() or ""
    return "dagshub" in uri


__all__ = [
    "end_run",
    "log_artifact",
    "log_dict",
    "log_hydra_config",
    "log_metrics",
    "log_params",
    "register_model",
    "running_in_dagshub",
    "setup_tracking",
    "start_run",
]
