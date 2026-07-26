"""Registry client + custom PyFunc ensemble artifact."""

from home_credit.registry.mlflow_client import (
    end_run,
    log_artifact,
    log_dict,
    log_hydra_config,
    log_metrics,
    log_params,
    register_model,
    running_in_dagshub,
    setup_tracking,
    start_run,
)
from home_credit.registry.pyfunc_ensemble import (
    EnsemblePyFunc,
    save_ensemble_artifact,
)

__all__ = [
    "EnsemblePyFunc",
    "end_run",
    "log_artifact",
    "log_dict",
    "log_hydra_config",
    "log_metrics",
    "log_params",
    "register_model",
    "running_in_dagshub",
    "save_ensemble_artifact",
    "setup_tracking",
    "start_run",
]
