"""Evaluation: metrics, drift monitor, fairness audit.

Metrics
  ``report(y_true, y_pred)`` — AUC, Brier, KS, expected loss, confusion

Drift
  ``psi`` / ``ks_drift`` — NaN-aware distribution shift (fix W16)
  ``drift_report`` — per-feature PSI + KS + NaN-rate shift

Fairness
  ``fairness_report`` — demographic parity by protected attribute (fix W12)
  ``merge_on_id`` — safe SK_ID_CURR join
"""

from home_credit.evaluate.drift import (
    NaN_SENTINEL,
    drift_report,
    ks_drift,
    psi,
)
from home_credit.evaluate.fairness import fairness_report, merge_on_id
from home_credit.evaluate.metrics import (
    auc,
    brier,
    confusion,
    expected_loss,
    ks_statistic,
    report,
)

__all__ = [
    "NaN_SENTINEL",
    "auc",
    "brier",
    "confusion",
    "drift_report",
    "expected_loss",
    "fairness_report",
    "ks_drift",
    "ks_statistic",
    "merge_on_id",
    "psi",
    "report",
]
