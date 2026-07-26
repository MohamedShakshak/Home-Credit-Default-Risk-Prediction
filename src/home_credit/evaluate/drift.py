"""Drift monitoring: PSI, KS, NaN-rate tracking (fix W16).

Fixes notebook W16: PSI/KS now **track NaN rates separately** instead of
calling ``.dropna()`` which hides missingness drift — a real signal for
credit data (e.g. ``EXT_SOURCE_3`` ~20% missing).

NaN values are imputed to ``NaN_SENTINEL`` before binning so that
missingness creates a dedicated PSI bin; additionally the function
reports the shift in NaN proportion between reference and current.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NaN_SENTINEL: float = -999.0


def _bin_edges(
    reference: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute percentile-based bin edges from the reference distribution.

    If the reference has too few unique values, falls back to uniform edges.
    """
    uniq = np.unique(reference)
    if len(uniq) < n_bins:
        return np.linspace(uniq.min(), uniq.max() + 1e-10, min(n_bins, len(uniq)) + 1)
    return np.asarray(np.percentile(reference, np.linspace(0, 100, n_bins + 1)))


def _bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values falling into each bin defined by ``edges``."""
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(float)


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """Population Stability Index between reference and current distributions.

    Returns ``(psi_value, nan_rate_shift)``.

    NaN values are imputed to ``NaN_SENTINEL`` to create a dedicated bin,
    so missingness drift contributes to the PSI.  The ``nan_rate_shift``
    separately reports the absolute difference in NaN proportion.
    """
    ref = np.asarray(reference).ravel().astype(float)
    cur = np.asarray(current).ravel().astype(float)

    ref_nan_rate = float(np.isnan(ref).mean())
    cur_nan_rate = float(np.isnan(cur).mean())
    nan_rate_shift = abs(cur_nan_rate - ref_nan_rate)

    # Impute NaN → sentinel for binning.
    ref_clean = np.where(np.isnan(ref), NaN_SENTINEL, ref)
    cur_clean = np.where(np.isnan(cur), NaN_SENTINEL, cur)

    edges = _bin_edges(ref_clean, n_bins=n_bins)
    ref_counts = _bin_counts(ref_clean, edges) + epsilon
    cur_counts = _bin_counts(cur_clean, edges) + epsilon

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi_value, nan_rate_shift


def ks_drift(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """Two-sample KS statistic measuring distribution shift.

    Returns ``(ks_stat, nan_rate_shift)``.
    """
    ref = np.asarray(reference).ravel().astype(float)
    cur = np.asarray(current).ravel().astype(float)

    ref_nan_rate = float(np.isnan(ref).mean())
    cur_nan_rate = float(np.isnan(cur).mean())
    nan_rate_shift = abs(cur_nan_rate - ref_nan_rate)

    # Impute NaN → sentinel.
    ref_clean = np.where(np.isnan(ref), NaN_SENTINEL, ref)
    cur_clean = np.where(np.isnan(cur), NaN_SENTINEL, cur)

    # Compute empirical CDF difference.
    combined = np.sort(np.concatenate([ref_clean, cur_clean]))
    cdf_ref = np.searchsorted(ref_clean, combined, side="right") / len(ref_clean)
    cdf_cur = np.searchsorted(cur_clean, combined, side="right") / len(cur_clean)
    ks_stat = float(np.max(np.abs(cdf_ref - cdf_cur)))
    return ks_stat, nan_rate_shift


def drift_report(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute PSI, KS, and NaN-rate shift for every feature.

    Returns::

        {"feature_name": {
            "psi": float, "psi_nan_shift": float,
            "ks": float, "ks_nan_shift": float,
            "ref_nan_rate": float, "cur_nan_rate": float,
        }, ...}
    """
    report: dict[str, Any] = {}
    common = set(reference) & set(current)
    for col in sorted(common):
        ref_arr = reference[col]
        cur_arr = current[col]
        psi_val, psi_nan = psi(ref_arr, cur_arr, n_bins=n_bins)
        ks_val, ks_nan = ks_drift(ref_arr, cur_arr)
        ref_nan = float(np.isnan(ref_arr).mean())
        cur_nan = float(np.isnan(cur_arr).mean())
        report[col] = {
            "psi": psi_val,
            "psi_nan_shift": psi_nan,
            "ks": ks_val,
            "ks_nan_shift": ks_nan,
            "ref_nan_rate": ref_nan,
            "cur_nan_rate": cur_nan,
        }
    return report


__all__ = [
    "NaN_SENTINEL",
    "drift_report",
    "ks_drift",
    "psi",
]
