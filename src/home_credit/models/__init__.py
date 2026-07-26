"""Model trainers (LGB, XGB), blender (nested CV), calibrator (held-out split).

Trainers
  ``train_lgb_cv``, ``train_lgb_fold`` — LightGBM with early stopping
  ``train_xgb_cv``, ``train_xgb_fold`` — XGBoost with native NaN (fix W7)

Blender
  ``optimize_blend_weights`` — nested-CV weight search (fix W6)
  ``blend_predictions`` — weighted average

Calibrator
  ``calibrate_with_split`` — fit/eval split, pick best sigmoid|isotonic (fix C2, W9)
"""

from home_credit.models.blender import blend_predictions, optimize_blend_weights
from home_credit.models.calibrator import calibrate_with_split
from home_credit.models.lgb_trainer import train_lgb_cv, train_lgb_fold
from home_credit.models.xgb_trainer import train_xgb_cv, train_xgb_fold

__all__ = [
    "blend_predictions",
    "calibrate_with_split",
    "optimize_blend_weights",
    "train_lgb_cv",
    "train_lgb_fold",
    "train_xgb_cv",
    "train_xgb_fold",
]
