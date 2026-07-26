"""Feature engineering, target encoding, fold-safe feature selection.

Encoders
  ``CategoricalEncoder`` — ordinal-encode ``object`` columns
  ``TargetEncoder`` — Bayesian target encoding with fold isolation (fix M6)

Selection
  ``select_features`` — per-fold feature selection using only ``(x_tr, y_tr)`` (fix W1)
  ``select_by_mutual_info``, ``select_by_variance``, ``select_constant_and_duplicate``

Engineering
  ``assign_column``, ``FEATURE_GROUPS``, ``group_columns`` — safe ops + metadata
"""

from home_credit.features.encoders import CategoricalEncoder, TargetEncoder
from home_credit.features.engineering import (
    FEATURE_GROUPS,
    assign_column,
    assign_columns,
    get_feature_group,
    group_columns,
)
from home_credit.features.selection import (
    select_by_mutual_info,
    select_by_variance,
    select_constant_and_duplicate,
    select_features,
)

__all__ = [
    "FEATURE_GROUPS",
    "CategoricalEncoder",
    "TargetEncoder",
    "assign_column",
    "assign_columns",
    "get_feature_group",
    "group_columns",
    "select_by_mutual_info",
    "select_by_variance",
    "select_constant_and_duplicate",
    "select_features",
]
