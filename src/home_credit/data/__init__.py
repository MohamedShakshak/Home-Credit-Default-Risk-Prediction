"""Data loaders and featurizers (application, bureau, previous).

Entrypoints:
- ``loader.load_application()`` — raw train/test CSVs, downcast, sentinel fix
- ``pipeline.build_full_features(train, test)`` — all featurization + merge
- ``pipeline.train_test_split_features(X, y, X_test)`` — separate target, align
"""

from home_credit.data.application import (
    add_age_employment_features,
    add_credit_income_ratios,
    add_ext_source_features,
    add_missing_indicators,
    add_social_document_features,
    engineer_application_features,
)
from home_credit.data.bureau import aggregate_bureau, aggregate_bureau_balance
from home_credit.data.loader import (
    fix_sentinels,
    load_application,
    load_bureau,
    load_bureau_balance,
    load_credit_card,
    load_csv,
    load_installments,
    load_pos_cash,
    load_previous_application,
    reduce_memory,
)
from home_credit.data.pipeline import (
    FullFeaturePipeline,
    build_full_features,
    train_test_split_features,
)
from home_credit.data.previous import (
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_application,
)

__all__ = [
    "FullFeaturePipeline",
    "add_age_employment_features",
    "add_credit_income_ratios",
    "add_ext_source_features",
    "add_missing_indicators",
    "add_social_document_features",
    "aggregate_bureau",
    "aggregate_bureau_balance",
    "aggregate_credit_card",
    "aggregate_installments",
    "aggregate_pos_cash",
    "aggregate_previous_application",
    "build_full_features",
    "engineer_application_features",
    "fix_sentinels",
    "load_application",
    "load_bureau",
    "load_bureau_balance",
    "load_credit_card",
    "load_csv",
    "load_installments",
    "load_pos_cash",
    "load_previous_application",
    "reduce_memory",
    "train_test_split_features",
]
