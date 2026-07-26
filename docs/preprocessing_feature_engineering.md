# Preprocessing & Feature Engineering — Complete Reference

## Dataset Overview

**7 relational tables** from Home Credit, combined into a flat feature matrix by aggregating auxiliary tables per `SK_ID_CURR` and left-joining onto the application table.

| Table | Rows | Description |
|---|---|---|
| `application_train.csv` | ~307k | Target + client demographics, loan details, external scores |
| `application_test.csv` | ~48k | Same schema, no target |
| `bureau.csv` | ~1.7M | Previous credits at other institutions |
| `bureau_balance.csv` | ~27M | Monthly payment statuses for bureau credits |
| `previous_application.csv` | ~1.6M | Previous applications at Home Credit |
| `POS_CASH_balance.csv` | ~10M | Monthly balances for POS/cash loans |
| `credit_card_balance.csv` | ~8M | Monthly credit card balances |
| `installments_payments.csv` | ~32M | Payment history for previous loans |

---

## 1. Raw Data Loading (`loader.py`)

### 1.1 Memory Reduction (`reduce_memory`)

Each column is downcast to the smallest numeric type that can hold its values:

| Original dtype | Target dtype(s) | Condition |
|---|---|---|
| `bool` | `int8` | Always |
| `int64` / `Int64` | `int8` → `int16` → `int32` | Scans `min()` / `max()` against `np.iinfo` bounds |
| `int64` / `Int64` (with NaN) | `float32` | Nullable ints with missing values cannot become numpy int → fallback to float |
| `float64` | `float32` | If `min()` / `max()` fit within `np.finfo(float32)` range |
| `object` | Unchanged | No numeric downcast attempted |

**Bug fix M1**: The notebook used `str(col_type)[:3] == 'int'` which fails on pandas nullable `Int64` types (string starts with `'Int'`, capital I). Production uses `pd.api.types.is_integer_dtype()` which correctly handles both `int64` and `Int64`.

**Memory savings**: Typical ~60-70% reduction (2.5 GB → ~800 MB).

### 1.2 Sentinel Fix (`fix_sentinels`)

Home Credit encodes "no end date" / "unemployed" with the sentinel value `365243` (roughly 1000 years in days). This appears in all `DAYS_*` columns:

```python
_DAYS_SENTINEL_COLS = [
    "DAYS_EMPLOYED",           # Application table
    "DAYS_CREDIT_ENDDATE",     # Bureau table
    "DAYS_FIRST_DRAWING",      # Previous application
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "DAYS_ENTRY_PAYMENT",      # Installments
    "DAYS_INSTALMENT",
]
```

Each column is checked; matching sentinels are replaced with `NaN`. This prevents ~365k-day garbage values in any subsequent arithmetic (e.g., `DAYS_CREDIT_ENDDATE - DAYS_CREDIT`).

**Bug fix W3**: Notebook computed `BUREAU_CREDIT_DURATION = DAYS_CREDIT_ENDDATE - DAYS_CREDIT` **before** fixing the sentinel — revolving credits with no end date produced durations of ~365,000 days instead of `NaN`.

### 1.3 Per-Table Loaders

| Function | Tables loaded | Extra fixes |
|---|---|---|
| `load_application()` | `application_train`, `application_test` | `CODE_GENDER='XNA'` → `NaN` (anonymised value) |
| `load_bureau()` | `bureau` | Generic sentinel fix only |
| `load_bureau_balance()` | `bureau_balance` | Generic sentinel fix only |
| `load_previous_application()` | `previous_application` | Additional sentinel cols applied post-load (fix W4: no `inplace=True`) |
| `load_installments()` | `installments_payments` | Generic sentinel fix only |
| `load_pos_cash()` | `POS_CASH_balance` | Generic sentinel fix only |
| `load_credit_card()` | `credit_card_balance` | Generic sentinel fix only |

---

## 2. Application Feature Engineering (`application.py`)

The core feature table. Raw columns describe income, credit amount, annuity, external risk scores, social circle statistics, document submission flags, and demographic information.

### 2.1 Credit & Income Ratios

| Feature | Formula | Rationale |
|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / (AMT_INCOME_TOTAL + 1)` | How large is the loan relative to annual income? Higher → higher risk |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)` | Monthly payment burden as % of income. DTIs > 40% are red flags |
| `CREDIT_TERM` | `AMT_ANNUITY / (AMT_CREDIT + 1)` | Implicit loan term. Higher = faster repayment |
| `GOODS_CREDIT_RATIO` | `AMT_GOODS_PRICE / (AMT_CREDIT + 1)` | Loan-to-value proxy: ratio of goods price to credit amount |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS + 1)` | Per-capita income — family financial pressure |
| `CREDIT_PER_PERSON` | `AMT_CREDIT / (CNT_FAM_MEMBERS + 1)` | Per-capita debt burden |
| `AMT_DOWN_PAYMENT_PROXY` | `AMT_GOODS_PRICE - AMT_CREDIT` | Absolute down payment (gap between goods price and loan) |

**Denominator guard**: All divisions use `_one_plus()`, which replaces `NaN` with 1 before adding 1, preventing division by zero or null propagation.

### 2.2 Age & Employment Features

All `DAYS_*` columns are negative counts of days before application. Convert to positive years:

| Feature | Formula |
|---|---|
| `AGE_YEARS` | `DAYS_BIRTH / -365` |
| `EMPLOYED_YEARS` | `DAYS_EMPLOYED / -365` (NaN for unemployed after sentinel fix) |
| `REGISTRATION_YEARS` | `DAYS_REGISTRATION / -365` |
| `ID_PUBLISH_YEARS` | `DAYS_ID_PUBLISH / -365` |
| `LAST_PHONE_CHANGE_YEARS` | `DAYS_LAST_PHONE_CHANGE / -365` |
| `EMPLOYED_TO_AGE_RATIO` | `EMPLOYED_YEARS / (AGE_YEARS + 1)` |
| `REGISTRATION_TO_AGE` | `REGISTRATION_YEARS / (AGE_YEARS + 1)` |
| `ID_TO_AGE_RATIO` | `ID_PUBLISH_YEARS / (AGE_YEARS + 1)` |
| `AGE_AT_EMPLOYMENT_START` | `AGE_YEARS - EMPLOYED_YEARS` |

**Interpretation**: `EMPLOYED_TO_AGE_RATIO` — high ratio = stable employment history (longer at current job relative to age). `AGE_AT_EMPLOYMENT_START` — very young age at job start may indicate informal labour.

### 2.3 External Source Features

`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are third-party credit scores (unknown origin, higher = lower risk). They are among the most predictive raw features.

| Feature | Formula | Rationale |
|---|---|---|
| `EXT_SOURCE_MEAN` | `mean(ext_cols)` | Aggregate creditworthiness |
| `EXT_SOURCE_STD` | `std(ext_cols)` | Score consistency — high std = mixed signals |
| `EXT_SOURCE_MIN` | `min(ext_cols)` | Weakest score |
| `EXT_SOURCE_MAX` | `max(ext_cols)` | Best score |
| `EXT_SOURCE_RANGE` | `max - min` | Score spread |
| `EXT_SOURCE_PRODUCT` | `prod(ext_cols)` | Rewards consistently high scores, heavily penalizes any low one |
| `EXT_SOURCE_WEIGHTED` | `(e1*0.5 + e2*2.0 + e3*1.5) / 4.0` | Weighted mean — `EXT_SOURCE_2` is most predictive (weight 2.0) |
| `EXT_SOURCE_COUNT` | `count(notna(ext_cols))` | How many scores available? Missing = thin credit file |
| `EXT_12_RATIO` | `e1 / (e2 + 1e-5)` | Disagreement between score sources |
| `EXT_23_RATIO` | `e2 / (e3 + 1e-5)` | |
| `EXT_13_RATIO` | `e1 / (e3 + 1e-5)` | |
| `EXT_SOURCE_1_MISSING` | `isna(e1).astype(int8)` | Missingness itself is a signal |
| `EXT_SOURCE_2_MISSING` | `isna(e2).astype(int8)` | |
| `EXT_SOURCE_3_MISSING` | `isna(e3).astype(int8)` | |
| `EXT_MEAN_X_CREDIT_RATIO` | `mean * (CREDIT / INCOME)` | Interaction between credit score and leverage |

**Weighted mean detail**: `EXT_SOURCE_2` has highest predictive power in EDA, so it gets 2.0 weight vs 0.5 for `EXT_SOURCE_1` and 1.5 for `EXT_SOURCE_3`. Missing values are imputed with the row's own mean before weighting.

### 2.4 Social & Document Features

| Feature | Formula | Rationale |
|---|---|---|
| `SOCIAL_CIRCLE_DEFAULT_TOTAL` | `DEF_30_CNT_SOCIAL_CIRCLE + DEF_60_CNT_SOCIAL_CIRCLE` (fillna 0) | Total defaults among known contacts |
| `SOCIAL_CIRCLE_DEFAULT_RATE` | `DEF_30 / (OBS_30 + 1)` (fillna 0/1) | Default rate in observed circle |
| `REGION_RATING_MEAN` | `(REGION_RATING_CLIENT + REGION_RATING_CLIENT_W_CITY) / 2` | Average regional risk |
| `REGION_CITY_DIFF` | `REGION_RATING_CLIENT_W_CITY - REGION_RATING_CLIENT` | Client's personal region rating vs their city's |
| `DOCUMENT_COUNT` | `sum(FLAG_DOCUMENT_*)` | How many documents submitted (max 10) |
| `DOCUMENT_MISSING` | `(DOCUMENT_COUNT == 0).astype(int8)` | No documents at all |
| `CONTACT_COUNT` | `sum(FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL)` | How many contact methods provided |
| `CHILDREN_RATIO` | `CNT_CHILDREN / (CNT_FAM_MEMBERS + 1)` | Children as fraction of household |
| `HAS_CHILDREN` | `(CNT_CHILDREN > 0).astype(int8)` | Binary children flag |

### 2.5 Missing Indicators

| Feature | Raw column | Missing rate (approx) |
|---|---|---|
| `EXT_SOURCE_1_MISSING` | `EXT_SOURCE_1` | ~56% |
| `EXT_SOURCE_3_MISSING` | `EXT_SOURCE_3` | ~19% |
| `AMT_GOODS_PRICE_MISSING` | `AMT_GOODS_PRICE` | ~0.1% |
| `AMT_ANNUITY_MISSING` | `AMT_ANNUITY` | Small but important |
| `OWN_CAR_AGE_MISSING` | `OWN_CAR_AGE` | ~66% (likely no car) |
| `OCCUPATION_TYPE_MISSING` | `OCCUPATION_TYPE` | ~31% (possible unemployment) |
| `CNT_FAM_MEMBERS_MISSING` | `CNT_FAM_MEMBERS` | Rare |
| `DAYS_LAST_PHONE_CHANGE_MISSING` | `DAYS_LAST_PHONE_CHANGE` | Rare |

**Imputation**: `OWN_CAR_AGE` filled with 0 where NaN (no car). Other columns' missingness is left as `NaN` in the original column — the tree-based models handle it natively; the `_MISSING` indicator provides the missingness signal explicitly.

**Why not impute everything?**: LightGBM and XGBoost handle NaN natively by learning an optimal missing-value branch direction. Imputation would collapse distinct signals: "missing because no car" vs "missing because data entry error".

---

## 3. Bureau Features (`bureau.py`)

### 3.1 Core Bureau Aggregation (`aggregate_bureau`)

Raw bureau table: each row is a credit account at another financial institution, linked by `SK_ID_BUREAU` → mapped to `SK_ID_CURR`.

**Per-bureau derived columns before aggregation:**

| Column | Formula | Description |
|---|---|---|
| `CREDIT_ACTIVE_BINARY` | `(CREDIT_ACTIVE == 'Active').astype(int8)` | Binary active flag |
| `CREDIT_OVERDUE_BINARY` | `(CREDIT_DAY_OVERDUE > 0).astype(int8)` | Binary overdue flag |
| `BUREAU_CREDIT_UTILIZATION` | `AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM` (safe div) | Debt-to-credit ratio per loan |
| `BUREAU_CREDIT_DURATION` | `DAYS_CREDIT_ENDDATE - DAYS_CREDIT` | Loan duration in days (fix W3: sentinel removed first) |

**Groupby aggregations** (per `SK_ID_CURR`):

**All loans:**
- `BUREAU_LOAN_COUNT` — total bureau loans
- `BUREAU_ACTIVE_COUNT` — active loans
- `BUREAU_OVERDUE_COUNT` — overdue loans
- `BUREAU_PROLONG_COUNT` — total prolongations
- `BUREAU_CREDIT_SUM_MEAN / MAX / TOTAL` — credit amounts
- `BUREAU_DEBT_MEAN / TOTAL / MAX` — current debt amounts
- `BUREAU_OVERDUE_AMT_MEAN / MAX` — overdue amounts
- `BUREAU_DAY_OVERDUE_MEAN / MAX` — severity of overdue
- `BUREAU_UTILIZATION_MEAN / MAX` — average and peak utilization
- `BUREAU_DAYS_CREDIT_MEAN / MAX / MIN` — recency of credit (days since application)

**Active loans only** (subset):
- `BUREAU_ACTIVE_DEBT_TOTAL / MEAN` — debt on active loans
- `BUREAU_ACTIVE_CREDIT_MEAN` — credit on active loans
- `BUREAU_ACTIVE_OVERDUE_MAX` — worst overdue among active
- `BUREAU_ACTIVE_UTIL_MEAN` — utilization on active only

**Post-aggregation ratio features:**
- `BUREAU_ACTIVE_RATIO` — active / total
- `BUREAU_OVERDUE_RATIO` — overdue / total
- `BUREAU_DEBT_CREDIT_RATIO` — total debt / total credit (global utilization)

### 3.2 Bureau Balance Aggregation (`aggregate_bureau_balance`)

Raw: monthly status records per `SK_ID_BUREAU`.

**Status mapping:**

| STATUS code | Meaning | Numeric value |
|---|---|---|
| `C` | Closed | 0 |
| `X` | Unknown | 0 |
| `0` | 0 days overdue | 0 |
| `1` | 1-30 days overdue | 1 |
| `2` | 31-60 days overdue | 2 |
| `3` | 61-90 days overdue | 3 |
| `4` | 91-120 days overdue | 4 |
| `5` | >120 days overdue | 5 |

**Derived per-record columns:**
- `IS_DPD` — `STATUS_NUM > 0` (any delinquency)
- `IS_BAD` — `STATUS_NUM >= 2` (serious delinquency)

**Per-bureau aggregations** (groupby `SK_ID_BUREAU`):
- `BB_MONTHS_COUNT` — months of history
- `BB_STATUS_MEAN` — average delinquency level
- `BB_STATUS_MAX` — worst delinquency level
- `BB_DPD_COUNT` — months with any DPD
- `BB_BAD_COUNT` — months with serious DPD
- `BB_DPD_RATE` — DPD months / total months
- `BB_BAD_RATE` — serious months / total months

**Merge via `SK_ID_BUREAU` → `SK_ID_CURR` mapping** (loads only `SK_ID_CURR`, `SK_ID_BUREAU` from bureau.csv to save memory).

**Per-client aggregations** (groupby `SK_ID_CURR`):
- `BB_NUM_BUREAU_CREDITS` — credits with balance data
- `BB_STATUS_MEAN_OF_MEANS` — average delinquency across all credits
- `BB_STATUS_MAX_EVER` — worst status across all credits
- `BB_TOTAL_DPD_MONTHS` — total DPD months
- `BB_TOTAL_BAD_MONTHS` — total serious months
- `BB_DPD_RATE_MEAN` — average DPD rate
- `BB_BAD_RATE_MEAN` — average bad rate
- `BB_DPD_RATE_MAX` — highest DPD rate among credits

---

## 4. Previous Application Features (`previous.py`)

### 4.1 Previous Application Aggregation (`aggregate_previous_application`)

Raw: all previous loan applications at Home Credit, each with contract status, credit amount, decision date.

**Per-record derived columns:**
- `PREV_APP_CREDIT_RATIO` — `AMT_APPLICATION / AMT_CREDIT` (what they asked for vs what they got)
- `PREV_DOWN_PAYMENT_RATE` — `AMT_DOWN_PAYMENT / AMT_GOODS_PRICE`
- `PREV_IS_RECENT` — `DAYS_DECISION >= -365` (applied within last year)

**All-status aggregations** (groupby `SK_ID_CURR`):
- `PREV_APP_COUNT` — total previous applications
- `PREV_AMT_CREDIT_MEAN / MAX` — credit amounts
- `PREV_AMT_ANNUITY_MEAN` — average annuity
- `PREV_AMT_APPLICATION_MEAN` — average amount requested
- `PREV_APP_CREDIT_RATIO_MEAN` — average gap between requested and granted
- `PREV_DOWN_PAYMENT_MEAN` — average down payment rate
- `PREV_DAYS_DECISION_MAX / MEAN` — recency of decisions
- `PREV_RECENT_COUNT` — how many recent applications
- `PREV_CNT_PAYMENT_MEAN` — average payment count

**Approved-only aggregations:**
- `PREV_APPROVED_COUNT` — approved applications
- `PREV_APPROVED_CREDIT_MEAN` — average credit of approved ones
- `PREV_APPROVED_ANNUITY_MEAN` — average annuity of approved
- `PREV_APPROVED_RATIO_MEAN` — ratio on approved only

**Refused-only aggregations:**
- `PREV_REFUSED_COUNT` — refused applications
- `PREV_REFUSED_CREDIT_MEAN` — average credit of refused
- `PREV_REFUSED_DAYS_MEAN` — average recency of refusals

**Post-aggregation ratios:**
- `PREV_APPROVAL_RATE` — approved / total
- `PREV_REFUSAL_RATE` — refused / total

### 4.2 Installments Aggregation (`aggregate_installments`)

Raw: every payment made (or missed) against previous loans.

**Per-record derived columns:**

| Column | Formula |
|---|---|
| `DAYS_LATE` | `max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)` |
| `DAYS_EARLY` | `-min(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)` |
| `PAYMENT_RATIO` | `AMT_PAYMENT / AMT_INSTALMENT` (safe div) |
| `PAYMENT_DIFF` | `AMT_INSTALMENT - AMT_PAYMENT` (0 if fully paid) |
| `IS_LATE` | `DAYS_LATE > 0` |
| `IS_VERY_LATE` | `DAYS_LATE > 30` |
| `IS_UNDERPAID` | `PAYMENT_RATIO < 0.95` |

**Groupby aggregations** (per `SK_ID_CURR`):
- `INS_PAYMENT_COUNT` — total installments tracked
- `INS_DAYS_LATE_MEAN / MAX / SUM` — severity of lateness
- `INS_DAYS_EARLY_MEAN` — average early payment (positive behavior)
- `INS_LATE_COUNT / VERY_LATE_COUNT / UNDERPAID_COUNT` — counts
- `INS_PAYMENT_RATIO_MEAN / MIN` — payment ratio stats
- `INS_PAYMENT_DIFF_MEAN / MAX` — underpayment amount stats
- `INS_DAYS_ENTRY_MAX` — most recent payment tracked

**Post-aggregation rates:**
- `INS_LATE_RATE` — late payments / total
- `INS_VERY_LATE_RATE` — very late / total
- `INS_UNDERPAID_RATE` — underpaid / total

### 4.3 POS Cash Aggregation (`aggregate_pos_cash`)

Raw: monthly snapshot of point-of-sale / cash loan balances.

**Per-record derived columns:**
- `IS_DPD` — `SK_DPD > 0` (days past due on this month)
- `IS_DPD_DEF` — `SK_DPD_DEF > 0` (days past due defined)

**Groupby aggregations** (per `SK_ID_CURR`):
- `POS_MONTHS_COUNT` — months of history
- `POS_SK_DPD_MEAN / MAX / SUM` — DPD severity
- `POS_SK_DPD_DEF_MEAN / MAX` — defined DPD
- `POS_DPD_MONTH_COUNT` — months with any DPD
- `POS_DPD_DEF_COUNT` — months with DPD defined
- `POS_COMPLETED_COUNT` — completed contracts
- `POS_ACTIVE_COUNT` — active contracts
- `POS_CNT_INSTALMENT_MEAN` — average total installments
- `POS_CNT_INSTALMENT_FUTURE_MEAN` — installments remaining

**Post-aggregation rates:**
- `POS_DPD_RATE` — DPD months / total months
- `POS_COMPLETED_RATE` — completed / total months

### 4.4 Credit Card Aggregation (`aggregate_credit_card`)

Raw: monthly credit card balance snapshots.

**Per-record derived columns:**

| Column | Formula |
|---|---|
| `CC_UTILIZATION` | `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` (safe div, clipped [0,1]) |
| `CC_MIN_PAYMENT_RATIO` | `AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY` (safe div) |
| `CC_DRAWING_TOTAL` | Sum of ATM + POS + current + other drawings |
| `CC_CASH_DRAW_RATIO` | `AMT_DRAWINGS_ATM / CC_DRAWING_TOTAL` (safe div) |
| `IS_DPD` | `SK_DPD > 0` |
| `IS_MAXED` | `CC_UTILIZATION > 0.95` |

**Groupby aggregations** (per `SK_ID_CURR`):
- `CC_MONTHS_COUNT` — months of history
- `CC_UTILIZATION_MEAN / MAX / MIN` — credit utilization profile
- `CC_AMT_BALANCE_MEAN / MAX` — outstanding balance
- `CC_PAYMENT_MEAN` — average payment amount
- `CC_MIN_PAYMENT_RATIO_MEAN` — average minimum payment ratio
- `CC_DRAWING_TOTAL_MEAN` — average monthly drawings
- `CC_CASH_DRAW_RATIO_MEAN` — cash advance behaviour
- `CC_DPD_MEAN / MAX` — DPD severity
- `CC_DPD_MONTH_COUNT` — months with DPD
- `CC_MAXED_COUNT` — months near maxed out
- `CC_LIMIT_MEAN / MAX` — credit limit stats

**Post-aggregation rates:**
- `CC_DPD_RATE` — DPD months / total
- `CC_MAXED_RATE` — maxed-out months / total

---

## 5. Feature Encoding (`features/encoders.py`)

### 5.1 Categorical Encoder

Wraps sklearn `OrdinalEncoder` with:

- `handle_unknown='use_encoded_value'`, `unknown_value=-1` — unseen categories at inference get `-1` (tree will branch left)
- `encoded_missing_value=-2` — `NaN` in categorical columns gets `-2` (distinct from both seen and unseen)
- Detects `object`-dtype columns automatically via `select_dtypes(include='object')`
- Applies only to columns seen during `fit()`; other columns pass through unchanged

### 5.2 Target Encoder

Bayesian smoothing formula:

```
encoded = (n * mean + m * global_mean) / (n + m)
```

Where:
- `n` = count of samples in this category
- `mean` = target mean (default rate) within this category
- `m` = smoothing factor (default 10.0)
- `global_mean` = overall target mean across training data

**Behaviour:**
- Low-count categories are pulled strongly toward the global mean (regularisation)
- High-count categories are close to their observed mean
- `m=10` means a category needs ~10+ observations before its own mean dominates
- Unknown categories at transform time get `global_mean`

**Fold isolation**: In the training pipeline, `TargetEncoder` is `fit()` on each fold's training data with the fold's `y_tr` target, then `transform()` is applied to the validation fold. No target information leaks across folds.

**Bug fix M6**: Notebook docstring described the Bayesian formula but the implementation used a sigmoid blend `1/(1+exp(-(count-1)/m))`. Production documentation matches the implementation.

---

## 6. Feature Selection (`features/selection.py`)

All selection operates on **training folds only** (`x_tr`, `y_tr`) — the target is used exclusively for mutual information computation, never for pre-filtering.

### 6.1 Pipeline (`select_features`)

Steps executed in order:

1. **Drop constant columns**: `nunique(dropna=True) <= 1` — single-value columns provide no signal
2. **Drop duplicate columns**: Hash-based comparison (`fillna(-999).to_numpy().tobytes()`) detects columns with identical values
3. **Drop near-zero-variance columns**: `variance < 1e-8`
4. **Mutual information ranking** (optional): `mutual_info_classif(x_tr.fillna(0), y_tr)`, select top-k features. Default `k=200`.

### 6.2 Bug Fix W1

**Notebook**: Selected `_MISSING` indicator columns based on their correlation with `TARGET` on the **full training set**, then reused the same selection inside each CV fold. This creates a mild optimistic bias: the selection criterion had access to target values from the validation fold.

**Production**: `select_features()` takes only `(x_tr, y_tr)` — the training fold's data and target. Every CV fold makes its own selection decision based on its own training data. No target information crosses the fold boundary.

---

## 7. Feature Engineering Metadata (`features/engineering.py`)

### 7.1 Feature Groups

The total feature vector (~200-250 features) is organized into logical groups for ablation studies and logging:

| Group | Prefix / Pattern | Example features |
|---|---|---|
| `credit_income_ratios` | Exact matches | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO` |
| `age_employment` | Exact matches | `AGE_YEARS`, `EMPLOYED_YEARS` |
| `ext_source` | Exact matches | `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD` |
| `social_document` | Exact matches | `SOCIAL_CIRCLE_DEFAULT_TOTAL` |
| `missing_indicators` | `_MISSING` suffix | `EXT_SOURCE_1_MISSING`, `OWN_CAR_AGE_MISSING` |
| `bureau` | `BUREAU_*` prefix | `BUREAU_LOAN_COUNT`, `BUREAU_ACTIVE_RATIO` |
| `bureau_balance` | `BB_*` prefix | `BB_STATUS_MEAN_OF_MEANS` |
| `previous_application` | `PREV_*` prefix | `PREV_APP_COUNT`, `PREV_APPROVAL_RATE` |
| `installments` | `INS_*` prefix | `INS_LATE_RATE`, `INS_PAYMENT_RATIO_MEAN` |
| `pos_cash` | `POS_*` prefix | `POS_DPD_RATE` |
| `credit_card` | `CC_*` prefix | `CC_UTILIZATION_MEAN` |

### 7.2 Safe Column Assignment

`assign_column(df, name, values)` → wraps `df.assign(**{name: values})` to produce a new DataFrame, avoiding the SettingWithCopyWarning that triggers on chained `.iloc` assignment to views.

**Bug fix W2**: Notebook used `log_df['auc_delta'].iloc[0] = 0.0` — chained assignment on a potentially sliced DataFrame. Under pandas 3.0 Copy-on-Write, this may silently not persist.

---

## 8. Pipeline Orchestration (`data/pipeline.py`)

### 8.1 Merge Strategy

All auxiliary aggregations are independent and can run in parallel. They are left-joined to the application table on `SK_ID_CURR`:

```
Application (train / test)
  ← left-join Bureau
  ← left-join Bureau Balance
  ← left-join Previous Applications
  ← left-join Installments
  ← left-join POS Cash
  ← left-join Credit Card
```

The result is a single wide table (~250 columns) with one row per `SK_ID_CURR`. Missing aggregates (applicants with no bureau history, etc.) produce `NaN`, which tree-based models handle natively.

### 8.2 DVC Featurize Stage

The `__main__` block writes to `data/interim/`:
- `train_fe.parquet` — engineered train features (includes `TARGET`)
- `test_fe.parquet` — engineered test features (no `TARGET`)
- `feature_names.parquet` — column names in training order

This enables the DVC `train` stage to reproduce from the cached parquet files without re-running feature engineering.

---

## 9. Total Feature Count (Approximate)

| Source | Features |
|---|---|
| Raw application columns (after dropping IDs) | ~120 |
| Credit/income ratios | 7 |
| Age/employment features | 9 |
| EXT_SOURCE interactions | 13-15 |
| Social/document features | 9 |
| Missing indicators | 8 |
| Bureau aggregations | ~30 |
| Bureau balance aggregations | 8 |
| Previous application aggregations | ~20 |
| Installments aggregations | 16 |
| POS cash aggregations | 12 |
| Credit card aggregations | 17 |
| **Total** | **~270** |

After feature selection (mutual info top-200) + one-hot/ordinal encoding: **~200-250 features**.

---

## 10. Key Design Decisions

1. **No target encoding in current pipeline**: The `TargetEncoder` is implemented and tested but not enabled by default. Current best results come from letting tree models handle categoricals with ordinal encoding (the implicit ordinality captures "education level" or "income type" relationships that target encoding would smooth away).

2. **Division guard pattern**: `_safe_div(a, b) = np.where(b != 0, a / b, 0.0)` prevents division by zero without propagating `inf`. `_one_plus(b) = b.where(b.notna(), 1.0) + 1` prevents null propagation in denominator+1 patterns.

3. **Aggregate robustness**: Aggregations use `fillna(0)` or `fillna(1)` before division to handle sparse data (e.g., `AMT_CREDIT_SUM_DEBT` may be 0 for fully paid loans). Empty groupby results return `NaN` which is preserved for tree-based learning.

4. **Memory hygiene**: Each aggregation function explicitly `del` large DataFrames and calls `gc.collect()` before returning, keeping peak memory below 8 GB for the full pipeline.
