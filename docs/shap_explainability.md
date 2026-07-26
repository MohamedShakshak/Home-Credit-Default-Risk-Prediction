# SHAP & Explainability — Complete Reference

## Table of Contents

1. [What is SHAP?](#1-what-is-shap)
2. [Why SHAP over other methods](#2-why-shap-over-other-methods)
3. [Architecture Overview](#3-architecture-overview)
4. [The SHAPExplainer Class](#4-the-shapexplainer-class)
5. [Explain Flow: Step by Step](#5-explain-flow-step-by-step)
6. [Feature Ranking (fix W11)](#6-feature-ranking-fix-w11)
7. [Top Features Selection](#7-top-features-selection)
8. [API Integration](#8-api-integration)
9. [Bug Fixes Detail](#9-bug-fixes-detail)
10. [Testing Strategy](#10-testing-strategy)
11. [Common Interview Questions](#11-common-interview-questions)

---

## 1. What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual predictions. Each prediction is decomposed into a sum of feature contributions plus a base value.

### Core idea

For a given prediction, SHAP computes how much each feature "contributed" to the difference between the actual prediction and the average prediction (base value). The sum of all feature contributions plus the base value equals the prediction:

```
prediction = base_value + Σ(shap_values)
```

The "base value" is the expected prediction over the background dataset — essentially, "what would the model predict if you knew nothing about this applicant?" It represents the average default risk across the training population.

### Why Shapley values?

SHAP is grounded in cooperative game theory. The Shapley value is the unique allocation of a payoff among players that satisfies four desirable properties:

| Property | Meaning | Why it matters for ML |
|---|---|---|
| **Efficiency** | The sum of all Shapley values + base = prediction | Explains the entire prediction, not just relative importances |
| **Symmetry** | If two features contribute equally, they get the same value | Fair attribution |
| **Dummy** | A feature that never changes the prediction gets zero | Irrelevant features are correctly ignored |
| **Additivity** | Explanations of independent models can be summed | The ensemble explanation is the sum of individual model explanations |

### For this project

SHAP answers: *"Why did this applicant get a PD of 0.082 instead of the average of 0.12? Because their high EXT_SOURCE_2 score reduced risk by -0.31 (in log-odds space), but their CREDIT_INCOME_RATIO increased risk by +0.14."*

---

## 2. Why SHAP over Other Methods

| Tool | Pros | Cons | Verdict |
|---|---|---|---|
| **SHAP** | Game-theoretic guarantees, consistent, TreeExplainer is fast for tree models | Background sampling can be slow for large datasets | **Best for tree ensembles** |
| **LIME** | Model-agnostic, simple to understand | Unstable (different random samples → different explanations), linear assumption locally | Used when you can't access model internals |
| **Permutation importance** | Global only, fast, model-agnostic | Gives global feature rankings, not per-prediction explanations | Complement to SHAP for global understanding |
| **Partial dependence plots** | Shows average marginal effect of a feature | Can hide heterogeneity (Simpson's paradox) | Good for regulatory reporting |
| **Tree interpreter (Saabas)** | Very fast for tree models | Not game-theoretic; can assign credit to features that had no effect | Not recommended |

### Key advantage for this project

TreeExplainer uses the tree structure directly (O(TLD) complexity per tree, where T = trees, L = leaves, D = depth) rather than sampling-based approximations. For our 10 models (5 LGB × 5 XGB, each with ~500 trees), TreeExplainer can compute per-prediction SHAP values in ~5-10 ms per row.

---

## 3. Architecture Overview

```
                  ┌──────────────┐
                  │ Raw input    │
                  │ DataFrame    │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ Assert column│  (fix C5)
                  │ order match  │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ SHAP values  │  (for each fold model, then average)
                  │ per fold     │  (fix W13: average across all folds)
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ raw_score =  │  base_value + sum(shap_values)
                  │ (log-odds)   │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ expit()      │  (fix C3: sigmoid to convert log-odds → prob)
                  │ → probability│
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ Calibrator   │  Optional: refine probability
                  │ predict_proba│
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │ Top-5 SHAP   │
                  │ reasons      │
                  └──────────────┘
```

### The data flow through SHAP

1. **Column assertion**: Verify the incoming feature columns match training order (C5)
2. **Convert to numpy**: `np.asarray(row).astype(float)` — TreeExplainer needs numpy arrays
3. **SHAP per fold**: Run `explainer.shap_values(row_np)` for each fold's TreeExplainer
4. **Shape normalization**: Handle list/2D/3D return formats (C4)
5. **Average across folds**: Element-wise mean of all fold SHAP matrices (W13)
6. **Raw score**: `base_value + sum(avg_shap)` = log-odds
7. **Sigmoid**: `expit(raw_score)` → probability in [0, 1] (C3)
8. **Calibrator**: Feed probability through the selected calibrator (Platt or isotonic)
9. **Top reasons**: Sort features by absolute SHAP value, take top 5

---

## 4. The SHAPExplainer Class

### Initialization

```python
class SHAPExplainer:
    def __init__(
        self,
        fold_models: list[Any],          # 5 LGB + 5 XGB (or any TreeExplainer-compatible models)
        feature_names: list[str],         # Column names in training order
        calibrator: Any | None = None,   # Optional fitted calibrator
        background_size: int = 200,      # Samples for SHAP importance ranking
        random_state: int = RANDOM_STATE, # Seeded RNG (fix M5)
    ):
```

**Why pass fold_models and not a single model?** (W13)

The notebook used `fold_models[-1]` — only the last fold's model. This has two problems:
1. It discards 80% of the ensemble's signal
2. Different folds may have different feature importances; averaging across folds gives a more stable explanation

**Why a background sample size of 200?** SHAP TreeExplainer doesn't strictly need background data for TreeExplainer (unlike KernelExplainer). The background is only used for `_compute_top_features()` which ranks features by global importance. 200 samples is enough for stable ranking and keeps computation under 1 second.

### The `fit()` method

```python
def fit(self, x_background: pd.DataFrame) -> SHAPExplainer:
    import shap

    # Create one TreeExplainer per fold model
    self._explainers = []
    for model in self._fold_models:
        explainer = shap.TreeExplainer(model)
        self._explainers.append(explainer)

    # Compute global feature ranking (fix W11)
    self._top_features = self._compute_top_features(x_background)
    return self
```

**shap is lazily imported** inside the method to avoid import-time overhead when SHAP is not needed (e.g., during training if explanation is skipped).

### The `explain()` method

```python
def explain(self, x_row: pd.DataFrame) -> dict[str, Any]:
    # 1. Column order assertion (fix C5)
    self._assert_column_order(x_row)

    row_np = np.asarray(x_row.iloc[:1]).astype(float)

    # 2-4. SHAP per fold + shape handling + averaging
    shap_values_list, base_values_list = [], []
    for explainer in self._explainers:
        sv = explainer.shap_values(row_np)

        # Handle list [class0, class1] (ancient SHAP versions)
        if isinstance(sv, list):
            sv = sv[-1]
        # Handle 3-D (n, n_features, n_outputs) (modern sklearn SHAP)
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        shap_values_list.append(sv[0])

        # Handle base_value similarly
        bv = explainer.expected_value
        if isinstance(bv, list):
            bv = bv[-1]
        elif isinstance(bv, np.ndarray) and bv.ndim == 1:
            bv = bv[-1]
        base_values_list.append(float(bv))

    # 5. Average across folds
    avg_shap = np.mean(shap_values_list, axis=0)
    avg_base = float(np.mean(base_values_list))

    # 6-7. Raw score → sigmoid
    raw_score = avg_base + float(avg_shap.sum())
    proba_raw = float(expit(raw_score))

    # 8. Calibrator
    pd_value = proba_raw
    if self._calibrator is not None:
        cal_pred = self._calibrator.predict_proba(np.array([[proba_raw]]))[0, 1]
        pd_value = float(cal_pred)

    # 9. Top-5 reasons
    shap_pairs = list(zip(self._feature_names, avg_shap.tolist(), strict=True))
    shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
    top_reasons = [
        {"feature": name, "shap": round(val, 6)}
        for name, val in shap_pairs_sorted[:5]
    ]

    return {
        "pd": round(pd_value, 6),
        "raw_score": round(raw_score, 6),
        "base_value": round(avg_base, 6),
        "shap_values": shap_pairs_sorted,
        "top_reasons": top_reasons,
    }
```

### The `save()` / `load()` methods

Persistence follows the same structure as the rest of the model:

```
explainer/
├── explainer_metadata.json    # feature_names, top_features
└── explainers.pkl             # List of TreeExplainer objects (joblib)
```

The fold models themselves are NOT saved inside the explainer directory — they're saved separately in `fold_models.pkl`. This avoids duplication since the same objects are needed by the ensemble predictor.

---

## 5. Explain Flow: Step by Step

### 5.1 Column Order Assertion (C5)

```python
def _assert_column_order(self, x_row: pd.DataFrame) -> None:
    incoming = list(x_row.columns)
    if incoming != self._feature_names:
        raise ValueError(
            f"Feature order mismatch: expected {len(self._feature_names)} "
            f"columns, got {len(incoming)}."
        )
```

**Why this matters**: If a production pipeline produces columns in a different order than training (e.g., `[AMT_CREDIT, AMT_INCOME_TOTAL]` vs `[AMT_INCOME_TOTAL, AMT_CREDIT]`), the SHAP values will be attached to the wrong feature names. A feature with a large negative SHAP value might be labelled "EXT_SOURCE_2" when it's actually "CREDIT_INCOME_RATIO". This is a silent bug — the numbers look correct, the labels look correct, but the pairs are wrong. The assertion makes this a hard crash rather than a silent misattribution.

### 5.2 SHAP Value Extraction

For each fold model, `explainer.shap_values(row_np)` returns the SHAP values for the single row.

**Before SHAP 0.42**: Returns a list of two arrays: `[class_0_values, class_1_values]`. Each array has shape `(1, n_features)`. We take `sv[-1]` to get the positive class.

**SHAP 0.42-0.43**: Returns a 2D array of shape `(1, n_features)`. This is the ideal format.

**SHAP 0.44+ for sklearn models**: Returns a 3D array of shape `(1, n_features, 2)` — two outputs (one per class) even for binary classification. We take `sv[:, :, -1]` to get the positive class.

The code handles all three formats with a simple guard chain.

### 5.3 Fold Averaging (W13)

```python
avg_shap = np.mean(shap_values_list, axis=0)
avg_base = float(np.mean(base_values_list))
```

`shap_values_list` contains one `(n_features,)` array per fold model. `np.mean(axis=0)` computes element-wise average across all 10 models. This gives each model equal weight, which is appropriate since the blend weight is very close to 0.5 anyway.

### 5.4 Raw Score

```python
raw_score = avg_base + float(avg_shap.sum())
```

This is the log-odds before the sigmoid. It's not a calibrated probability — it's the raw output of the ensemble in the model's internal space. For LightGBM, this is log-odds of default. For XGBoost, this is also log-odds (for binary logistic objective). The base value is the average raw score over the background population.

### 5.5 Sigmoid Transformation (C3)

```python
proba_raw = float(expit(raw_score))
```

`expit(x) = 1 / (1 + exp(-x))` converts log-odds to probability in [0, 1].

**Why this is necessary**: The SHAP values explain the raw model output (log-odds). The calibrator was fit during training on blended OOF **probabilities** (after the sigmoid). Feeding raw log-odds (range -∞ to +∞) into the calibrator is a domain mismatch. An isotonic calibrator clips any value outside its training range to the nearest extreme — so log-odds < -5 would get the same calibrated PD as log-odds < -2, losing all ranking signal. The sigmoid bridges this gap.

### 5.6 Calibrator

```python
if self._calibrator is not None:
    cal_pred = self._calibrator.predict_proba(np.array([[proba_raw]]))[0, 1]
    pd_value = float(cal_pred)
```

The calibrator (LogisticRegression for Platt scaling, IsotonicRegression for isotonic) refines the sigmoid-transformed probability. This is optional — if no calibrator was fitted, the sigmoid-transformed probability is returned directly.

### 5.7 Top Reasons

```python
shap_pairs = list(zip(self._feature_names, avg_shap.tolist(), strict=True))
shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
top_reasons = [
    {"feature": name, "shap": round(val, 6)}
    for name, val in shap_pairs_sorted[:5]
]
```

Sorted by **absolute SHAP value** (magnitude), not signed value. This means the top reason could be either risk-increasing (positive SHAP) or risk-decreasing (negative SHAP). The sign tells the direction.

---

## 6. Feature Ranking (fix W11)

### Problem

The notebook identified `top_features` as the first 15 columns of the processed DataFrame: `X_processed.columns[:15].tolist()`. This is completely arbitrary — columns happen to be in a particular order after merge/encoding. The drift monitoring section then used these arbitrary columns as the set to monitor, missing genuinely important features.

### Solution

```python
def _compute_top_features(self, x_background: pd.DataFrame) -> list[str]:
    sample = x_background.sample(n=min(self._background_size, len(x_background)), ...)
    sample_np = np.asarray(sample).astype(float)

    all_importances = []
    for explainer in self._explainers:
        sv = explainer.shap_values(sample_np)
        # Shape normalization (same as in explain())
        if isinstance(sv, list):
            sv = sv[-1]
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        all_importances.append(mean_abs)

    avg_importance = np.mean(all_importances, axis=0)
    ranked = [self._feature_names[i] for i in np.argsort(avg_importance)[::-1]]
    return ranked
```

This computes the **mean absolute SHAP value** for each feature across (a) the background sample and (b) all fold models. Features are then ranked by this importance. The top-ranked features are used to populate `self._top_features`, which is available to the API for feature selection, drift monitoring, and reporting.

---

## 7. Top Features Selection

### How top_features are used

The `top_features` property returns features ranked by global SHAP importance:

```python
@property
def top_features(self) -> list[str]:
    return list(self._top_features)
```

### Expected top features (based on literature and EDA)

| Rank | Feature | Expected SHAP pattern |
|---|---|---|
| 1 | `EXT_SOURCE_2` | Strong negative SHAP when high (reduces PD) |
| 2 | `EXT_SOURCE_3` | Similar but weaker |
| 3 | `EXT_SOURCE_1` | Similar but weaker |
| 4 | `CREDIT_INCOME_RATIO` | Positive SHAP when high (higher leverage → higher PD) |
| 5 | `EXT_SOURCE_WEIGHTED` | Composite of the three scores |
| 6 | `DAYS_BIRTH` | Older applicants → lower PD |
| 7 | `ANNUITY_INCOME_RATIO` | Higher burden → higher PD |
| 8 | `BUREAU_DEBT_CREDIT_RATIO` | Higher utilization → higher PD |
| 9 | `PREV_APPROVAL_RATE` | Higher approval rate → higher PD? (complex) |
| 10 | `ACTIVE_LOAN_COUNT` | More active debt → higher PD |

---

## 8. API Integration

### The `/explain` endpoint

```python
@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    model: Any = Depends(get_model),
) -> ExplainResponse:
    df = pd.DataFrame([request.application.model_dump(exclude_none=True)])
    result = model.predict(df, params={"return_shap": True})
    row = result.iloc[0]

    # Extract SHAP reasons
    top_reasons = []
    for i in range(1, 6):
        feat = row.get(f"shap_feature_{i}")
        val = row.get(f"shap_value_{i}")
        if feat is not None and val is not None and not pd.isna(val):
            top_reasons.append(ShapReason(feature=str(feat), shap=float(val)))

    return ExplainResponse(
        SK_ID_CURR=...,
        predicted_pd=float(row["predicted_pd"]),
        raw_score=float(row.get("raw_score", 0.0)),
        base_value=float(row.get("base_value", 0.0)),
        top_reasons=top_reasons,
    )
```

### How the PyFunc returns SHAP data

The `EnsemblePyFunc.predict()` method, when called with `params={"return_shap": True}`, appends SHAP columns to the output DataFrame:

```python
if _return_shap and self._explainer is not None:
    for i in range(len(df)):
        explanation = self._explainer.explain(row_df)
        top_reasons = explanation.get("top_reasons", [])
        for j, reason in enumerate(top_reasons):
            result.loc[i, f"shap_feature_{j+1}"] = reason["feature"]
            result.loc[i, f"shap_value_{j+1}"] = reason["shap"]
```

The API endpoint extracts these columns and formats them into the response schema.

### Example response

```json
{
  "SK_ID_CURR": 1001,
  "predicted_pd": 0.082,
  "raw_score": -2.15,
  "base_value": -1.47,
  "top_reasons": [
    {"feature": "EXT_SOURCE_2", "shap": -0.315},
    {"feature": "CREDIT_INCOME_RATIO", "shap": 0.142},
    {"feature": "EXT_SOURCE_3", "shap": -0.098},
    {"feature": "DAYS_BIRTH", "shap": -0.076},
    {"feature": "ANNUITY_INCOME_RATIO", "shap": 0.054}
  ],
  "shap_values": [
    {"feature": "EXT_SOURCE_2", "shap": -0.315},
    {"feature": "CREDIT_INCOME_RATIO", "shap": 0.142},
    ...
  ]
}
```

### Interpreting the response

- **predicted_pd = 0.082**: This applicant has an 8.2% chance of default
- **raw_score = -2.15**: The ensemble log-odds before sigmoid (negative → less likely than average)
- **base_value = -1.47**: The average log-odds across the training population (also negative → defaults are rare)
- **EXT_SOURCE_2 = -0.315**: Their strong credit score reduces the log-odds by 0.315 (lower PD)
- **CREDIT_INCOME_RATIO = +0.142**: Their high loan-to-income ratio increases the log-odds by 0.142 (higher PD)
- The sum of all SHAP values + base = raw_score: `-1.47 + (-0.315 + 0.142 + ...) = -2.15`

---

## 9. Bug Fixes Detail

### C3: Log-odds → Probability Domain Mismatch

**Bug**: SHAP explains raw log-odds from the tree ensemble. The calibrator was fit on blended OOF probabilities (in [0, 1]). The notebook fed raw log-odds directly into the calibrator:

```python
# Notebook (WRONG)
raw_score = base_value + sum(shap_values)
pd_value = calibrator.predict_proba([[raw_score]])[0, 1]
```

For isotonic calibration, any log-odds < -5 maps to the same calibrated value as log-odds < -2 (both are below the lowest training probability). This destroys ranking.

**Fix**:
```python
# Production (CORRECT)
raw_score = base_value + sum(shap_values)
proba_raw = expit(raw_score)  # sigmoid: log-odds → [0, 1]
pd_value = calibrator.predict_proba([[proba_raw]])[0, 1]
```

**Impact**: Without the fix, applicants with log-odds between -5 and +5 (the vast majority) get roughly the same calibrated PD. The top-5 SHAP reasons are identical but the PD values are compressed into ~3 distinct buckets instead of a smooth distribution.

### C4: SHAP Return Shape Version Handling

**Bug**: The notebook assumed `shap_values()` returns `(n_samples, n_features)`. But:

```python
# SHAP < 0.42 (ancient):
shap_values = [class0_array, class1_array]    # list of 2 arrays

# SHAP 0.42 - 0.43 (modern single-output):
shap_values = array(1, n_features)            # 2D array

# SHAP >= 0.44 for sklearn models (multi-output):
shap_values = array(1, n_features, 2)         # 3D array
```

The notebook's `[0]` index got the class-0 (non-default) values in the ancient format.

**Fix**: Guard chain:
```python
if isinstance(sv, list):
    sv = sv[-1]      # positive class (for ancient format)
if sv.ndim == 3:
    sv = sv[:, :, -1]  # positive class (for modern sklearn multi-output)
if sv.ndim == 1:
    sv = sv[np.newaxis, :]  # ensure 2D
```

Also pinned `shap>=0.44` in pyproject.toml for production deployments, but the backward-compatible guards remain for safety.

### C5: Feature Name Persistence and Assertion

**Bug**: The notebook's `SHAPExplainer.save()` pickled the model and calibrator but saved NO feature metadata. When deploying a model to production, if the production pipeline produces columns in a different order (or with different column names), the SHAP values are silently attached to the wrong feature names.

**Fix**: Persist `feature_names.json` alongside the model:

```python
metadata = {
    "feature_names": self._feature_names,
    "top_features": self._top_features,
}
with open(path / "explainer_metadata.json", "w") as f:
    json.dump(metadata, f)
```

And assert on every `explain()` call:

```python
def _assert_column_order(self, x_row: pd.DataFrame) -> None:
    incoming = list(x_row.columns)
    if incoming != self._feature_names:
        raise ValueError("Feature order mismatch: ...")
```

### W11: Top Features from Arbitrary Order

**Bug**: `top_features = X_processed.columns[:15].tolist()` — the first 15 columns after feature processing, sorted arbitrarily by merge/encoding order.

**Fix**: Compute SHAP importance on a background sample, rank features by mean absolute SHAP. The top features become `self._top_features` and are used by the drift monitoring module.

### W13: Single-Fold SHAP

**Bug**: `SHAPExplainer.__init__` used `fold_models[-1]` — only the last fold's model. Per-applicant explanations may not represent the ensemble's decision.

**Fix**: Create a `TreeExplainer` for every fold model, average the SHAP values element-wise. This gives a 10-model average explanation (5 LGB × 5 XGB).

### M5: Non-Deterministic SHAP Sampling

**Bug**: `np.random.choice(len(X_sel_final), 5000, replace=False)` uses the global RNG with no seed → non-reproducible SHAP sample. Every run produces slightly different feature rankings.

**Fix**: Create a seeded generator:
```python
self._rng = np.random.default_rng(RANDOM_STATE)
# In _compute_top_features:
sample = x_background.sample(
    n=min(self._background_size, len(x_background)),
    random_state=int(self._rng.integers(0, 2**31 - 1)),
)
```

---

## 10. Testing Strategy

9 tests in `tests/test_explain.py`:

| Test | What it validates | Bug regressions |
|---|---|---|
| `test_explainer_init` | Constructor sets correct defaults, empty state | — |
| `test_explainer_fit_computes_top_features` | `fit()` produces ranked `top_features`, order differs from input column order | W11 |
| `test_explainer_explain_returns_all_keys` | Every expected key in the response dict | — |
| `test_explainer_column_order_assertion` | Wrong column order raises `ValueError` | C5 |
| `test_explainer_save_load` | Round-trip: save → load → explain produces same structure | — |
| `test_explainer_with_calibrator` | Calibrator changes the PD value; PD stays in [0, 1] | C3 |
| `test_explainer_shap_value_length` | Number of SHAP values = number of features | C4 |
| `test_explainer_fold_average` | Multiple fold models produce a single averaged result | W13 |
| `test_explainer_seeded` | Same seed → same top features | M5 |

### Test fixture

```python
@pytest.fixture
def syn_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small dataset for SHAP tests. 4 features, 100 rows."""
    rng = np.random.default_rng(42)
    n = 100
    x = pd.DataFrame({
        "feature_a": np.where(rng.random(n) < 0.5, rng.normal(2, 0.8, n), rng.normal(-2, 0.8, n)),
        "feature_b": rng.normal(scale=2, size=n),
        "feature_c": rng.normal(scale=2, size=n),
        "feature_d": rng.exponential(2, size=n),
    })
    y = pd.Series(rng.binomial(1, 0.4, n))
    return x, y
```

Uses a `RandomForestClassifier(n_estimators=50, max_depth=5)` as a lightweight proxy for the LightGBM fold models. RF is TreeExplainer-compatible. 3 models are trained on bootstrapped samples to simulate the 10-model ensemble (5 LGB + 5 XGB).

### Known SHAP deprecation warnings

SHAP's color modules produce `PendingDeprecationWarning` about `set_bad`/`set_over`/`set_under`. These are from SHAP's internal `matplotlib` usage and don't affect correctness. They're suppressed in pytest output by the conftest.

---

## 11. Common Interview Questions

**Q: How is SHAP different from feature importance (gain/cover)?**

A: Gain and cover are global measures that tell you "on average across all trees, how much did this feature reduce loss?" They're useful for model understanding but they don't explain individual predictions. SHAP gives per-prediction decomposition, which is essential for credit decisions where you need to tell a customer *why* they were rejected. Also, gain can be biased toward high-cardinality features; SHAP doesn't have this bias.

**Q: Why do you need the sigmoid between SHAP and the calibrator?**

A: SHAP explains the model's raw output, which for log-loss objective is log-odds (range -∞ to +∞). The calibrator (Platt scaling or isotonic regression) was trained on blended OOF probabilities in range [0, 1]. If you feed log-odds directly into an isotonic calibrator, values less than the minimum training probability all get the same calibrated output — you lose ranking. The sigmoid converts log-odds to probability, matching the calibrator's expected input domain.

**Q: Why average SHAP across all folds instead of refitting on full data?**

A: You could refit a single model on the full training data and use that for SHAP. But the ensemble's prediction is the average of 10 fold models. If you explain only one refitted model, you're explaining a model that's slightly different from the actual ensemble. Averaging SHAP across all fold models gives explanations that are consistent with the ensemble's actual behaviour. The computational cost is 10x higher per explanation, but at ~5-10 ms per row, it's still acceptable.

**Q: How do you validate that SHAP explanations are correct?**

A: Two sanity checks. (1) The additive property: `expit(base_value + sum(shap_values))` should equal the calibrated PD (within floating-point tolerance). If not, there's a bug in the pipeline. (2) Consistency across similar inputs: similar applicants should have similar top reasons. I test this by cloning an applicant's row, slightly perturbing a feature, and checking that the perturbed feature's SHAP changes proportionally.

**Q: What would happen if you passed a row with features in the wrong order?**

A: With the C5 fix, it raises a `ValueError` immediately with a message listing the mismatched columns. Without the fix (as in the notebook), it would compute SHAP values and attach them to the wrong feature names silently. A feature with a large negative SHAP (reducing PD) might be labelled "EXT_SOURCE_2" when it's actually "CREDIT_INCOME_RATIO" — the numbers are internally consistent but the labels are swapped, leading to a misleading explanation for the customer.

**Q: Can SHAP explain a blended (LGB + XGB) ensemble?**

A: Yes, because SHAP values satisfy the additivity property: the SHAP values for a weighted ensemble are the weighted average of the individual models' SHAP values. We compute SHAP for each fold model independently, then average them. The base value is similarly averaged. This is mathematically exact — no approximation.
