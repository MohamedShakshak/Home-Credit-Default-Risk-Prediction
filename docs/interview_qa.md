# Interview Q&A — Home Credit Default Risk Prediction

## General & Motivation

**Q: Why this competition? What's the business problem?**

A: Home Credit is a non-bank lender in emerging markets. Their clients often lack formal credit history (no credit score, no bank account). The competition provides 7 relational tables from their loan book. The business problem is binary classification: predict whether an applicant will default on their loan (`TARGET=1`, ~8% of the population). The cost asymmetry is significant: a false negative (reject a good applicant) loses a customer and potential interest income; a false positive (approve a defaulter) loses the loan principal. The expected loss metric captures this: I used FP_cost=$1,000, FN_cost=$5,000 ratios in the project.

**Q: Why refactor notebooks instead of starting from scratch?**

A: The notebooks contained 3+ months of feature engineering iteration and domain knowledge. Starting from scratch would lose that. However, the notebooks had 5 critical bugs and 16 warnings identified during review: hardcoded Kaggle paths, calibrator evaluated on its own training data, SHAP log-odds fed directly into a probability-space calibrator, blend weights overfit to the OOF they were evaluated on, etc. The refactor preserves the feature logic while fixing all leakage bugs and wrapping everything in production infrastructure.

**Q: What was the hardest bug to find and fix?**

A: C3 — the SHAP log-odds / calibrator mismatch. The notebook computed SHAP values (which explain raw model output in log-odds space), then fed those log-odds directly into a calibrator that was fit on probabilities in [0, 1]. An isotonic calibrator clips out-of-range values to the nearest extreme, so most applicants got either ~0.01 or ~0.99 PD regardless of their actual risk. The fix was subtle: apply `expit()` (the logistic sigmoid) to the raw score before the calibrator call. This is invisible in the final API response because the calibrated PD looks plausible either way — but the pre-fix version had effectively random ranking within each "bucket".

---

## Data & Features

**Q: Walk me through the feature engineering process.**

A: The raw data has 7 tables. I engineered features in two layers:

1. **Application-level** (~50 features on the main table): Credit-to-income ratios, age/employment durations converted from days to years, external credit score interactions (mean, std, product, weighted mean where EXT_SOURCE_2 gets double weight because it's most predictive), social circle default rates, document submission counts, and missing-value indicators for columns with >5% missingness.

2. **Auxiliary aggregation** (~120 features from 6 side tables): Each table (bureau loans, bureau monthly balances, previous applications, installments payments, POS cash balances, credit card balances) is aggregated per customer using groupby with count/sum/mean/max on numeric columns and rate calculations (e.g., late payment rate = late installments / total installments). These aggregates are left-joined onto the application table.

The total feature count after selection is ~200-250.

**Q: How did you handle the sentinel value 365243?**

A: Home Credit uses 365243 (≈ 1000 years in days) as a sentinel for "no end date" on revolving credits and "unemployed" on DAYS_EMPLOYED. The notebook had a bug: it computed `BUREAU_CREDIT_DURATION = DAYS_CREDIT_ENDDATE - DAYS_CREDIT` before replacing this sentinel, producing ~365,000-day garbage values for revolving credits. My `fix_sentinels()` function replaces 365243 with NaN in all DAYS_* columns as the first step after loading, before any arithmetic. There's a known list of 9 column names that carry this sentinel across all tables.

**Q: Why missing-value indicators instead of imputation?**

A: For tree-based models, NaN is informative. LightGBM and XGBoost learn an optimal missing-value branch direction during training — the model decides whether samples with missing EXT_SOURCE are more like the left-branch population or the right-branch population. Imputing with mean/median collapses this signal. However, I also add explicit binary `_MISSING` flags for high-signal columns because the trees need enough samples with missing values to learn a reliable branch direction. The flag gives them a shortcut: "if this is missing, here's a direct feature".

**Q: The feature list mentions ~270 raw features but you select ~200. What's your selection strategy?**

A: I use a three-stage pipeline inside each CV fold (never on the full dataset — that was W1). First, drop constant and duplicate columns via hash comparison. Second, drop near-zero-variance columns (threshold 1e-8). Third, rank by mutual information with the fold's training target and keep the top 200. The per-fold selection is crucial because doing it on the full training set would leak validation-fold target information through the mutual information computation.

---

## Model Training

**Q: Why both LightGBM and XGBoost? Why not just pick one?**

A: They have fundamentally different tree-growing strategies: LGB is leaf-wise (grows the leaf with the largest loss reduction), XGB is level-wise (grows all leaves at the current depth before going deeper). This produces diverse predictions — their individual OOF AUCs are ~0.785-0.787, but the weighted blend reaches ~0.791. The correlation between their predictions is around 0.92, so there's real diversity to exploit. I keep both in the pipeline and let the nested-CV blender find the optimal weight (typically 0.55-0.65 for LGB, meaning LGB contributes slightly more).

**Q: How does the nested-CV blending work?**

A: The outer 5-fold CV produces OOF predictions from LGB and XGB separately. Then I run a 3-fold inner CV on those OOF predictions. For each inner fold: split OOF into inner-train and inner-val, use L-BFGS-B to find the blend weight w ∈ [0, 1] that maximises AUC on inner-train, evaluate that w on inner-val. The final weight is the median of the 3 inner-fold weights. This ensures the weight search doesn't overfit to the OOF data that I later use to report the blended AUC — the notebook bug W6 was tuning weights on the same OOF used for reporting, making the reported AUC optimistically biased by ~0.003-0.005.

**Q: What's the difference between the sigmoid and isotonic calibrators? Which one wins?**

A: The sigmoid (Platt scaling) fits a logistic regression on the uncalibrated probabilities — it's parametric and assumes the distortion follows a sigmoid shape. The isotonic fit is a non-parametric monotonic step function — more flexible but can overfit on small data. I split the OOF into cal_fit (50%) and cal_eval (50%), train both on cal_fit, evaluate both on cal_eval, and pick the one with lower Brier. In practice, sigmoid usually wins by a small margin (Brier ~0.085 vs ~0.087) because the ensemble probabilities are already reasonably calibrated and don't need the isotonic flexibility.

**Q: Your OOF AUC after calibration is ~0.790. Is that good?**

A: For this dataset, yes. The Kaggle competition leaderboard had top scores around 0.80 for single models and 0.805 for ensembles. My pipeline achieves 0.790 with only 5-fold CV on 200 features and no neural networks or stacking. The gap to 0.805 is largely explainable by: (1) the competition allowed using external data sources; (2) top solutions used 20+ fold models with heavy feature selection; (3) some used neural network embeddings on categoricals. For a production pipeline that prioritises reproducibility and maintainability over squeezing the last 0.005 AUC, this is a solid result.

---

## MLOps & Engineering

**Q: How do you prevent data leakage through the entire pipeline?**

A: Every transformation follows a strict contract: fit on training folds only, transform on validation. Enforcement points: target encoder takes `y_tr` and can't see `y_val`. Feature selection takes `(x_tr, y_tr)` — no target-informed pre-filtering. Blend weights use nested CV — the weight search never sees the outer OOF. Calibrator splits OOF into cal_fit/cal_eval — disjoint sets. SHAP explainer fits a separate TreeExplainer per fold on that fold's training data. The test CSV is only touched at the final scoring step, never during any training or selection.

**Q: How would you deploy this in production?**

A: The registered MLflow PyFunc artifact is loaded by a FastAPI service. On startup, the `lifespan` handler calls `mlflow.pyfunc.load_model('models:/home_credit_default/Staging')`. Each POST request receives raw application data (same columns as the training CSV), the model runs the full pipeline internally (featurize, encode, align columns, ensemble, blend, calibrate, optionally explain with SHAP), and returns `{predicted_pd, decision, top_reasons}`. The API container is a multi-stage Docker image (~250 MB) that can be deployed on Kubernetes, Cloud Run, or any container orchestrator. For canary deployments, you promote the new model to "Staging" in MLflow, route 10% traffic to the Staging-serving pod, compare metrics, then promote to "Production".

**Q: How is this project tested?**

A: 121 tests across 13 test files. Three layers: (1) unit tests for individual functions (encoders, metrics, drift computation) using synthetic DataFrames with controlled noise; (2) integration tests for the API using `fastapi.testclient.TestClient` with a monkeypatched mock model (no real MLflow connection needed); (3) slow tests marked `@pytest.mark.slow` that require real CSV files in `data/raw/` and are excluded from CI. The CI gate runs ruff, mypy, and `pytest --cov --cov-fail-under=50`. Current coverage is ~77%.

**Q: You used Dagshub for MLflow and DVC. What if the platform goes down?**

A: Good question — it's a single point of failure for experiment tracking and model registry. Mitigations: (1) The `.dvc/config` stores the remote URL and can be pointed at any S3-compatible storage. (2) MLflow runs can be exported and re-imported. (3) The trained PyFunc artifact is also saved as a local directory via `save_ensemble_artifact()`, so the production API could load from a local path as fallback. (4) For a production deployment, I'd host MLflow on a dedicated PostgreSQL + S3 backend.

---

## Trade-offs & Design Decisions

**Q: What's the biggest trade-off you made?**

A: Single PyFunc artifact versus separately deployable components. Packaging the full pipeline (featurizers + models + blend weights + calibrator + encoder) into one artifact simplifies deployment enormously — one `mlflow.pyfunc.load_model()` call gives you a complete inference pipeline. But it means you can't update the encoder without redeploying the entire model, and the artifact is ~500 MB (10 tree models saved as pickles). For a credit model that changes infrequently, this is acceptable. If you needed nightly retraining, you'd split into feature-service and model-service.

**Q: What would you improve next?**

A: Three things. (1) The Dockerfile and GitHub Actions workflows are documented but not implemented — Phase 10. (2) I'd add automated model performance monitoring: track AUC on production data by comparing predictions against eventually-received ground truth labels (loans that matured). (3) I'd implement the fairness audit as a scheduled job that checks demographic parity on a weekly cadence and alerts if disparity exceeds a threshold.

**Q: How would you handle a model that degrades in production?**

A: Three lines of defence. (1) The drift detection endpoint (`POST /drift/report`) compares new feature distributions against the training reference. If PSI > 0.1 or KS > 0.2 on more than 10% of features, it triggers an investigation. (2) The NaN-aware drift monitor catches missingness-shift specifically — a drift in `EXT_SOURCE_3` missingness from 20% to 30% means the data vendor changed their reporting, which likely affects model reliability. (3) Ground-truth monitoring: when loans mature (6-24 months later), compare the predicted PD against the actual outcome. If AUC drops below a threshold, retrain and promote the new model.

---

## Quick Reference: Key Numbers

| Metric | Value | Comment |
|---|---|---|
| Population default rate | ~8% | Class imbalance |
| LGB OOF AUC (mean ± std) | 0.785 ± 0.004 | 5-fold StratifiedKFold |
| XGB OOF AUC (mean ± std) | 0.787 ± 0.004 | 5-fold StratifiedKFold |
| Blended OOF AUC | ~0.791 | Nested-CV weight (honest estimate) |
| Calibrated Brier | ~0.085 | Held-out eval split |
| Expected loss at 0.15 threshold | ~0.31 | FP_cost=$1K, FN_cost=$5K |
| Number of features | ~200-250 | After mutual-info selection |
| Training time | ~15-20 min | 5 folds × 2 models, early stopping |
| Inference time (single row) | ~50 ms | Including SHAP explanation |
| Test count | 121 | 13 files, ~77% coverage |

## Quick Reference: Bug Fix Summary

| ID | Component | One-line description |
|---|---|---|
| C1 | Paths | Hardcoded Kaggle path → env-relative |
| C2 | Calibrator | Fit + evaluate on same OOF → held-out split |
| C3 | SHAP | Log-odds fed to probability calibrator → expit() bridge |
| C4 | SHAP | Return shape varies by version → handle list/2D/3D |
| C5 | SHAP | No column validation → assert on every explain() |
| W1 | Selection | Target-informed pre-filtering → per-fold selection |
| W6 | Blender | Weights tuned on reporting OOF → nested CV |
| W7 | XGB | fillna(-999) → native NaN |
| W8 | XGB | scale_pos_weight=11 → 2.5 |
| W12 | Fairness | .values alignment → SK_ID_CURR merge |
| W14 | Training | Hardcoded AUC → dynamic |
| W15 | Training | Hardcoded threshold → tuned |
| W16 | Drift | .dropna() hides NaN shift → NaN-aware binning |
| M1 | Loader | str(col_type)[:3] → is_integer_dtype() |
| M5 | SHAP | Global RNG → seeded generator |
| M6 | Encoder | Docstring mismatch → Bayesian smoothing formula |
