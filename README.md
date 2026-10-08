# Credit Risk Assessment: Predicting Loan Defaults on LendingClub

End-to-end supervised classification pipeline on **2,260,701 LendingClub loans**, from raw data to a **leakage-free, explainable LightGBM model** (SHAP + calibration analysis).

| | |
|---|---|
| **Task** | Binary classification (Fully Paid vs. Charged Off / Default) |
| **Dataset** | [LendingClub Accepted Loans 2007–2018Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| **Labeled loans** | 1,345,350 (80.04% solvent / 19.96% defaulted) |
| **Champion model** | Optimized LightGBM |
| **Test ROC-AUC / AP** | **0.7248 / 0.3961** |

## Key Highlights
- **Two rounds of data leakage found and fixed** using forensic SHAP analysis. ROC-AUC dropped from a fake 0.9999 to a realistic, deployable 0.7248.
- Full pipeline: missingness policy, automated multicollinearity pruning, winsorization, MICE imputation, encoding, class-weighted boosting, tuning, SHAP, calibration.
- 6 models compared on a stratified 80/20 split (1,076,280 train / 269,070 test).

## Pipeline

| Stage | Columns |
|---|---|
| Raw dataset | 152 |
| Drop columns with >30% missing | 94 |
| Automated collinearity filter (\|r\| > 0.95, keep higher single-variable AUC) | 83 |
| Categorical/ID cleanup + encoding | 74 |
| Leakage purge (8 features) | **62 model-ready features** |

- **Missing data:** diagnosed as MAR (null-correlation matrix), imputed with MICE (IterativeImputer + DecisionTreeRegressor) in 5 chunks (2,172,128 missing cells to 0).
- **Outliers:** 99th-percentile winsorization (e.g. `annual_inc` max $10.99M capped at $250K, `dti` 999 capped at 38.5). No rows deleted.
- **Encoding:** ordinal (`sub_grade`), one-hot (`home_ownership`, `verification_status`), binary (`application_type`), frequency (`purpose`, `addr_state`).
- **Imbalance:** class weighting (`scale_pos_weight = 4.0088`, `class_weight='balanced'`) instead of SMOTE.

## The Leakage Investigation

| Round | Features removed | Test ROC-AUC | Test AP | Verdict |
|---|---|---|---|---|
| 1. No purge | none | 0.9999 | 0.9998 | Too good to be true |
| 2. Partial purge | 6 post-origination fields (`recoveries`, `total_rec_prncp`, `collection_recovery_fee`, `last_pymnt_amnt`, `total_rec_int`, `total_rec_late_fee`) | 0.9519 | 0.8306 | Still implausible |
| 3. Full purge (final) | 8 total (+ `last_fico_range_high`, `last_fico_range_low`) | **0.7248** | **0.3961** | Realistic, consistent with published benchmarks |

Only the leakage-feature set changed between rounds. Model, split and class weighting were identical.

## Final Results (held-out test set, leakage-free)

| Model | ROC-AUC | Avg. Precision |
|---|---|---|
| Logistic Regression | 0.6359 | 0.2937 |
| Decision Tree | 0.6927 | 0.3533 |
| Random Forest | 0.7132 | 0.3789 |
| XGBoost | 0.7214 | 0.3921 |
| LightGBM (baseline) | 0.7204 | 0.3901 |
| **Optimized LightGBM** | **0.7248** | **0.3961** |

- **Tuning:** `RandomizedSearchCV` (10 candidates, 3-fold stratified CV, scored on ROC-AUC).
- **Overfitting check:** train vs. test mean predicted probability differs by less than 0.03 percentage points for all models.

## Interpretability (SHAP)
Top drivers: `sub_grade`, loan term, `dti`, `funded_amnt`, `acc_open_past_24mths`. All are known at origination, which confirms the leakage purge worked. Waterfall plots explain individual applicant decisions.

## Calibration
Brier score = 0.2126. The model ranks risk well but under-predicts at low-to-mid probabilities (typical for class-weighted boosting). Platt scaling or isotonic regression is recommended before using raw outputs as literal default probabilities.

## Limitations & Next Steps
- Post-hoc probability calibration
- Temporal (vintage-based) validation instead of a random split
- Cost-based decision threshold (Type I vs. Type II errors)
- Fairness / disparate-impact testing on `addr_state` and `home_ownership`

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · matplotlib · seaborn

## How to Run
```bash
git clone https://github.com/armanzanganeh/credit-risk-classification.git
cd credit-risk-classification
pip install -r requirements.txt
jupyter notebook
```
Download the dataset from Kaggle (link above, too large for this repo), then open `Credit_Risk_Assessment.ipynb` and run all cells. MICE imputation is memory-heavy, so it was run in chunks.

## Author
**Arman Zanganeh**, MSc Data Science, Università degli Studi di Napoli "Federico II"
[LinkedIn](https://linkedin.com/in/arman-zanganeh2000) · [GitHub](https://github.com/armanzanganeh)
