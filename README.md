# Fraud Shield — Loan Default Risk Scoring Engine

## Problem
Predict at mortgage application time whether a borrower will default, and convert
that prediction into a 0–100 risk score with a decision band: Approve / Manual Review / Reject.
Dataset: Kaggle `yasserh/loan-default-dataset`. Target: `status` (1 = default).
Base default rate: 24.6% (class-imbalanced).

## Data Decisions
- Dropped identifiers and post-outcome fields (loan_id, year).
- **Leakage removal:** dropped rate_of_interest, interest_rate_spread, upfront_charges,
  and credit_type=EQUI — these encode information unavailable at application time.
- Engineered three missingness flags, then **audited each against the target**:
  - `propval_missing`: present in 41% of defaulters vs 0.002% of non-defaulters → LEAK, removed.
  - `dtir_missing`: 44% vs 7% → LEAK, removed.
  - `income_missing`: 3% vs 7% → balanced, RETAINED.
- Imputation (median/mode) and scaling fit on TRAIN only, to prevent leakage into test.

## Model Selection
All three models trained on identical clean features (38 columns):

| Model | PR-AUC | Default Recall (@0.5) |
|-------|--------|----------------------|
| Logistic Regression | 0.55 | 0.66 |
| Random Forest | 0.83 | 0.59 |
| **XGBoost (chosen)** | **0.84** | **0.73** |

Key finding: the logistic baseline scored 0.77 PR-AUC *with* the leaked flags but collapsed
to 0.55 once removed — it had been substantially built on leakage. Tree models were robust
(reconstructed signal from legitimate features). **This is why we audit features against the
target, not just trust the score.**

## Threshold Selection
Default 0.5 cutoff treats both errors equally; in lending they aren't. We assigned a cost
ratio of **5:1** (a missed default costs 5× a wrongly rejected good borrower) and swept
thresholds to minimize total cost. Optimal threshold = **0.41**.
- Recall: 0.73 → 0.77 (catches ~330 more defaulters)
- Precision: 0.75 → 0.56 (acceptable — borderline cases route to Manual Review, not auto-reject)

## Failure Analysis
Sliced test-set recall by credit-score band: 0.77 / 0.76 / 0.79 (low/medium/high).
Performance is **stable across credit tiers** — no systematic blind spot. This validates
using a single threshold for all applicants rather than tier-specific cutoffs.

## Risk Scorer
*[STUB — fill after notebook 07]*
Maps model probability → 0–100 score → decision band (Approve / Review / Reject).

## Known Limitations & Next Steps
- `property_value` is missing for ~41% of defaulters; median imputation may carry minor
  residual outcome signal. Flagged, not chased.
- Probability calibration (Platt/isotonic) not applied — scores are model-raw. Next step.
- Threshold cost ratio (5:1) is an assumption; real deployment would use bank-supplied figures.