# Fraud Shield — Decision Log

A per-notebook record of every significant decision and the reasoning behind it.
Purpose: a forensic audit trail — for defending any choice under questioning, and for
reconstructing the pipeline's logic later.

Project: Loan Default Risk Scoring Engine. Dataset: Kaggle `yasserh/loan-default-dataset`.
Target: `status` (1 = default). Base default rate: 24.6%.

---

## 01_eda

| Decision | Why |
|----------|-----|
| Renamed columns to readable names | Raw column names were cryptic; readability reduces downstream error. |
| Dropped `loan_id` | Identifier — no predictive signal, risk of the model memorizing rows. |
| Dropped `year` | Single-value / artifact column — no variance, no signal. |
| Ran dominance + missingness + correlation analysis | Identify near-constant columns, missing-data patterns, and redundant correlated features before modeling. |
| Saved `loan_eda.csv` | Checkpoint after cleaning so feature work starts from a stable file. |

---

## 02_features

| Decision | Why |
|----------|-----|
| Dropped `rate_of_interest`, `interest_rate_spread`, `upfront_charges` | **Leakage** — these are set/known after loan approval, not available at application time. Using them lets the model "see the future." |
| Dropped `credit_type = EQUI` | **Label leak** — this credit type encoded the outcome directly. |
| Dropped `security_type`, `open_credit`, `credit_worthiness`, `construction_type`, `secured_by` | Dominated / artifact columns — near-constant, little to no predictive variance. |
| Created `income_missing`, `dtir_missing`, `propval_missing` flags | Hypothesis: *whether* a field is missing might itself carry signal. (Later audited — see 04.) |
| Saved `loan_features.csv` WITH NaNs (26 cols) | Deliberately did NOT impute here. Imputation must be fit on train only (see 03) to avoid leaking test statistics into training. Saving with NaNs preserves that boundary. |

**Principle locked:** leakage is about *timing/causation*, not correlation strength. A feature
is leaky if it wouldn't exist at the moment you score a live applicant.

---

## 03_baseline

| Decision | Why |
|----------|-----|
| Pipeline order: split (stratified) → impute → one-hot → scale → model | Each step's parameters (medians, categories, scaler) fit on TRAIN ONLY, then applied to test. Prevents test information leaking into training. |
| Stratified split | Preserves the 24.6% default ratio in both train and test, so metrics are representative. |
| Impute medians/modes from TRAIN only | Fitting imputation on the full dataset would leak test distribution into training. |
| Scaler fit on TRAIN only | Same leakage reason as imputation. |
| Baseline model = Logistic Regression + `class_weight='balanced'` | Simple, interpretable floor to beat. `balanced` counters the 24.6% imbalance so the model doesn't ignore the minority (default) class. |
| Saved X/y artifacts + `scaler.pkl` + `logreg_balanced.pkl` | Frozen artifacts so later notebooks build forward without re-running preprocessing. |

**Baseline result (with leaky flags still present):** ROC-AUC 0.844, PR-AUC 0.768, recall 0.66.
*(Note: this PR-AUC was later found to be inflated by leakage — see 04.)*

**Principle locked:** accuracy is misleading under imbalance (predicting all-majority scores
75% accuracy while catching zero defaults). PR-AUC and recall are the metrics that matter.

---

## 04_models_threshold

### Model comparison

| Decision | Why |
|----------|-----|
| Reused scaled artifacts for tree models (RF, XGBoost) | Trees split on value *order*, not magnitude. Scaling is monotonic — it doesn't change order — so scaled features are fine. Avoided regenerating unscaled CSVs (would reopen frozen preprocessing). |
| Tested RF with and without `class_weight='balanced'` | To measure sensitivity. Result: barely moved PR-AUC (0.829 vs 0.830) — trees already split on the minority class, so reweighting matters less than for LR. |
| Added XGBoost with `scale_pos_weight ≈ 3.06` | XGBoost's imbalance lever (negatives÷positives). Tabular problems often favor gradient boosting. |
| **Chosen model: XGBoost** | Best on all three metrics simultaneously: PR-AUC 0.841, recall 0.73, balanced precision 0.75. Only model that's both top-scoring and leak-immune. |

### Leakage audit (triggered by feature-importance plot)

| Decision | Why |
|----------|-----|
| Investigated `propval_missing` dominating feature importance | A missingness flag out-ranking real risk drivers (DTI, LTV, credit score) is a leak signal, not a finding. |
| Audited all three missingness flags against the target | `propval_missing`: 41% of defaulters vs 0.002% non-defaulters → **LEAK, dropped**. `dtir_missing`: 44% vs 7% → **LEAK, dropped**. `income_missing`: 3% vs 7% → balanced → **RETAINED**. |
| Re-ran LR / RF / XGBoost on clean 38-col features | To make the model comparison fair — all three on identical leak-free features. |
| Kept XGBoost despite residual `property_value` imputation risk | `property_value` is a legitimate application-time field and can't be dropped. Its missingness (~41% of defaulters) may carry minor residual signal — documented as a known limitation rather than chased (out of scope under deadline). |

**Key finding:** LR collapsed from 0.768 → 0.553 PR-AUC once leaky flags were removed — it had
been substantially built on leakage. RF/XGBoost were robust (reconstructed signal from real
features). *This is why we audit features against the target, not just trust the score.*

**Clean comparison (38 features):**

| Model | PR-AUC | Recall @0.5 |
|-------|--------|-------------|
| Logistic Regression | 0.553 | 0.66 |
| Random Forest | 0.827 | 0.59 |
| **XGBoost (chosen)** | **0.841** | **0.73** |

### Threshold selection

| Decision | Why |
|----------|-----|
| Moved threshold below 0.5 | In lending, a missed default (lose principal) costs more than a wrongly rejected good borrower (lose interest). Catching more defaulters means flagging default more readily = lower threshold. |
| Cost ratio FN:FP = 5:1 | A missed default assumed 5× costlier than a wrongful rejection. Stated assumption — a real deployment would use bank-supplied figures. |
| Swept thresholds, minimized total cost → **0.41** | Cost-optimal cutoff. Not "as low as possible" (that approves nobody) — the point where prevented expensive errors outweigh added cheap ones. |
| Accepted precision drop 0.75 → 0.56 at 0.41 | Recall rose 0.73 → 0.77 (caught ~330 more defaulters). Lower precision is acceptable because borderline cases route to Manual Review, not auto-rejection. |
| Saved `decision_threshold.pkl` (0.41) | So the scorer (07) uses the exact tuned value, not a hardcoded guess. |
| Skipped probability calibration | Polish step; decisions already work. Documented as next step under deadline. |

---

## 05_error

| Decision | Why |
|----------|-----|
| Sliced recall by credit-score band only | Depth-over-breadth under deadline; one clean finding rather than many shallow ones. |
| Binned on z-scores (`credit_score` is scaled) | The column went through StandardScaler, so values are ~ -1.7 to +1.7, not 300–850. Bins set in z-score space (`-0.6`, `0.6`) to get three balanced bands. |
| Reported result as "stable, no blind spot" | Recall was 0.77 / 0.76 / 0.79 across low/medium/high — a 3-point spread = noise, not a pattern. Did not manufacture a weakness the data doesn't support. |
| Framed stability as design validation | Uniform performance across credit tiers justifies using a single threshold for all applicants rather than tier-specific cutoffs. |

**Principle locked:** a null result reported honestly beats a dramatic result gone fishing for.

---

## Cross-cutting principles internalized

- Leakage = timing/causation, not correlation strength.
- Impute and scale fit on TRAIN only.
- Stratify to preserve class ratio.
- Accuracy paradox under imbalance → use PR-AUC + recall.
- A suspiciously perfect score (0.95+) or a missingness flag beating real features = check for a leak.
- Freeze *preferences* (scope, taste), chase *bugs* (leaks, errors).
- Threshold direction follows cost asymmetry; threshold magnitude follows the cost ratio.