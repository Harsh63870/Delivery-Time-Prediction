# Cross-Validation and Stability Report

## The question

On a single 80/20 train–test split (`random_state=42`), Linear Regression looked like the best model (holdout MAE **6.06 minutes**). That ranking could be an accident of which 200 rows landed in the test set.

This report checks that claim with **five-fold cross-validation on the training data only**. The unused 20% test set is never mixed into the folds. It is reported at the end only as a comparison with the original experiment.

**Short answer:** Linear Regression is genuinely the most reliable of the five models on this dataset. It has the lowest mean validation MAE, the lowest spread among the real predictors, and it wins **every one of the five folds**. The original split was slightly optimistic, but it did not create a false winner.

---

## 1. Why a single split is not enough

A holdout evaluation answers: *how well did this model do on this particular 200-row slice?*

It does **not** answer:

- Would another random 200-row slice give the same ranking?
- How much does error jump when the validation rows change?
- Is a model winning because it generalizes, or because the test slice happened to be easy?

Cross-validation repeats the train/validate cycle on several partitions of the **same training set**. The mean MAE is a more honest estimate of typical error. The standard deviation of MAE across folds is a direct measure of **stability**.

---

## 2. Experimental design

### 2.1 Data used for CV

| Item | Value |
|---|---|
| Dataset | `Food_Delivery_Times.csv` |
| Target | `Delivery_Time_min` |
| Dropped identifier | `Order_ID` |
| Full rows | 1,000 |
| Training rows used in CV | 800 |
| Held-out test rows (not used in CV) | 200 |
| Split | `train_test_split(test_size=0.20, random_state=42)` |

The 80/20 split matches the original pipeline so the CV result can be compared with the earlier holdout numbers.

### 2.2 Features

**Numeric:** `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`

**Categorical:** `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`

### 2.3 Preprocessing (inside each fold)

A `Pipeline` wraps preprocessing and the estimator so transformers are **fit only on that fold’s training rows**:

1. Numeric: median imputation, then `StandardScaler`
2. Categorical: most-frequent imputation, then one-hot encoding (`handle_unknown='ignore'`)

This avoids the leakage that would occur if the scaler or encoder saw the validation fold first.

### 2.4 Same folds for every model

Splitter:

```text
KFold(n_splits=5, shuffle=True, random_state=42)
```

The same `KFold` object is passed to every model. Shuffle plus a fixed seed makes the five partitions identical across Dummy, Linear Regression, Decision Tree, Random Forest, and XGBoost. Differences in MAE are therefore due to the models, not due to different validation rows.

Each fold uses 640 training rows and 160 validation rows (800 / 5).

### 2.5 Models

| Model | Role | Settings |
|---|---|---|
| Dummy Regressor | Baseline that always predicts the training-fold mean | `strategy='mean'` |
| Linear Regression | Linear additive model | sklearn default |
| Decision Tree | Fully grown tree | `random_state=42` |
| Random Forest | Bagged trees | `n_estimators=100`, `random_state=42` |
| XGBoost | Gradient-boosted trees | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42` |

Dummy and Decision Tree were not in the original three-model pipeline. They are included here because a baseline and an unregularized tree are needed to interpret whether Linear Regression is actually good, or merely “less bad than two ensembles.”

### 2.6 Metric

**MAE (mean absolute error)** in minutes, on the validation fold.

- Lower is better.
- Sample standard deviation (`ddof=1`) is used for the five fold scores, because we are describing variation among folds, not a population of infinite folds.

Script: `cross_validation.py`  
Saved tables: `output/cv_summary.csv`, `output/cv_fold_mae.csv`, `output/cv_results.json`  
Plots: `plots/cv_mae_mean_std.png`, `plots/cv_mae_per_fold.png`

---

## 3. Per-fold validation MAE

These are the five **shared** validation folds. Read down a column to see how a model moves when the validation slice changes. Read across a row to compare models on the **same** 160 rows.

| Fold | Dummy | Linear Regression | Decision Tree | Random Forest | XGBoost |
|---|---:|---:|---:|---:|---:|
| 1 | 17.1975 | **6.4011** | 11.4812 | 7.4231 | 7.2218 |
| 2 | 17.8419 | **6.4735** | 11.5000 | 7.7222 | 7.3424 |
| 3 | 18.0743 | **7.1306** | 11.7937 | 8.4769 | 8.5846 |
| 4 | 18.1295 | **7.4353** | 10.4375 | 8.6924 | 8.6027 |
| 5 | 18.3571 | **6.5972** | 9.7188 | 7.2281 | 7.5918 |

Linear Regression has the lowest MAE on **all five folds**. The original ranking was not an artifact of one lucky test set.

Folds 3 and 4 are harder for every real model (higher MAE). Dummy also rises there, which means those folds are slightly harder slices of the target, not a Linear Regression-specific failure.

---

## 4. Mean and standard deviation of validation MAE

| Model | Mean validation MAE | Std of validation MAE | Min | Max | Mean train MAE |
|---|---:|---:|---:|---:|---:|
| **Linear Regression** | **6.8075** | **0.4526** | 6.4011 | 7.4353 | 6.6437 |
| XGBoost | 7.8687 | 0.6752 | 7.2218 | 8.6027 | 2.8107 |
| Random Forest | 7.9085 | 0.6463 | 7.2281 | 8.6924 | 3.0611 |
| Decision Tree | 10.9863 | 0.8759 | 9.7188 | 11.7937 | 0.0000 |
| Dummy Regressor | 17.9200 | 0.4435 | 17.1975 | 18.3571 | 17.8843 |

### How to read the two numbers

- **Mean MAE** = typical absolute error, in minutes, when the model is trained on 4/5 of the training data and scored on the remaining fifth.
- **Std MAE** = how much that error moves when the validation fifth changes. Small std means the model’s quality is not hostage to which rows were held out.

Dummy has the smallest std (0.44) only because predicting the mean is almost equally wrong on every fold. Stability without accuracy is not useful. Among models that actually fit the data, **Linear Regression is both the most accurate and the most stable**.

---

## 5. Interpretation by model

### Dummy Regressor (baseline)

Mean validation MAE **17.92**. That is roughly the average absolute deviation of delivery time from its mean. Any serious model must beat this by a wide margin. Linear Regression does: about **11 minutes** better on average.

### Linear Regression

Mean **6.81**, std **0.45**. Train MAE (6.64) is almost the same as validation MAE (6.81). That pair is the signature of a well-specified, low-variance model: it is not memorizing the training fold.

This is consistent with the EDA. Delivery time is dominated by a nearly linear relationship with `Distance_km` (correlation ≈ 0.78), plus a smaller linear contribution from preparation time. Extra tree complexity does not buy much once those effects are captured.

### Decision Tree

Train MAE is **0.00** and validation MAE is **10.99**. The unrestricted tree memorizes the training fold and then fails on new rows. It is also the least stable real model (std 0.88). This is classic overfitting, not a data bug.

### Random Forest and XGBoost

Both beat Dummy and the single tree, but both sit about **1.0–1.1 minutes** behind Linear Regression on mean CV MAE.

They also overfit, though less than the single tree:

- Random Forest: train 3.06 vs validation 7.91
- XGBoost: train 2.81 vs validation 7.87

Default-ish tree ensembles on 800 rows, with a mostly linear signal, add variance without reducing bias. Their fold-to-fold std (0.65 and 0.68) is higher than Linear Regression’s 0.45, so they are also less stable.

---

## 6. Did the original 80/20 split create a false winner?

Holdout test MAE (same 200 rows as the original pipeline, **not** used in CV):

| Model | 5-fold mean CV MAE | Original holdout MAE | Difference (holdout − CV) |
|---|---:|---:|---:|
| Linear Regression | 6.8075 | 6.0634 | −0.74 |
| Random Forest | 7.9085 | 6.8831 | −1.03 |
| XGBoost | 7.8687 | 7.0328 | −0.84 |
| Decision Tree | 10.9863 | 11.2900 | +0.30 |
| Dummy Regressor | 17.9200 | 17.5014 | −0.42 |

Two facts matter:

1. **Ranking is unchanged.** Linear Regression still first, then the two ensembles, then Decision Tree, then Dummy.
2. **The original test set was a bit easy** for Linear Regression, Random Forest, and XGBoost (holdout MAE lower than CV mean). That is normal sampling variation. It made Linear Regression look about 0.74 minutes better than its CV estimate, but it did not invent the win. Linear Regression already leads on every CV fold, including the two harder folds (3 and 4).

So: the particular split made Linear Regression look *slightly better than it really is*, not *better than the other models by accident*.

---

## 7. Direct answer

**Is Linear Regression genuinely the most reliable model, or did it win because of the particular train–test split?**

It is genuinely the most reliable model in this comparison.

Evidence:

1. Lowest **mean** validation MAE (**6.81** minutes).
2. Lowest **std** among models that actually fit the data (**0.45**).
3. Best MAE on **all five** shared folds.
4. Almost no train/validation gap (6.64 vs 6.81), unlike the trees.
5. Same ranking as the original holdout, so the 80/20 split did not manufacture the conclusion.

Dummy is “stable” only in the weak sense that a constant predictor is consistently bad. Decision Tree is unstable and overfit. Random Forest and XGBoost are useful but strictly worse than Linear Regression here, both in accuracy and in fold-to-fold spread.

A fair next step, if model choice is still open, would be to keep Linear Regression as the default and only revisit trees after regularization or a smaller `max_depth` / `min_samples_leaf`, evaluated with the **same** five folds.

---

## 8. How to reproduce

```text
python cross_validation.py
```

Outputs:

- `output/cv_summary.csv` — mean, std, min, max, train MAE, holdout MAE
- `output/cv_fold_mae.csv` — every fold for every model
- `output/cv_results.json` — machine-readable copy of the same numbers
- `plots/cv_mae_mean_std.png` — mean MAE with ±1 std error bars
- `plots/cv_mae_per_fold.png` — MAE trajectory across the five shared folds
