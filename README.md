# Food Delivery Time Prediction: Technical Analysis & Modeling Report

This repository contains the end-to-end empirical investigation into predicting food delivery times from tabular order, environmental, and courier attributes (`Food_Delivery_Times.csv`).

---

## Executive Summary of Findings

Across five sequential investigations, **Linear Regression** emerged as the most reliable, robust, and generalizable model for food delivery time estimation.

| Model / Phase | Mean 5-Fold CV MAE | Std CV MAE | Holdout Test MAE ($N=200$) | Bias / Variance Diagnosis |
|---|---|---|---|---|
| **Dummy Baseline (Mean)** | 17.92 min | ± 0.44 min | 17.50 min | Extreme Underfitting (High Bias) |
| **Decision Tree (Unconstrained)** | 10.99 min | ± 0.88 min | 11.29 min | Severe Overfitting (Train MAE = 0.00 min) |
| **Shallow Decision Tree (Depth 3)** | 8.84 min | ± 0.65 min | 8.71 min | Moderate Underfitting (Constrained Splits) |
| **Random Forest (100 Trees)** | 7.91 min | ± 0.65 min | 6.88 min | Mild Overfitting (Train MAE = 3.06 min) |
| **XGBoost (Depth 5)** | 7.87 min | ± 0.68 min | 7.03 min | Mild Overfitting (Train MAE = 2.81 min) |
| **Decision Tree (Tuned, depth 5, leaf 5)** | 9.12 min | ± 0.85 min | 8.84 min | Regularized (gap 11.0 → 1.5 min) |
| **Random Forest (Tuned, depth 12, leaf 5, 100 trees)** | 7.77 min | ± 0.60 min | 6.75 min | Modest gain vs original RF (−0.14 min) |
| **XGBoost (Tuned, lr 0.05, depth 3, 100 trees)** | 7.39 min | ± 0.62 min | 6.67 min | Best tree model; still behind linear |
| **Linear Regression (Original)** | **6.81 min** | **± 0.45 min** | **6.06 min** | **Optimal Trade-off (Best Model)** |
| **Linear Regression (Engineered)** | 6.82 min | ± 0.47 min | 6.07 min | Neutral (Base features already sufficient) |

---

## Section-by-Section Synthesis

### Part 1: Cross-Validation and Stability
- **The Question:** Did Linear Regression win on the initial 80/20 train-test split purely due to random sampling luck?
- **Methodology:** 5-fold cross-validation strictly on the training set (800 rows). Preprocessing (median imputation, standard scaling, and one-hot encoding) was executed inside each fold pipeline to eliminate data leakage. Identical folds were shared across all estimators.
- **Result:** Linear Regression was the top performer on **every single fold** (Fold MAEs: 6.40m, 6.74m, 6.46m, 7.44m, 6.99m), with the lowest mean error (**6.81 min**) and the tightest standard deviation (**±0.45 min**). Its victory is genuine and statistically stable.

### Part 2: Bias and Variance Through Learning Curves
- **The Question:** How can we tell whether a model is too simple (underfitting) or too complex (overfitting)?
- **Methodology:** Empirical learning curves evaluated training error versus 5-fold validation error across sample slices from 10% to 100% of the training set.
- **Findings:**
  - **Unrestricted Decision Tree:** Training error remained exactly 0.00 min across all training sizes, while validation error plateaued near ~11.0 min. The vast, persistent gap is the textbook hallmark of **high variance (severe overfitting)**.
  - **Linear Regression:** Training MAE (6.64 min) and validation MAE (6.81 min) converged rapidly with an ultra-narrow gap ($\approx 0.17\text{ min}$). It avoids memorization and extracts genuine data signal.
  - **Tree Depth Trade-off:** Systematically sweeping tree depth confirmed that shallow trees (depth 3–5) underfit by missing feature interactions, while trees beyond depth 8 memorize training noise without validation gains.

### Part 3: Feature Engineering
- **The Question:** Can domain-informed transformations improve predictions?
- **Features Created:**
  1. `Distance_x_Traffic`: Distance multiplied by an ordinal traffic factor ($1, 2, 3$).
  2. `Prep_plus_Travel_Proxy`: Preparation time plus estimated travel time based on vehicle base speeds (Bike: 12 km/h, Scooter: 15 km/h, Car: 20 km/h) modulated by traffic speed factors.
- **Result:** Linear Regression showed negligible difference (**6.81 min → 6.82 min**), and XGBoost deteriorated slightly (7.87 min → 7.99 min). Tree models captured tiny gains (Tree: -0.36 min, RF: -0.15 min). Because the underlying data generator is largely linear-additive, adding manual interactions provided no substantial new information over the raw features.

### Part 4: Residual and Error Analysis
- **The Question:** Which types of deliveries does the best model predict poorly?
- **Residual Diagnostics:**
  - **Actual vs. Predicted:** Strong alignment along $y = x$ from 30 to 70 minutes; underpredicts long-tail delays (80–140 min).
  - **Residuals vs. Fitted:** Variance expands at higher predicted delivery times (heteroscedasticity).
- **Subgroup Findings:**
  - **Distance Tier:** Error strictly escalates with route length:
    - Short (<7 km): **MAE = 4.08 min** ($N=68$)
    - Medium (7–14 km): **MAE = 5.71 min** ($N=73$)
    - Long (>14 km): **MAE = 8.78 min** ($N=59$)
  - **Traffic Conditions:** Error jumps significantly in High traffic (**MAE = 8.39 min**, RMSE = 13.84 min) compared to Medium traffic (**MAE = 5.01 min**).
- **Extreme Failure Analysis:**
  - **Severe Underpredictions (Actual >> Predicted):** The largest positive residuals (+30 to +53 minutes) occur when **High Traffic** coincides with **Adverse Weather (Foggy, Snowy, Windy)** and **Two-Wheeled Vehicles (Bicycle/Scooter)**. The model's linear addition underestimates the compounding friction of bad weather and heavy traffic on vulnerable couriers.
  - **Severe Overpredictions (Actual << Predicted):** Occur at long distances (18–20 km) with **Low Traffic**, where highway travel allows couriers to arrive 15–20 minutes faster than the average per-km linear rate.

### Part 5: Controlled Hyperparameter Tuning
- **The Question:** Can we improve Decision Tree, Random Forest, and XGBoost systematically (small grid + the same 5 folds) rather than by trial and error?
- **Search:** DT `max_depth` × `min_samples_leaf` (16 cells). RF same two knobs with `n_estimators` **fixed at 100** from the original experiment (12 cells). XGBoost `learning_rate` × `n_estimators` × the previously studied `max_depth` (27 cells).
- **Result:** All three improve. Decision Tree **10.99 → 9.12** min, XGBoost **7.87 → 7.39** min, Random Forest **7.91 → 7.77** min. The selected settings are more regularized (shallower trees, larger leaves, slower boosting). Linear Regression (**6.81 min**) still wins; tuning shrinks tree overfitting but does not change the model-family ranking.

### Part 6: Final Model Selection, Evaluation, and Prediction Demo
- **The Question:** Which model should be selected for production, how does it perform on the untouched test set, and how does it handle normal, long-distance, and edge-case orders?
- **Model Selected:** **Linear Regression Pipeline** (lowest CV MAE 6.81 min, tightest CV std ±0.45 min, minimal generalization gap 0.16 min).
- **Refit:** Retrained the full preprocessor + estimator pipeline on all 800 training observations.
- **Untouched Test Evaluation ($N=200$):**
  - **MAE:** **6.06 min** | **RMSE:** **8.95 min** | **$R^2$:** **0.821** | **Median AE:** **4.37 min**
  - **SLA Reliability:** **55.0%** within ±5 min, **82.5%** within ±10 min, **94.0%** within ±15 min.
  - **Overfitting check:** Test MAE (6.06 min) < Train MAE (6.65 min), confirming robust out-of-sample generalization.
- **Prediction Demo Scenarios:**
  1. *Normal Order* (5.5 km, 15m prep, 5y exp, Clear, Med traffic, Scooter): **39.52 min** (68% band: [33.5, 45.6] min).
  2. *Long-Distance Order* (18.5 km, 25m prep, 3y exp, Windy, High traffic, Car): **98.30 min** (68% band: [92.2, 104.4] min).
  3. *Unusual Combination Stress Test* (14.0 km, 28m prep, 0.5y rookie, Snowy, High traffic, Bike, Night): **94.19 min** (68% band: [88.1, 100.3] min).
- **Critical Finding on Edge Case:** The linear model's additive structure underpredicts severe compound disruptions (snow + gridlock + bicycle over 14 km), which realistically require 130–160+ minutes. Recommended automated dispatch guards to restrict bicycle routing in adverse weather.

---

## Project Structure & Deliverables

```
_TEMP_TAI/
├── Food_Delivery_Times.csv              # Primary dataset (1,000 orders)
│
├── cross_validation.py                  # Part 1: 5-Fold Cross-Validation implementation
├── CROSS_VALIDATION_REPORT.md           # Part 1: Comprehensive analytical report
│
├── learning_curves.py                   # Part 2: Bias-Variance & Learning Curves script
├── LEARNING_CURVES_REPORT.md            # Part 2: Detailed Learning Curves report
│
├── feature_engineering.py               # Part 3: Feature transforms & pipeline comparison
├── FEATURE_ENGINEERING_REPORT.md        # Part 3: Feature Engineering experimental report
│
├── residual_analysis.py                 # Part 4: Residuals, subgroup MAE & outlier diagnostics
├── RESIDUAL_ANALYSIS_REPORT.md          # Part 4: Full Residual & Error Analysis report
│
├── model_tuning.py                      # Part 5: Small-grid CV hyperparameter search
├── MODEL_TUNING_REPORT.md               # Part 5: Original vs tuned comparison report
│
├── final_model_demo.py                  # Part 6: Final pipeline refit, test evaluation & demo
├── FINAL_MODEL_REPORT.md                # Part 6: Complete final model & prediction report
│
├── plots/                               # Generated high-resolution diagnostic charts
│   ├── actual_vs_predicted.png
│   ├── cv_mae_mean_std.png
│   ├── cv_mae_per_fold.png
│   ├── learning_curves_comparison.png
│   ├── learning_curves_validation_overlay.png
│   ├── tree_complexity_bias_variance.png
│   ├── feature_engineering_cv_mae.png
│   ├── residual_actual_vs_predicted.png
│   ├── residual_residuals_vs_predicted.png
│   ├── residual_mae_by_subgroups.png
│   ├── residual_comprehensive_diagnostics.png
│   ├── tuning_original_vs_tuned.png
│   ├── tuning_dt_heatmap.png
│   ├── tuning_rf_heatmap.png
│   ├── tuning_xgb_heatmaps.png
│   ├── final_model_actual_vs_predicted_test.png
│   ├── final_model_coefficients_importance.png
│   ├── final_model_scenario_waterfall.png
│   └── final_model_error_distribution.png
│
├── output/                              # Structured CSV and JSON experimental outputs
│   ├── cv_summary.csv
│   ├── cv_fold_mae.csv
│   ├── learning_curve_data.csv
│   ├── feature_engineering_comparison.csv
│   ├── residual_mae_by_distance_test.csv
│   ├── residual_mae_by_traffic_test.csv
│   ├── residual_analysis_results.json
│   ├── tuning_comparison.csv
│   ├── tuning_grid_decision_tree.csv
│   ├── tuning_grid_random_forest.csv
│   ├── tuning_grid_xgboost.csv
│   ├── tuning_results.json
│   ├── final_model_test_metrics.json
│   ├── final_model_test_metrics.csv
│   ├── final_model_coefficients.csv
│   ├── final_model_test_predictions.csv
│   └── final_model_scenario_predictions.csv
│
└── README.md                            # Complete executive summary and project brief
```

---

## How to Reproduce

All scripts are self-contained and run in standard Python 3.10+ environments with `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `xgboost`.

```bash
cd _TEMP_TAI

# Part 1: Cross-Validation
python cross_validation.py

# Part 2: Bias-Variance and Learning Curves
python learning_curves.py

# Part 3: Feature Engineering
python feature_engineering.py

# Part 4: Residual and Error Analysis
python residual_analysis.py

# Part 5: Controlled Hyperparameter Tuning
python model_tuning.py

# Part 6: Final Model Selection, Test Evaluation & Prediction Demo
python final_model_demo.py
```

