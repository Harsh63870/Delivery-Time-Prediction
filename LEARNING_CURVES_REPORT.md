# Bias and Variance Analysis Through Learning Curves

## The Question

**How can we tell whether a model is too simple or too complex?**

When evaluating a machine learning model, a single performance score cannot reveal *why* a model fails or succeeds. A model might fail because it is too rigid to learn the underlying relationships (**high bias / underfitting**) or because it is so flexible that it memorizes noise and idiosyncrasies of the training sample (**high variance / overfitting**).

**Learning curves** solve this by plotting both **training error** and **cross-validation (validation) error** as a function of the amount of training data used to train the model.

In this experiment, we analyze:
1. **Linear Regression**
2. **A Shallow Decision Tree** (`max_depth=3`)
3. **An Unrestricted / Deeper Decision Tree** (`max_depth=None`)

All evaluations use the exact same 5-fold cross-validation scheme and leakage-free preprocessing pipeline established in Part 1.

---

## 1. Fundamentals: Training Error vs. Validation Error

| Metric | Definition | What It Measures | Behavioral Goal |
|---|---|---|---|
| **Training Error** | The model's error (MAE) evaluated on the exact same data points used to fit its parameters. | **Fit / Memorization capacity.** How well the model represents the data it was explicitly shown. | Should be reasonably low, but not artificially zero. |
| **Validation Error** | The model's error (MAE) evaluated on unseen hold-out folds not used during parameter fitting. | **Generalization ability.** How well the learned patterns transfer to new, independent data. | Should be minimized. This is our true benchmark of model quality. |
| **Generalization Gap** | $\text{Validation Error} - \text{Training Error}$ | **Degree of overfitting (variance penalty).** | Should be small. A large gap indicates the model cannot replicate its training success on new data. |

### The Core Difference

- **Training error is optimistic**: Because optimization algorithms minimize loss directly on the training set, training error is almost always lower than true error on unseen data.
- **Validation error is honest**: It simulates real-world deployment by evaluating predictions strictly on unseen examples.
- **The trajectory matters**: As training data size $N$ increases:
  - Training error generally rises or stays constant from very small $N$ because it becomes harder for a constrained model to fit every single observation perfectly.
  - Validation error decreases because more training examples allow the model to learn more robust, representative patterns.
  - The gap between the two curves narrows.

---

## 2. Diagnosing Underfitting and Overfitting via Learning Curves

```
      HIGH BIAS (Underfitting)                   HIGH VARIANCE (Overfitting)
   Error                                      Error
     ^                                          ^
     |    Validation Error (High)               |    Validation Error (High)
     |  ----------------------------            |  \
     |  ----------------------------            |   \-----------------------  <- Large Gap
     |    Training Error (High)                 |
     |                                          |    Training Error (Near 0)
     |                                          |  -------------------------
     +------------------------------>           +-------------------------->
                   Data Size (N)                              Data Size (N)
   Both curves plateau at high error;         Huge gap between train and validation;
   gap is small; more data does NOT help.     model memorizes noise; needs regularization.
```

### Underfitting (High Bias)
- **Visual Signature**: Both training error and validation error are **high** and converge close to each other. The generalization gap is small, but both curves plateau well above the desired error threshold.
- **Root Cause**: The hypothesis class is too constrained or simplistic to capture the true underlying functional relationship (e.g. attempting to model a non-linear relationship with a constant or a very shallow tree).
- **Effect of Adding More Data**: **Ineffective.** Once the curves have flattened, collecting additional training examples will not reduce validation error because the model fundamentally lacks representational capacity.
- **Remedy**: Increase model complexity (e.g. increase tree depth, add polynomial or interaction terms, switch to a more expressive algorithm).

### Overfitting (High Variance)
- **Visual Signature**: Training error is **extremely low** (often near zero), while validation error is **substantially higher**. A **large, persistent generalization gap** remains even as sample size increases.
- **Root Cause**: The model has too much capacity and too few constraints relative to the dataset size and noise level. It fits random fluctuations and noise in the training set rather than the genuine underlying signal.
- **Effect of Adding More Data**: **Helpful, but slow.** Increasing training data forces the model to generalize better and slowly narrows the gap, but if capacity remains unrestricted, severe overfitting will persist unless data size is massive.
- **Remedy**: Constrain or regularize the model (e.g. restrict `max_depth`, increase `min_samples_leaf`, apply $L_1/L_2$ penalties, use ensemble averaging like Random Forests).

---

## 3. Experimental Setup

To ensure strict comparability with Part 1:
- **Dataset**: `Food_Delivery_Times.csv` (1,000 total rows; 800 training rows, 200 holdout test rows).
- **Cross-Validation Scheme**: 5-Fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`), run on the 800 training rows.
- **Data Sizes Evaluated**: 10 equally spaced intervals from 10% to 100% of available fold training data ($N \in [64, 128, 192, 256, 320, 384, 448, 512, 576, 640]$).
- **Leakage Prevention**: All transformations (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fit inside each fold using `sklearn.pipeline.Pipeline`.
- **Metric**: Mean Absolute Error (MAE) in minutes on both training subsets and validation folds.

---

## 4. Empirical Results & Learning Curve Data

### 4.1 Summary of Final Model Performance (at $N = 640$ training samples)

| Model | Model Archetype | Train MAE (min) | Validation MAE (min) | Generalization Gap (min) | Diagnosis |
|---|---|---:|---:|---:|---|
| **Linear Regression** | Well-specified / Parametric | **6.6437** | **6.8075** | **0.1638** | **Optimal Fit (Low Bias, Low Variance)** |
| **Shallow Decision Tree** (`max_depth=3`) | Restricted capacity | 9.1779 | 9.5554 | 0.3774 | **Underfitting (High Bias)** |
| **Unrestricted Decision Tree** (`max_depth=None`) | Unconstrained capacity | **0.0000** | 10.9863 | **10.9863** | **Severe Overfitting (High Variance)** |

### 4.2 Full Learning Curve Progression Across Sample Sizes

The table below records mean MAE across the 5 CV folds for each sample size:

| Training Size ($N$) | Linear Reg: Train MAE | Linear Reg: Val MAE | Shallow Tree: Train MAE | Shallow Tree: Val MAE | Full Tree: Train MAE | Full Tree: Val MAE |
|---:|---:|---:|---:|---:|---:|---:|
| **64** | 5.1647 | 8.0553 | 5.7896 | 13.3528 | 0.0000 | 13.0062 |
| **128** | 6.4232 | 7.0461 | 8.1469 | 10.7397 | 0.0000 | 12.3613 |
| **192** | 7.3487 | 7.3101 | 9.2619 | 11.1980 | 0.0000 | 13.1987 |
| **256** | 6.9027 | 7.2295 | 8.9695 | 10.0470 | 0.0000 | 12.7263 |
| **320** | 6.9639 | 6.9989 | 9.0886 | 10.0086 | 0.0000 | 12.1900 |
| **384** | 6.8226 | 6.9089 | 9.2497 | 10.1035 | 0.0000 | 12.4962 |
| **448** | 6.9738 | 6.9464 | 9.4875 | 10.0467 | 0.0000 | 12.8138 |
| **512** | 6.7877 | 6.8719 | 9.2686 | 9.7683 | 0.0000 | 10.8000 |
| **576** | 6.7013 | 6.8738 | 9.2233 | 10.0052 | 0.0000 | 11.0863 |
| **640** | **6.6437** | **6.8075** | **9.1779** | **9.5554** | **0.0000** | **10.9863** |

---

## 5. Model-by-Model Analysis

### 5.1 Linear Regression: Low Bias, Low Variance (Balanced Fit)
- **Observations**:
  - At small sample size ($N=64$), train MAE is 5.16 min and validation MAE is 8.06 min (gap = 2.89 min).
  - As $N$ expands from 64 to 640, validation MAE drops steadily from 8.06 min down to **6.81 min**, while training MAE stabilizes at **6.64 min**.
  - By $N=320$, the curves have essentially converged. The final generalization gap is negligible (**0.16 minutes**).
- **Diagnosis**:
  - Linear Regression strikes the ideal balance for this problem. The underlying data generating process has strong linear relationships (notably between `Distance_km`, `Preparation_Time_min`, and delivery time).
  - The model does not suffer from high bias because its validation error is the lowest of all candidate models.
  - It does not suffer from high variance because its training and validation curves are practically touching.

### 5.2 Shallow Decision Tree (`max_depth=3`): High Bias (Underfitting)
- **Observations**:
  - At $N=64$, train MAE is 5.79 min, but validation error is 13.35 min.
  - As $N$ increases to 640, both training error (9.18 min) and validation error (9.56 min) flatten out.
  - The generalization gap shrinks to only **0.38 minutes**, showing very low variance.
  - However, **both training error and validation error are substantially worse** than Linear Regression (9.56 min vs. 6.81 min — almost 3 minutes slower per delivery!).
- **Diagnosis**:
  - **Classic Underfitting (High Bias)**. Restricting the tree to depth 3 allows at most $2^3 = 8$ terminal leaves. Dividing a 7-dimensional feature space into only 8 constant-prediction regions cannot capture the smooth, continuous impact of delivery distance and prep time.
  - Notice that between $N=256$ and $N=640$, validation error barely changes (10.05 min $\rightarrow$ 9.56 min). This confirms that **more data cannot rescue an underfit model**; only increasing model capacity can.

### 5.3 Unrestricted Decision Tree (`max_depth=None`): High Variance (Overfitting)
- **Observations**:
  - Training MAE is **0.0000 minutes** across every single sample size from $N=64$ to $N=640$. The tree creates leaves with single samples to achieve zero training error.
  - Validation MAE remains dismal across the entire curve: starting at 13.01 min and ending at **10.99 min** at $N=640$.
  - The generalization gap is **10.99 minutes** — the gap alone is larger than Linear Regression's entire error!
- **Diagnosis**:
  - **Severe Overfitting (High Variance)**. An unrestricted tree has virtually unbounded capacity. It eagerly splits on noisy, sample-specific variations rather than learning general patterns.
  - Despite having 640 training examples, validation error (10.99 min) is worse than even the high-bias shallow tree (9.56 min) and far worse than Linear Regression (6.81 min).
  - The zero training error combined with an enormous generalization gap is the textbook proof of high variance.

---

## 6. Depth vs. Complexity Analysis (The Bias-Variance Tradeoff Curve)

To verify the continuum between bias and variance, we evaluated Decision Tree regressors across depths ranging from 1 to 15 up to unrestricted (`None`):

| max_depth | Train MAE (min) | Validation MAE (min) | Generalization Gap (min) | Regime |
|---:|---:|---:|---:|---|
| **1** | 12.3297 | 12.5897 | 0.2599 | Severe Underfitting (High Bias) |
| **2** | 10.6877 | 11.1759 | 0.4883 | Underfitting (High Bias) |
| **3** | 9.1779 | 9.5554 | 0.3774 | Underfitting (High Bias) |
| **4** | 8.3642 | 9.5449 | 1.1807 | Moderate Bias |
| **5** | 7.5041 | **9.3021** | 1.7980 | **Single Tree Minimum Val Error** |
| **6** | 6.3643 | 9.4393 | 3.0750 | Variance begins dominating |
| **7** | 4.9814 | 10.1049 | 5.1235 | Overfitting |
| **8** | 3.4026 | 10.5382 | 7.1356 | Severe Overfitting |
| **10** | 1.2684 | 10.7872 | 9.5188 | Severe Overfitting |
| **12** | 0.3435 | 11.1939 | 10.8504 | Extreme Overfitting |
| **15** | 0.0249 | 11.0734 | 11.0485 | Memorization |
| **None** | **0.0000** | **10.9863** | **10.9863** | Complete Memorization (High Variance) |

### Key Takeaways from the Complexity Curve:
1. **The U-Shaped Validation Curve**: Validation error starts high at depth 1 (12.59 min), reaches a local optimum at depth 5 (9.30 min), and then rises back up to ~11.0 minutes as tree depth increases.
2. **Monotonic Training Error**: Training error strictly decreases from 12.33 min down to 0.00 min as depth increases. A lower training error does *not* indicate a superior model.
3. **Even the Best Single Tree Underperforms Linear Regression**: The best possible single tree (`max_depth=5`, Val MAE 9.30 min) is still **2.49 minutes worse** than simple Linear Regression (Val MAE 6.81 min). This confirms that axis-aligned step functions are structurally inferior to a linear model for continuous physical relationships like delivery distance and travel time.

---

## 7. Direct Answers to the Assignment Questions

### Q1: How can we tell whether a model is too simple or too complex?
- **Too Simple (Underfitting / High Bias)**:
  - Both training error and validation error are **high**.
  - The gap between training and validation error is **small**.
  - Adding more training data causes both curves to flatten out without reaching the target accuracy.
- **Too Complex (Overfitting / High Variance)**:
  - Training error is **exceptionally low** (close to zero).
  - Validation error is **much higher**, creating a **large generalization gap**.
  - The model fits noise in the training set and fails to generalize to fresh data.

### Q2: What is the difference between training error and validation error?
- **Training error** evaluates predictions on data the model already saw during training. It measures internal fit and memorization.
- **Validation error** evaluates predictions on hold-out folds the model never saw. It measures external generalization and real-world predictive utility.
- A model is only useful if its **validation error** is low; a low training error by itself is meaningless if the validation error is high.

### Q3: What is the evidence of underfitting or overfitting in our models?
- **Evidence of Underfitting in the Shallow Tree (`max_depth=3`)**:
  - Training MAE is high (9.18 min) and validation MAE is high (9.56 min).
  - The generalization gap is small (0.38 min), proving low variance, but the high asymptotic error proves the model is too simple to capture the feature interactions.
- **Evidence of Overfitting in the Unrestricted Tree (`max_depth=None`)**:
  - Training MAE is **0.0000 min** (zero error / 100% memorization).
  - Validation MAE is **10.99 min** (the worst of all real models).
  - The massive generalization gap of **10.99 minutes** is direct empirical evidence of excessive complexity.
- **Evidence of Balanced Fit in Linear Regression**:
  - Validation MAE is the lowest overall (**6.81 min**).
  - Training MAE is **6.64 min**, yielding a minimal gap of **0.16 min**.
  - The curves converge smoothly and remain stable as sample size grows.

---

## 8. Artifacts and Plot Locations

- **Execution Script**: [`learning_curves.py`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/learning_curves.py)
- **Visual Plots**:
  - [`plots/learning_curves_comparison.png`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/plots/learning_curves_comparison.png): 3-panel comparative learning curves (Linear Regression vs. Shallow Tree vs. Unrestricted Tree) with ±1 std confidence bands.
  - [`plots/learning_curves_validation_overlay.png`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/plots/learning_curves_validation_overlay.png): Direct overlay of validation MAE curves across training data sizes.
  - [`plots/tree_complexity_bias_variance.png`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/plots/tree_complexity_bias_variance.png): Model complexity curve showing training vs. validation MAE across Decision Tree depths (1 through 15 and Unrestricted).
- **Numerical Data**:
  - [`output/learning_curve_data.csv`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/output/learning_curve_data.csv): Training and validation MAE at every sample size.
  - [`output/learning_curve_summary.json`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/output/learning_curve_summary.json): Machine-readable results and final metrics.
  - [`output/tree_depth_tradeoff.csv`](file:///c:/Users/Admin/OneDrive/College%20Assignments%20and%20pdf/Semester-5/TAI/_TEMP_TAI_ZIP/_TEMP_TAI/output/tree_depth_tradeoff.csv): Full depth vs error data.

