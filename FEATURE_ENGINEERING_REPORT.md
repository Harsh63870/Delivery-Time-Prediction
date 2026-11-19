# Feature Engineering Report

## The question

Can one or two simple, meaningful transformations improve delivery-time predictions?

This experiment adds two features that have a physical story, then compares that feature set with the original one. The comparison uses **the same five training-set folds** as the stability study, so a change in MAE is due to the features, not a different split.

**Short answer:** not in a way that matters. Linear Regression, still the best model, does not improve (mean CV MAE **6.81 → 6.82**). Random Forest and Decision Tree pick up small, noisy gains. XGBoost gets slightly worse. The original features already capture the same information these transforms try to spell out.

---

## 1. What stayed fixed

| Item | Value |
|---|---|
| Training rows | 800 (20% holdout never used) |
| Splitter | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| Models | Dummy, Linear Regression, Decision Tree, Random Forest, XGBoost |
| Metric | Validation MAE (minutes) |
| Preprocessing | Median impute + scale numerics; most-frequent impute + one-hot categoricals, **fit inside each fold** |

The engineered columns are created by a transformer that is also **fit inside each fold** (only the traffic-mode fill is learned; speeds are not learned from the target).

---

## 2. Original feature set

**Numeric:** `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`

**Categorical:** `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`

Linear Regression already gets a different intercept for each traffic level through one-hot encoding. What it does **not** get is a different *slope* of distance for each traffic level. That is the gap the first new feature is meant to fill.

---

## 3. Engineered feature set

The original columns are kept. Two columns are added.

### 3.1 `Distance_x_Traffic`

**Idea:** extra distance should cost more time when traffic is heavier.

Traffic is mapped to an ordinal scale, then multiplied by distance:

| Traffic_Level | Ordinal |
|---|---:|
| Low | 1 |
| Medium | 2 |
| High | 3 |

\[
\texttt{Distance\_x\_Traffic} = \texttt{Distance\_km} \times \text{ordinal}(\texttt{Traffic\_Level})
\]

Missing `Traffic_Level` is filled with the **training-fold mode** (usually Medium). The target is never used.

**Why this is a restricted interaction.** The product assumes the distance slope rises in equal steps Low → Medium → High. That is simpler than three separate `Distance × dummy` terms. On the training data, distance still correlates strongly with delivery time in every traffic band (about 0.78–0.82), and minutes per kilometre rise from Low to High, so the direction of the feature is justified. The equal-step assumption is an approximation.

### 3.2 `Prep_plus_Travel_Proxy`

**Idea:** observed delivery time is roughly kitchen time plus on-road time. A proxy for that sum can be built without looking at the target.

\[
\texttt{Prep\_plus\_Travel\_Proxy}
= \texttt{Preparation\_Time\_min}
+ \texttt{Distance\_km} \times \text{base min/km}(\text{vehicle}) \times \text{traffic multiplier}
\]

**Domain speeds (chosen a priori, not fit on `Delivery_Time_min`):**

| Vehicle | Assumed medium-traffic speed | Base min/km |
|---|---|---:|
| Bike | ~12 km/h | 5.0 |
| Scooter | ~15 km/h | 4.0 |
| Car | ~20 km/h | 3.0 |

| Traffic | Multiplier | Meaning |
|---|---:|---|
| Low | 0.80 | about 20% faster than medium |
| Medium | 1.00 | reference |
| High | 1.40 | about 40% slower than medium |

Example: a 10 km scooter trip in high traffic  
`10 × 4.0 × 1.40 = 56` minutes of travel proxy, plus the recorded preparation time.

**Assumptions (and where they can fail):**

1. Total time is **additive**: prep and travel do not overlap. If the courier waits in traffic after the food is ready, this is reasonable; if the kitchen is slow while the courier is already moving, it is not.
2. Travel time is **linear in distance**. No congestion “fixed delay,” no highway vs last-mile mix.
3. Vehicle changes **base speed**, traffic **scales** that speed. Interaction of vehicle and traffic is only a product, not a richer table.
4. Speeds are **urban food-delivery guesses**, not GPS measurements. They were **not** estimated from this dataset’s target, on purpose, so the feature is not a leaked copy of `y`.
5. Missing traffic uses the fold mode; missing vehicle would fall back to Scooter (none are missing here).

These speeds are close to, but not copied from, training-set minutes-per-km medians (Low ≈ 3.5, Medium ≈ 4.2, High ≈ 4.6). Using the medians themselves would have mixed the target into the feature.

Vehicle is in the proxy so the second feature is not a linear clone of `Distance_x_Traffic`. If the proxy had used only `3 / 4 / 5 min per km` by traffic, it would have been `2 × Distance + Distance_x_Traffic`, which adds nothing new for Linear Regression.

---

## 4. Results on the same five folds

Negative delta = engineered set is better (lower MAE).

| Model | Mean MAE original | Std original | Mean MAE engineered | Std engineered | Δ mean MAE |
|---|---:|---:|---:|---:|---:|
| Dummy Regressor | 17.9200 | 0.4435 | 17.9200 | 0.4435 | 0.0000 |
| **Linear Regression** | **6.8075** | **0.4526** | **6.8198** | 0.4653 | **+0.0123** |
| Decision Tree | 10.9863 | 0.8759 | 10.6300 | 1.3468 | −0.3563 |
| Random Forest | 7.9085 | 0.6463 | 7.7583 | 0.7350 | −0.1502 |
| XGBoost | 7.8687 | 0.6752 | 7.9861 | 0.7851 | +0.1174 |

Dummy is unchanged. That is the control: a mean predictor ignores features, so if Dummy had moved, the folds would not have been the same.

### Linear Regression, fold by fold

| Fold | Original | Engineered |
|---|---:|---:|
| 1 | 6.4011 | 6.4166 |
| 2 | 6.4735 | 6.5029 |
| 3 | 7.1306 | 7.1176 |
| 4 | 7.4353 | 7.4921 |
| 5 | 6.5972 | 6.5700 |

Three folds slightly worse, two slightly better. The net is a **0.01 minute** loss — noise, not a real improvement.

Linear Regression remains first under both feature sets (engineered RF is still 7.76).

---

## 5. Why the “meaningful” features barely helped

**Linear Regression already had the main effects.** Distance, prep time, traffic dummies, and vehicle dummies are in the original matrix. The ordinal interaction only helps if the *distance slope* changes a lot by traffic. In this sample it does not change much (correlations stay high in every band). The travel proxy is a hand-weighted mix of distance, vehicle, and traffic — columns the linear model can already combine. Extra collinearity can even nudge MAE the wrong way, which is what +0.012 looks like.

**Trees can use the composite as a shortcut.** Decision Tree mean MAE fell by 0.36 minutes, Random Forest by 0.15. Both still overfit (tree train MAE remains 0.00). Decision Tree’s fold std **rose** from 0.88 to 1.35: fold 1 improved a lot (11.48 → 8.73), fold 3 got worse (11.79 → 12.00). That is not stable structure; it is a greedy tree latching onto a new split. XGBoost, which already builds its own interactions, got slightly worse and less stable.

**Best model did not change.** Even the friendliest reading (RF improved) leaves Linear Regression on the original features ahead by about a minute.

---

## 6. Direct answer

Simple, meaningful transforms **did not improve the model we should ship.**

- `Distance_x_Traffic` is a fair test of “does extra kilometre hurt more in heavy traffic?” The equal-step interaction is too weak to beat one-hot traffic plus distance.
- `Prep_plus_Travel_Proxy` is a defensible additive story, but its ingredients were already in the table. Using domain speeds kept the feature honest (no target leakage); it also kept it from being a second copy of `y`.
- On the **same folds**, Linear Regression is essentially unchanged and still the most accurate and most stable model.

Keep the original feature set. If a later experiment wants a richer interaction, the next honest test is `Distance_km ×` each traffic dummy (two extra slopes), still evaluated on these five folds — not a larger pile of unmotivated transforms.

---

## 7. How to reproduce

```text
python feature_engineering.py
```

Outputs:

- `output/feature_engineering_comparison.csv` — mean/std MAE and delta
- `output/feature_engineering_fold_mae.csv` — every fold, both feature sets
- `output/feature_engineering_results.json`
- `plots/feature_engineering_cv_mae.png`
