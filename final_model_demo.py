"""
Part 6: Final Model Selection, Full Retraining, Untouched Test Evaluation,
and Prediction Demo on Real-World and Edge Scenarios.

This script implements:
1. Formal model selection based on 5-fold cross-validation results from Parts 1-5.
2. Refitting the selected pipeline (Linear Regression) on the full training dataset (N=800).
3. Extracting and interpreting both standardized and physical model coefficients.
4. Single evaluation on the untouched holdout test set (N=200) with full metrics.
5. Testing three distinct scenarios:
   - Scenario 1: Normal / typical urban dinner order.
   - Scenario 2: Long-distance suburban transit order.
   - Scenario 3: Unusual combination of inputs (severe edge case).
6. Generating publication-quality diagnostic plots and structured tabular outputs.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set aesthetic style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300
palette = sns.color_palette("deep")

os.makedirs("plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "Delivery_Time_min"

# ==============================================================================
# 1. LOAD DATASET & REPRODUCE IDENTICAL SPLIT
# ==============================================================================
df = pd.read_csv("Food_Delivery_Times.csv")

feature_num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
feature_cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]

if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Identical train/test split as Parts 1 to 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Dataset Loaded: Total={len(df)}, Train={len(X_train)}, Test={len(X_test)}")

# ==============================================================================
# 2. DEFINE AND REFIT THE CHOSEN PIPELINE ON FULL TRAINING DATA
# ==============================================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_num_cols),
        ("cat", categorical_transformer, feature_cat_cols),
    ]
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

# Refit on the full 800-instance training set
final_pipeline.fit(X_train, y_train)
print("Chosen Linear Regression pipeline successfully refit on full training dataset (N=800).")

# ==============================================================================
# 3. EXTRACT AND INTERPRET MODEL COEFFICIENTS
# ==============================================================================
model_step = final_pipeline.named_steps["model"]
prep_step = final_pipeline.named_steps["preprocessor"]

cat_encoder = prep_step.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = cat_encoder.get_feature_names_out(feature_cat_cols)
all_feature_names = feature_num_cols + list(cat_feature_names)

coef_scaled = model_step.coef_
intercept = model_step.intercept_

# Unscale numerical coefficients to derive physical interpretation
num_scaler = prep_step.named_transformers_["num"].named_steps["scaler"]
num_scale_means = num_scaler.mean_
num_scale_stds = num_scaler.scale_

coef_unscaled = []
for i, name in enumerate(all_feature_names):
    if i < len(feature_num_cols):
        # unscaled coef = scaled_coef / std
        unscaled = coef_scaled[i] / num_scale_stds[i]
        coef_unscaled.append(unscaled)
    else:
        # one-hot coefficients directly represent delta in minutes
        coef_unscaled.append(coef_scaled[i])

coef_df = pd.DataFrame(
    {
        "Feature": all_feature_names,
        "Scaled_Coefficient": coef_scaled,
        "Physical_Coefficient": coef_unscaled,
        "Abs_Scaled_Importance": np.abs(coef_scaled),
    }
).sort_values(by="Abs_Scaled_Importance", ascending=False)

coef_df.to_csv("output/final_model_coefficients.csv", index=False)
print("\nTop 8 Feature Coefficients (by standardized impact):")
print(coef_df.head(8).to_string(index=False))

# ==============================================================================
# 4. SINGLE EVALUATION ON THE UNTOUCHED TEST SET
# ==============================================================================
y_train_pred = final_pipeline.predict(X_train)
y_test_pred = final_pipeline.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_medae = median_absolute_error(y_test, y_test_pred)
test_evs = explained_variance_score(y_test, y_test_pred)

test_residuals = y_test - y_test_pred
test_abs_residuals = np.abs(test_residuals)
max_error = float(np.max(test_abs_residuals))

pct_within_5 = float(np.mean(test_abs_residuals <= 5.0) * 100)
pct_within_10 = float(np.mean(test_abs_residuals <= 10.0) * 100)
pct_within_15 = float(np.mean(test_abs_residuals <= 15.0) * 100)

test_metrics = {
    "Model": "Linear Regression (Final Production Pipeline)",
    "Train_Samples": len(X_train),
    "Test_Samples": len(X_test),
    "Train_MAE_min": round(train_mae, 4),
    "Train_RMSE_min": round(train_rmse, 4),
    "Train_R2": round(train_r2, 4),
    "5Fold_CV_MAE_Mean_min": 6.8075,
    "5Fold_CV_MAE_Std_min": 0.4526,
    "Test_MAE_min": round(test_mae, 4),
    "Test_RMSE_min": round(test_rmse, 4),
    "Test_MSE_min2": round(test_mse, 4),
    "Test_R2": round(test_r2, 4),
    "Test_MedAE_min": round(test_medae, 4),
    "Test_Explained_Variance": round(test_evs, 4),
    "Max_Absolute_Error_min": round(max_error, 4),
    "Generalization_Gap_MAE_min": round(test_mae - train_mae, 4),
    "Pct_Within_5min": round(pct_within_5, 2),
    "Pct_Within_10min": round(pct_within_10, 2),
    "Pct_Within_15min": round(pct_within_15, 2),
    "Intercept_Minutes": round(float(intercept), 4),
}

with open("output/final_model_test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv("output/final_model_test_metrics.csv", index=False)

# Save test predictions details
test_pred_df = X_test.copy()
test_pred_df["Actual_Delivery_Time"] = y_test
test_pred_df["Predicted_Delivery_Time"] = y_test_pred
test_pred_df["Residual"] = test_residuals
test_pred_df["Absolute_Error"] = test_abs_residuals
test_pred_df.to_csv("output/final_model_test_predictions.csv", index=False)

print("\n--- UNTOUCHED TEST SET EVALUATION METRICS ---")
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

# ==============================================================================
# 5. PREDICTION DEMO: THREE DISTINCT ORDERS
# ==============================================================================
# Case 1: Normal Order
# Case 2: Long-Distance Order
# Case 3: Unusual Combination of Inputs
demo_orders = pd.DataFrame(
    [
        {
            "Scenario": "1. Normal Order",
            "Distance_km": 5.5,
            "Weather": "Clear",
            "Traffic_Level": "Medium",
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Scooter",
            "Preparation_Time_min": 15.0,
            "Courier_Experience_yrs": 5.0,
            "Description": "Standard urban dinner order, moderate distance, smooth conditions, scooter courier.",
        },
        {
            "Scenario": "2. Long-Distance Order",
            "Distance_km": 18.5,
            "Weather": "Windy",
            "Traffic_Level": "High",
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Car",
            "Preparation_Time_min": 25.0,
            "Courier_Experience_yrs": 3.0,
            "Description": "Suburban perimeter delivery near outer limit (18.5 km), rush hour, car transit, large order.",
        },
        {
            "Scenario": "3. Unusual Combination",
            "Distance_km": 14.0,
            "Weather": "Snowy",
            "Traffic_Level": "High",
            "Time_of_Day": "Night",
            "Vehicle_Type": "Bike",
            "Preparation_Time_min": 28.0,
            "Courier_Experience_yrs": 0.5,
            "Description": "Severe stress test: rookie courier on bicycle in heavy snow & gridlock over 14 km at night.",
        },
    ]
)

# Predict demo orders
demo_features = demo_orders.drop(columns=["Scenario", "Description"])
demo_preds = final_pipeline.predict(demo_features)

demo_orders["Predicted_Time_min"] = np.round(demo_preds, 2)
demo_orders["Lower_Bound_68pct"] = np.round(demo_preds - test_mae, 2)
demo_orders["Upper_Bound_68pct"] = np.round(demo_preds + test_mae, 2)
demo_orders["Lower_Bound_95pct"] = np.round(demo_preds - 1.96 * test_mae, 2)
demo_orders["Upper_Bound_95pct"] = np.round(demo_preds + 1.96 * test_mae, 2)

# Decompose predictions into component contributions:
# y_hat = intercept + sum_j (coef_scaled_j * X_prep_j)
X_demo_prep = prep_step.transform(demo_features)
contributions = []

for i, row in demo_orders.iterrows():
    x_prep_row = X_demo_prep[i]
    feat_contribs = {"Scenario": row["Scenario"], "Intercept": round(intercept, 2)}
    
    # Distance
    dist_val = x_prep_row[feature_num_cols.index("Distance_km")] * coef_scaled[feature_num_cols.index("Distance_km")]
    feat_contribs["Distance_Effect"] = round(dist_val, 2)
    
    # Prep Time
    prep_val = x_prep_row[feature_num_cols.index("Preparation_Time_min")] * coef_scaled[feature_num_cols.index("Preparation_Time_min")]
    feat_contribs["Prep_Time_Effect"] = round(prep_val, 2)
    
    # Experience
    exp_val = x_prep_row[feature_num_cols.index("Courier_Experience_yrs")] * coef_scaled[feature_num_cols.index("Courier_Experience_yrs")]
    feat_contribs["Experience_Effect"] = round(exp_val, 2)
    
    # Weather
    weather_idx = [j for j, f in enumerate(all_feature_names) if f.startswith("Weather_") and x_prep_row[j] == 1]
    feat_contribs["Weather_Effect"] = round(coef_scaled[weather_idx[0]], 2) if weather_idx else 0.0
    
    # Traffic
    traffic_idx = [j for j, f in enumerate(all_feature_names) if f.startswith("Traffic_Level_") and x_prep_row[j] == 1]
    feat_contribs["Traffic_Effect"] = round(coef_scaled[traffic_idx[0]], 2) if traffic_idx else 0.0
    
    # Time of Day
    time_idx = [j for j, f in enumerate(all_feature_names) if f.startswith("Time_of_Day_") and x_prep_row[j] == 1]
    feat_contribs["Time_of_Day_Effect"] = round(coef_scaled[time_idx[0]], 2) if time_idx else 0.0
    
    # Vehicle
    veh_idx = [j for j, f in enumerate(all_feature_names) if f.startswith("Vehicle_Type_") and x_prep_row[j] == 1]
    feat_contribs["Vehicle_Effect"] = round(coef_scaled[veh_idx[0]], 2) if veh_idx else 0.0
    
    feat_contribs["Total_Sum"] = round(sum(v for k, v in feat_contribs.items() if k != "Scenario"), 2)
    contributions.append(feat_contribs)

contrib_df = pd.DataFrame(contributions)
demo_results_df = demo_orders.merge(contrib_df, on="Scenario")
demo_results_df.to_csv("output/final_model_scenario_predictions.csv", index=False)

print("\n--- PREDICTION DEMO RESULTS ---")
for _, r in demo_orders.iterrows():
    print(f"\n[{r['Scenario']}]")
    print(f"  Description: {r['Description']}")
    print(f"  Inputs: Dist={r['Distance_km']}km, Prep={r['Preparation_Time_min']}m, Exp={r['Courier_Experience_yrs']}y, Weather={r['Weather']}, Traffic={r['Traffic_Level']}, Vehicle={r['Vehicle_Type']}, Time={r['Time_of_Day']}")
    print(f"  Point Prediction: {r['Predicted_Time_min']} min")
    print(f"  68% Band (±1 MAE): [{r['Lower_Bound_68pct']} - {r['Upper_Bound_68pct']}] min")
    print(f"  95% Band (±1.96 MAE): [{r['Lower_Bound_95pct']} - {r['Upper_Bound_95pct']}] min")

# ==============================================================================
# 6. PLOTTING: DIAGNOSTICS AND PREDICTION DEMO VISUALIZATIONS
# ==============================================================================

# PLOT 1: Actual vs Predicted on Untouched Holdout Test Set
fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(
    y_test,
    y_test_pred,
    c=test_abs_residuals,
    cmap="magma_r",
    alpha=0.85,
    edgecolors="w",
    s=55,
    zorder=3,
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Absolute Error (minutes)", fontsize=11)

lims = [0, 160]
ax.plot(lims, lims, "r--", linewidth=2, label="Ideal (y = x)", zorder=4)
ax.fill_between(lims, [l - 5 for l in lims], [l + 5 for l in lims], color="green", alpha=0.12, label="Within ±5 min")
ax.fill_between(lims, [l - 10 for l in lims], [l + 10 for l in lims], color="blue", alpha=0.06, label="Within ±10 min")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Delivery Time (min)", fontsize=12, fontweight="bold")
ax.set_ylabel("Predicted Delivery Time (min)", fontsize=12, fontweight="bold")
ax.set_title(
    "Final Pipeline Evaluation on Untouched Test Set (N=200)\nModel: Linear Regression",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

stats_text = (
    f"Test MAE: {test_mae:.2f} min\n"
    f"Test RMSE: {test_rmse:.2f} min\n"
    f"Test R²: {test_r2:.3f}\n"
    f"Median AE: {test_medae:.2f} min\n"
    f"Within ±5m: {pct_within_5:.1f}%\n"
    f"Within ±10m: {pct_within_10:.1f}%"
)
ax.text(
    0.05,
    0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=10.5,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.9),
)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()
plt.savefig("plots/final_model_actual_vs_predicted_test.png", dpi=300)
plt.close()
print("Saved plots/final_model_actual_vs_predicted_test.png")

# PLOT 2: Feature Importance & Coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot A: Standardized Feature Importances
top_scaled = coef_df.sort_values(by="Abs_Scaled_Importance", ascending=True)
colors = ["#2b5c8f" if c >= 0 else "#c44e52" for c in top_scaled["Scaled_Coefficient"]]
bars1 = ax1.barh(top_scaled["Feature"], top_scaled["Scaled_Coefficient"], color=colors, edgecolor="black", alpha=0.85)
ax1.axvline(0, color="black", linestyle="--", linewidth=1)
ax1.set_xlabel("Standardized Coefficient (Minutes / 1 Std Dev)", fontsize=11, fontweight="bold")
ax1.set_title("Standardized Feature Coefficients\n(Global Relative Effect)", fontsize=12, fontweight="bold")
for bar, val in zip(bars1, top_scaled["Scaled_Coefficient"]):
    ax1.text(
        val + (0.2 if val >= 0 else -0.3),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=9,
        fontweight="bold",
    )

# Subplot B: Physical Intuitive Effects (Unscaled)
phys_features = [
    ("Distance_km", "Distance (per km)"),
    ("Preparation_Time_min", "Prep Time (per min)"),
    ("Courier_Experience_yrs", "Experience (per yr)"),
    ("Traffic_Level_High", "Traffic: High (vs base)"),
    ("Traffic_Level_Medium", "Traffic: Medium (vs base)"),
    ("Weather_Snowy", "Weather: Snowy (vs base)"),
    ("Weather_Rainy", "Weather: Rainy (vs base)"),
    ("Weather_Clear", "Weather: Clear (vs base)"),
    ("Vehicle_Type_Bike", "Vehicle: Bike (vs base)"),
    ("Vehicle_Type_Car", "Vehicle: Car (vs base)"),
]

phys_df = []
for f_key, label in phys_features:
    row = coef_df[coef_df["Feature"] == f_key]
    if not row.empty:
        phys_df.append({"Label": label, "Val": row["Physical_Coefficient"].values[0]})
phys_df = pd.DataFrame(phys_df).sort_values(by="Val", ascending=True)

colors2 = ["#2b5c8f" if c >= 0 else "#c44e52" for c in phys_df["Val"]]
bars2 = ax2.barh(phys_df["Label"], phys_df["Val"], color=colors2, edgecolor="black", alpha=0.85)
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.set_xlabel("Physical Impact on Delivery Time (Minutes)", fontsize=11, fontweight="bold")
ax2.set_title("Physical Interpretation of Key Drivers\n(Minutes Added / Subtracted)", fontsize=12, fontweight="bold")
for bar, val in zip(bars2, phys_df["Val"]):
    ax2.text(
        val + (0.15 if val >= 0 else -0.2),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}m",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("plots/final_model_coefficients_importance.png", dpi=300)
plt.close()
print("Saved plots/final_model_coefficients_importance.png")

# PLOT 3: Prediction Demo Scenario Breakdown
fig, ax = plt.subplots(figsize=(10, 6.5))

categories = [
    "Distance_Effect",
    "Prep_Time_Effect",
    "Experience_Effect",
    "Weather_Effect",
    "Traffic_Effect",
    "Vehicle_Effect",
    "Time_of_Day_Effect",
]
cat_labels = [
    "Distance",
    "Prep Time",
    "Courier Exp",
    "Weather",
    "Traffic",
    "Vehicle",
    "Time of Day",
]

bar_width = 0.25
x_indices = np.arange(len(categories))

colors_scenario = ["#4C72B0", "#DD8452", "#55A868"]

for idx, row in contrib_df.iterrows():
    vals = [row[c] for c in categories]
    rects = ax.bar(
        x_indices + (idx - 1) * bar_width,
        vals,
        bar_width,
        label=f"{row['Scenario']}: Total {row['Total_Sum']} min",
        color=colors_scenario[idx],
        edgecolor="black",
        alpha=0.88,
    )
    for r in rects:
        h = r.get_height()
        ax.annotate(
            f"{h:+.1f}",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3 if h >= 0 else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x_indices)
ax.set_xticklabels(cat_labels, fontsize=10.5, fontweight="bold")
ax.set_ylabel("Contribution to Delivery Time (Minutes)", fontsize=11, fontweight="bold")
ax.set_title(
    f"Prediction Breakdown Across Demo Scenarios (Base Intercept = {intercept:.1f} min)",
    fontsize=12.5,
    fontweight="bold",
    pad=12,
)
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
plt.tight_layout()
plt.savefig("plots/final_model_scenario_waterfall.png", dpi=300)
plt.close()
print("Saved plots/final_model_scenario_waterfall.png")

# PLOT 4: Residual Distribution on Test Set
fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5.5))

# Residual Histogram & KDE
sns.histplot(test_residuals, kde=True, color="#4C72B0", ax=ax_hist, bins=25, edgecolor="black")
ax_hist.axvline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Bias Line")
ax_hist.axvline(test_residuals.mean(), color="black", linestyle=":", linewidth=1.5, label=f"Mean Error: {test_residuals.mean():+.2f}m")
ax_hist.set_xlabel("Residual (Actual - Predicted) [min]", fontsize=11, fontweight="bold")
ax_hist.set_ylabel("Frequency Count", fontsize=11, fontweight="bold")
ax_hist.set_title("Test Set Residual Distribution\n(Untouched Holdout N=200)", fontsize=12, fontweight="bold")
ax_hist.legend(loc="upper right")

# Cumulative Absolute Error Curve
sorted_abs_err = np.sort(test_abs_residuals)
cdf = np.arange(1, len(sorted_abs_err) + 1) / len(sorted_abs_err) * 100
ax_cdf.plot(sorted_abs_err, cdf, color="#2b5c8f", linewidth=2.5, label="Empirical CDF")
ax_cdf.axvline(5.0, color="green", linestyle="--", label=f"Within 5 min: {pct_within_5:.1f}%")
ax_cdf.axvline(10.0, color="orange", linestyle="--", label=f"Within 10 min: {pct_within_10:.1f}%")
ax_cdf.axvline(15.0, color="red", linestyle="--", label=f"Within 15 min: {pct_within_15:.1f}%")
ax_cdf.set_xlabel("Absolute Error Threshold (min)", fontsize=11, fontweight="bold")
ax_cdf.set_ylabel("Percentage of Orders (%)", fontsize=11, fontweight="bold")
ax_cdf.set_title("Cumulative Absolute Error Distribution\n(SLA Compliance Curve)", fontsize=12, fontweight="bold")
ax_cdf.legend(loc="lower right")
ax_cdf.set_ylim([0, 105])

plt.tight_layout()
plt.savefig("plots/final_model_error_distribution.png", dpi=300)
plt.close()
print("Saved plots/final_model_error_distribution.png")

print("\n--- ALL PART 6 COMPUTATIONS, SCENARIOS & PLOTS GENERATED SUCCESSFULLY ---")

