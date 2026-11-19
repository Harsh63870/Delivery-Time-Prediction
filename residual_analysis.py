"""
Part 4: Residual and Error Analysis for the Best Model (Linear Regression)

Questions addressed:
1. Plot actual delivery time against predicted delivery time.
2. Plot residuals against predicted values.
3. Calculate MAE for short, medium, and long distances.
4. Compare errors across low, medium, and high traffic levels.
5. Identify concrete examples of underprediction and overprediction,
   explaining the root causes and conditions under which the model struggles.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("deep")

os.makedirs('plots', exist_ok=True)
os.makedirs('output', exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TARGET_COL = "Delivery_Time_min"

# 1. Load Data
df = pd.read_csv("Food_Delivery_Times.csv")

feature_num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
feature_cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]

if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Consistent 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 2. Preprocessing pipeline matching Parts 1-3
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

best_model = LinearRegression()

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ]
)

# 3. Fit on Training set and Evaluate on Test Set (Holdout Analysis)
full_pipeline.fit(X_train, y_train)
y_test_pred = full_pipeline.predict(X_test)
test_residuals = y_test - y_test_pred  # residual = Actual - Predicted

# 4. Out-of-fold Cross-Validation predictions on the Training Set (800 rows)
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
y_train_oof_pred = cross_val_predict(full_pipeline, X_train, y_train, cv=cv)
train_oof_residuals = y_train - y_train_oof_pred

# Combine into analysis DataFrames
test_analysis_df = X_test.copy()
test_analysis_df["Actual_Delivery_Time"] = y_test
test_analysis_df["Predicted_Delivery_Time"] = y_test_pred
test_analysis_df["Residual"] = test_residuals
test_analysis_df["Abs_Error"] = np.abs(test_residuals)

train_oof_df = X_train.copy()
train_oof_df["Actual_Delivery_Time"] = y_train
train_oof_df["Predicted_Delivery_Time"] = y_train_oof_pred
train_oof_df["Residual"] = train_oof_residuals
train_oof_df["Abs_Error"] = np.abs(train_oof_residuals)

# Function to categorize distance into Short, Medium, Long
def categorize_distance(d):
    if d < 7.0:
        return "Short (<7 km)"
    elif d <= 14.0:
        return "Medium (7-14 km)"
    else:
        return "Long (>14 km)"

for df_eval in [test_analysis_df, train_oof_df]:
    df_eval["Distance_Tier"] = df_eval["Distance_km"].apply(categorize_distance)
    # Traffic Level filled for grouping
    df_eval["Traffic_Level_Clean"] = df_eval["Traffic_Level"].fillna("Missing/Unknown")

# -------------------------------------------------------------
# Distance & Traffic Subgroup Error Analysis
# -------------------------------------------------------------
tier_order = ["Short (<7 km)", "Medium (7-14 km)", "Long (>14 km)"]
traffic_order = ["Low", "Medium", "High"]

def get_subgroup_metrics(df_sub, group_col, order=None):
    grouped = df_sub.groupby(group_col)
    res = []
    keys = order if order else sorted(df_sub[group_col].unique())
    for k in keys:
        if k in grouped.groups:
            sub = grouped.get_group(k)
            mae = mean_absolute_error(sub["Actual_Delivery_Time"], sub["Predicted_Delivery_Time"])
            rmse = np.sqrt(mean_squared_error(sub["Actual_Delivery_Time"], sub["Predicted_Delivery_Time"]))
            mean_res = sub["Residual"].mean()
            std_res = sub["Residual"].std()
            count = len(sub)
            res.append({
                "Group": k,
                "Count": count,
                "MAE": round(float(mae), 3),
                "RMSE": round(float(rmse), 3),
                "Mean_Residual": round(float(mean_res), 3),
                "Std_Residual": round(float(std_res), 3)
            })
    return pd.DataFrame(res)

test_dist_metrics = get_subgroup_metrics(test_analysis_df, "Distance_Tier", tier_order)
oof_dist_metrics = get_subgroup_metrics(train_oof_df, "Distance_Tier", tier_order)

test_traffic_metrics = get_subgroup_metrics(
    test_analysis_df[test_analysis_df["Traffic_Level_Clean"].isin(traffic_order)],
    "Traffic_Level_Clean", traffic_order
)
oof_traffic_metrics = get_subgroup_metrics(
    train_oof_df[train_oof_df["Traffic_Level_Clean"].isin(traffic_order)],
    "Traffic_Level_Clean", traffic_order
)

# -------------------------------------------------------------
# Extreme Error Cases: Underprediction & Overprediction
# -------------------------------------------------------------
# Underprediction: Actual > Predicted (Residual > 0, largest positive residuals)
# Overprediction: Actual < Predicted (Residual < 0, largest negative residuals)
top_underpredictions = test_analysis_df.sort_values(by="Residual", ascending=False).head(5)
top_overpredictions = test_analysis_df.sort_values(by="Residual", ascending=True).head(5)

top_oof_under = train_oof_df.sort_values(by="Residual", ascending=False).head(5)
top_oof_over = train_oof_df.sort_values(by="Residual", ascending=True).head(5)

# Save tables to CSV
test_dist_metrics.to_csv("output/residual_mae_by_distance_test.csv", index=False)
oof_dist_metrics.to_csv("output/residual_mae_by_distance_oof.csv", index=False)
test_traffic_metrics.to_csv("output/residual_mae_by_traffic_test.csv", index=False)
oof_traffic_metrics.to_csv("output/residual_mae_by_traffic_oof.csv", index=False)
test_analysis_df.to_csv("output/residual_test_predictions.csv", index=False)

# -------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------

# Plot 1: Actual vs Predicted Delivery Time (Holdout Test Set)
plt.figure(figsize=(8, 7))
plt.scatter(
    test_analysis_df["Actual_Delivery_Time"],
    test_analysis_df["Predicted_Delivery_Time"],
    alpha=0.7,
    color="#2b5c8f",
    edgecolors="w",
    s=65,
    label="Deliveries (Test Set)"
)
min_val = min(test_analysis_df["Actual_Delivery_Time"].min(), test_analysis_df["Predicted_Delivery_Time"].min()) - 2
max_val = max(test_analysis_df["Actual_Delivery_Time"].max(), test_analysis_df["Predicted_Delivery_Time"].max()) + 2
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")

plt.title("Actual vs Predicted Delivery Time (Linear Regression - Test Set)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Actual Delivery Time (min)", fontsize=11)
plt.ylabel("Predicted Delivery Time (min)", fontsize=11)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.savefig("plots/residual_actual_vs_predicted.png", dpi=300)
plt.close()

# Plot 2: Residuals vs Predicted Values (Holdout Test Set)
plt.figure(figsize=(9, 6))
plt.scatter(
    test_analysis_df["Predicted_Delivery_Time"],
    test_analysis_df["Residual"],
    alpha=0.7,
    color="#e26d5c",
    edgecolors="k",
    linewidths=0.5,
    s=65
)
plt.axhline(0, color="black", linestyle="--", lw=1.8, label="Zero Error Line (Residual = 0)")

# Annotate overprediction vs underprediction zones
plt.text(
    test_analysis_df["Predicted_Delivery_Time"].min() + 2,
    test_analysis_df["Residual"].max() - 3,
    "Underprediction Zone (Actual > Predicted)",
    fontsize=10,
    fontweight="semibold",
    color="#8a1c14",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fce8e6", ec="#8a1c14", alpha=0.8)
)
plt.text(
    test_analysis_df["Predicted_Delivery_Time"].min() + 2,
    test_analysis_df["Residual"].min() + 2,
    "Overprediction Zone (Actual < Predicted)",
    fontsize=10,
    fontweight="semibold",
    color="#1d4e89",
    bbox=dict(boxstyle="round,pad=0.3", fc="#e6f0fa", ec="#1d4e89", alpha=0.8)
)

plt.title("Residuals vs Predicted Values (Linear Regression - Test Set)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Predicted Delivery Time (min)", fontsize=11)
plt.ylabel("Residual (Actual - Predicted, min)", fontsize=11)
plt.legend(frameon=True, loc="upper right")
plt.tight_layout()
plt.savefig("plots/residual_residuals_vs_predicted.png", dpi=300)
plt.close()

# Plot 3: Subgroup MAE Bar Charts (Distance and Traffic)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Distance Tier MAE
sns.barplot(
    data=test_dist_metrics,
    x="Group",
    y="MAE",
    palette="Blues_d",
    ax=axes[0],
    edgecolor="black"
)
axes[0].set_title("MAE by Distance Tier (Test Set)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Distance Tier", fontsize=11)
axes[0].set_ylabel("Mean Absolute Error (min)", fontsize=11)
for i, row in test_dist_metrics.iterrows():
    axes[0].text(i, row["MAE"] + 0.15, f"{row['MAE']:.2f} m\n(n={row['Count']})", ha="center", fontsize=9.5, fontweight="bold")
axes[0].set_ylim(0, test_dist_metrics["MAE"].max() * 1.25)

# Traffic Level MAE
sns.barplot(
    data=test_traffic_metrics,
    x="Group",
    y="MAE",
    palette="Oranges_d",
    ax=axes[1],
    edgecolor="black"
)
axes[1].set_title("MAE by Traffic Level (Test Set)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Traffic Level", fontsize=11)
axes[1].set_ylabel("Mean Absolute Error (min)", fontsize=11)
for i, row in test_traffic_metrics.iterrows():
    axes[1].text(i, row["MAE"] + 0.15, f"{row['MAE']:.2f} m\n(n={row['Count']})", ha="center", fontsize=9.5, fontweight="bold")
axes[1].set_ylim(0, test_traffic_metrics["MAE"].max() * 1.25)

plt.tight_layout()
plt.savefig("plots/residual_mae_by_subgroups.png", dpi=300)
plt.close()

# Plot 4: Comprehensive 4-panel diagnostic plot (Actual vs Pred, Res vs Pred, Res vs Distance, Res by Traffic)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Actual vs Pred (CV OOF 800 rows for large sample visibility)
axes[0, 0].scatter(train_oof_df["Actual_Delivery_Time"], train_oof_df["Predicted_Delivery_Time"], alpha=0.5, color="#1f77b4", edgecolors="none", s=30)
oof_min = min(train_oof_df["Actual_Delivery_Time"].min(), train_oof_df["Predicted_Delivery_Time"].min())
oof_max = max(train_oof_df["Actual_Delivery_Time"].max(), train_oof_df["Predicted_Delivery_Time"].max())
axes[0, 0].plot([oof_min, oof_max], [oof_min, oof_max], 'r--', lw=2, label="Perfect Fit")
axes[0, 0].set_title("OOF Actual vs Predicted (N=800 Train Set)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("Actual Delivery Time (min)")
axes[0, 0].set_ylabel("Predicted Delivery Time (min)")
axes[0, 0].legend()

# Panel 2: Residuals vs Predicted
axes[0, 1].scatter(train_oof_df["Predicted_Delivery_Time"], train_oof_df["Residual"], alpha=0.5, color="#ff7f0e", edgecolors="none", s=30)
axes[0, 1].axhline(0, color="black", linestyle="--", lw=1.5)
axes[0, 1].set_title("OOF Residuals vs Predicted (N=800 Train Set)", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("Predicted Delivery Time (min)")
axes[0, 1].set_ylabel("Residual (Actual - Predicted, min)")

# Panel 3: Residuals vs Distance_km
axes[1, 0].scatter(train_oof_df["Distance_km"], train_oof_df["Residual"], alpha=0.5, color="#2ca02c", edgecolors="none", s=30)
axes[1, 0].axhline(0, color="black", linestyle="--", lw=1.5)
axes[1, 0].set_title("OOF Residuals vs Distance (km)", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Distance (km)")
axes[1, 0].set_ylabel("Residual (min)")

# Panel 4: Boxplot of Residuals by Traffic Level
sns.boxplot(
    data=train_oof_df[train_oof_df["Traffic_Level_Clean"].isin(traffic_order)],
    x="Traffic_Level_Clean",
    y="Residual",
    order=traffic_order,
    palette="Set2",
    ax=axes[1, 1]
)
axes[1, 1].axhline(0, color="red", linestyle="--", lw=1.2)
axes[1, 1].set_title("OOF Residual Distribution by Traffic Level", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Traffic Level")
axes[1, 1].set_ylabel("Residual (min)")

plt.tight_layout()
plt.savefig("plots/residual_comprehensive_diagnostics.png", dpi=300)
plt.close()

# Save JSON summary with detailed cases and metrics
summary_json = {
    "test_set_overall": {
        "N": len(test_analysis_df),
        "MAE": round(float(mean_absolute_error(y_test, y_test_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
        "R2": round(float(r2_score(y_test, y_test_pred)), 4),
        "Mean_Residual": round(float(test_residuals.mean()), 4),
        "Std_Residual": round(float(test_residuals.std()), 4)
    },
    "oof_train_overall": {
        "N": len(train_oof_df),
        "MAE": round(float(mean_absolute_error(y_train, y_train_oof_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_train, y_train_oof_pred))), 4),
        "R2": round(float(r2_score(y_train, y_train_oof_pred)), 4),
        "Mean_Residual": round(float(train_oof_residuals.mean()), 4),
        "Std_Residual": round(float(train_oof_residuals.std()), 4)
    },
    "test_distance_tiers": test_dist_metrics.to_dict(orient="records"),
    "oof_distance_tiers": oof_dist_metrics.to_dict(orient="records"),
    "test_traffic_levels": test_traffic_metrics.to_dict(orient="records"),
    "oof_traffic_levels": oof_traffic_metrics.to_dict(orient="records"),
    "extreme_test_underpredictions": top_underpredictions[
        ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", 
         "Preparation_Time_min", "Courier_Experience_yrs", "Actual_Delivery_Time", 
         "Predicted_Delivery_Time", "Residual"]
    ].to_dict(orient="records"),
    "extreme_test_overpredictions": top_overpredictions[
        ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", 
         "Preparation_Time_min", "Courier_Experience_yrs", "Actual_Delivery_Time", 
         "Predicted_Delivery_Time", "Residual"]
    ].to_dict(orient="records"),
}

with open("output/residual_analysis_results.json", "w") as f:
    json.dump(summary_json, f, indent=4)

print("=" * 70)
print("RESIDUAL & ERROR ANALYSIS COMPLETE")
print("=" * 70)
print(f"Test Set Overall MAE: {summary_json['test_set_overall']['MAE']} min | RMSE: {summary_json['test_set_overall']['RMSE']} min | R2: {summary_json['test_set_overall']['R2']}")
print("\n--- Distance Tier MAE (Test Set) ---")
print(test_dist_metrics.to_string(index=False))
print("\n--- Traffic Level MAE (Test Set) ---")
print(test_traffic_metrics.to_string(index=False))
print("\n--- Out-of-Fold Distance Tier MAE (N=800 Train) ---")
print(oof_dist_metrics.to_string(index=False))
print("\n--- Out-of-Fold Traffic Level MAE (N=800 Train) ---")
print(oof_traffic_metrics.to_string(index=False))
print("\n--- Top 3 Underpredictions (Actual >> Predicted) in Test Set ---")
for idx, row in top_underpredictions.head(3).iterrows():
    print(f"Actual: {row['Actual_Delivery_Time']:.1f}m, Pred: {row['Predicted_Delivery_Time']:.1f}m, Res: +{row['Residual']:.1f}m | Dist: {row['Distance_km']:.1f}km, Traffic: {row['Traffic_Level']}, Weather: {row['Weather']}, Vehicle: {row['Vehicle_Type']}")
print("\n--- Top 3 Overpredictions (Actual << Predicted) in Test Set ---")
for idx, row in top_overpredictions.head(3).iterrows():
    print(f"Actual: {row['Actual_Delivery_Time']:.1f}m, Pred: {row['Predicted_Delivery_Time']:.1f}m, Res: {row['Residual']:.1f}m | Dist: {row['Distance_km']:.1f}km, Traffic: {row['Traffic_Level']}, Weather: {row['Weather']}, Vehicle: {row['Vehicle_Type']}")

