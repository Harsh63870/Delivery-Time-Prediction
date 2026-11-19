"""
Learning curves and bias-variance analysis.

Question: How can we tell whether a model is too simple or too complex?
We compare:
  1. Linear Regression
  2. Shallow Decision Tree (max_depth=3)
  3. Unrestricted Decision Tree (max_depth=None)

We plot training error and cross-validation error against the amount
of training data, and examine the gap between them to identify
underfitting (high bias) vs overfitting (high variance).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

os.makedirs("plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TARGET_COL = "Delivery_Time_min"

# Load dataset
df = pd.read_csv("Food_Delivery_Times.csv")

feature_num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
feature_cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]

if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Split 80/20 train/test. CV and learning curves use only training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Preprocessing pipeline inside each fold to prevent data leakage
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

# 5-fold cross validation splitter
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# 10 training subset sizes from 10% to 100% of available fold training data (64 to 640 rows)
train_sizes = np.linspace(0.1, 1.0, 10)

models_to_compare = {
    "Linear Regression": LinearRegression(),
    "Shallow Decision Tree (max_depth=3)": DecisionTreeRegressor(
        max_depth=3, random_state=RANDOM_STATE
    ),
    "Unrestricted Decision Tree (max_depth=None)": DecisionTreeRegressor(
        max_depth=None, random_state=RANDOM_STATE
    ),
}

results_records = []
curve_summary = {}

for name, estimator in models_to_compare.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipe,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    
    # neg_mean_absolute_error -> positive MAE in minutes
    train_mae_mean = -np.mean(train_scores, axis=1)
    train_mae_std = np.std(train_scores, axis=1)
    val_mae_mean = -np.mean(test_scores, axis=1)
    val_mae_std = np.std(test_scores, axis=1)
    gap = val_mae_mean - train_mae_mean
    
    curve_summary[name] = {
        "train_sizes": train_sizes_abs.tolist(),
        "train_mae_mean": np.round(train_mae_mean, 4).tolist(),
        "train_mae_std": np.round(train_mae_std, 4).tolist(),
        "val_mae_mean": np.round(val_mae_mean, 4).tolist(),
        "val_mae_std": np.round(val_mae_std, 4).tolist(),
        "gap": np.round(gap, 4).tolist(),
        "final_train_mae": round(float(train_mae_mean[-1]), 4),
        "final_val_mae": round(float(val_mae_mean[-1]), 4),
        "final_gap": round(float(gap[-1]), 4),
    }
    
    for i in range(len(train_sizes_abs)):
        results_records.append(
            {
                "Model": name,
                "Train_Size": int(train_sizes_abs[i]),
                "Train_MAE_Mean": round(float(train_mae_mean[i]), 4),
                "Train_MAE_Std": round(float(train_mae_std[i]), 4),
                "Val_MAE_Mean": round(float(val_mae_mean[i]), 4),
                "Val_MAE_Std": round(float(val_mae_std[i]), 4),
                "Generalization_Gap": round(float(gap[i]), 4),
            }
        )

results_df = pd.DataFrame(results_records)
results_df.to_csv("output/learning_curve_data.csv", index=False)

with open("output/learning_curve_summary.json", "w") as f:
    json.dump(curve_summary, f, indent=4)

# -------------------------------------------------------------
# PLOT 1: Individual 3-Panel Learning Curves Comparison
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

panel_configs = [
    ("Linear Regression", axes[0], "Balanced / Low Bias & Low Variance"),
    ("Shallow Decision Tree (max_depth=3)", axes[1], "Underfitting / High Bias"),
    ("Unrestricted Decision Tree (max_depth=None)", axes[2], "Overfitting / High Variance"),
]

for name, ax, subtitle in panel_configs:
    data = results_df[results_df["Model"] == name]
    sizes = data["Train_Size"]
    
    # Train curve
    ax.plot(
        sizes,
        data["Train_MAE_Mean"],
        "o-",
        color="#d95f02",
        linewidth=2,
        label="Training Error (MAE)",
    )
    ax.fill_between(
        sizes,
        data["Train_MAE_Mean"] - data["Train_MAE_Std"],
        data["Train_MAE_Mean"] + data["Train_MAE_Std"],
        alpha=0.15,
        color="#d95f02",
    )
    
    # Validation curve
    ax.plot(
        sizes,
        data["Val_MAE_Mean"],
        "s-",
        color="#2b5c8f",
        linewidth=2,
        label="Validation Error (5-Fold CV MAE)",
    )
    ax.fill_between(
        sizes,
        data["Val_MAE_Mean"] - data["Val_MAE_Std"],
        data["Val_MAE_Mean"] + data["Val_MAE_Std"],
        alpha=0.15,
        color="#2b5c8f",
    )
    
    # Annotate final values
    final_tr = data["Train_MAE_Mean"].iloc[-1]
    final_val = data["Val_MAE_Mean"].iloc[-1]
    final_gap = data["Generalization_Gap"].iloc[-1]
    
    ax.annotate(
        f"Train: {final_tr:.2f} min\nVal: {final_val:.2f} min\nGap: {final_gap:.2f} min",
        xy=(sizes.iloc[-1], final_val),
        xytext=(sizes.iloc[-1] - 180, (final_val + final_tr) / 2 if abs(final_val - final_tr) > 2 else final_val + 1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="gray"),
        fontsize=9,
    )
    
    ax.set_title(f"{name}\n({subtitle})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Set Size (samples)", fontsize=10)
    ax.set_xticks(sizes[::2])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

axes[0].set_ylabel("Mean Absolute Error (minutes)", fontsize=10)
plt.suptitle(
    "Learning Curves: Training vs. Cross-Validation Error against Training Data Size",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("plots/learning_curves_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# PLOT 2: Direct Overlay of Validation MAE Curves
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = {
    "Linear Regression": "#2ca02c",
    "Shallow Decision Tree (max_depth=3)": "#ff7f0e",
    "Unrestricted Decision Tree (max_depth=None)": "#d62728",
}

for name in models_to_compare.keys():
    data = results_df[results_df["Model"] == name]
    ax.plot(
        data["Train_Size"],
        data["Val_MAE_Mean"],
        marker="o",
        linewidth=2.2,
        label=f"{name} (Val MAE: {data['Val_MAE_Mean'].iloc[-1]:.2f}m)",
        color=colors[name],
    )
    ax.fill_between(
        data["Train_Size"],
        data["Val_MAE_Mean"] - data["Val_MAE_Std"],
        data["Val_MAE_Mean"] + data["Val_MAE_Std"],
        alpha=0.12,
        color=colors[name],
    )

ax.set_title("Cross-Validation MAE by Training Data Size across Models", fontsize=12, fontweight="bold")
ax.set_xlabel("Training Set Size (samples)", fontsize=10)
ax.set_ylabel("Validation MAE (minutes) — Lower is Better", fontsize=10)
ax.set_xticks(data["Train_Size"][::2])
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig("plots/learning_curves_validation_overlay.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------------------
# PLOT 3: Model Complexity Curve (Bias-Variance vs Tree Depth)
# -------------------------------------------------------------
depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
depth_labels = [str(d) if d is not None else "None (Full)" for d in depths]
tree_complexity_rows = []

for d, d_lbl in zip(depths, depth_labels):
    estimator = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    sizes, train_scores, val_scores = learning_curve(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=[1.0],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    t_mae = -float(np.mean(train_scores))
    v_mae = -float(np.mean(val_scores))
    tree_complexity_rows.append(
        {
            "Max_Depth": d_lbl,
            "Train_MAE": round(t_mae, 4),
            "Val_MAE": round(v_mae, 4),
            "Gap": round(v_mae - t_mae, 4),
        }
    )

depth_df = pd.DataFrame(tree_complexity_rows)
depth_df.to_csv("output/tree_depth_tradeoff.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5.5))
x_indices = np.arange(len(depth_labels))

ax.plot(
    x_indices,
    depth_df["Train_MAE"],
    "o-",
    color="#d95f02",
    linewidth=2,
    label="Training MAE (Decreases to ~0)",
)
ax.plot(
    x_indices,
    depth_df["Val_MAE"],
    "s-",
    color="#2b5c8f",
    linewidth=2,
    label="Validation MAE (U-shaped)",
)

# Reference line for Linear Regression validation MAE
lr_val_mae = curve_summary["Linear Regression"]["final_val_mae"]
ax.axhline(
    lr_val_mae,
    color="#2ca02c",
    linestyle="--",
    linewidth=1.8,
    label=f"Linear Regression Val MAE ({lr_val_mae:.2f}m)",
)

# Annotate underfitting and overfitting regimes
ax.annotate(
    "High Bias\n(Underfitting)",
    xy=(1, depth_df.loc[depth_df['Max_Depth']=='2', 'Val_MAE'].values[0]),
    xytext=(0.5, 12.5),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9.5,
    fontweight="bold",
    ha="center",
)

ax.annotate(
    "High Variance\n(Severe Overfitting)",
    xy=(len(depth_labels)-1, depth_df.loc[depth_df['Max_Depth']=='None (Full)', 'Val_MAE'].values[0]),
    xytext=(len(depth_labels)-2, 13),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9.5,
    fontweight="bold",
    ha="center",
)

ax.set_xticks(x_indices)
ax.set_xticklabels(depth_labels, rotation=30)
ax.set_xlabel("Decision Tree max_depth (Model Complexity)", fontsize=10)
ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=10)
ax.set_title("Model Complexity Curve: Bias-Variance Tradeoff across Decision Tree Depths", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="center right", fontsize=9.5)
plt.tight_layout()
plt.savefig("plots/tree_complexity_bias_variance.png", dpi=300, bbox_inches="tight")
plt.close()

print("=" * 75)
print("BIAS AND VARIANCE THROUGH LEARNING CURVES - EXPERIMENT COMPLETE")
print("=" * 75)
print("Models evaluated:")
for name in models_to_compare.keys():
    s = curve_summary[name]
    print(
        f"  * {name:45s} | Final Train MAE: {s['final_train_mae']:.4f}m | "
        f"Final Val MAE: {s['final_val_mae']:.4f}m | Gap: {s['final_gap']:.4f}m"
    )

print("\nTree Depth Tradeoff (Final Sample Size 640):")
print(depth_df.to_string(index=False))

