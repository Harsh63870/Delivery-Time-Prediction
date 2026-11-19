"""
Five-fold cross-validation on the training set.

Question: is Linear Regression genuinely the most reliable model,
or did it win only because of the particular 80/20 train-test split?

All models share the same KFold splits. Preprocessing is fit inside
each fold (via Pipeline) so there is no leakage from validation folds.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

os.makedirs("plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TARGET_COL = "Delivery_Time_min"

df = pd.read_csv("Food_Delivery_Times.csv")

feature_num_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
feature_cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]

if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# Same holdout split as the original pipeline. CV uses only the training portion.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

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

# Same folds for every model. shuffle=True + random_state makes the
# 5 partitions fixed and reusable across estimators.
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}

fold_rows = []
summary_rows = []
per_model_fold_mae = {}

for name, estimator in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    cv_results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        return_train_score=True,
        n_jobs=1,
    )
    val_mae = -cv_results["test_score"]
    train_mae = -cv_results["train_score"]
    per_model_fold_mae[name] = val_mae.tolist()

    for fold_i, (v, t) in enumerate(zip(val_mae, train_mae), start=1):
        fold_rows.append(
            {
                "Model": name,
                "Fold": fold_i,
                "Validation_MAE": round(float(v), 4),
                "Train_MAE": round(float(t), 4),
            }
        )

    summary_rows.append(
        {
            "Model": name,
            "Mean_Validation_MAE": round(float(val_mae.mean()), 4),
            "Std_Validation_MAE": round(float(val_mae.std(ddof=1)), 4),
            "Min_Validation_MAE": round(float(val_mae.min()), 4),
            "Max_Validation_MAE": round(float(val_mae.max()), 4),
            "Mean_Train_MAE": round(float(train_mae.mean()), 4),
        }
    )

fold_df = pd.DataFrame(fold_rows)
summary_df = pd.DataFrame(summary_rows).sort_values("Mean_Validation_MAE")

# Optional: one-shot holdout MAE on the unused test set, using the same
# pipelines, so CV rankings can be compared with the original split.
holdout_rows = []
for name, estimator in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    holdout_rows.append(
        {
            "Model": name,
            "Holdout_Test_MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }
    )
holdout_df = pd.DataFrame(holdout_rows)
comparison_df = summary_df.merge(holdout_df, on="Model")

fold_df.to_csv("output/cv_fold_mae.csv", index=False)
comparison_df.to_csv("output/cv_summary.csv", index=False)

payload = {
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "same_folds_for_all_models": True,
    "cv_splitter": "KFold(n_splits=5, shuffle=True, random_state=42)",
    "per_fold_validation_mae": per_model_fold_mae,
    "summary": comparison_df.to_dict(orient="records"),
    "best_by_mean_cv_mae": comparison_df.iloc[0]["Model"],
    "lowest_cv_std": comparison_df.sort_values("Std_Validation_MAE").iloc[0]["Model"],
}

with open("output/cv_results.json", "w") as f:
    json.dump(payload, f, indent=4)

# Plot: mean CV MAE with standard-deviation error bars
plot_df = comparison_df.sort_values("Mean_Validation_MAE")
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(plot_df))
ax.bar(
    x_pos,
    plot_df["Mean_Validation_MAE"],
    yerr=plot_df["Std_Validation_MAE"],
    capsize=6,
    color=["#4C78A8" if m != "Dummy Regressor" else "#E45756" for m in plot_df["Model"]],
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_df["Model"], rotation=20, ha="right")
ax.set_ylabel("Validation MAE (minutes)")
ax.set_title("5-Fold CV Mean Validation MAE (± 1 std)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/cv_mae_mean_std.png", dpi=300)
plt.close()

# Plot: per-fold MAE so fold-to-fold movement is visible
pivot = fold_df.pivot(index="Fold", columns="Model", values="Validation_MAE")
fig, ax = plt.subplots(figsize=(9, 5))
for col in pivot.columns:
    ax.plot(pivot.index, pivot[col], marker="o", label=col)
ax.set_xlabel("Fold")
ax.set_ylabel("Validation MAE (minutes)")
ax.set_title("Validation MAE on the Same 5 Folds")
ax.set_xticks(list(pivot.index))
ax.legend(frameon=True)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/cv_mae_per_fold.png", dpi=300)
plt.close()

print("=" * 72)
print("FIVE-FOLD CROSS-VALIDATION ON TRAINING DATA")
print("=" * 72)
print(f"Training rows: {len(X_train)} | Held-out test rows (not used in CV): {len(X_test)}")
print(f"Splitter: {payload['cv_splitter']}")
print("Same folds used for every model: YES")
print()
print("Per-fold validation MAE:")
print(pivot.to_string())
print()
print("Mean and standard deviation of validation MAE:")
print(
    comparison_df[
        [
            "Model",
            "Mean_Validation_MAE",
            "Std_Validation_MAE",
            "Min_Validation_MAE",
            "Max_Validation_MAE",
            "Holdout_Test_MAE",
        ]
    ].to_string(index=False)
)
print()
print(f"Lowest mean CV MAE: {payload['best_by_mean_cv_mae']}")
print(f"Most stable (lowest std): {payload['lowest_cv_std']}")
