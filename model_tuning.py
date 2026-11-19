"""
Controlled hyperparameter tuning with a small grid and the same
five training-set folds as the earlier experiments.

Question: can we improve Decision Tree, Random Forest, and XGBoost
systematically rather than by trial and error?

Search is confined to the knobs that previous studies already flagged:
tree depth / leaf size for DT and RF; depth, learning rate, and
n_estimators for XGBoost. Random Forest n_estimators stays at 100,
the value used throughout the original pipeline.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
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
RF_N_ESTIMATORS = 100  # fixed from the original / CV experiments

FEATURE_NUM = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
FEATURE_CAT = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

df = pd.read_csv("Food_Delivery_Times.csv")
X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]
if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

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
        ("num", numeric_transformer, FEATURE_NUM),
        ("cat", categorical_transformer, FEATURE_CAT),
    ]
)

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_pipe(estimator):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def flatten_params(prefixed):
    """Drop the Pipeline 'model__' prefix for readable tables."""
    return {k.replace("model__", ""): v for k, v in prefixed.items()}


def params_to_label(params):
    parts = []
    for k in sorted(params):
        parts.append(f"{k}={params[k]}")
    return ", ".join(parts)


def grid_table(search):
    rows = []
    for i, params in enumerate(search.cv_results_["params"]):
        clean = flatten_params(params)
        row = {k: ("None" if v is None else v) for k, v in clean.items()}
        row["Mean_Validation_MAE"] = round(
            float(-search.cv_results_["mean_test_score"][i]), 4
        )
        row["Std_Validation_MAE"] = round(
            float(search.cv_results_["std_test_score"][i]), 4
        )
        row["Mean_Train_MAE"] = round(
            float(-search.cv_results_["mean_train_score"][i]), 4
        )
        row["Rank"] = int(search.cv_results_["rank_test_score"][i])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Rank", "Mean_Validation_MAE"])


def evaluate_config(name, estimator, config_kind, params):
    pipe = make_pipe(estimator)
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

    pipe.fit(X_train, y_train)
    holdout = float(mean_absolute_error(y_test, pipe.predict(X_test)))

    return {
        "Model": name,
        "Config": config_kind,
        "Params": params_to_label(params),
        "Mean_Validation_MAE": round(float(val_mae.mean()), 4),
        "Std_Validation_MAE": round(float(val_mae.std(ddof=1)), 4),
        "Mean_Train_MAE": round(float(train_mae.mean()), 4),
        "Generalization_Gap": round(float(val_mae.mean() - train_mae.mean()), 4),
        "Holdout_Test_MAE": round(holdout, 4),
        "Fold_MAE": [round(float(v), 4) for v in val_mae],
    }


# -----------------------------------------------------------------------------
# Original configurations (same as cross_validation.py)
# -----------------------------------------------------------------------------
original_specs = {
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {"max_depth": None, "min_samples_leaf": 1},
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "params": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
    },
}

# Small, assignment-aligned search spaces.
search_specs = {
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 5, 10, 20],
        },
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "param_grid": {
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "param_grid": {
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__n_estimators": [50, 100, 200],
        },
    },
}

grid_tables = {}
searches = {}

print("=" * 72)
print("CONTROLLED HYPERPARAMETER TUNING (5-FOLD GRID SEARCH)")
print("=" * 72)
print(f"Training rows: {len(X_train)} | Held-out test rows: {len(X_test)}")
print("Splitter: KFold(n_splits=5, shuffle=True, random_state=42)")
print(f"Random Forest n_estimators fixed at {RF_N_ESTIMATORS}")
print()

for name, spec in search_specs.items():
    n_combos = int(np.prod([len(v) for v in spec["param_grid"].values()]))
    print(f"Searching {name}: {n_combos} configurations...")
    search = GridSearchCV(
        estimator=make_pipe(spec["estimator"]),
        param_grid=spec["param_grid"],
        scoring="neg_mean_absolute_error",
        cv=cv,
        return_train_score=True,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    searches[name] = search
    table = grid_table(search)
    grid_tables[name] = table
    slug = name.lower().replace(" ", "_")
    table.to_csv(f"output/tuning_grid_{slug}.csv", index=False)
    best = flatten_params(search.best_params_)
    print(f"  best params: {best}")
    print(f"  best mean CV MAE: {round(float(-search.best_score_), 4)}")
    print()

# Rebuild the selected estimators explicitly so original vs tuned
# share the same evaluation path (same folds, same holdout).
tuned_estimators = {
    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE, **flatten_params(searches["Decision Tree"].best_params_)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **{
            k: v
            for k, v in flatten_params(searches["Random Forest"].best_params_).items()
            if k != "n_estimators"
        },
    ),
    "XGBoost": XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
        **flatten_params(searches["XGBoost"].best_params_),
    ),
}

comparison_rows = []
for name in original_specs:
    orig = original_specs[name]
    comparison_rows.append(
        evaluate_config(name, orig["estimator"], "original", orig["params"])
    )
    tuned_params = flatten_params(searches[name].best_params_)
    if name == "Random Forest":
        tuned_params = {"n_estimators": RF_N_ESTIMATORS, **tuned_params}
    comparison_rows.append(
        evaluate_config(name, tuned_estimators[name], "tuned", tuned_params)
    )

# Linear Regression is not tuned; it is the current champion and the
# reference every tree improvement has to beat.
lr_row = evaluate_config(
    "Linear Regression", LinearRegression(), "reference (untuned)", {}
)
lr_row["Params"] = "sklearn defaults"

comparison_df = pd.DataFrame(comparison_rows)
# Keep Linear Regression at the bottom of the tree-model block for the CSV,
# but also persist a combined ranking table.
comparison_export = comparison_df.drop(columns=["Fold_MAE"])
comparison_export.to_csv("output/tuning_comparison.csv", index=False)

lr_export = {k: v for k, v in lr_row.items() if k != "Fold_MAE"}
pd.DataFrame([lr_export]).to_csv("output/tuning_linear_reference.csv", index=False)

payload = {
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "rf_n_estimators_fixed": RF_N_ESTIMATORS,
    "search_spaces": {
        name: {k.replace("model__", ""): v for k, v in spec["param_grid"].items()}
        for name, spec in search_specs.items()
    },
    "n_configurations": {
        name: int(np.prod([len(v) for v in spec["param_grid"].values()]))
        for name, spec in search_specs.items()
    },
    "best_params": {
        name: flatten_params(searches[name].best_params_) for name in searches
    },
    "comparison": comparison_export.to_dict(orient="records"),
    "linear_regression_reference": lr_export,
    "fold_mae": {
        f"{row['Model']} | {row['Config']}": row["Fold_MAE"] for row in comparison_rows
    },
}
with open("output/tuning_results.json", "w") as f:
    json.dump(payload, f, indent=4)

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
# 1. Original vs tuned mean CV MAE
tree_names = ["Decision Tree", "Random Forest", "XGBoost"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(tree_names))
width = 0.35
orig_vals = [
    comparison_df[
        (comparison_df["Model"] == m) & (comparison_df["Config"] == "original")
    ]["Mean_Validation_MAE"].iloc[0]
    for m in tree_names
]
orig_std = [
    comparison_df[
        (comparison_df["Model"] == m) & (comparison_df["Config"] == "original")
    ]["Std_Validation_MAE"].iloc[0]
    for m in tree_names
]
tuned_vals = [
    comparison_df[
        (comparison_df["Model"] == m) & (comparison_df["Config"] == "tuned")
    ]["Mean_Validation_MAE"].iloc[0]
    for m in tree_names
]
tuned_std = [
    comparison_df[
        (comparison_df["Model"] == m) & (comparison_df["Config"] == "tuned")
    ]["Std_Validation_MAE"].iloc[0]
    for m in tree_names
]
ax.bar(
    x - width / 2,
    orig_vals,
    width,
    yerr=orig_std,
    capsize=5,
    label="Original",
    color="#E45756",
    edgecolor="black",
    linewidth=0.6,
)
ax.bar(
    x + width / 2,
    tuned_vals,
    width,
    yerr=tuned_std,
    capsize=5,
    label="Tuned",
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.6,
)
ax.axhline(
    lr_row["Mean_Validation_MAE"],
    color="#54A24B",
    linestyle="--",
    linewidth=1.4,
    label=f"Linear Regression ({lr_row['Mean_Validation_MAE']:.2f} min)",
)
ax.set_xticks(x)
ax.set_xticklabels(tree_names)
ax.set_ylabel("Mean 5-fold validation MAE (minutes)")
ax.set_title("Original vs Tuned Configurations (± 1 std)")
ax.legend(frameon=True)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/tuning_original_vs_tuned.png", dpi=300)
plt.close()


def plot_heatmap(table, index, columns, title, path, fmt="{:.2f}"):
    pivot = table.pivot(index=index, columns=columns, values="Mean_Validation_MAE")
    # Keep a readable row order
    if index == "max_depth":
        pivot = pivot.reindex(table["max_depth"].drop_duplicates().tolist())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(pivot.values, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticklabels([str(r) for r in pivot.index])
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean CV MAE (min)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


plot_heatmap(
    grid_tables["Decision Tree"],
    index="max_depth",
    columns="min_samples_leaf",
    title="Decision Tree: mean CV MAE by max_depth × min_samples_leaf",
    path="plots/tuning_dt_heatmap.png",
)
plot_heatmap(
    grid_tables["Random Forest"],
    index="max_depth",
    columns="min_samples_leaf",
    title=f"Random Forest (n_estimators={RF_N_ESTIMATORS}): mean CV MAE",
    path="plots/tuning_rf_heatmap.png",
)

# XGBoost: one panel per max_depth (the depth already studied)
xgb = grid_tables["XGBoost"].copy()
depths = sorted(xgb["max_depth"].unique())
fig, axes = plt.subplots(1, len(depths), figsize=(12, 4.2), sharey=True)
vmin = xgb["Mean_Validation_MAE"].min()
vmax = xgb["Mean_Validation_MAE"].max()
for ax, depth in zip(axes, depths):
    sub = xgb[xgb["max_depth"] == depth]
    pivot = sub.pivot(
        index="n_estimators", columns="learning_rate", values="Mean_Validation_MAE"
    )
    pivot = pivot.sort_index()
    im = ax.imshow(pivot.values, cmap="YlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("learning_rate")
    ax.set_title(f"max_depth={depth}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
axes[0].set_ylabel("n_estimators")
fig.suptitle("XGBoost: mean CV MAE over learning_rate × n_estimators")
fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.04, label="MAE (min)")
fig.subplots_adjust(top=0.82, wspace=0.25)
plt.savefig("plots/tuning_xgb_heatmaps.png", dpi=300, bbox_inches="tight")
plt.close()

print("Comparison (original vs tuned):")
print(comparison_export.to_string(index=False))
print()
print("Linear Regression reference:")
print(pd.DataFrame([lr_export]).to_string(index=False))
print()
print("Wrote output/tuning_comparison.csv and plots/tuning_original_vs_tuned.png")
