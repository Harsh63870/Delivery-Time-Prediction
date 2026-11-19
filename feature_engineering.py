"""
Compare the original feature set with a version that adds two
hand-designed features, using the same five CV folds as before.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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

FEATURE_NUM_ORIG = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]
FEATURE_CAT = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
FEATURE_NUM_ENG = FEATURE_NUM_ORIG + [
    "Distance_x_Traffic",
    "Prep_plus_Travel_Proxy",
]

# Ordinal map for Distance × Traffic. Equal spacing is an assumption,
# not a claim that High traffic is exactly 3× Low.
TRAFFIC_ORDINAL = {"Low": 1.0, "Medium": 2.0, "High": 3.0}

# Domain travel-time proxy. Speeds are a priori urban-delivery values,
# not estimated from Delivery_Time_min (avoids target leakage).
# Medium-traffic minutes per km = 60 / assumed_km_h.
VEHICLE_BASE_MIN_PER_KM = {
    "Bike": 5.0,  # ~12 km/h
    "Scooter": 4.0,  # ~15 km/h
    "Car": 3.0,  # ~20 km/h
}
TRAFFIC_SPEED_MULTIPLIER = {
    "Low": 0.80,  # faster than medium
    "Medium": 1.00,
    "High": 1.40,  # slower than medium
}
DEFAULT_VEHICLE = "Scooter"


class SimpleFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add Distance_x_Traffic and Prep_plus_Travel_Proxy.

    Missing Traffic_Level is filled with the training-fold mode so the
    interaction is defined on every row without using the target.
    """

    def fit(self, X, y=None):
        traffic = X["Traffic_Level"]
        mode = traffic.mode(dropna=True)
        self.traffic_fill_ = mode.iloc[0] if len(mode) else "Medium"
        return self

    def transform(self, X):
        X = X.copy()
        traffic = X["Traffic_Level"].fillna(self.traffic_fill_)
        ordinal = traffic.map(TRAFFIC_ORDINAL).astype(float)
        X["Distance_x_Traffic"] = X["Distance_km"] * ordinal

        vehicle = X["Vehicle_Type"].fillna(DEFAULT_VEHICLE)
        base_mpk = vehicle.map(VEHICLE_BASE_MIN_PER_KM).fillna(
            VEHICLE_BASE_MIN_PER_KM[DEFAULT_VEHICLE]
        )
        traffic_mult = traffic.map(TRAFFIC_SPEED_MULTIPLIER).astype(float)
        travel_proxy = X["Distance_km"] * base_mpk * traffic_mult
        X["Prep_plus_Travel_Proxy"] = X["Preparation_Time_min"] + travel_proxy
        return X


def make_preprocessor(numeric_cols):
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, FEATURE_CAT),
        ]
    )


def make_pipeline(estimator, engineered):
    steps = []
    if engineered:
        steps.append(("engineer", SimpleFeatureEngineer()))
        prep = make_preprocessor(FEATURE_NUM_ENG)
    else:
        prep = make_preprocessor(FEATURE_NUM_ORIG)
    steps.append(("preprocessor", prep))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


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

# Identical folds to the stability experiment.
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

feature_sets = {
    "original": False,
    "engineered": True,
}

fold_rows = []
summary_rows = []
per_model = {}

for set_name, engineered in feature_sets.items():
    per_model[set_name] = {}
    for model_name, estimator in models.items():
        pipe = make_pipeline(estimator, engineered)
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
        per_model[set_name][model_name] = [float(v) for v in val_mae]

        for fold_i, (v, t) in enumerate(zip(val_mae, train_mae), start=1):
            fold_rows.append(
                {
                    "Feature_Set": set_name,
                    "Model": model_name,
                    "Fold": fold_i,
                    "Validation_MAE": round(float(v), 4),
                    "Train_MAE": round(float(t), 4),
                }
            )

        summary_rows.append(
            {
                "Feature_Set": set_name,
                "Model": model_name,
                "Mean_Validation_MAE": round(float(val_mae.mean()), 4),
                "Std_Validation_MAE": round(float(val_mae.std(ddof=1)), 4),
                "Mean_Train_MAE": round(float(train_mae.mean()), 4),
            }
        )

fold_df = pd.DataFrame(fold_rows)
summary_df = pd.DataFrame(summary_rows)

orig = summary_df[summary_df["Feature_Set"] == "original"].set_index("Model")
eng = summary_df[summary_df["Feature_Set"] == "engineered"].set_index("Model")
compare = pd.DataFrame(
    {
        "Mean_MAE_Original": orig["Mean_Validation_MAE"],
        "Std_MAE_Original": orig["Std_Validation_MAE"],
        "Mean_MAE_Engineered": eng["Mean_Validation_MAE"],
        "Std_MAE_Engineered": eng["Std_Validation_MAE"],
    }
)
compare["Delta_Mean_MAE"] = (
    compare["Mean_MAE_Engineered"] - compare["Mean_MAE_Original"]
)
compare["Improved"] = compare["Delta_Mean_MAE"] < -1e-4
compare = compare.reset_index()

# Training-only description of the new columns (after a full-train fit of
# the engineer, for reporting — not used as a CV input).
engineer_all_train = SimpleFeatureEngineer().fit(X_train)
X_train_eng = engineer_all_train.transform(X_train)
desc = (
    X_train_eng[["Distance_km", "Distance_x_Traffic", "Prep_plus_Travel_Proxy"]]
    .describe()
    .round(3)
    .to_dict()
)

fold_df.to_csv("output/feature_engineering_fold_mae.csv", index=False)
compare.to_csv("output/feature_engineering_comparison.csv", index=False)

payload = {
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "same_folds_as_stability_cv": True,
    "cv_splitter": "KFold(n_splits=5, shuffle=True, random_state=42)",
    "original_numeric": FEATURE_NUM_ORIG,
    "engineered_numeric_added": [
        "Distance_x_Traffic",
        "Prep_plus_Travel_Proxy",
    ],
    "traffic_ordinal": TRAFFIC_ORDINAL,
    "vehicle_base_min_per_km": VEHICLE_BASE_MIN_PER_KM,
    "traffic_speed_multiplier": TRAFFIC_SPEED_MULTIPLIER,
    "train_feature_describe": desc,
    "per_fold_validation_mae": per_model,
    "comparison": compare.to_dict(orient="records"),
}

with open("output/feature_engineering_results.json", "w") as f:
    json.dump(payload, f, indent=4)

# Grouped bar: original vs engineered mean MAE
order = [
    "Dummy Regressor",
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]
plot_df = compare.set_index("Model").loc[order]
x = np.arange(len(order))
width = 0.36
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar(
    x - width / 2,
    plot_df["Mean_MAE_Original"],
    width,
    yerr=plot_df["Std_MAE_Original"],
    capsize=4,
    label="Original features",
    color="#4C78A8",
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x + width / 2,
    plot_df["Mean_MAE_Engineered"],
    width,
    yerr=plot_df["Std_MAE_Engineered"],
    capsize=4,
    label="Engineered features",
    color="#F58518",
    edgecolor="black",
    linewidth=0.5,
)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=15, ha="right")
ax.set_ylabel("Validation MAE (minutes)")
ax.set_title("Original vs Engineered Features (same 5 folds, mean ± 1 std)")
ax.legend()
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/feature_engineering_cv_mae.png", dpi=300)
plt.close()

print("=" * 72)
print("FEATURE ENGINEERING: ORIGINAL vs ENGINEERED (SAME 5 FOLDS)")
print("=" * 72)
print(compare.to_string(index=False))
print()
print("Negative Delta_Mean_MAE means the engineered set improved MAE.")
print()
print("Per-fold Linear Regression MAE:")
lr_fold = fold_df[fold_df["Model"] == "Linear Regression"].pivot(
    index="Fold", columns="Feature_Set", values="Validation_MAE"
)
print(lr_fold.to_string())
