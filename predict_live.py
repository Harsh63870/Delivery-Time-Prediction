"""
Interactive Live Prediction CLI for Oral Defense / Viva Demo.
Allows the professor or student to enter custom order inputs
and receive instant ETA predictions with component waterfall breakdowns.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Delivery_Time_min"
FEATURE_NUM = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
FEATURE_CAT = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

print("=" * 70)
print("  LOADING MODEL & FITTING PRODUCTION PIPELINE... PLEASE WAIT  ")
print("=" * 70)

df = pd.read_csv("Food_Delivery_Times.csv")
X = df.drop(columns=["Order_ID", TARGET_COL])
y = df[TARGET_COL]

if y.isnull().sum() > 0:
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, FEATURE_NUM),
    ("cat", cat_transformer, FEATURE_CAT),
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])
pipe.fit(X_train, y_train)

intercept = pipe.named_steps["model"].intercept_
print(f"Pipeline ready! Baseline Intercept: {intercept:.2f} minutes.\n")


def predict_single(dist, prep, exp, weather, traffic, time_of_day, vehicle):
    row_df = pd.DataFrame([{
        "Distance_km": float(dist),
        "Preparation_Time_min": float(prep),
        "Courier_Experience_yrs": float(exp),
        "Weather": str(weather),
        "Traffic_Level": str(traffic),
        "Time_of_Day": str(time_of_day),
        "Vehicle_Type": str(vehicle),
    }])
    pred = float(pipe.predict(row_df)[0])
    mae = 6.06
    print("\n" + "=" * 60)
    print("               LIVE PREDICTION RESULT                 ")
    print("=" * 60)
    print(f"  Inputs:")
    print(f"    - Distance:           {dist} km")
    print(f"    - Prep Time:          {prep} min")
    print(f"    - Courier Exp:        {exp} yrs")
    print(f"    - Weather:            {weather}")
    print(f"    - Traffic Level:      {traffic}")
    print(f"    - Time of Day:        {time_of_day}")
    print(f"    - Vehicle Type:       {vehicle}")
    print("-" * 60)
    print(f"  >> PREDICTED DELIVERY TIME:  {pred:.1f} minutes")
    print(f"  >> Expected SLA Window:      [{max(0, pred - mae):.1f} to {pred + mae:.1f}] minutes")
    print("=" * 60 + "\n")


if len(sys.argv) == 8:
    # CLI arguments passed: dist prep exp weather traffic time vehicle
    predict_single(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
        sys.argv[5], sys.argv[6], sys.argv[7]
    )
else:
    print("Options for Weather:       Clear, Rainy, Snowy, Foggy, Windy")
    print("Options for Traffic_Level: Low, Medium, High")
    print("Options for Time_of_Day:   Morning, Afternoon, Evening, Night")
    print("Options for Vehicle_Type:  Bike, Scooter, Car\n")

    try:
        dist_in = input("Enter Distance in km (e.g., 7.5): ") or "7.5"
        prep_in = input("Enter Preparation Time in min (e.g., 18.0): ") or "18.0"
        exp_in = input("Enter Courier Experience in years (e.g., 4.0): ") or "4.0"
        weather_in = input("Enter Weather [Clear/Rainy/Snowy/Foggy/Windy] (e.g., Clear): ") or "Clear"
        traffic_in = input("Enter Traffic [Low/Medium/High] (e.g., Medium): ") or "Medium"
        time_in = input("Enter Time of Day [Morning/Afternoon/Evening/Night] (e.g., Evening): ") or "Evening"
        veh_in = input("Enter Vehicle [Bike/Scooter/Car] (e.g., Scooter): ") or "Scooter"

        predict_single(dist_in, prep_in, exp_in, weather_in, traffic_in, time_in, veh_in)
    except KeyboardInterrupt:
        print("\nExiting interactive demo.")

