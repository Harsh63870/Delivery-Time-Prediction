import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("deep")

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('output', exist_ok=True)

# -------------------------------------------------------------
# STEP 1 & 2: Load and Inspect Dataset
# -------------------------------------------------------------
df = pd.read_csv('Food_Delivery_Times.csv')

shape_info = df.shape
columns_info = list(df.columns)
dtypes_info = df.dtypes.astype(str).to_dict()
head_5 = df.head().to_dict(orient='records')

desc_num = df.describe().T
desc_cat = (
    df.describe(include=['str', 'object']).T
    if len(df.select_dtypes(
        include=['str', 'object']
    ).columns) > 0
    else pd.DataFrame()
)

# -------------------------------------------------------------
# STEP 4: Missing Values
# -------------------------------------------------------------
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_pct
})

# -------------------------------------------------------------
# STEP 5: Duplicate Rows
# -------------------------------------------------------------
duplicate_count = int(df.duplicated().sum())

# -------------------------------------------------------------
# STEP 6: Categorical & Numerical Columns Identification
# -------------------------------------------------------------
target_col = 'Delivery_Time_min'
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['str', 'object', 'category']).columns.tolist()
# Order_ID is an identifier, not a statistical numerical feature
feature_num_cols = [c for c in num_cols if c not in ['Order_ID', target_col]]
feature_cat_cols = cat_cols.copy()

# -------------------------------------------------------------
# STEP 7 & 8: Outliers, Distributions, Skewness
# -------------------------------------------------------------
skewness = {}
outlier_info = {}

for col in feature_num_cols + [target_col]:
    val = df[col].dropna()
    skew_val = float(val.skew())
    skewness[col] = skew_val
    
    Q1 = float(val.quantile(0.25))
    Q3 = float(val.quantile(0.75))
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = val[(val < lower) | (val > upper)]
    
    outlier_info[col] = {
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower_Bound': round(lower, 2),
        'Upper_Bound': round(upper, 2),
        'Outlier_Count': int(len(outliers)),
        'Outlier_Pct': round((len(outliers) / len(val)) * 100, 2),
        'Skewness': round(skew_val, 3),
        'Mean': round(float(val.mean()), 2),
        'Median': round(float(val.median()), 2),
        'Std': round(float(val.std()), 2)
    }

# -------------------------------------------------------------
# VISUALIZATIONS FOR EDA
# -------------------------------------------------------------

# Plot 1: Missing values chart
plt.figure(figsize=(8, 4))
sns.barplot(x=missing_df.index, y=missing_df['Missing_Percentage'], color='coral')
plt.title('Missing Value Percentage by Column', fontsize=14, fontweight='bold')
plt.ylabel('Missing Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/missing_values.png', dpi=300)
plt.close()

# Plot 2: Numerical Distributions & Boxplots
fig, axes = plt.subplots(len(feature_num_cols + [target_col]), 2, figsize=(12, 3.5 * len(feature_num_cols + [target_col])))
for i, col in enumerate(feature_num_cols + [target_col]):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0], color='teal')
    axes[i, 0].set_title(f'Distribution of {col} (Skew: {skewness[col]:.2f})', fontsize=11)
    
    sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='lightcoral')
    axes[i, 1].set_title(f'Boxplot of {col}', fontsize=11)
plt.tight_layout()
plt.savefig('plots/numerical_distributions.png', dpi=300)
plt.close()

# Plot 3: Categorical Count Plots
fig, axes = plt.subplots(1, len(feature_cat_cols), figsize=(4.5 * len(feature_cat_cols), 4))
if len(feature_cat_cols) == 1:
    axes = [axes]
for i, col in enumerate(feature_cat_cols):
    sns.countplot(data=df,x=col,hue=col,ax=axes[i],palette='viridis',legend=False)
    axes[i].set_title(f'Value Counts: {col}', fontsize=12)
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plots/categorical_counts.png', dpi=300)
plt.close()

# Plot 4: Correlation Matrix
corr_matrix = df[feature_num_cols + [target_col]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.8, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of Numerical Features & Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# STEP 12, 13, 14, 15, 16, 17: Preprocessing & Data Preparation
# -------------------------------------------------------------

# Features (X) and Target (y)
X = df.drop(columns=['Order_ID', target_col])
y = df[target_col]

# We need to handle missing values in target if any, or drop rows where target is missing
if y.isnull().sum() > 0:
    valid_target_idx = y.dropna().index
    X = X.loc[valid_target_idx]
    y = y.loc[valid_target_idx]

# Train-Test Split FIRST to avoid data leakage! (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Imputation Strategy Decisions based on EDA:
# Distance_km: check skewness and outliers -> median if skewed/outliers present, else mean
# Preparation_Time_min: median / mean
# Courier_Experience_yrs: median / mean
# Categorical columns: SimpleImputer with strategy='most_frequent'

num_imputer_strategy = 'median' # Robust to any potential outliers and mild skewness

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_num_cols),
        ('cat', categorical_transformer, feature_cat_cols)
    ]
)

# Fit preprocessor ONLY on X_train and transform X_train and X_test
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Get feature names after One-Hot Encoding
ohe_feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(feature_cat_cols))
all_feature_names = feature_num_cols + ohe_feature_names

# -------------------------------------------------------------
# STEP 18, 19, 20: Model Training, Evaluation & Comparison
# -------------------------------------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    # Fit model on training data
    model.fit(X_train_prep, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_prep)
    predictions[name] = y_pred
    
    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }

results_df = pd.DataFrame(results).T

# -------------------------------------------------------------
# Visualizations for Model Performance
# -------------------------------------------------------------

# Plot 5: Actual vs Predicted Delivery Time for Best Model
best_model_name = results_df['R2'].idxmax()
best_preds = predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color='dodgerblue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title(f'Actual vs Predicted Delivery Time ({best_model_name})', fontsize=14, fontweight='bold')
plt.xlabel('Actual Delivery Time (min)', fontsize=12)
plt.ylabel('Predicted Delivery Time (min)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=300)
plt.close()

# Plot 6: Feature Importances for Tree Models (XGBoost / Random Forest)
if 'Random Forest Regressor' in models:
    rf_model = models['Random Forest Regressor']
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature',  hue='Feature', palette='crest', legend=False)
    plt.title('Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/feature_importances.png', dpi=300)
    plt.close()

# Save JSON results summary
summary_data = {
    'shape': shape_info,
    'columns': columns_info,
    'dtypes': dtypes_info,
    'missing_values': missing_df.to_dict(),
    'duplicates': duplicate_count,
    'num_cols': feature_num_cols,
    'cat_cols': feature_cat_cols,
    'target_col': target_col,
    'outlier_info': outlier_info,
    'correlations': corr_matrix[target_col].to_dict(),
    'models_results': results,
    'best_model': best_model_name
}

with open('output/summary_results.json', 'w') as f:
    json.dump(summary_data, f, indent=4)

print("\n" + "="*60)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Shape: {shape_info}")
print(f"Duplicates: {duplicate_count}")
print("\nMissing Values:")
print(missing_df)
print("\nOutliers Summary:")
print(pd.DataFrame(outlier_info).T[['Mean', 'Median', 'Skewness', 'Outlier_Count', 'Outlier_Pct']])
print("\nModel Evaluation Comparison:")
print(results_df)
