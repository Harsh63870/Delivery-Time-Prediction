import os
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create directory for plots
os.makedirs('plots', exist_ok=True)

# 1. Load dataset
df = pd.read_csv('Food_Delivery_Times.csv')

print("="*60)
print("1. DATASET INSPECTION")
print("="*60)

# 2. Shape, column names, data types, first 5 rows, descriptive stats
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDescriptive Statistics (Numerical):")
print(df.describe().T)

print("\nDescriptive Statistics (Categorical):")
cat_mask = df.select_dtypes(
    include=['str', 'object', 'category']
).columns

if len(cat_mask) > 0:
    print(df[cat_mask].describe().T)
else:
    print(df.describe(include='all').T)

# 4. Check missing values for EVERY column
print("\n" + "="*60)
print("4. MISSING VALUES CHECK FOR EVERY COLUMN")
print("="*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing Percentage (%)': missing_pct})
print(missing_df)

# 5. Duplicate rows
print("\n" + "="*60)
print("5. DUPLICATE ROWS CHECK")
print("="*60)
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# 6. Identify numerical and categorical columns
print("\n" + "="*60)
print("6. NUMERICAL & CATEGORICAL COLUMNS IDENTIFICATION")
print("="*60)

# Note: Order_ID is an identifier, not a real numerical feature. We should identify it as ID.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(
    include=['str', 'object', 'category']
).columns.tolist()

print(f"All Numerical Columns: {num_cols}")
print(f"All Categorical Columns: {cat_cols}")

# Target variable identification
target_col = 'Delivery_Time_min'
print(f"Target Variable: {target_col}")

# Features columns
feature_num_cols = [col for col in num_cols if col not in ['Order_ID', target_col]]
feature_cat_cols = cat_cols.copy()

print(f"Feature Numerical Columns: {feature_num_cols}")
print(f"Feature Categorical Columns: {feature_cat_cols}")

# 8. Check outliers in numerical columns using IQR
print("\n" + "="*60)
print("8. OUTLIER ANALYSIS USING IQR")
print("="*60)

outlier_summary = {}
for col in feature_num_cols + [target_col]:
    valid_data = df[col].dropna()
    Q1 = valid_data.quantile(0.25)
    Q3 = valid_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
    outlier_summary[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'Lower Bound': lower_bound, 'Upper Bound': upper_bound,
        'Outlier Count': len(outliers),
        'Outlier Percentage (%)': (len(outliers) / len(valid_data)) * 100
    }
outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df)

# Plotting distributions and boxplots
sns.set_theme(style="whitegrid")

# Plot 1: Numerical distributions & Boxplots
fig, axes = plt.subplots(len(feature_num_cols + [target_col]), 2, figsize=(12, 4 * len(feature_num_cols + [target_col])))
for i, col in enumerate(feature_num_cols + [target_col]):
    # Histogram + KDE
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Distribution of {col}')
    # Boxplot
    sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='salmon')
    axes[i, 1].set_title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('plots/numerical_distributions_boxplots.png', dpi=300)
plt.close()

# Plot 2: Categorical distributions
if feature_cat_cols:
    fig, axes = plt.subplots(1, len(feature_cat_cols), figsize=(5 * len(feature_cat_cols), 4))
    if len(feature_cat_cols) == 1:
        axes = [axes]
    for i, col in enumerate(feature_cat_cols):
        sns.countplot(data=df, x=col, hue=col, ax=axes[i], palette='Set2',  legend=False)
        axes[i].set_title(f'Value Counts: {col}')
        axes[i].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('plots/categorical_counts.png', dpi=300)
    plt.close()

# 11. Check correlations between numerical features and delivery-time target
print("\n" + "="*60)
print("11. CORRELATION ANALYSIS WITH TARGET")
print("="*60)
corr_matrix = df[feature_num_cols + [target_col]].corr()
print("Correlation Matrix:")
print(corr_matrix)
print("\nCorrelation with Target (Delivery_Time_min):")
print(corr_matrix[target_col].sort_values(ascending=False))

# Plot Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300)
plt.close()

# Save script summary log
with open('eda_summary.txt', 'w') as f:
    f.write("EDA Completed successfully.\n")

print("\nEDA and Visualizations Complete. Saved plots to 'plots/' directory.")
