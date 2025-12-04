import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load data
train_file_path = "data/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head(3))

# Dropping data with too many missing values
columns_to_drop = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 
                   'FireplaceQu', 'LotFrontage', 'GarageType', 'GarageYrBlt', 
                   'GarageFinish', 'GarageQual', 'GarageCond']
dataset_df = dataset_df.drop(columns_to_drop, axis=1)

# New features
dataset_df['TotalSqFt'] = dataset_df['GrLivArea'] + dataset_df['TotalBsmtSF']
dataset_df['TotalBaths'] = (dataset_df['FullBath'] + 
                            (0.5 * dataset_df['HalfBath']) + 
                            dataset_df['BsmtFullBath'] + 
                            (0.5 * dataset_df['BsmtHalfBath']))

# Identify columns to encode
columns_to_encode = ['MSZoning', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 
                     'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                     'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 
                     'CentralAir', 'Electrical', 'PavedDrive', 'SaleType', 'SaleCondition']

# Handle missing values
for col in columns_to_encode:
    if col in dataset_df.columns:
        dataset_df[col] = dataset_df[col].fillna('None')

# One hot encoding
dataset_df = pd.get_dummies(dataset_df, columns=columns_to_encode, drop_first=True)
print(f"Dataset shape after one hot encoding: {dataset_df.shape}")


numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns
dataset_df[numeric_cols] = dataset_df[numeric_cols].fillna(dataset_df[numeric_cols].median())
non_numeric_cols = dataset_df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    dataset_df = pd.get_dummies(dataset_df, columns=non_numeric_cols, drop_first=True)

# Standardization
columns_to_standardize = ['GarageArea', 'OpenPorchSF', '1stFlrSF', 'TotalSqFt']
scaler = StandardScaler()

columns_to_standardize = [col for col in columns_to_standardize if col in dataset_df.columns]
if columns_to_standardize:
    dataset_df[columns_to_standardize] = scaler.fit_transform(dataset_df[columns_to_standardize])

# Function to visualize data
def visualize_data():
    print("\n=== SalePrice Statistics ===")
    print(dataset_df['SalePrice'].describe())
    
    plt.figure(figsize=(9, 8))
    sns.histplot(dataset_df['SalePrice'], color='g', bins=100, alpha=0.4)
    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.show()
    
    df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
    df_num.hist(figsize=(16, 12), bins=50, xlabelsize=8, ylabelsize=8)
    plt.suptitle("Distributions of Numeric Features", fontsize=12)
    plt.tight_layout(pad=1.0)
    plt.show()

# visualize_data()


# TRAIN/VAL/TEST Split
X = dataset_df.drop('SalePrice', axis=1)
y = dataset_df['SalePrice']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print(f"\nData types in X:")
print(X.dtypes.value_counts())

# Check for any object/string columns
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
assert X.select_dtypes(include=['object']).shape[1] == 0, "Non-numeric columns still exist!"
print("✓ All features are numeric")


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# Train random forest model
print("\n Training Random Forest ")

rf_baseline = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_baseline.fit(X_train, y_train)

y_train_pred = rf_baseline.predict(X_train)
y_val_pred = rf_baseline.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Baseline Model Performance:")
print(f"  Train RMSE: ${train_rmse:,.2f}")
print(f"  Val RMSE: ${val_rmse:,.2f}")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Val R²: {val_r2:.4f}")


# Hyperparameter Tuning

print("\n Hyperparameter Tuning with RandomizedSearchCV ")

param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=20,  
    cv=3,  
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

print("Running RandomizedSearchCV (this may take several minutes)...")
rf_random.fit(X_train, y_train)
print(f"\nBest parameters found: {rf_random.best_params_}")
print(f"Best CV RMSE: ${np.sqrt(-rf_random.best_score_):,.2f}")

final_model = rf_random.best_estimator_


# Model Eval

print("\n Final Model Evaluation ")

y_train_pred_final = final_model.predict(X_train)
y_val_pred_final = final_model.predict(X_val)
y_test_pred_final = final_model.predict(X_test)

train_rmse_final = np.sqrt(mean_squared_error(y_train, y_train_pred_final))
val_rmse_final = np.sqrt(mean_squared_error(y_val, y_val_pred_final))
test_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred_final))

train_mae_final = mean_absolute_error(y_train, y_train_pred_final)
val_mae_final = mean_absolute_error(y_val, y_val_pred_final)
test_mae_final = mean_absolute_error(y_test, y_test_pred_final)

train_r2_final = r2_score(y_train, y_train_pred_final)
val_r2_final = r2_score(y_val, y_val_pred_final)
test_r2_final = r2_score(y_test, y_test_pred_final)

print("\nFinal Tuned Model Performance:")
print(f"{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
print("-" * 60)
print(f"{'RMSE':<15} ${train_rmse_final:<14,.2f} ${val_rmse_final:<14,.2f} ${test_rmse_final:<14,.2f}")
print(f"{'MAE':<15} ${train_mae_final:<14,.2f} ${val_mae_final:<14,.2f} ${test_mae_final:<14,.2f}")
print(f"{'R²':<15} {train_r2_final:<15.4f} {val_r2_final:<15.4f} {test_r2_final:<15.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n Top 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Visualization of predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_val_pred_final, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Validation Set: Actual vs Predicted\nR² = {val_r2_final:.4f}')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_final, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2_final:.4f}')

plt.tight_layout()
plt.show()

# Save the model
joblib.dump(final_model, 'final_housing_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n Model and scaler saved")