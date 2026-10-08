# Task 3: Linear Regression - House Price Prediction
# Tools: Scikit-learn, Pandas, Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import and preprocess the dataset
df = pd.read_csv("house_price_dataset.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Numeric columns are already suitable for linear regression.
# Fill any accidental missing numeric values with the column median.
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

features = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Age_years",
    "Distance_km", "Location_score", "Parking"
]
target = "Price"

X = df[features]
y = df[target]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Fit multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 4. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Multiple Linear Regression Results ---")
print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R^2  : {r2:.4f}")

print("\nIntercept:", f"{model.intercept_:,.2f}")
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:18s}: {coef:,.2f}")

# 5. Simple linear regression and regression line
X_simple = df[["Area_sqft"]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.20, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
y_pred_s = simple_model.predict(X_test_s)

print("\n--- Simple Linear Regression (Area -> Price) ---")
print(f"Intercept : {simple_model.intercept_:,.2f}")
print(f"Coefficient (Area): {simple_model.coef_[0]:,.2f}")
print(f"R^2       : {r2_score(y_test_s, y_pred_s):.4f}")

order = np.argsort(X_test_s["Area_sqft"].to_numpy())
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s["Area_sqft"], y_test_s, alpha=0.55, label="Actual")
plt.plot(
    X_test_s["Area_sqft"].to_numpy()[order],
    y_pred_s[order],
    linewidth=2,
    label="Regression line"
)
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression: Area vs House Price")
plt.legend()
plt.tight_layout()
plt.show()

# Multiple regression: actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.55)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, linewidth=2, label="Ideal prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
