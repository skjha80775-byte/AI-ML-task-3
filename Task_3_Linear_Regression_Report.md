# Task 3: Linear Regression — House Price Prediction

## Objective
Implement and understand simple and multiple linear regression using Pandas,
Scikit-learn, and Matplotlib.

## Dataset
A self-contained synthetic house-price dataset with 1,000 observations was used.
Features: Area, Bedrooms, Bathrooms, Age, Distance, Location Score, and Parking.
Target: House Price.

## Method
1. Loaded the CSV using Pandas.
2. Checked the shape, first rows, and missing values.
3. Split the data into 80% training and 20% testing sets using `random_state=42`.
4. Trained `LinearRegression` for multiple linear regression.
5. Evaluated using MAE, MSE, RMSE, and R².
6. Trained a simple regression using Area only and plotted its regression line.
7. Interpreted the learned coefficients.

## Multiple Regression Results
- MAE: 191,936.03
- MSE: 61,148,362,211.38
- RMSE: 247,281.95
- R²: 0.9970

## Learned Model
Price = 755,409.17 +4,505.89×Area_sqft +185,630.47×Bedrooms +247,194.40×Bathrooms -20,695.30×Age_years -35,146.05×Distance_km +299,678.97×Location_score +91,514.09×Parking

## Simple Regression Results
- Feature: Area_sqft
- R²: 0.9508
- Area coefficient: 4,462.65

## Interpretation
The multiple regression model estimates house price as a linear combination of
all seven input features. A positive coefficient means that, while holding the
other features constant, increasing that feature is associated with a higher
predicted price; a negative coefficient means the opposite.

R² indicates the proportion of variance in the test target explained by the
model. MAE is the average absolute prediction error, while MSE gives greater
weight to larger errors.

## Conclusion
Multiple linear regression is more informative than using house area alone
because it combines several property characteristics. The test metrics show
how accurately the fitted model generalizes to unseen houses.
