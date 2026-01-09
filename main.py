import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#These are all random dates, it is not trained on real-world data, but it simulates real world scenarios

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
n = len(dates)

# (Higher sales in Dec, Lower in Jan)
seasonality = np.sin(np.linspace(0, 3.14 * 4, n)) * 10000

# (Business growing over time)
trend = np.linspace(0, 5000, n)

# (holidays)
holidays = np.zeros(n)
holidays[[5, 47, 51, 57, 99, 103]] = 1 # Random weeks marked as holidays since this is just a simulation
holiday_effect = holidays * 20000  # $20k spike on holidays

# Base Sales + Noise
base_sales = 50000 + np.random.normal(0, 2000, n)
weekly_sales = base_sales + seasonality + trend + holiday_effect

# DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Weekly_Sales': weekly_sales,
    'Is_Holiday': holidays,
    'Temperature': np.random.uniform(30, 90, n), # Random Temp
    'Fuel_Price': np.random.uniform(3.0, 4.5, n) # Random Fuel
})

print("✅ Data Generated Successfully (Rows: 105)")


df['Week'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# "What were sales last week?"
df['Lag_1'] = df['Weekly_Sales'].shift(1)
df['Lag_2'] = df['Weekly_Sales'].shift(2)

# "What was the average of the last 4 weeks?"
df['Rolling_Mean_4'] = df['Weekly_Sales'].rolling(window=4).mean()

# Drop NaN
df.dropna(inplace=True)

# Predictors (X) and Target (y)
features = ['Is_Holiday', 'Temperature', 'Fuel_Price', 'Week', 'Month', 'Lag_1', 'Lag_2', 'Rolling_Mean_4']
X = df[features]
y = df['Weekly_Sales']

# (Train on 2023, Test on 2024)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
predictions = rf.predict(X_test)


# Accuracy Metrics
mae = mean_absolute_error(y_test, predictions)
accuracy = 100 - (mae / y_test.mean() * 100)

print("-" * 30)
print(f"Model Accuracy: {accuracy:.2f}%")
print(f"Mean Absolute Error: ${mae:,.2f}")
print("-" * 30)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['Date'].iloc[train_size:], y_test, label='Actual Sales', color='blue', linewidth=2)
plt.plot(df['Date'].iloc[train_size:], predictions, label='AI Forecast', color='orange', linestyle='--', linewidth=2)
plt.title('Retail Demand Forecast: AI vs Actuals')
plt.xlabel('Date')
plt.ylabel('Weekly Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature Importance Plot
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('What Drives Sales? (Feature Importance)')
plt.barh(range(len(indices)), importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()