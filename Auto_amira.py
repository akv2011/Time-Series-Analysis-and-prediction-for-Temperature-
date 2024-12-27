import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

# Load the dataset
df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/Train_concatenated_file.csv')

# Convert 'Datetime' to datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

# Set Datetime as index for time series analysis
df.set_index('Datetime', inplace=True)

# Perform stationarity test
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df['Temperature'])
print("p-value for Temperature stationarity:", pvalue)
print("If p-value > 0.05, data is not stationary")

# Use auto_arima to find best model parameters
arima_model = auto_arima(
    df['Temperature'], 
    start_p=1, d=1, start_q=1, 
    max_p=5, max_q=5, max_d=5, 
    seasonal=True, 
    m=365,  # assuming monthly seasonality 
    trace=True, 
    error_action='ignore',   
    suppress_warnings=True,  
    stepwise=True
)

print(arima_model.summary())

# Split data into train and test
size = int(len(df) * 0.8)
train_data = df['Temperature'][:size]
test_data = df['Temperature'][size:]

# Fit SARIMAX model based on auto_arima suggestion
# Note: You might need to adjust these parameters based on auto_arima output
model = SARIMAX(
    train_data,  
    order=(1, 1, 1),  # p,d,q from auto_arima
    seasonal_order=(1, 1, 1, 12)  # P,D,Q,m 
)

results = model.fit()
print(results.summary())

# Make predictions
train_prediction = results.predict(start=0, end=len(train_data)-1)
test_prediction = results.predict(start=len(train_data), end=len(df)-1)

# Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(train_data, train_prediction))
test_rmse = math.sqrt(mean_squared_error(test_data, test_prediction))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Forecast for future periods
forecast_periods = 24  # adjust as needed
forecast = results.predict(
    start=len(df), 
    end=len(df) + forecast_periods, 
    typ='levels'
).rename('Forecast')

# Visualization
plt.figure(figsize=(15,8))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Temperature Forecast')
plt.legend()
plt.show()

# Prepare final predictions for submission
# Load test dataset
test_df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/test_modi_c0.csv')

# Create results DataFrame
results_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Predicted_Temperature': forecast[:len(test_df)]
})
results_df.to_csv('predicted_temperatures_arima.csv', index=False)

# Optional: Detailed results visualization
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
plt.title('Training and Test Data vs Predictions')
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data')
plt.plot(test_prediction, label='Test Predictions', linestyle='--')
plt.legend()

plt.subplot(2,1,2)
plt.title('Forecast')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.tight_layout()
plt.show()