import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.plot import add_changepoints_to_plot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.ensemble import IsolationForest

class ProphetVisualizer:
    def __init__(self, model, train_df, test_df, forecast, train_forecast):
        """
        Initialize the visualizer with model and data.
        """
        self.model = model
        self.train_df = train_df
        self.test_df = test_df
        self.forecast = forecast
        self.train_forecast = train_forecast
        
        sns.set_style("whitegrid")  # or any other style like "darkgrid", "white", "dark", "ticks"
        
    def plot_components(self, output_path='prophet_components.png'):
        """Plot the components of the Prophet forecast."""
        fig = self.model.plot_components(self.forecast)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_forecast_vs_actual(self, output_path='forecast_vs_actual.png'):
        """Plot the forecast against actual values with improved visualization."""
        plt.figure(figsize=(15, 8))
        
        # Plot training data with alpha for better visibility
        plt.plot(self.train_df['ds'], self.train_df['y'], 
                label='Training Actual', color='blue', alpha=0.6)
        plt.plot(self.train_df['ds'], self.train_forecast['yhat'], 
                label='Training Predicted', color='red', alpha=0.6)
        
        # Add overlap period with different color
        overlap_mask = (self.train_df['ds'] >= self.train_df['ds'].max() - pd.Timedelta(days=7))
        plt.plot(self.train_df.loc[overlap_mask, 'ds'], 
                self.train_df.loc[overlap_mask, 'y'],
                color='purple', alpha=0.5, label='Overlap Period')
        
        # Plot forecast
        plt.plot(self.test_df['ds'], self.forecast['yhat'], 
                label='Forecast', color='green', alpha=0.8)
        
        # Add confidence intervals
        plt.fill_between(self.forecast['ds'], 
                        self.forecast['yhat_lower'], 
                        self.forecast['yhat_upper'], 
                        color='green', alpha=0.2, label='Confidence Interval')
        
        plt.title('Temperature Forecast vs Actual Values')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_residuals(self, output_path='residuals_analysis.png'):
        """Plot enhanced residuals analysis."""
        residuals = self.train_df['y'] - self.train_forecast['yhat']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time with trend line
        ax1.plot(self.train_df['ds'], residuals, 'o', alpha=0.5)
        z = np.polyfit(range(len(residuals)), residuals, 1)
        p = np.poly1d(z)
        ax1.plot(self.train_df['ds'], p(range(len(residuals))), "r--", alpha=0.8)
        ax1.set_title('Residuals over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residual')
        ax1.tick_params(axis='x', rotation=45)
        
        # Residuals histogram with KDE
        sns.histplot(residuals, kde=True, ax=ax2, bins=50, alpha=0.5)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Count')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        
        # Residuals vs Fitted with LOWESS
        ax4.scatter(self.train_forecast['yhat'], residuals, alpha=0.5)
        from scipy.stats import binned_statistic
        bins = np.linspace(min(self.train_forecast['yhat']), 
                          max(self.train_forecast['yhat']), 20)
        mean_residuals, _, _ = binned_statistic(self.train_forecast['yhat'], 
                                              residuals, 'mean', bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax4.plot(bin_centers, mean_residuals, 'r-', linewidth=2)
        ax4.set_title('Residuals vs Fitted')
        ax4.set_xlabel('Fitted Values')
        ax4.set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def detect_and_handle_outliers(data, columns, contamination=0.1):
    """
    Enhanced outlier detection and handling using Isolation Forest with rolling statistics.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    for column in columns:
        # Use rolling statistics for more context-aware outlier replacement
        rolling_median = data[column].rolling(window=24, center=True, min_periods=1).median()
        rolling_std = data[column].rolling(window=24, center=True, min_periods=1).std()
        
        X = data[column].values.reshape(-1, 1)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Replace outliers with value within 2 standard deviations of rolling median
        outlier_mask = outlier_labels == -1
        replacement_values = rolling_median[outlier_mask] + \
                           np.random.normal(0, rolling_std[outlier_mask] * 0.5)
        data.loc[outlier_mask, column] = replacement_values
    return data

def create_advanced_features(data, is_train=True):
    """
    Create advanced features with smooth transitions and better handling of temporal aspects.
    Modified to handle both training and test data.
    """
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
    # Enhanced temporal features
    data['hour'] = data['Datetime'].dt.hour
    data['month'] = data['Datetime'].dt.month
    data['day_of_week'] = data['Datetime'].dt.dayofweek
    
    # Improved cyclic features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Enhanced pollution index with exponential moving averages
    data['pollution_index'] = (data['SO2_concentration'] + 
                             data['NO2_concentration'] + 
                             data['CO_concentration']) / 3
    
    # Use EMA for smoother transitions
    for window in [3, 6, 12, 24]:
        data[f'pollution_index_ema_{window}'] = (
            data['pollution_index'].ewm(span=window, adjust=False).mean())
        data[f'pollution_index_volatility_{window}'] = (
            data['pollution_index'].ewm(span=window, adjust=False).std())
    
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data

def prepare_data_for_prophet(train_data, test_data, target_column="Temperature"):
    """
    Prepare data for Prophet with improved feature engineering and scaling.
    """
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # Create advanced features
    train_df = create_advanced_features(train_df, is_train=True)
    test_df = create_advanced_features(test_df, is_train=False)
    
    # Scale features
    scaler = StandardScaler()
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Temperature', 'hour', 'month', 'day_of_week', 'ID']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    
    # Rename columns for Prophet
    train_df = train_df.rename(columns={"Datetime": "ds", target_column: "y"})
    test_df = test_df.rename(columns={"Datetime": "ds"})
    
    # Sort by date
    train_df = train_df.sort_values('ds').reset_index(drop=True)
    test_df = test_df.sort_values('ds').reset_index(drop=True)
    
    # Select regressors (excluding certain columns)
    exclude_cols = ['ds', 'y', 'hour', 'month', 'day_of_week', 'Datetime', 'ID', 'Temperature']
    regressors = [col for col in train_df.columns if col not in exclude_cols]
    
    return train_df, test_df, regressors



print("Loading data...")
train_data = pd.read_csv("Train_concatenated_file.csv")
test_data = pd.read_csv("Test_concatenated_file.csv")

# Prepare data with overlap period
print("Preparing data...")
train_df, test_df, regressors = prepare_data_for_prophet(train_data, test_data)

# Add overlap period
overlap_days = 7
train_end = train_df['ds'].max()
overlap_start = train_end - pd.Timedelta(days=overlap_days)

# Configure Prophet model
print("Training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    changepoint_prior_scale=0.5,
    changepoint_range=0.98,
    interval_width=0.95
)

# Add custom seasonalities
model.add_seasonality(
    name='hourly',
    period=24,
    fourier_order=8
)

# Add regressors with appropriate modes
for regressor in regressors:
    if 'rolling' in regressor or 'ema' in regressor:
        mode = 'additive'
    else:
        mode = 'multiplicative'
    model.add_regressor(regressor, mode=mode)

# Fit model
model.fit(train_df)

# Cross-validation
print("Performing cross-validation...")
df_cv = cross_validation(model, initial='180 days', 
                       period='30 days', horizon='7 days')
df_p = performance_metrics(df_cv)
print("\nCross-validation metrics:")
print(df_p)

# Generate forecasts
print("Generating forecasts...")
future = test_df[['ds']].copy()
for regressor in regressors:
    future[regressor] = test_df[regressor]

forecast = model.predict(future)
train_forecast = model.predict(train_df[['ds'] + regressors])

# Evaluate
mse = mean_squared_error(train_df['y'], train_forecast['yhat'])
mae = mean_absolute_error(train_df['y'], train_forecast['yhat'])
print(f"\nTrain MSE: {mse:.4f}")
print(f"Train MAE: {mae:.4f}")

# Create visualizations
print("Creating visualizations...")
visualizer = ProphetVisualizer(model, train_df, test_df, forecast, train_forecast)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Generate individual plots
print("Generating component plots...")
visualizer.plot_components('plots/prophet_components.png')

print("Generating forecast vs actual plots...")
visualizer.plot_forecast_vs_actual('plots/forecast_vs_actual.png')

print("Generating residuals analysis...")
visualizer.plot_residuals('plots/residuals_analysis.png')

# Save predictions
print("Saving predictions...")
predictions_df = pd.DataFrame({
    'ID': test_data['ID'],
    'Predicted_Temperature': forecast['yhat']
})
predictions_df.to_csv('Phrophet_predictions.csv', index=False)
print("Predictions saved successfully.")

# Print final success message
print("\nAll operations completed successfully!")
print(f"Plots saved in: {os.path.abspath('plots')}")
print(f"Predictions saved in: {os.path.abspath('temperature_predictions.csv')}")


    
    
