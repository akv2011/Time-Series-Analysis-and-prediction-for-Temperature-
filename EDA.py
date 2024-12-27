

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('dark_background')


df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/train_modi_c0.csv')
print(df.dtypes)






df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')


print(df.dtypes)

print("\nNumber of NaN values in each column:")
print(df.isna().sum())

df.info()

df.set_index('Datetime', inplace=True) 
plt.plot(df['Temperature'])



for column in df.columns:
    plt.plot(df[column], label=column)
    plt.title(f"Plot of {column}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()
    plt.show() 


from statsmodels.tsa.stattools import adfuller
for column in df.columns:
    adf_result = adfuller(df[column].dropna())  
    adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adf_result
    print(f"Results for {column}:")
    print("ADF Statistic:", adf)
    print("p-value:", pvalue)
    print("Critical Values:", critical_values_)
    if pvalue > 0.05:
        print("The data is not stationary (p > 0.05)\n")
    else:
        print("The data is stationary (p <= 0.05)\n")



df['City_ID'].unique()
df['Precipitation'].unique()


df['year'] = [d.year for d in df.index]

def convert_to_month(df):
    """
    Converts datetime index to month abbreviations, handling NaT values.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe with datetime index
        
    Returns:
    list: Month abbreviations with 'N/A' for NaT values
    """
    return ['N/A' if pd.isna(d) else d.strftime('%b') for d in df.index]

df['month'] = convert_to_month(df)


def convert_to_hour(df):
    """
    Converts datetime index to hours (24-hour format), handling NaT values.
    
    Parameters:
    df : pandas.DataFrame
        The dataframe with datetime index
        
    Returns:
    list: Hours with 'N/A' for NaT values
    """
    return ['N/A' if pd.isna(d) else d.strftime('%H') for d in df.index]


df['hour'] = convert_to_hour(df)

years = df['year'].unique()
print(years)


def analyze_duplicates(df):
    
    
    duplicate_timestamps = df.index.duplicated(keep=False)
    duplicate_rows = df[duplicate_timestamps]
    
    print("=== Duplicate Analysis ===")
    print(f"\nTotal number of rows: {len(df)}")
    print(f"Number of duplicate timestamps: {duplicate_timestamps.sum()}")
    print(f"Percentage of duplicates: {(duplicate_timestamps.sum()/len(df)*100):.2f}%")
    
    if len(duplicate_rows) > 0:
        print("\nFirst few rows with duplicate timestamps:")
        print(duplicate_rows.head())
        
        
        duplicate_counts = duplicate_rows.groupby(level=0).size()
        print("\nFrequency of duplications:")
        print(duplicate_counts.value_counts().sort_index())
        
       
        print("\nExample of duplicate records:")
        for idx in duplicate_counts.head().index:
            print(f"\nDuplicates for timestamp {idx}:")
            print(df.loc[idx])


analyze_duplicates(df)


def plot_boxplot_unique_index(df):
    """
    Creates a boxplot while maintaining data structure by creating unique index
    """
    df_unique = df.copy()
    df_unique.index = pd.Index(range(len(df_unique)))
    return sns.boxplot(x='hour', y='Temperature', data=df_unique)

plot_boxplot_unique_index(df)



from statsmodels.tsa.seasonal import seasonal_decompose 
def plot_time_series_decomposition(df, column='Temperature', model='additive'):
    
    decomposed = seasonal_decompose(df[column],
                                  model=model,
                                  period=24*30*12)  
    
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    
    
    fig, axes = plt.subplots(4, 1, figsize=(200, 100))
    
   
    axes[0].plot(df[column], label='Original', color='yellow')
    axes[0].set_title(f'Original {column} Time Series')
    axes[0].legend(loc='upper left')
    
 
    axes[1].plot(trend, label='Trend', color='yellow')
    axes[1].set_title('Trend')
    axes[1].legend(loc='upper left')
    

    axes[2].plot(seasonal, label='Seasonal', color='yellow')
    axes[2].set_title('Seasonal')
    axes[2].legend(loc='upper left')
    
   
    axes[3].plot(residual, label='Residual', color='yellow')
    axes[3].set_title('Residual')
    axes[3].legend(loc='upper left')
    
    
    plt.tight_layout()
    
    return fig, axes, decomposed


fig, axes, decomposed = plot_time_series_decomposition(df)
plt.show()









from statsmodels.tsa.stattools import acf

acf_144 = acf(df.Temperature    , nlags=2160)


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Moisture_percent   ) 





############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

plt.style.use('dark_background')

# Load the dataset
df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/train_modi_c1.csv')

# Convert the Datetime column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

# Check for any NaN values
print("\nNumber of NaN values in each column:")
print(df.isna().sum())

# Set the Datetime column as the index
df.set_index('Datetime', inplace=True)

# Add year, month, and hour columns
df['year'] = [d.year for d in df.index]
df['month'] = ['N/A' if pd.isna(d) else d.strftime('%b') for d in df.index]
df['hour'] = ['N/A' if pd.isna(d) else d.strftime('%H') for d in df.index]

# Generate autocorrelation and partial autocorrelation plots for each column
for column in df.columns:
    acf_values = acf(df[column], nlags=50)
    pacf_values = pacf(df[column], nlags=50)

    # Calculate the 95% confidence intervals
    n = len(df[column])
    confidence = 1.96 / np.sqrt(n)

    # Print the lags where the autocorrelation is significant
    print(f"Significant Autocorrelation Lags for {column}:")
    for lag, acf_value in enumerate(acf_values):
        if abs(acf_value) > confidence:
            print(f"Lag {lag+1}: {acf_value:.2f}")

    # Partial autocorrelation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(np.arange(1, len(acf_values) + 1), acf_values)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=confidence, linestyle='--', color='gray')
    ax1.axhline(y=-confidence, linestyle='--', color='gray')
    ax1.set_title(f"Autocorrelation Plot for {column}")
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')

    ax2.plot(np.arange(1, len(pacf_values) + 1), pacf_values)
    ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.axhline(y=confidence, linestyle='--', color='gray')
    ax2.axhline(y=-confidence, linestyle='--', color='gray')
    ax2.set_title(f"Partial Autocorrelation Plot for {column}")
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')

    # Print the lags where the partial autocorrelation is significant
    print(f"\nSignificant Partial Autocorrelation Lags for {column}:")
    for lag, pacf_value in enumerate(pacf_values):
        if abs(pacf_value) > confidence:
            print(f"Lag {lag+1}: {pacf_value:.2f}")

    plt.tight_layout()
    plt.show()
    
    


#######################################################################################


import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TimeSeriesOutlierDetector:
    def __init__(self, data, timestamp_column=None, value_column=None):
        
        self.data = data.copy()
        if timestamp_column is not None:
            self.data.set_index(timestamp_column, inplace=True)
        self.value_column = value_column
        self.residuals = None
        
        
    def plot_outliers(self, outliers, title="Time Series with Outliers"):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        
        ax1.plot(self.data.index, self.data[self.value_column], 
                label='Original', alpha=0.5)
        outlier_points = self.data[self.value_column][outliers]
        ax1.scatter(outlier_points.index, outlier_points.values, 
                   color='red', label='Outliers')
        ax1.set_title(f"{title} - Original Series")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.data.index, self.residuals, 
                label='Residuals', alpha=0.5)
        residual_outliers = self.residuals[outliers]
        ax2.scatter(residual_outliers.index, residual_outliers.values, 
                   color='red', label='Outliers')
        ax2.set_title(f"{title} - Residuals")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def compute_residuals(self, period=24):
        
        decomposition = seasonal_decompose(
            self.data[self.value_column], 
            period=period,
            extrapolate_trend='freq'
        )
        self.residuals = decomposition.resid
        return self.residuals
    
    def get_dynamic_window(self):
        "
        data_length = len(self.data)
        
        if isinstance(self.data.index, pd.DatetimeIndex):
            avg_daily_points = self.data.resample('D').size().mean()
            window_size = int(max(24, min(avg_daily_points * 2, data_length * 0.1)))
        else:
            window_size = int(max(24, min(data_length * 0.1, 168)))
            
        return window_size

    def detect_stl_decomposition(self, threshold=2.5):
        
        if self.residuals is None:
            self.compute_residuals()
        
        outliers = np.abs(self.residuals) > threshold * np.std(self.residuals)
        return outliers
    
    def detect_rolling_zscore_outliers(self, threshold=2.5):
        
        if self.residuals is None:
            self.compute_residuals()
            
        window = self.get_dynamic_window()
        

        rolling_mean = self.residuals.rolling(window=window, center=True).mean()
        rolling_std = self.residuals.rolling(window=window, center=True).std()
        

        z_scores = np.abs((self.residuals - rolling_mean) / rolling_std)
        
        return z_scores > threshold
    
    def detect_rolling_percentile_outliers(self):
        
        if self.residuals is None:
            self.compute_residuals()
            
        window = self.get_dynamic_window()
        

        rolling_data = self.residuals.rolling(window=window, center=True)
        rolling_1st = rolling_data.quantile(0.01)
        rolling_99th = rolling_data.quantile(0.99)
        

        lower_outliers = self.residuals < rolling_1st
        upper_outliers = self.residuals > rolling_99th
        
        return lower_outliers | upper_outliers
    
    def detect_isolation_forest_outliers(self, contamination=0.1):
        
        if self.residuals is None:
            self.compute_residuals()
            

        X = pd.DataFrame({'residuals': self.residuals})
        

        if isinstance(self.data.index, pd.DatetimeIndex):
            X['hour'] = self.data.index.hour
            X['month'] = self.data.index.month
            X['year'] = self.data.index.year
            
   
        X = X.fillna(X.mean())
            
 
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
      
        predictions = iso_forest.fit_predict(X)
        return predictions == -1
    
    def detect_mad_outliers(self, threshold=3.5):
        
        if self.residuals is None:
            self.compute_residuals()
            
        median = np.median(self.residuals)
        mad = np.median(np.abs(self.residuals - median))
        modified_zscore = 0.6745 * (self.residuals - median) / mad
        return np.abs(modified_zscore) > threshold
    
    
    def handle_outliers(self, outliers, method='interpolate'):
        
        data_cleaned = self.data[self.value_column].copy()
        
        if method == 'interpolate':
            data_cleaned[outliers] = np.nan
            data_cleaned = data_cleaned.interpolate(method='time')
        elif method == 'rolling_median':
            window = self.get_dynamic_window()
            rolling_median = self.data[self.value_column].rolling(
                window=window, center=True).median()
            data_cleaned[outliers] = rolling_median[outliers]
            
     
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data[self.value_column], 
                label='Original', alpha=0.5)
        plt.plot(self.data.index, data_cleaned, 
                label='Cleaned', alpha=0.8)
        plt.title(f"Original vs Cleaned Data - {self.value_column}")
        plt.legend()
        plt.grid(True)
        plt.show()
            
        return data_cleaned

def analyze_city_data(df, value_columns, City_ID, output_path=None):
    
    try:
      
        city_data = df[df['City_ID'] == City_ID].copy()
        
        if city_data.empty:
            raise ValueError(f"No data found for City {City_ID}")
        
 
        cleaned_city_data = city_data.copy()
            
        for column in value_columns:
            column = column.strip()
            
            if column not in city_data.columns:
                print(f"Warning: Column {column} not found in dataset. Skipping...")
                continue
                
            print(f"\nAnalyzing {column} for City {City_ID}")
            

            missing_count = city_data[column].isna().sum()
            if missing_count > 0:
                print(f"Warning: Found {missing_count} missing values in {column}")
                city_data[column] = city_data[column].interpolate(method='time')
            
          
            detector = TimeSeriesOutlierDetector(city_data, value_column=column)
            detector.compute_residuals()
            
         
            stl_outliers = detector.detect_stl_decomposition()
            zscore_outliers = detector.detect_rolling_zscore_outliers()
            percentile_outliers = detector.detect_rolling_percentile_outliers()
            isolation_forest_outliers = detector.detect_isolation_forest_outliers()
            mad_outliers = detector.detect_mad_outliers()
            
   
            combined_outliers = (stl_outliers.astype(int) + 
                               zscore_outliers.astype(int) + 
                               percentile_outliers.astype(int) +
                               isolation_forest_outliers.astype(int) +
                               mad_outliers.astype(int) >= 5)
            
            
            detector.plot_outliers(combined_outliers, 
                                 f"Outliers in {column} - City {City_ID}")
            
            cleaned_data = detector.handle_outliers(combined_outliers, method='rolling_median')
            cleaned_city_data[column] = cleaned_data
            
            print("\nOutlier Statistics:")
            print(f"Total points: {len(combined_outliers)}")
            print(f"Outliers detected: {combined_outliers.sum()}")
            print(f"Percentage: {(combined_outliers.sum()/len(combined_outliers))*100:.2f}%")
            
  
        if output_path:
            filename = f"cleaned_data_city_{City_ID}.csv"
            full_path = output_path.rstrip('/') + '/' + filename
            cleaned_city_data = cleaned_city_data.reset_index()
            cleaned_city_data.to_csv(full_path, index=False)
            print(f"\nCleaned data saved to: {full_path}")
            
        return cleaned_city_data
                
    except Exception as e:
        print(f"Error processing City {City_ID}: {str(e)}")
    
    


                
    

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/Test_concatenated_file.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)

    columns_to_analyze = [
        'Particulate_matter',
        'SO2_concentration',
        'O3_concentration',
        'CO_concentration',
        'NO2_concentration',
        'Presure',
        'Dew_point',
        'Precipitation',
        'Wind_speed',
        'Moisture_percent',
        'Temperature'
    ]

    for City_ID in [0, 1]:
        analyze_city_data(df, columns_to_analyze, City_ID)
        
        
output_directory = '/home/systemx86/Desktop/Hack/Spy_Zen_aws/'
import os
os.makedirs(output_directory, exist_ok=True)

cleaned_data_all_cities = pd.DataFrame()
for City_ID in [0, 1]:
    cleaned_city_data = analyze_city_data(
        df, 
        columns_to_analyze, 
        City_ID, 
        output_path=output_directory
    )
    if cleaned_city_data is not None:
        cleaned_data_all_cities = pd.concat([cleaned_data_all_cities, cleaned_city_data])


cleaned_data_all_cities.to_csv(f"{output_directory}/cleaned_data_all_cities.csv", index=False)
print(f"\nCombined cleaned data for all cities saved to: {output_directory}/cleaned_data_all_cities.csv")
