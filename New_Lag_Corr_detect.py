import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.signal import correlate
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(df, threshold=5.0):
    
    print("\nChecking for multicollinearity...\n")
    
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(numeric_df.shape[1])]
    
    print("Variance Inflation Factors:")
    print(vif_data.sort_values('VIF', ascending=False))
    print("\nFeatures with high VIF (> {}):".format(threshold))
    print(vif_data[vif_data["VIF"] > threshold].sort_values('VIF', ascending=False))
    

    correlation_matrix = numeric_df.corr()
    
    
    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:  
                high_correlation_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })
    
  
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    if high_correlation_pairs:
        print("\nHighly correlated feature pairs (|correlation| > 0.8):")
        for pair in high_correlation_pairs:
            print(f"{pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated feature pairs found (threshold: 0.8)")
    
    return {
        'vif_data': vif_data,
        'high_correlation_pairs': high_correlation_pairs,
        'correlation_matrix': correlation_matrix
    }

def preprocess_data(df):
    print("\nPreprocessing data: Handling missing values and outliers...\n")
    df_clean = df.copy()
    
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['category', 'object']).columns
    
    print(f"Numerical columns: {list(numeric_cols)}")
    print(f"Categorical columns: {list(categorical_cols)}\n")
    
   
    print("Interpolating missing values for numerical columns using time method...")
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
    
    
    print("Handling missing values for categorical columns using forward fill...")
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna(method='ffill')
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna(method='bfill')
    
    # Handle outliers using IQR method for numerical columns
    #def remove_outliers(col):
     #   Q1 = col.quantile(0.25)
      #  Q3 = col.quantile(0.75)
       # IQR = Q3 - Q1
       # lower = Q1 - 1.5 * IQR
       # upper = Q3 + 1.5 * IQR
       # print(f"Removing outliers in column '{col.name}': Lower bound = {lower:.2f}, Upper bound = {upper:.2f}")
       # return col.clip(lower=lower, upper=upper)
    
 #   print(f"\nProcessing outliers for numeric columns...\n")
  #  for col in numeric_cols:
   #     if df_clean[col].nunique() > 1:  # Only process columns with multiple unique values
    #        print(f"Processing column '{col}' for outliers...")
     #       df_clean[col] = remove_outliers(df_clean[col])
      #  else:
       #     print(f"Skipping column '{col}' as it contains constant values.\n")
    #
   # print("\nData preprocessing completed.\n")
    return df_clean



def analyze_temperature_relationships(df):
 
    print("Starting analysis for temperature relationships...\n")
    
    print("Step 1: Preprocessing the data...")
    df_clean = preprocess_data(df)
    
    
    print("\nStep 2: Checking for multicollinearity...")
    multicollinearity_results = check_multicollinearity(df_clean)
    
 
    print("\nStep 3: Running correlation analysis...")
    pearson_corr, spearman_corr = analyze_correlations(df_clean)
    
    print("\nStep 4: Running Granger causality analysis...")
    granger_results, significant_features = improved_granger_causality(df_clean)
    print(f"\nSignificant features causing 'Temperature': {significant_features}")
    
    print("\nStep 5: Running cross-correlation analysis...")
    cross_corr_results = improved_cross_correlation(df_clean)
    
   
    print("\nAnalysis completed. Returning results...\n")
    return {
        'multicollinearity': multicollinearity_results,
        'correlations': {'pearson': pearson_corr, 'spearman': spearman_corr},
        'granger_causality': {'results': granger_results, 'significant_features': significant_features},
        'cross_correlation': cross_corr_results,
        'cleaned_data': df_clean
    }


def get_feature_recommendations(multicollinearity_results, vif_threshold=5.0, corr_threshold=0.8):
    """
    Provide recommendations for handling multicollinearity
    """
    recommendations = []
    
    
    high_vif_features = multicollinearity_results['vif_data'][
        multicollinearity_results['vif_data']['VIF'] > vif_threshold
    ]['Feature'].tolist()
    
    if high_vif_features:
        recommendations.append(f"\nFeatures with high VIF (>{vif_threshold}):")
        for feature in high_vif_features:
            recommendations.append(f"- Consider removing or transforming: {feature}")
    
    
    if multicollinearity_results['high_correlation_pairs']:
        recommendations.append(f"\nHighly correlated feature pairs (>{corr_threshold}):")
        for pair in multicollinearity_results['high_correlation_pairs']:
            recommendations.append(
                f"- Consider keeping only one of: {pair['feature1']} <-> {pair['feature2']} "
                f"(correlation: {pair['correlation']:.3f})"
            )
    
    if not recommendations:
        recommendations.append("\nNo significant multicollinearity issues detected.")
    
    return "\n".join(recommendations)

def analyze_correlations(df, target='Temperature'):
    print(f"\nAnalyzing correlations with target '{target}'...\n")
    
    
    numeric_df = df.select_dtypes(include=[np.number])
    
   
    print("Calculating Pearson correlation...")
    pearson_corr = numeric_df.corr(method='pearson')[target].sort_values(ascending=False)
    
    
    print("Calculating Spearman correlation...")
    spearman_corr = numeric_df.corr(method='spearman')[target].sort_values(ascending=False)
    
    
    print("\nTop correlated features (Pearson):\n", pearson_corr.head())
    print("\nTop correlated features (Spearman):\n", spearman_corr.head())
    
    
    print("\nPlotting Pearson vs. Spearman correlations...\n")
    plt.figure(figsize=(12, 6))
    pd.DataFrame({
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    }).plot(kind='bar')
    plt.title(f'Feature Correlations with {target}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return pearson_corr, spearman_corr

def improved_granger_causality(df, target='Temperature', maxlag=5, significance_level=0.05):
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    significant_features = []
    results = {}
    
    print(f"\nPerforming Granger causality tests with target '{target}' and maxlag={maxlag}:\n")
    
    for col in numeric_df.columns:
        if col != target and numeric_df[col].nunique() > 1:
            print(f"Testing if '{col}' causes '{target}'...\n")
            
            try:
                
                test_result = grangercausalitytests(numeric_df[[target, col]], maxlag=maxlag, verbose=True)
                
                print(f"\nGranger Causality Test for {col} and {target} completed successfully.\n")
                print(f"Results for '{col}' causing '{target}':\n")
                
                
                for lag in range(1, maxlag + 1):
                    f_test = test_result[lag][0]['ssr_ftest']
                    chi2_test = test_result[lag][0]['ssr_chi2test']
                    lr_test = test_result[lag][0]['lrtest']
                    param_test = test_result[lag][0]['params_ftest']
                    
                    print(f"Lag {lag} Results:")
                    print(f"  - F test:         F={f_test[0]:.4f}, p={f_test[1]:.4f}, df_num={f_test[2]}, df_denom={f_test[3]}")
                    print(f"  - Chi2 test:      chi2={chi2_test[0]:.4f}, p={chi2_test[1]:.4f}, df={chi2_test[2]}")
                    print(f"  - Likelihood ratio test: chi2={lr_test[0]:.4f}, p={lr_test[1]:.4f}, df={lr_test[2]}")
                    print(f"  - Parameter F test: F={param_test[0]:.4f}, p={param_test[1]:.4f}, df_num={param_test[2]}, df_denom={param_test[3]}")
                    print("---------------------------------------------------")
                
                
                p_values = [test_result[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
                results[col] = p_values

                
                if any(p < significance_level for p in p_values):
                    print(f"'{col}' significantly causes '{target}' at least at one lag (p < {significance_level}).\n")
                    significant_features.append(col)
                else:
                    print(f"No significant Granger causality found for '{col}' causing '{target}' (p >= {significance_level}).\n")
            except Exception as e:
                print(f"Granger causality test failed for '{col}': {e}")
                results[col] = [np.nan] * maxlag

    print("\nGranger Causality Test Summary:")
    print(f"Significant features causing '{target}': {significant_features if significant_features else 'None'}")
    
    return results, significant_features

def improved_cross_correlation(df, target='Temperature', max_lag=50):
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    print(f"\nPerforming cross-correlation analysis with target '{target}'...\n")
    
    results = {}
    n_cols = len([col for col in numeric_df.columns if col != target and numeric_df[col].nunique() > 1])
    n_rows = (n_cols - 1) // 3 + 1
    
    plt.figure(figsize=(15, 4 * n_rows))
    plot_idx = 1

    for col in numeric_df.columns:
        if col != target and numeric_df[col].nunique() > 1:
            print(f"Analyzing cross-correlation between '{target}' and '{col}'...\n")
            
            
            x = (numeric_df[target] - numeric_df[target].mean()) / numeric_df[target].std()
            y = (numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std()
            
            
            corr = correlate(x, y, mode='full')
            lags = np.arange(-max_lag, max_lag + 1)
            corr = corr[len(corr)//2 - max_lag:len(corr)//2 + max_lag + 1]
            
            
            max_corr_idx = np.argmax(np.abs(corr))
            max_corr = corr[max_corr_idx]
            optimal_lag = lags[max_corr_idx]
            
            print(f"  - Maximum correlation: {max_corr:.4f} at lag {optimal_lag}\n")
            results[col] = {'max_correlation': max_corr, 'optimal_lag': optimal_lag}
            
            
            plt.subplot(n_rows, 3, plot_idx)
            plt.plot(lags, corr)
            plt.title(f'{col}\nMax Corr: {max_corr:.2f} at lag {optimal_lag}')
            plt.xlabel('Lag')
            plot_idx += 1
            
    plt.tight_layout()
    plt.show()
    
    print("\nCross-correlation analysis completed.\n")
    return results




df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/Train_concatenated_file.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
df['Day'] = df['Datetime'].dt.day
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month
df['Year'] = df['Datetime'].dt.year
df['pollution_index'] = (df['SO2_concentration'] + df['NO2_concentration'] + df['CO_concentration']) / 3
df['SO2_to_NO2_ratio'] = df['SO2_concentration'] / df['NO2_concentration']
df['O3_to_NO2_ratio'] = df['O3_concentration'] / df['NO2_concentration']
df['weighted_pollution_index'] = (
        df['SO2_concentration'] * 0.4 +
        df['NO2_concentration'] * 0.3 +
        df['CO_concentration'] * 0.3
    )

df['time_of_day'] = pd.cut(
        df['Hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )

df['is_weekend'] = df['Day_of_Week'].isin([6, 7]).astype(int)
df['season'] = pd.cut(
        df['Month'],
        bins=[0, 3, 6, 9, 12],
        labels=['winter', 'spring', 'summer', 'fall']
    )

df['particulate_pollution_ratio'] = df['Particulate_matter'] / df['pollution_index']

df.set_index('Datetime', inplace=True)


results = analyze_temperature_relationships(df)
