import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import stats
import optuna
import warnings
import xgboost as xgb
from optuna.pruners import MedianPruner
warnings.filterwarnings('ignore')

class TimeSeriesTemperaturePredictor:
    def __init__(self, forecast_horizon=1):
        self.forecast_horizon = forecast_horizon
        self.xgb_model = None 
        
        self.numeric_transformer = RobustScaler()
        self.categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        self.feature_importance = None
        self.timestamp_column = 'Datetime'
        self.required_columns = [
            'Presure', 'Dew_point', 'Moisture_percent', 
            'SO2_concentration', 'NO2_concentration', 'CO_concentration'
        ]
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_columns = None
        
    def validate_input_data(self, df, for_training=True):
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if for_training and 'Temperature' not in df.columns:
            raise ValueError("Training data must include 'Temperature' column")
            
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Missing timestamp column: {self.timestamp_column}")
            
        try:
            pd.to_datetime(df[self.timestamp_column])
        except:
            raise ValueError(f"Column {self.timestamp_column} must be convertible to datetime")
            
    def get_time_series_cv_splits(self, X, test_size=24*30, n_splits=12, min_train_months=12):
        
        n_samples = len(X)
        min_train_size = 24 * 30 * min_train_months  
        
        if n_samples < min_train_size + test_size:
            raise ValueError(f"Not enough samples ({n_samples}) for minimum training size ({min_train_size}) and test size ({test_size})")
        
        if test_size < 24:
            raise ValueError("Test size should be at least 24 hours")
            
       
        max_splits = (n_samples - min_train_size) // test_size
        n_splits = min(n_splits, max_splits)
        
        splits = []
        for i in range(n_splits):
            
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            train_start = max(0, test_start - min_train_size - i * min_train_size)
            
         
            if test_start - train_start < min_train_size:
                break
                
            train_indices = np.arange(train_start, test_start)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
       
        return splits[::-1]
    
    def create_time_features(self, df):
        
        df = df.copy()
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column],format='%Y-%m-%d %H:%M:%S')
        df.set_index(self.timestamp_column, inplace=True)
        
        
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week
        
       
        for col, max_val in [('hour', 24), ('day', 31), ('month', 12), 
                           ('day_of_week', 7), ('week_of_year', 52)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])
        df['season'] = pd.cut(df['month'],
                            bins=[0, 3, 6, 9, 12],
                            labels=['winter', 'spring', 'summer', 'fall'])
        
        return df
    
    def create_lagged_features(self, df, lag_features):
        
        df = df.copy()
        
        feature_groups = {
            'hour': [1, 2, 3, 6, 12, 24],
            'day': [1, 2, 3, 7],
            'week': [1, 2]
        }
        
        try:
            # Feature lags
            for feature in lag_features:
                if feature not in df.columns:
                    warnings.warn(f"Feature {feature} not found in dataframe")
                    continue
                    
                for lag in feature_groups['hour']:
                    df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
                
                # Rolling statistics
                for window in [3, 6, 12, 24]:
                    df[f'{feature}_rolling_mean_{window}h'] = df[feature].rolling(window=window).mean()
                    df[f'{feature}_rolling_std_{window}h'] = df[feature].rolling(window=window).std()
                    df[f'{feature}_rolling_min_{window}h'] = df[feature].rolling(window=window).min()
                    df[f'{feature}_rolling_max_{window}h'] = df[feature].rolling(window=window).max()
            
            # Handle NaN values created by lagging
            df = df.fillna(method='bfill').fillna(method='ffill')
            
        except Exception as e:
            raise ValueError(f"Error creating lagged features: {str(e)}")
        
        return df
    
    def create_additional_features(self, df):
        
        df = df.copy()
        
        try:
            # Environmental indices
            df['weather_stability'] = (
                df['Presure'] * -0.83 +
                df['Dew_point'] * 0.81 +
                df['Moisture_percent'] * -0.45
            )
            
            #df['pollution_severity'] = (
             #   df['SO2_concentration'] * 0.893 +
              #  df['NO2_concentration'] * 0.886 +
               # df['CO_concentration'] * 0.855
           # )
            
            # Interaction terms
            df['pressure_moisture'] = df['Presure'] * df['Moisture_percent']
            df['dew_moisture'] = df['Dew_point'] * df['Moisture_percent']
            
            df.drop('City_ID',axis=1)
            
            
            df = df.fillna(method='bfill').fillna(method='ffill')
            
        except Exception as e:
            raise ValueError(f"Error creating additional features: {str(e)}")
        
        return df
    
    def create_temporal_features(self, df):
        
        df = df.copy()
        
        try:
            # Weather stability index
            df['weather_stability'] = (
                df['Presure'] * -0.83 +
                df['Dew_point'] * 0.81 +
                df['Moisture_percent'] * -0.45
            )
            
            # Pollution index
            ##   df['SO2_concentration'] +
              #  df['NO2_concentration'] +
               # df['CO_concentration']
            #) / 3
            
            
            for col in ['Presure', 'NO2_concentration']:
                df[f'{col}_diff_1h'] = df[col].diff()
                df[f'{col}_diff_24h'] = df[col].diff(24)
            
           
            df = df.fillna(method='bfill').fillna(method='ffill')
            
        except Exception as e:
            raise ValueError(f"Error creating temporal features: {str(e)}")
        
        return df
    
    

    def fit(self, train_data, target_col='Temperature', optimize=True, n_splits=12, test_size=24*30, min_train_months=12):
        
     
        self.validate_input_data(train_data, for_training=True)
        
        print("Preparing time series features...")
        df = self.create_time_features(train_data)
        df = self.create_lagged_features(df, self.required_columns)
        df = self.create_temporal_features(df)
        df = self.create_additional_features(df)
        
        df = df.sort_index()
        
        self.feature_columns = [col for col in df.columns 
                              if col != target_col 
                              and col != self.timestamp_column]
        
        
        y = df[target_col]
        X = df[self.feature_columns]
        
      
        self.categorical_features = ['time_of_day', 'season']
        self.numeric_features = [col for col in X.columns 
                               if col not in self.categorical_features]
        
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=True
        )
        
        
        X_processed = self.preprocessor.fit_transform(X)
        
        
        cv_splits = self.get_time_series_cv_splits(
            X_processed, 
            test_size=test_size,
            n_splits=n_splits,
            min_train_months=min_train_months
        )
        
        
        if optimize:
            print("Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X_processed, y, cv_splits)
            if best_params:
                print("Using optimized parameters")
                model_params = best_params
            else:
                print("Using default parameters")
                model_params = {
                    'max_depth': 6,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 100,
                    'tree_method': 'hist',
                    'objective': 'reg:squarederror'
                }
        else:
            model_params = {
                'max_depth': 8,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 1500,
                'tree_method': 'hist',
                'objective': 'reg:squarederror'
            }
        
        
        dtrain = xgb.DMatrix(X_processed, label=y)
        
        
        self.xgb_model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, 'train')],
            verbose_eval=50
        )
        
       
        importance_scores = self.xgb_model.get_score(importance_type='weight')
        self.feature_importance = pd.DataFrame(
            [(k, v) for k, v in importance_scores.items()],
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return self
    
    def optimize_hyperparameters(self, X, y, cv_splits, n_trials=20, plot_history=True):
        "
        pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=10,
            interval_steps=5,
            n_min_trials=5
        )
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 18),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 2.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 2.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2.0, log=True)
            }
            
            
            num_boost_round = trial.suggest_int('num_boost_round', 200, 5000)
            
            print(f"\nTrial {trial.number}: Testing parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")
            print(f"num_boost_round: {num_boost_round}")
            
            metrics = {
                'mse': [],
                'mae': [],
                'r2': [],
                'direction_accuracy': []
            }
            
            try:
                X_array = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
                y_array = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
                
                for fold, (train_idx, val_idx) in enumerate(cv_splits):
                    X_train, X_val = X_array[train_idx], X_array[val_idx]
                    y_train, y_val = y_array[train_idx], y_array[val_idx]
                    
                    if np.isnan(X_train).any() or np.isnan(y_train).any():
                        print("Warning: NaN values detected in training data")
                        return float('inf')
                    
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)
                    
                    model_params = {
                        **params,
                        'objective': 'reg:squarederror',
                        'eval_metric': ['rmse', 'mae']
                    }
                    
                    bst = xgb.train(
                        model_params,
                        dtrain,
                        num_boost_round=num_boost_round,  # Use num_boost_round here instead of n_estimators
                        evals=[(dtrain, 'train'), (dval, 'eval')],
                        early_stopping_rounds=200,
                        verbose_eval=50
                    )
                    
                    pred = bst.predict(dval)
                    
                    metrics['mse'].append(mean_squared_error(y_val, pred))
                    metrics['mae'].append(mean_absolute_error(y_val, pred))
                    metrics['r2'].append(r2_score(y_val, pred))
                    
                    y_val_diff = np.diff(y_val)
                    pred_diff = np.diff(pred)
                    direction_accuracy = np.mean((y_val_diff * pred_diff) > 0)
                    metrics['direction_accuracy'].append(direction_accuracy)
                    
                    print(f"\nFold {fold + 1} Metrics:")
                    print(f"MSE: {metrics['mse'][-1]:.4f}")
                    print(f"MAE: {metrics['mae'][-1]:.4f}")
                    print(f"R2: {metrics['r2'][-1]:.4f}")
                    print(f"Direction Accuracy: {metrics['direction_accuracy'][-1]:.4f}")
                    
                    trial.report(metrics['mse'][-1], fold)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                final_score = (
                    0.4 * np.mean(metrics['mse']) +
                    0.3 * np.mean(metrics['mae']) +
                    0.2 * (1 - np.mean(metrics['r2'])) +
                    0.1 * (1 - np.mean(metrics['direction_accuracy']))
                )
                
                print(f"\nTrial {trial.number} final score: {final_score:.4f}")
                return final_score
                
            except Exception as e:
                print(f"\nError in trial {trial.number}: {str(e)}")
                return float('inf')
        
        print("\nStarting hyperparameter optimization...")
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=42
            )
        )
        
        study.optimize(
            objective,
            n_trials=max(10, n_trials),
            n_jobs=2,
            show_progress_bar=True
        )
        
        if len(study.trials) > 0:
            print("\nOptimization Summary:")
            print(f"Number of completed trials: {len(study.trials)}")
            print(f"Best trial: #{study.best_trial.number}")
            print(f"Best value: {study.best_value:.4f}")
            print("\nBest parameters:")
            for key, value in study.best_params.items():
                print(f"    {key}: {value}")
            
            if plot_history and study.best_value != float('inf'):
                plt.figure(figsize=(12, 8))
                optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.title('Hyperparameter Optimization History')
                plt.show()
            
            return study.best_params
        else:
            print("\nNo successful trials completed")
            return None
    
    def predict(self, test_data, return_confidence=False):
        
        if self.xgb_model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            
            self.validate_input_data(test_data, for_training=False)
            df = self.create_time_features(test_data)
            df = self.create_lagged_features(df, self.required_columns)
            df = self.create_temporal_features(df)
            df = self.create_additional_features(df)
            
            X = df[self.feature_columns]
            X_processed = self.preprocessor.transform(X)
            
            
            dtest = xgb.DMatrix(X_processed)
            
           
            predictions = self.xgb_model.predict(dtest)
            
            if return_confidence:
                
                n_iterations = 100
                bootstrap_predictions = []
                
                for _ in range(n_iterations):
                   
                    feature_mask = np.random.binomial(1, 0.8, size=X_processed.shape[1])
                    if not np.any(feature_mask):  #
                        feature_mask[0] = 1
                        
                    X_bootstrap = X_processed[:, feature_mask == 1]
                    dtest_bootstrap = xgb.DMatrix(X_bootstrap)
                    bootstrap_predictions.append(self.xgb_model.predict(dtest_bootstrap))
                
                bootstrap_predictions = np.array(bootstrap_predictions)
                lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
                upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)
                
                return predictions, (lower_bound, upper_bound)
            
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
    
    
    def plot_feature_importance(self, top_n=20):
        
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before plotting feature importance")
            
        plt.figure(figsize=(12, 6))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top Feature Importances')
        plt.tight_layout()
        return plt
    
    
    
    
    def evaluate(self, test_data, predictions, target_col='Temperature'):
        
        try:
            
            if target_col not in test_data.columns:
                print(f"\nNote: {target_col} not found in test data. Saving predictions only.")
                
              
                predictions_df = pd.DataFrame({
                    'Datetime': test_data[self.timestamp_column],
                    'Predicted_Temperature': predictions
                })
                
                
                output_file = 'XGB_predictions.csv'
                predictions_df.to_csv(output_file, index=False)
                print(f"Predictions saved to '{output_file}'")
                
                return None
                
            else:
               
                y_true = test_data[target_col]
                metrics = {
                    'MSE': mean_squared_error(y_true, predictions),
                    'RMSE': np.sqrt(mean_squared_error(y_true, predictions)),
                    'MAE': mean_absolute_error(y_true, predictions),
                    'R2': r2_score(y_true, predictions)
                }
                
                # Add MAPE if no zero values
                #if not np.any(y_true == 0):
                 #   metrics['MAPE'] = np.mean(np.abs((y_true - predictions) / y_true)) * 100
                    
                    
                 # Add directional accuracy
                direction_true = np.diff(y_true) > 0
                direction_pred = np.diff(predictions) > 0
                metrics['Direction_Accuracy'] = np.mean(direction_true == direction_pred) * 100
                
                print("\nModel Performance Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                return metrics
                    
                print("\nModel Performance Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                    
                return metrics
                
        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            return None
    

    
    def save_predictions_to_csv(self, predictions, test_data, filename='RF_predictions.csv'):
        
        predictions_df = pd.DataFrame({
            'ID': test_data['ID'] if 'ID' in test_data.columns else range(len(predictions)),
            'Predicted_Temperature': predictions
        })
        predictions_df.to_csv(filename, index=False)
        print(f"\nPredictions saved to '{filename}'")

   
    
    


    


TRAIN_PATH = 'Train_concatenated_file.csv'
TEST_PATH = 'Test_concatenated_file.csv'

print("Loading data...")
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_data.drop('SO2_concentration',axis=1)
test_data.drop('SO2_concentration',axis=1)

print("Converting datetime...")
train_data['Datetime'] = pd.to_datetime(train_data['Datetime'], format='%Y-%m-%d %H:%M:%S')
test_data['Datetime'] = pd.to_datetime(test_data['Datetime'], format='%Y-%m-%d %H:%M:%S')

print("Initializing model...")
predictor = TimeSeriesTemperaturePredictor(forecast_horizon=1)

print("Fitting model...")
predictor.fit(train_data, target_col='Temperature', optimize=True)


print("Generating predictions...")
predictions = predictor.predict(test_data)

print("Evaluating model...")
metrics = predictor.evaluate(test_data, predictions)


print("Process completed successfully!")



importance = xgb.get_score(importance_type='weight')  

importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)


for feature, score in importance:
    print(f"Feature: {feature}, Score: {score}")
