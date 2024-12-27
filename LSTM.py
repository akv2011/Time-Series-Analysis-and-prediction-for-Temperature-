
        
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class LSTMTemperaturePredictor:
    def __init__(self, sequence_length=24, forecast_horizon=24):
        # Previous initialization remains the same
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.timestamp_column = 'Datetime'
        self.target_column = 'Temperature'
        self.categorical_columns = ['time_of_day', 'season']
        self.required_columns = [
            'Presure', 'Dew_point', 'Moisture_percent',
            'SO2_concentration', 'NO2_concentration', 'CO_concentration'
        ]
        self.history = None
        self.feature_columns = None
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def check_nulls(self, df, stage=""):
        """Helper function to check for null values and print detailed information"""
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"\nNull values found at {stage}:")
            for col in df.columns[df.isnull().any()]:
                print(f"{col}: {df[col].isnull().sum()} null values")
            return True
        return False

    def create_time_features(self, df):
        """Create time-based features with null checking"""
        print("\nCreating time features...")
        df = df.copy()
        
        # Convert timestamp if string
        if isinstance(df.index, pd.DatetimeIndex):
            df_index = df.index
        else:
            df_index = pd.to_datetime(df.index)
        
        # Create basic time features
        df['hour_sin'] = np.sin(2 * np.pi * df_index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df_index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df_index.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df_index.day / 31)
        df['month_sin'] = np.sin(2 * np.pi * df_index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df_index.month / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df_index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df_index.dayofweek / 7)
        
        self.check_nulls(df, "after time features")
        return df

    def handle_missing_values(self, df):
        """Comprehensive missing value handling"""
        print("\nHandling missing values...")
        df = df.copy()
        
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            # First try forward fill
            df[col] = df[col].fillna(method='ffill')
            # Then backward fill
            df[col] = df[col].fillna(method='bfill')
            # Finally, use mean for any remaining nulls
            df[col] = df[col].fillna(df[col].mean())
            
            # Check if any nulls remain
            if df[col].isnull().any():
                print(f"Warning: Could not fill all nulls in {col}")
        
        # Handle categorical columns
        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('missing')
            
        return df

    def fit(self, train_data, validation_split=0.2, epochs=100, batch_size=32):
        """Train LSTM with comprehensive null handling and debugging"""
        try:
            print("\nStarting data preprocessing...")
            df = train_data.copy()
            
            # Initial null check
            print("\nChecking initial data for nulls...")
            self.check_nulls(df, "initial data")
            
            # Verify required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Set datetime index if not already set
            if self.timestamp_column in df.columns:
                df.set_index(self.timestamp_column, inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a datetime index or a datetime column named 'Datetime'")
            
            # Handle missing values first
            df = self.handle_missing_values(df)
            
            # Create features
            print("\nApplying feature engineering...")
            df = self.create_time_features(df)
            
            # Check nulls after feature creation
            if self.check_nulls(df, "after feature engineering"):
                # Additional handling for any new nulls
                df = self.handle_missing_values(df)
            
            # Create categorical features using index
            df['time_of_day'] = pd.cut(df.index.hour, 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['night', 'morning', 'afternoon', 'evening'],
                                     include_lowest=True)
            df['season'] = pd.cut(df.index.month, 
                                bins=[0, 3, 6, 9, 12], 
                                labels=['winter', 'spring', 'summer', 'fall'],
                                include_lowest=True)
            
            # Final null check
            if self.check_nulls(df, "final check"):
                raise ValueError("Null values remain after all preprocessing steps")
            
            # Store feature columns
            self.feature_columns = [col for col in df.columns if col != self.target_column]
            
            # Split data
            print("\nSplitting data...")
            total_timestamps = len(df)
            split_idx = int(total_timestamps * (1 - validation_split))
            
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            # Prepare sequences
            print("\nPreparing sequences...")
            X_train, y_train = self.prepare_sequences(train_df, is_training=True)
            X_val, y_val = self.prepare_sequences(val_df, is_training=True)
            
            print(f"\nTraining shape: X={X_train.shape}, y={y_train.shape}")
            print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
            
            # Build and train model
            print("\nBuilding model...")
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001),
                ModelCheckpoint('best_lstm_model.keras', save_best_only=True)
            ]
            
            print("\nStarting training...")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return self
            
        except Exception as e:
            print(f"\nDetailed error in fit method: {str(e)}")
            # Print the full shape of the dataframe and its columns
            if 'df' in locals():
                print("\nDataframe information:")
                print(f"Shape: {df.shape}")
                print("\nColumns:")
                for col in df.columns:
                    print(f"{col}: {df[col].dtype}")
                print("\nNull counts:")
                print(df.isnull().sum())
            raise RuntimeError(f"Error during model fitting: {str(e)}")

    def prepare_sequences(self, data, is_training=True):
        """Modified sequence preparation to handle edge cases and maintain full length"""
        X = []
        y = []
    
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
    
        feature_cols = [col for col in data.columns if col != self.target_column]
    
        try:
            if is_training:
                X_df = data[feature_cols].copy()
                y_df = data[[self.target_column]].copy()
    
                # Handle categorical columns
                for cat_col in self.categorical_columns:
                    current_categories = X_df[cat_col].astype('category').cat.categories
                    new_categories = current_categories.union(['missing'])
                    X_df[cat_col] = pd.Categorical(
                        X_df[cat_col],
                        categories=new_categories,
                        ordered=True
                    )
                    X_df[cat_col] = X_df[cat_col].fillna('missing')
    
                # Handle numeric features
                numeric_cols = X_df.select_dtypes(include=np.number).columns
                X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].mean())
    
                # One-Hot Encode categorical features
                categorical_data = X_df[self.categorical_columns]
                if is_training:
                    X_encoded = self.encoder.fit_transform(categorical_data)
                else:
                    X_encoded = self.encoder.transform(categorical_data)
                    
                X_numeric = X_df.drop(self.categorical_columns, axis=1).values
                X_combined = np.concatenate([X_numeric, X_encoded], axis=1)
                
                # Scale features and target
                if is_training:
                    X_scaled = self.feature_scaler.fit_transform(X_combined)
                    y_scaled = self.target_scaler.fit_transform(y_df)
                else:
                    X_scaled = self.feature_scaler.transform(X_combined)
                    y_scaled = y_df
    
                # Create sequences with padding for the first sequence_length-1 rows
                padded_data = np.pad(X_scaled, ((self.sequence_length-1, 0), (0, 0)), mode='edge')
                
                # Create sequences including the first sequence_length-1 rows
                for i in range(len(data)):
                    start_idx = i
                    end_idx = i + self.sequence_length
                    X.append(padded_data[start_idx:end_idx])
                    if is_training:
                        y.append(y_scaled[i])
    
            else:  # Prediction mode
                X_df = data[feature_cols].copy()
                
                # Handle categorical columns
                for cat_col in self.categorical_columns:
                    X_df[cat_col] = pd.Categorical(
                        X_df[cat_col],
                        categories=pd.Categorical(X_df[cat_col]).categories.union(['missing']),
                        ordered=True
                    )
                    X_df[cat_col] = X_df[cat_col].fillna('missing')
    
                # Handle numeric features
                numeric_cols = X_df.select_dtypes(include=np.number).columns
                X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].mean())
    
                # Transform features using stored encoder and scaler
                X_encoded = self.encoder.transform(X_df[self.categorical_columns])
                X_numeric = X_df.drop(self.categorical_columns, axis=1).values
                X_combined = np.concatenate([X_numeric, X_encoded], axis=1)
                X_scaled = self.feature_scaler.transform(X_combined)
    
                # Pad the beginning to handle the first sequence_length-1 rows
                padded_data = np.pad(X_scaled, ((self.sequence_length-1, 0), (0, 0)), mode='edge')
                
                # Create sequences including the first sequence_length-1 rows
                for i in range(len(data)):
                    start_idx = i
                    end_idx = i + self.sequence_length
                    X.append(padded_data[start_idx:end_idx])
    
            X = np.array(X)
            y = np.array(y) if is_training else None
    
            if is_training:
                print(f"Sequence shapes - X: {X.shape}, y: {y.shape}")
            
            return X, y
    
        except Exception as e:
            print(f"Error in prepare_sequences: {str(e)}")
            raise
        
           
    def predict(self, test_data):
        """Generate predictions."""
        try:
            df = test_data.copy()

            if self.timestamp_column not in df.columns:
                raise ValueError(f"Missing timestamp column: {self.timestamp_column}")

            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
            df.set_index(self.timestamp_column, inplace=True)

            # Feature engineering (same as in fit)
            df = self.create_time_features(df)
            df = self.create_lagged_features(df, self.required_columns)
            df = self.create_additional_features(df)
            df = self.create_temporal_features(df)
            df['time_of_day'] = pd.cut(df.index.hour, bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
            df['season'] = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'fall'])

            # Handle NaNs *after* all feature engineering (same as in fit)
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(df[numeric_cols].mean())
            #categorical_cols = df.select_dtypes(include=['category', 'object']).columns
            #df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode()[0])
            
            # Verify features match training
            if self.feature_columns is None:
                raise ValueError("Model must be trained before prediction")
            
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in test data: {missing_cols}")
                
            df = df[self.feature_columns] #Align test features with training


            X_test, _ = self.prepare_sequences(df, is_training=False) # Prepare test sequences

            predictions = self.model.predict(X_test)
            predictions = self.target_scaler.inverse_transform(predictions)  # Inverse transform to original scale
            return predictions

        except Exception as e:
            print("Prediction error:")
            print(f"Feature columns during training: {self.feature_columns}")
            print(f"Test data columns: {df.columns.tolist() if 'df' in locals() else 'N/A'}")
            raise RuntimeError(f"Error during prediction: {str(e)}")
           
    
    def create_lagged_features(self, df, lag_features):
        """
        Create lagged features without using temperature
        """
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
            
            
        except Exception as e:
            raise ValueError(f"Error creating lagged features: {str(e)}")
        
        return df
    
    def create_additional_features(self, df):
        """
        Create additional derived features without temperature dependencies
        """
        df = df.copy()
        
        try:
            # Environmental indices
            df['weather_stability'] = (
                df['Presure'] * -0.83 +
                df['Dew_point'] * 0.81 +
                df['Moisture_percent'] * -0.45
            )
            
            df['pollution_severity'] = (
                df['SO2_concentration'] * 0.893 +
                df['NO2_concentration'] * 0.886 +
                df['CO_concentration'] * 0.855
            )
            
            # Interaction terms
            df['pressure_moisture'] = df['Presure'] * df['Moisture_percent']
            df['dew_moisture'] = df['Dew_point'] * df['Moisture_percent']
            
            
        except Exception as e:
            raise ValueError(f"Error creating additional features: {str(e)}")
        
        return df
    
    def create_temporal_features(self, df):
        """
        Create temporal interaction features without temperature dependencies
        """
        df = df.copy()
        
        try:
            # Weather stability index
            df['weather_stability'] = (
                df['Presure'] * -0.83 +
                df['Dew_point'] * 0.81 +
                df['Moisture_percent'] * -0.45
            )
            
            # Pollution index
            df['pollution_index'] = (
                df['SO2_concentration'] +
                df['NO2_concentration'] +
                df['CO_concentration']
            ) / 3
            
            # Temporal differences (only for non-temperature features)
            for col in ['Presure', 'pollution_index']:
                df[f'{col}_diff_1h'] = df[col].diff()
                df[f'{col}_diff_24h'] = df[col].diff(24)
            
            
            
        except Exception as e:
            raise ValueError(f"Error creating temporal features: {str(e)}")
        
        return df


    def build_model(self, input_shape):
        """Regular LSTM architecture for single-step prediction"""
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(32),
            Dropout(0.3),
            
            # Dense layers for prediction
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Single output for temperature prediction
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
    
    def plot_training_history(self):
        """Plot training metrics"""
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
            
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, test_data, predictions):
        """Plot test predictions against actual values"""
        plt.figure(figsize=(15, 6))
        
        actual = test_data[self.target_column].values[self.sequence_length:]
        pred = predictions.flatten()
        
        plt.plot(actual[:len(pred)], label='Actual', alpha=0.7)
        plt.plot(pred, label='Predicted', alpha=0.7)
        
        plt.title('Temperature Predictions vs Actual Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Calculate metrics
        if len(actual) >= len(pred):
            mse = mean_squared_error(actual[:len(pred)], pred)
            mae = mean_absolute_error(actual[:len(pred)], pred)
            r2 = r2_score(actual[:len(pred)], pred)
            
            print(f"\nTest Metrics:")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")

    def save_model(self, filepath):
        """Save the LSTM model and scalers"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Save Keras model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scalers
        np.save(f"{filepath}_feature_scaler.npy", 
                [self.feature_scaler.scale_, self.feature_scaler.min_])
        np.save(f"{filepath}_target_scaler.npy", 
                [self.target_scaler.scale_, self.target_scaler.min_])
        
        print(f"Model and scalers saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a saved LSTM model and scalers"""
        try:
            # Create instance
            instance = cls()
            
            # Load Keras model
            instance.model = load_model(f"{filepath}_model.h5")
            
            # Load scalers
            feature_scaler_params = np.load(f"{filepath}_feature_scaler.npy")
            target_scaler_params = np.load(f"{filepath}_target_scaler.npy")
            
            instance.feature_scaler.scale_ = feature_scaler_params[0]
            instance.feature_scaler.min_ = feature_scaler_params[1]
            instance.target_scaler.scale_ = target_scaler_params[0]
            instance.target_scaler.min_ = target_scaler_params[1]
            
            print(f"Model loaded successfully from {filepath}")
            return instance
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")


    def save_predictions_to_csv(self, predictions, test_data, filename='RF_predictions.csv'):
        """
        Save predictions to a CSV file
        
        Args:
            predictions: Model predictions
            test_data: Original test dataset
            filename: Output CSV filename
        """
        predictions_df = pd.DataFrame({
            'ID': test_data['ID'] if 'ID' in test_data.columns else range(len(predictions)),
            'Predicted_Temperature': predictions
        })
        predictions_df.to_csv(filename, index=False)
        print(f"\nPredictions saved to '{filename}'")


# Define paths
TRAIN_PATH = '/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/Train_concatenated_file.csv'
TEST_PATH = '/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/Test_concatenated_file.csv'

# Load training and test dat
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)


print("Data loaded successfully:")
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Convert datetime
print("\nConverting datetime...")

train_data['Datetime'] = pd.to_datetime(train_data['Datetime'], format='%Y-%m-%d %H:%M:%S')
test_data['Datetime'] = pd.to_datetime(test_data['Datetime'], format='%Y-%m-%d %H:%M:%S')


# Initialize model
print("\nInitializing LSTM model...")
predictor = LSTMTemperaturePredictor(sequence_length=24, forecast_horizon=24)

# Train model
print("\nTraining model...")

predictor.fit(
train_data,
validation_split=0.2,
epochs=1,
batch_size=8
)


predictions = predictor.predict(test_data)
print(f"Predictions shape: {predictions.shape}")


# Plot training history
print("\nPlotting training history...")
predictor.plot_training_history()

# Plot predictions
print("\nPlotting predictions vs actual values...")
predictor.plot_predictions(test_data, predictions)



# Save predictions to CSV

predictor.save_predictions_to_csv(predictions, test_data, 'lstm_predictions.csv')

# Save model
print("\nSaving model...")

predictor.save_model('lstm_temperature_predictor')
print("Model saved successfully!")
   
   

# Save predictions to CSV
print("\nSaving predictions to CSV...")
   
predictor.save_predictions_to_csv(predictions, test_data, 'lstm_predictions.csv')



