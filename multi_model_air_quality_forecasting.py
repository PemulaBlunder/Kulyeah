import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import json
import os
from datetime import datetime, date
from zoneinfo import ZoneInfo
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class ModelConfig:
    """Configuration class for different model types."""
    
    def __init__(self, model_type: str, model_params: Dict, 
                 file_paths: Dict[str, str], pollutant: str):
        self.model_type = model_type
        self.model_params = model_params
        self.file_paths = file_paths
        self.pollutant = pollutant
    
    def __repr__(self):
        return f"ModelConfig({self.model_type}, {self.pollutant})"


class BaseModel(ABC):
    """Abstract base class for all model types."""
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, int], **kwargs) -> Sequential:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model type name."""
        pass


class LSTMModel(BaseModel):
    """LSTM model implementation."""
    
    def build_model(self, input_shape: Tuple[int, int], units: int = 50, 
                   dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        model = Sequential()
        model.add(LSTM(units, input_shape=input_shape, kernel_regularizer=l2(1e-4)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "LSTM"


class GRUModel(BaseModel):
    """GRU model implementation."""
    
    def build_model(self, input_shape: Tuple[int, int], units: int = 50, 
                   dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        model = Sequential()
        model.add(GRU(units, input_shape=input_shape, kernel_regularizer=l2(1e-4)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "GRU"


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model implementation."""
    
    def build_model(self, input_shape: Tuple[int, int], cnn_filters: int = 64, 
                   lstm_units: int = 50, dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        model = Sequential()
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(lstm_units, kernel_regularizer=l2(1e-4)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "CNN-LSTM"


class DeepLSTMModel(BaseModel):
    """Deep LSTM model with multiple layers."""
    
    def build_model(self, input_shape: Tuple[int, int], units: List[int] = [50, 30], 
                   dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(1e-4)))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        if len(units) > 1:
            model.add(LSTM(units[1], kernel_regularizer=l2(1e-4)))
            model.add(Dropout(dropout))
        
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "Deep-LSTM"


class BidirectionalLSTMModel(BaseModel):
    """Bidirectional LSTM model implementation."""
    
    def build_model(self, input_shape: Tuple[int, int], units: int = 50, 
                   dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential()
        model.add(Bidirectional(LSTM(units, kernel_regularizer=l2(1e-4)), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "Bidirectional-LSTM"


class SimpleDenseModel(BaseModel):
    """Simple dense neural network model."""
    
    def build_model(self, input_shape: Tuple[int, int], units: List[int] = [64, 32], 
                   dropout: float = 0.2, lr: float = 0.001, **kwargs) -> Sequential:
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        
        for unit in units:
            model.add(Dense(unit, activation='relu', kernel_regularizer=l2(1e-4)))
            model.add(Dropout(dropout))
        
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model
    
    def get_model_name(self) -> str:
        return "Dense-NN"


class ModelFactory:
    """Factory class for creating different model types."""
    
    _models = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'cnn_lstm': CNNLSTMModel,
        'deep_lstm': DeepLSTMModel,
        'bidirectional_lstm': BidirectionalLSTMModel,
        'dense': SimpleDenseModel
    }
    
    @classmethod
    def create_model(cls, model_type: str) -> BaseModel:
        """Create a model instance by type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}")
        return cls._models[model_type]()
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._models.keys())


class AirQualityDataFetcher:
    """Handles fetching air quality data from the API."""
    
    def __init__(self, base_url: str = "https://airnet.waqi.info/airnet/sse/historic/daily/420154"):
        self.base_url = base_url
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.pollutants = ['pm25', 'pm10', 'co', 'no2', 'o3', 'so2']
    
    def fetch_pollutant_data(self, pollutant: str) -> pd.DataFrame:
        """Fetch data for a specific pollutant."""
        url = f"{self.base_url}?specie={pollutant}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {pollutant}: {response.status_code}")
        
        text_data = response.text
        raw_matches = re.findall(r'data:\s*(\{.*?\})', text_data)
        
        day_list = []
        median_list = []
        
        for obj in raw_matches:
            cleaned = obj.strip()
            cleaned = cleaned.replace("NaN", "null").replace("Infinity", "null")
            cleaned = re.sub(r",\s*}", "}", cleaned)
            
            try:
                data = json.loads(cleaned)
                if 'day' in data and 'median' in data:
                    day_list.append(data['day'])
                    median_list.append(data['median'])
            except json.JSONDecodeError:
                continue
        
        df = pd.DataFrame({
            'day': day_list,
            'median': median_list
        })
        
        print(f"{pollutant}: {len(df)} data points successfully fetched.")
        return df
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all pollutants."""
        data_dict = {}
        
        for pollutant in self.pollutants:
            try:
                data_dict[pollutant] = self.fetch_pollutant_data(pollutant)
            except Exception as e:
                print(f"Error fetching {pollutant}: {e}")
                continue
        
        return data_dict


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self):
        self.wib = ZoneInfo("Asia/Jakarta")
    
    def merge_pollutant_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all pollutant data into a single DataFrame."""
        df_merged = None
        
        for pollutant, df in data_dict.items():
            df = df.rename(columns={'median': pollutant})
            if df_merged is None:
                df_merged = df
            else:
                df_merged = pd.merge(df_merged, df, on='day', how='outer')
        
        df_merged = df_merged.sort_values('day').reset_index(drop=True)
        return df_merged
    
    def clean_and_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and perform interpolation."""
        df_cleaned = df.interpolate(method='linear', limit_direction='both')
        df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
        return df_cleaned
    
    def remove_today_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove today's data from the DataFrame."""
        today_wib = datetime.now(self.wib).date()
        df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
        today = pd.to_datetime(today_wib)
        
        df_filtered = df[df['day'] != today].copy()
        return df_filtered
    
    def reorder_columns(self, df: pd.DataFrame, new_order: List[str]) -> pd.DataFrame:
        """Reorder DataFrame columns."""
        return df[new_order].copy()
    
    def preprocess_data(self, data_dict: Dict[str, pd.DataFrame], 
                       column_order: List[str] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        if column_order is None:
            column_order = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
        
        df_merged = self.merge_pollutant_data(data_dict)
        df_cleaned = self.clean_and_interpolate(df_merged)
        df_filtered = self.remove_today_data(df_cleaned)
        df_final = self.reorder_columns(df_filtered, column_order)
        
        return df_final


class MultiModelManager:
    """Manages multiple model types for different pollutants."""
    
    def __init__(self, model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
        self.model_factory = ModelFactory()
        self.retrained_models = {}
    
    def load_model_components(self, pollutant: str) -> Tuple[Any, Dict, Any, ModelConfig]:
        """Load model, parameters, scaler, and config for a pollutant."""
        if pollutant not in self.model_configs:
            raise ValueError(f"No model configuration found for pollutant: {pollutant}")
        
        config = self.model_configs[pollutant]
        
        # Load parameters
        with open(config.file_paths['params'], "r") as f:
            best_params_dict = json.load(f)
        
        # Load model
        model = load_model(config.file_paths['model'])
        
        # Load scaler
        scaler = joblib.load(config.file_paths['scaler'])
        
        return model, best_params_dict, scaler, config
    
    def retrain_model(self, pollutant: str, df_supervised: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame]:
        """Retrain model with updated data."""
        model, best_params_dict, scaler, config = self.load_model_components(pollutant)
        
        # Create model instance
        model_builder = self.model_factory.create_model(config.model_type)
        
        # Create lag features
        lag_offsets = best_params_dict.get("lags", [1, 2, 3])
        df_supervised_final = df_supervised[['y']].copy()
        
        for offset in lag_offsets:
            df_supervised_final[f'lag_{offset}'] = df_supervised_final['y'].shift(offset)
        df_supervised_final = df_supervised_final.dropna().reset_index(drop=True)
        
        # Prepare features
        feature_columns = ['y'] + [f'lag_{offset}' for offset in lag_offsets]
        
        # Scaling
        scaled_data = scaler.transform(df_supervised_final[feature_columns])
        X = scaled_data[:, 1:]
        y = scaled_data[:, 0]
        X = X.reshape(X.shape[0], len(lag_offsets), 1)
        
        # Split train/validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build and retrain model
        retrained_model = model_builder.build_model(
            input_shape=(len(lag_offsets), 1),
            **config.model_params
        )
        
        retrained_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=int(best_params_dict.get("epochs", 50)),
            batch_size=32,
            verbose=1
        )
        
        print(f"✅ Retrained {config.model_type} model for {pollutant}! Samples: {len(df_supervised_final)}")
        
        # Store retrained model
        self.retrained_models[pollutant] = retrained_model
        
        return retrained_model, scaler, df_supervised_final
    
    def save_retrained_model(self, pollutant: str, model: Any, save_dir: str) -> str:
        """Save retrained model."""
        config = self.model_configs[pollutant]
        model_name = f"{pollutant}_{config.model_type}_retrained.keras"
        save_path = os.path.join(save_dir, model_name)
        model.save(save_path)
        print(f"✅ Retrained model saved as: {save_path}")
        return save_path


class MultiModelForecaster:
    """Handles forecasting with multiple model types."""
    
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
    
    def sequential_forecast_with_retrain(self, pollutant: str, df_supervised: pd.DataFrame,
                                       num_steps: int = 7) -> Tuple[List, Any, Any, pd.DataFrame]:
        """Perform sequential forecasting with retraining at each step."""
        model, best_params_dict, scaler, config = self.model_manager.load_model_components(pollutant)
        model_builder = self.model_manager.model_factory.create_model(config.model_type)
        
        lag_offsets = best_params_dict.get("lags", [1, 2, 3])
        num_lags = len(lag_offsets)
        num_features = num_lags + 1
        extended_df = df_supervised.copy()
        
        # Create lag columns if missing
        feature_columns = ['y'] + [f'lag_{offset}' for offset in lag_offsets]
        missing_lags = [f'lag_{offset}' for offset in lag_offsets if f'lag_{offset}' not in extended_df.columns]
        
        if missing_lags:
            print(f"Creating missing lag columns for {pollutant}: {missing_lags}")
            if 'y' not in extended_df.columns:
                raise ValueError("df_supervised must have column 'y'.")
            
            max_offset = max(lag_offsets)
            if len(extended_df) < max_offset:
                raise ValueError(f"Data length ({len(extended_df)}) is too short for max lag offset ({max_offset})")
            
            for offset in lag_offsets:
                col_name = f'lag_{offset}'
                extended_df[col_name] = extended_df['y'].shift(offset)
            
            initial_len = len(extended_df)
            extended_df = extended_df.dropna().reset_index(drop=True)
            dropped_rows = initial_len - len(extended_df)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with NaN after creating lags.")
        
        current_scaler = scaler
        current_model = model
        forecasts = []
        
        # Initialize last_lags_unscaled
        last_lags_unscaled = []
        for offset in lag_offsets:
            col_name = f'lag_{offset}'
            last_lags_unscaled.append(extended_df[col_name].iloc[-1])
        last_lags_unscaled = np.array(last_lags_unscaled)
        
        # Dummy for initial scaled lags
        dummy = np.zeros((1, num_features))
        dummy[0, 1:] = last_lags_unscaled
        dummy[0, 0] = extended_df['y'].iloc[-1]
        scaled_dummy = current_scaler.transform(dummy)
        last_row_scaled = scaled_dummy[0, 1:]
        
        for day in range(num_steps):
            # Predict
            current_seq = last_row_scaled.reshape(1, num_lags, 1)
            pred_scaled = current_model.predict(current_seq, verbose=0)[0][0]
            
            # Inverse transform
            dummy_inv = np.zeros((1, num_features))
            dummy_inv[0, 0] = pred_scaled
            dummy_inv[0, 1:] = last_lags_unscaled
            pred_original = current_scaler.inverse_transform(dummy_inv)[0, 0]
            forecasts.append(pred_original)
            
            # Create new row
            new_row_dict = {'y': pred_original}
            for offset in lag_offsets:
                prev_index = len(extended_df) - offset
                if prev_index >= 0:
                    lag_value = extended_df['y'].iloc[prev_index]
                else:
                    lag_value = 0.0
                    print(f"Warning: Insufficient data for lag_{offset} at day {day+1}, using 0.")
                new_row_dict[f'lag_{offset}'] = lag_value
            
            new_row = pd.DataFrame([new_row_dict])
            extended_df = pd.concat([extended_df, new_row], ignore_index=True)
            
            # Retrain model
            total_len = len(extended_df)
            train_size = int(total_len * 0.8)
            val_size = total_len - train_size
            
            if val_size < 1:
                val_size = 1
                train_size = total_len - 1
            
            train_ext = extended_df.iloc[:train_size]
            val_ext = extended_df.iloc[train_size:]
            
            # Create new scaler
            new_scaler = MinMaxScaler()
            train_ext_scaled = new_scaler.fit_transform(train_ext[feature_columns])
            val_ext_scaled = new_scaler.transform(val_ext[feature_columns])
            
            X_train_ext = train_ext_scaled[:, 1:]
            y_train_ext = train_ext_scaled[:, 0]
            X_val_ext = val_ext_scaled[:, 1:]
            y_val_ext = val_ext_scaled[:, 0]
            
            X_train_ext = X_train_ext.reshape(X_train_ext.shape[0], num_lags, 1)
            X_val_ext = X_val_ext.reshape(X_val_ext.shape[0], num_lags, 1)
            
            # Build and retrain model
            retrained = model_builder.build_model(
                input_shape=(num_lags, 1),
                **config.model_params
            )
            retrained.fit(
                X_train_ext, y_train_ext,
                validation_data=(X_val_ext, y_val_ext),
                epochs=int(best_params_dict.get("epochs", 50)),
                batch_size=32,
                verbose=0
            )
            current_model = retrained
            current_scaler = new_scaler
            
            # Update for next iteration
            new_lags_unscaled = []
            for offset in lag_offsets:
                new_lags_unscaled.append(extended_df[f'lag_{offset}'].iloc[-1])
            new_lags_unscaled = np.array(new_lags_unscaled)
            
            dummy_next = np.zeros((1, num_features))
            dummy_next[0, 1:] = new_lags_unscaled
            dummy_next[0, 0] = pred_original
            scaled_dummy_next = current_scaler.transform(dummy_next)
            last_row_scaled = scaled_dummy_next[0, 1:]
        
        return forecasts, current_model, current_scaler, extended_df


class MultiModelAirQualityPipeline:
    """Main pipeline class for multi-model air quality forecasting."""
    
    def __init__(self, model_configs: Dict[str, ModelConfig], drive_mount_path: str = None):
        self.model_configs = model_configs
        self.drive_mount_path = drive_mount_path
        
        # Initialize components
        self.data_fetcher = AirQualityDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.model_manager = MultiModelManager(model_configs)
        self.forecaster = MultiModelForecaster(self.model_manager)
        
        # Mount Google Drive if path provided
        if drive_mount_path:
            self._mount_drive()
    
    def _mount_drive(self):
        """Mount Google Drive if using Colab."""
        try:
            from google.colab import drive
            drive.mount(self.drive_mount_path)
            print("✅ Google Drive mounted successfully")
        except ImportError:
            print("⚠️ Google Colab not detected, skipping drive mount")
    
    def fetch_and_preprocess_data(self) -> pd.DataFrame:
        """Fetch and preprocess air quality data."""
        print("🔄 Fetching air quality data...")
        data_dict = self.data_fetcher.fetch_all_data()
        
        print("🔄 Preprocessing data...")
        df_final = self.preprocessor.preprocess_data(data_dict)
        
        return df_final
    
    def retrain_all_models(self, df_final: pd.DataFrame, save_dir: str = None) -> Dict[str, str]:
        """Retrain all pollutant models with their respective model types."""
        print("🔄 Retraining all models with different architectures...")
        retrained_models = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        pollutants = df_final.columns.tolist()
        
        for pollutant in pollutants:
            if pollutant not in self.model_configs:
                print(f"⚠️ No model configuration found for {pollutant}, skipping...")
                continue
                
            print(f"🔄 Retraining {pollutant} with {self.model_configs[pollutant].model_type} model...")
            
            # Prepare data
            data_tis = df_final[pollutant].copy()
            df_supervised = pd.DataFrame({'y': data_tis.values})
            
            # Retrain model
            retrained_model, _, _ = self.model_manager.retrain_model(pollutant, df_supervised)
            
            # Save retrained model
            if save_dir:
                save_path = self.model_manager.save_retrained_model(pollutant, retrained_model, save_dir)
                retrained_models[pollutant] = save_path
        
        return retrained_models
    
    def generate_forecasts(self, df_final: pd.DataFrame, num_steps: int = 14) -> Dict[str, List]:
        """Generate forecasts for all pollutants using their respective models."""
        print(f"🔄 Generating {num_steps}-day forecasts with different model types...")
        forecast_dict = {}
        
        pollutants = df_final.columns.tolist()
        
        for pollutant in pollutants:
            if pollutant not in self.model_configs:
                print(f"⚠️ No model configuration found for {pollutant}, skipping...")
                continue
                
            print(f"🔄 Forecasting {pollutant} with {self.model_configs[pollutant].model_type} model...")
            
            # Prepare data
            data_tis = df_final[pollutant].copy()
            df_supervised = pd.DataFrame({'y': data_tis.values})
            
            # Generate forecasts
            forecasts, _, _, _ = self.forecaster.sequential_forecast_with_retrain(
                pollutant, df_supervised, num_steps=num_steps
            )
            
            forecast_dict[pollutant] = forecasts
        
        return forecast_dict
    
    def run_complete_pipeline(self, num_forecast_days: int = 14, 
                            save_retrained: bool = True) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """Run the complete multi-model air quality forecasting pipeline."""
        print("🚀 Starting Multi-Model Air Quality Forecasting Pipeline...")
        print("=" * 60)
        
        # Display model configurations
        print("📋 Model Configurations:")
        for pollutant, config in self.model_configs.items():
            print(f"  {pollutant.upper()}: {config.model_type}")
        print()
        
        # Step 1: Fetch and preprocess data
        df_final = self.fetch_and_preprocess_data()
        
        # Step 2: Retrain all models
        save_dir = "/content/drive/MyDrive/Pengmas/Model/Retrained_Multi" if save_retrained else None
        retrained_models = self.retrain_all_models(df_final, save_dir)
        
        # Step 3: Generate forecasts
        forecasts = self.generate_forecasts(df_final, num_forecast_days)
        
        print("✅ Multi-model pipeline completed successfully!")
        return df_final, forecasts


def create_model_configs(base_dir: str) -> Dict[str, ModelConfig]:
    """Create model configurations for 6 different model types."""
    
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    model_types = ['lstm', 'gru', 'cnn_lstm', 'deep_lstm', 'bidirectional_lstm', 'dense']
    
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        model_type = model_types[i % len(model_types)]  # Cycle through model types
        
        # Model-specific parameters
        model_params = {
            'lstm': {'units': 50, 'dropout': 0.2, 'lr': 0.001},
            'gru': {'units': 60, 'dropout': 0.3, 'lr': 0.001},
            'cnn_lstm': {'cnn_filters': 64, 'lstm_units': 50, 'dropout': 0.2, 'lr': 0.001},
            'deep_lstm': {'units': [50, 30], 'dropout': 0.2, 'lr': 0.001},
            'bidirectional_lstm': {'units': 50, 'dropout': 0.2, 'lr': 0.001},
            'dense': {'units': [64, 32], 'dropout': 0.2, 'lr': 0.001}
        }
        
        # File paths
        file_paths = {
            'params': os.path.join(base_dir, f"best_params_{pollutant.upper()}_1.json"),
            'model': os.path.join(base_dir, f"final_model_{pollutant.upper()}_1.keras"),
            'scaler': os.path.join(base_dir, f"scaler_{pollutant.upper()}.pkl")
        }
        
        configs[pollutant] = ModelConfig(
            model_type=model_type,
            model_params=model_params[model_type],
            file_paths=file_paths,
            pollutant=pollutant
        )
    
    return configs


# Example usage
if __name__ == "__main__":
    # Create model configurations
    model_directory = "/content/drive/MyDrive/Pengmas/Model"
    model_configs = create_model_configs(model_directory)
    
    # Initialize pipeline
    pipeline = MultiModelAirQualityPipeline(model_configs, "/content/drive")
    
    # Run complete pipeline
    df_final, forecasts = pipeline.run_complete_pipeline(num_forecast_days=14)
    
    # Display results
    print("\n📊 Multi-Model Forecast Results:")
    for pollutant, forecast_values in forecasts.items():
        model_type = model_configs[pollutant].model_type
        print(f"{pollutant.upper()} ({model_type}): {len(forecast_values)} forecast values")
        print(f"  Sample values: {[round(x, 2) for x in forecast_values[:3]]}...")