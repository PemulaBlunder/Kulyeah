# Air Quality Forecasting - Object-Oriented Implementation

This project provides an object-oriented implementation of an air quality forecasting system using LSTM neural networks. The system fetches air quality data, preprocesses it, retrains models, and generates sequential forecasts.

## 🏗️ Architecture

The code is organized into several main classes:

### 1. `AirQualityDataFetcher`
- **Purpose**: Handles API data retrieval from air quality monitoring services
- **Key Methods**:
  - `fetch_pollutant_data()`: Fetches data for a specific pollutant
  - `fetch_all_data()`: Fetches data for all pollutants

### 2. `DataPreprocessor`
- **Purpose**: Handles data cleaning, merging, and preprocessing
- **Key Methods**:
  - `merge_pollutant_data()`: Combines data from all pollutants
  - `clean_and_interpolate()`: Performs data cleaning and interpolation
  - `remove_today_data()`: Removes current day data
  - `preprocess_data()`: Complete preprocessing pipeline

### 3. `LSTMModel`
- **Purpose**: Static class for building LSTM model architecture
- **Key Methods**:
  - `build_lstm()`: Creates LSTM model with specified parameters

### 4. `ModelManager`
- **Purpose**: Manages model loading, saving, and retraining
- **Key Methods**:
  - `load_model_components()`: Loads model, parameters, and scaler
  - `retrain_model()`: Retrains model with new data
  - `save_retrained_model()`: Saves retrained model

### 5. `AirQualityForecaster`
- **Purpose**: Handles sequential forecasting with retraining
- **Key Methods**:
  - `sequential_forecast_with_retrain()`: Performs step-by-step forecasting with model retraining

### 6. `AirQualityPipeline`
- **Purpose**: Main orchestrator class that coordinates the entire process
- **Key Methods**:
  - `fetch_and_preprocess_data()`: Complete data pipeline
  - `retrain_all_models()`: Retrains all pollutant models
  - `generate_forecasts()`: Generates forecasts for all pollutants
  - `run_complete_pipeline()`: Runs the entire forecasting pipeline

## 🚀 Usage

### Basic Usage

```python
from air_quality_forecasting import AirQualityPipeline

# Initialize pipeline
pipeline = AirQualityPipeline(
    model_dir="/path/to/models",
    drive_mount_path="/content/drive"  # Optional for Google Colab
)

# Run complete pipeline
df_final, forecasts = pipeline.run_complete_pipeline(num_forecast_days=14)
```

### Step-by-Step Usage

```python
# Step 1: Fetch and preprocess data
df_final = pipeline.fetch_and_preprocess_data()

# Step 2: Retrain models
retrained_models = pipeline.retrain_all_models(df_final)

# Step 3: Generate forecasts
forecasts = pipeline.generate_forecasts(df_final, num_steps=14)
```

### Individual Component Usage

```python
# Use individual components
data_fetcher = AirQualityDataFetcher()
data_dict = data_fetcher.fetch_all_data()

preprocessor = DataPreprocessor()
df_clean = preprocessor.preprocess_data(data_dict)

model_manager = ModelManager("/path/to/models")
model, params, scaler = model_manager.load_model_components("pm25")
```

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### Model Directory Structure
```
/path/to/models/
├── best_params_CO_1.json
├── best_params_NO2_1.json
├── best_params_O3_1.json
├── best_params_SO2_1.json
├── best_params_PM25_1.json
├── best_params_PM10_1.json
├── final_model_CO_1.keras
├── final_model_NO2_1.keras
├── final_model_O3_1.keras
├── final_model_SO2_1.keras
├── final_model_PM25_1.keras
├── final_model_PM10_1.keras
├── scaler_CO.pkl
├── scaler_NO2.pkl
├── scaler_O3.pkl
├── scaler_SO2.pkl
├── scaler_PM25.pkl
├── scaler_PM10.pkl
└── Retrained/
    ├── co_retrained.keras
    ├── no2_retrained.keras
    ├── o3_retrained.keras
    ├── so2_retrained.keras
    ├── pm25_retrained.keras
    └── pm10_retrained.keras
```

## 🌟 Key Features

1. **Modular Design**: Each component has a specific responsibility
2. **Easy to Extend**: Add new pollutants or forecasting methods easily
3. **Error Handling**: Robust error handling throughout the pipeline
4. **Flexible Configuration**: Easy to modify parameters and settings
5. **Google Colab Compatible**: Works seamlessly in Google Colab environment
6. **Sequential Forecasting**: Advanced forecasting with model retraining at each step

## 🔄 Workflow

1. **Data Fetching**: Retrieves air quality data from API
2. **Preprocessing**: Cleans, merges, and prepares data
3. **Model Retraining**: Retrains LSTM models with latest data
4. **Sequential Forecasting**: Generates forecasts with continuous model updates
5. **Results**: Returns forecasted values for all pollutants

## 📊 Output

The pipeline returns:
- `df_final`: Preprocessed historical data
- `forecasts`: Dictionary with forecast values for each pollutant

Example output:
```python
{
    'co': [1.2, 1.3, 1.1, ...],
    'no2': [15.5, 16.2, 14.8, ...],
    'o3': [45.2, 47.1, 43.9, ...],
    # ... other pollutants
}
```

## 🛠️ Customization

### Adding New Pollutants
1. Add pollutant name to `AirQualityDataFetcher.pollutants`
2. Ensure model files exist in the model directory
3. Update column order in `DataPreprocessor.preprocess_data()`

### Modifying Model Architecture
1. Update `LSTMModel.build_lstm()` method
2. Adjust parameters in model configuration files

### Changing Forecasting Strategy
1. Modify `AirQualityForecaster.sequential_forecast_with_retrain()`
2. Implement new forecasting methods in the class

## 📝 Notes

- The system is designed for Google Colab but can be adapted for other environments
- Model files should be pre-trained and saved in the specified directory structure
- The system automatically handles missing data through interpolation
- Sequential forecasting retrains the model at each prediction step for better accuracy