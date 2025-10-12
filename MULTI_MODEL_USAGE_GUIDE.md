# Multi-Model Air Quality Forecasting - Usage Guide

This guide shows you how to use 6 different model types for air quality forecasting with the OOP implementation.

## 🏗️ Architecture Overview

The multi-model system supports 6 different neural network architectures:

1. **LSTM** - Long Short-Term Memory networks
2. **GRU** - Gated Recurrent Units  
3. **CNN-LSTM** - Convolutional + LSTM hybrid
4. **Deep LSTM** - Multi-layer LSTM
5. **Bidirectional LSTM** - Bidirectional LSTM
6. **Dense NN** - Simple dense neural network

## 🚀 Quick Start

### Basic Usage with 6 Different Models

```python
from multi_model_air_quality_forecasting import MultiModelAirQualityPipeline, create_model_configs

# Create model configurations (automatically assigns 6 different model types)
model_configs = create_model_configs("/content/drive/MyDrive/Pengmas/Model")

# Initialize pipeline
pipeline = MultiModelAirQualityPipeline(model_configs, "/content/drive")

# Run complete pipeline
df_final, forecasts = pipeline.run_complete_pipeline(num_forecast_days=14)
```

### Custom Model Assignments

```python
from model_configurations import create_custom_configs

# Define which model to use for each pollutant
model_assignments = {
    'co': 'lstm',
    'no2': 'gru', 
    'o3': 'cnn_lstm',
    'so2': 'deep_lstm',
    'pm25': 'bidirectional_lstm',
    'pm10': 'dense'
}

# Create custom configurations
configs = create_custom_configs("/content/drive/MyDrive/Pengmas/Model", model_assignments)

# Use with pipeline
pipeline = MultiModelAirQualityPipeline(configs, "/content/drive")
df_final, forecasts = pipeline.run_complete_pipeline()
```

## 📋 Predefined Configurations

### 1. Standard Configurations
```python
from model_configurations import create_standard_configs

configs = create_standard_configs("/content/drive/MyDrive/Pengmas/Model")
# Automatically assigns: CO→LSTM, NO2→GRU, O3→CNN-LSTM, SO2→Deep-LSTM, PM25→Bidirectional-LSTM, PM10→Dense
```

### 2. High Performance Configurations
```python
from model_configurations import create_high_performance_configs

configs = create_high_performance_configs("/content/drive/MyDrive/Pengmas/Model")
# Uses larger, more complex models with optimized parameters
```

### 3. Fast Configurations (for testing)
```python
from model_configurations import create_fast_configs

configs = create_fast_configs("/content/drive/MyDrive/Pengmas/Model")
# Uses smaller, faster models for quick testing
```

### 4. LSTM-Only Configurations
```python
from model_configurations import create_lstm_only_configs

configs = create_lstm_only_configs("/content/drive/MyDrive/Pengmas/Model")
# Uses only LSTM models with different architectures
```

## 🔧 Advanced Usage

### Custom Parameters

```python
from model_configurations import create_custom_configs

# Define custom parameters for specific model types
custom_params = {
    'lstm': {'units': 100, 'dropout': 0.3, 'lr': 0.0005},
    'gru': {'units': 80, 'dropout': 0.25, 'lr': 0.001},
    'cnn_lstm': {'cnn_filters': 128, 'lstm_units': 80, 'dropout': 0.3, 'lr': 0.0005}
}

configs = create_custom_configs(
    "/content/drive/MyDrive/Pengmas/Model", 
    model_assignments=model_assignments,
    custom_params=custom_params
)
```

### Step-by-Step Pipeline

```python
# Initialize pipeline
pipeline = MultiModelAirQualityPipeline(configs, "/content/drive")

# Step 1: Fetch and preprocess data
df_final = pipeline.fetch_and_preprocess_data()

# Step 2: Retrain all models
retrained_models = pipeline.retrain_all_models(df_final)

# Step 3: Generate forecasts
forecasts = pipeline.generate_forecasts(df_final, num_steps=14)
```

## 📊 Model Types and Their Characteristics

| Model Type | Best For | Characteristics |
|------------|----------|-----------------|
| **LSTM** | General time series | Good balance of performance and speed |
| **GRU** | Fast training | Similar to LSTM but faster |
| **CNN-LSTM** | Complex patterns | Good for spatial-temporal patterns |
| **Deep LSTM** | Complex sequences | Multiple layers for complex patterns |
| **Bidirectional LSTM** | Context-aware | Uses both past and future context |
| **Dense NN** | Simple patterns | Fastest, good for simple relationships |

## 🎯 Recommended Model Assignments

### For Maximum Accuracy
```python
model_assignments = {
    'co': 'deep_lstm',           # Complex patterns
    'no2': 'bidirectional_lstm', # Context-aware
    'o3': 'cnn_lstm',           # Spatial-temporal patterns
    'so2': 'deep_lstm',         # Complex patterns
    'pm25': 'bidirectional_lstm', # Context-aware
    'pm10': 'cnn_lstm'          # Spatial-temporal patterns
}
```

### For Fast Processing
```python
model_assignments = {
    'co': 'lstm',
    'no2': 'gru',
    'o3': 'lstm',
    'so2': 'gru',
    'pm25': 'lstm',
    'pm10': 'dense'
}
```

### For Balanced Performance
```python
model_assignments = {
    'co': 'lstm',
    'no2': 'gru',
    'o3': 'cnn_lstm',
    'so2': 'deep_lstm',
    'pm25': 'bidirectional_lstm',
    'pm10': 'dense'
}
```

## 📈 Results Analysis

### View Forecast Results
```python
# Display forecast summary
for pollutant, forecast_values in forecasts.items():
    model_type = configs[pollutant].model_type
    print(f"{pollutant.upper()} ({model_type}): {len(forecast_values)} forecasts")
    print(f"  Range: {min(forecast_values):.2f} - {max(forecast_values):.2f}")
    print(f"  Mean: {np.mean(forecast_values):.2f}")
```

### Compare Model Performance
```python
# Group forecasts by model type
model_performance = {}
for pollutant, forecast_values in forecasts.items():
    model_type = configs[pollutant].model_type
    if model_type not in model_performance:
        model_performance[model_type] = []
    model_performance[model_type].extend(forecast_values)

# Analyze each model type
for model_type, values in model_performance.items():
    print(f"{model_type}: Mean={np.mean(values):.2f}, Std={np.std(values):.2f}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Model file not found**: Ensure all model files exist in the specified directory
2. **Memory issues**: Use `create_fast_configs()` for smaller models
3. **Slow training**: Reduce epochs or use simpler models
4. **Poor forecasts**: Try different model assignments or parameters

### Debug Mode

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.INFO)

# Run with debug information
pipeline = MultiModelAirQualityPipeline(configs, "/content/drive")
df_final, forecasts = pipeline.run_complete_pipeline()
```

## 📝 Example Scripts

### Complete Example
```python
from multi_model_air_quality_forecasting import MultiModelAirQualityPipeline
from model_configurations import create_standard_configs

# Create configurations
configs = create_standard_configs("/content/drive/MyDrive/Pengmas/Model")

# Initialize and run pipeline
pipeline = MultiModelAirQualityPipeline(configs, "/content/drive")
df_final, forecasts = pipeline.run_complete_pipeline(num_forecast_days=14)

# Display results
print("Forecast Results:")
for pollutant, values in forecasts.items():
    print(f"{pollutant}: {values[:5]}...")
```

### Custom Model Testing
```python
# Test individual models
for pollutant in ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']:
    config = configs[pollutant]
    print(f"Testing {pollutant} with {config.model_type} model...")
    
    # Your testing code here
```

## 🎉 Benefits of Multi-Model Approach

1. **Diversity**: Different models capture different patterns
2. **Robustness**: If one model fails, others continue working
3. **Optimization**: Each pollutant can use its best-suited model
4. **Flexibility**: Easy to experiment with different combinations
5. **Scalability**: Easy to add new model types or pollutants

This multi-model approach gives you the flexibility to use the best model for each pollutant while maintaining a clean, organized codebase!