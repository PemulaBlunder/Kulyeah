"""
Configuration file for setting up different model combinations for air quality forecasting.
This file provides easy-to-use functions for creating various model configurations.
"""

import os
from multi_model_air_quality_forecasting import ModelConfig

def create_standard_configs(base_dir: str) -> dict:
    """Create standard model configurations with 6 different model types."""
    
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    model_types = ['lstm', 'gru', 'cnn_lstm', 'deep_lstm', 'bidirectional_lstm', 'dense']
    
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        model_type = model_types[i]
        
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


def create_custom_configs(base_dir: str, 
                         model_assignments: dict = None,
                         custom_params: dict = None) -> dict:
    """Create custom model configurations with user-defined assignments."""
    
    # Default model assignments
    if model_assignments is None:
        model_assignments = {
            'co': 'lstm',
            'no2': 'gru', 
            'o3': 'cnn_lstm',
            'so2': 'deep_lstm',
            'pm25': 'bidirectional_lstm',
            'pm10': 'dense'
        }
    
    # Default parameters
    default_params = {
        'lstm': {'units': 50, 'dropout': 0.2, 'lr': 0.001},
        'gru': {'units': 60, 'dropout': 0.3, 'lr': 0.001},
        'cnn_lstm': {'cnn_filters': 64, 'lstm_units': 50, 'dropout': 0.2, 'lr': 0.001},
        'deep_lstm': {'units': [50, 30], 'dropout': 0.2, 'lr': 0.001},
        'bidirectional_lstm': {'units': 50, 'dropout': 0.2, 'lr': 0.001},
        'dense': {'units': [64, 32], 'dropout': 0.2, 'lr': 0.001}
    }
    
    # Use custom parameters if provided
    if custom_params:
        for model_type, params in custom_params.items():
            if model_type in default_params:
                default_params[model_type].update(params)
    
    configs = {}
    
    for pollutant, model_type in model_assignments.items():
        file_paths = {
            'params': os.path.join(base_dir, f"best_params_{pollutant.upper()}_1.json"),
            'model': os.path.join(base_dir, f"final_model_{pollutant.upper()}_1.keras"),
            'scaler': os.path.join(base_dir, f"scaler_{pollutant.upper()}.pkl")
        }
        
        configs[pollutant] = ModelConfig(
            model_type=model_type,
            model_params=default_params[model_type],
            file_paths=file_paths,
            pollutant=pollutant
        )
    
    return configs


def create_lstm_only_configs(base_dir: str, 
                           units_list: list = [30, 50, 70, 90, 110, 130]) -> dict:
    """Create configurations with only LSTM models but different architectures."""
    
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        units = units_list[i % len(units_list)]
        
        file_paths = {
            'params': os.path.join(base_dir, f"best_params_{pollutant.upper()}_1.json"),
            'model': os.path.join(base_dir, f"final_model_{pollutant.upper()}_1.keras"),
            'scaler': os.path.join(base_dir, f"scaler_{pollutant.upper()}.pkl")
        }
        
        configs[pollutant] = ModelConfig(
            model_type='lstm',
            model_params={'units': units, 'dropout': 0.2, 'lr': 0.001},
            file_paths=file_paths,
            pollutant=pollutant
        )
    
    return configs


def create_high_performance_configs(base_dir: str) -> dict:
    """Create high-performance model configurations with optimized parameters."""
    
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    model_types = ['deep_lstm', 'bidirectional_lstm', 'cnn_lstm', 'deep_lstm', 'bidirectional_lstm', 'cnn_lstm']
    
    # High-performance parameters
    high_perf_params = {
        'deep_lstm': {'units': [100, 50, 25], 'dropout': 0.3, 'lr': 0.0005},
        'bidirectional_lstm': {'units': 100, 'dropout': 0.3, 'lr': 0.0005},
        'cnn_lstm': {'cnn_filters': 128, 'lstm_units': 100, 'dropout': 0.3, 'lr': 0.0005}
    }
    
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        model_type = model_types[i]
        
        file_paths = {
            'params': os.path.join(base_dir, f"best_params_{pollutant.upper()}_1.json"),
            'model': os.path.join(base_dir, f"final_model_{pollutant.upper()}_1.keras"),
            'scaler': os.path.join(base_dir, f"scaler_{pollutant.upper()}.pkl")
        }
        
        configs[pollutant] = ModelConfig(
            model_type=model_type,
            model_params=high_perf_params[model_type],
            file_paths=file_paths,
            pollutant=pollutant
        )
    
    return configs


def create_fast_configs(base_dir: str) -> dict:
    """Create fast model configurations with smaller architectures for quick testing."""
    
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    model_types = ['lstm', 'gru', 'dense', 'lstm', 'gru', 'dense']
    
    # Fast parameters
    fast_params = {
        'lstm': {'units': 20, 'dropout': 0.1, 'lr': 0.01},
        'gru': {'units': 20, 'dropout': 0.1, 'lr': 0.01},
        'dense': {'units': [32, 16], 'dropout': 0.1, 'lr': 0.01}
    }
    
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        model_type = model_types[i]
        
        file_paths = {
            'params': os.path.join(base_dir, f"best_params_{pollutant.upper()}_1.json"),
            'model': os.path.join(base_dir, f"final_model_{pollutant.upper()}_1.keras"),
            'scaler': os.path.join(base_dir, f"scaler_{pollutant.upper()}.pkl")
        }
        
        configs[pollutant] = ModelConfig(
            model_type=model_type,
            model_params=fast_params[model_type],
            file_paths=file_paths,
            pollutant=pollutant
        )
    
    return configs


def create_ensemble_configs(base_dir: str) -> dict:
    """Create ensemble configurations with multiple models per pollutant."""
    
    # This is a simplified version - in practice, you'd need to modify the pipeline
    # to handle multiple models per pollutant
    pollutants = ['co', 'no2', 'o3', 'so2', 'pm25', 'pm10']
    model_types = ['lstm', 'gru', 'cnn_lstm', 'deep_lstm', 'bidirectional_lstm', 'dense']
    
    configs = {}
    
    for i, pollutant in enumerate(pollutants):
        model_type = model_types[i]
        
        # Use different parameters for ensemble diversity
        model_params = {
            'lstm': {'units': 50 + i*10, 'dropout': 0.2 + i*0.05, 'lr': 0.001},
            'gru': {'units': 60 + i*10, 'dropout': 0.3 + i*0.05, 'lr': 0.001},
            'cnn_lstm': {'cnn_filters': 64 + i*16, 'lstm_units': 50 + i*10, 'dropout': 0.2 + i*0.05, 'lr': 0.001},
            'deep_lstm': {'units': [50 + i*10, 30 + i*5], 'dropout': 0.2 + i*0.05, 'lr': 0.001},
            'bidirectional_lstm': {'units': 50 + i*10, 'dropout': 0.2 + i*0.05, 'lr': 0.001},
            'dense': {'units': [64 + i*16, 32 + i*8], 'dropout': 0.2 + i*0.05, 'lr': 0.001}
        }
        
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


# Example usage functions
def get_available_configs():
    """Get list of available configuration functions."""
    return [
        'create_standard_configs',
        'create_custom_configs', 
        'create_lstm_only_configs',
        'create_high_performance_configs',
        'create_fast_configs',
        'create_ensemble_configs'
    ]


def print_config_summary(configs: dict):
    """Print a summary of model configurations."""
    print("\n📋 Model Configuration Summary:")
    print("-" * 40)
    for pollutant, config in configs.items():
        print(f"{pollutant.upper():>6} → {config.model_type.upper():>20} → {config.model_params}")


if __name__ == "__main__":
    # Example usage
    base_dir = "/content/drive/MyDrive/Pengmas/Model"
    
    print("🔧 Model Configuration Examples")
    print("=" * 40)
    
    # Standard configurations
    print("\n1. Standard Configurations:")
    standard_configs = create_standard_configs(base_dir)
    print_config_summary(standard_configs)
    
    # Custom configurations
    print("\n2. Custom Configurations:")
    custom_assignments = {
        'co': 'lstm',
        'no2': 'gru',
        'o3': 'cnn_lstm',
        'so2': 'deep_lstm',
        'pm25': 'bidirectional_lstm',
        'pm10': 'dense'
    }
    custom_configs = create_custom_configs(base_dir, custom_assignments)
    print_config_summary(custom_configs)
    
    # High performance configurations
    print("\n3. High Performance Configurations:")
    high_perf_configs = create_high_performance_configs(base_dir)
    print_config_summary(high_perf_configs)