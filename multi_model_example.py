"""
Example usage of the Multi-Model Air Quality Forecasting system.
This script demonstrates how to use 6 different model types for forecasting.
"""

from multi_model_air_quality_forecasting import (
    MultiModelAirQualityPipeline, 
    ModelConfig, 
    ModelFactory,
    create_model_configs
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_custom_model_configs():
    """Create custom model configurations for 6 different pollutants."""
    
    base_dir = "/content/drive/MyDrive/Pengmas/Model"
    
    # Define 6 different model configurations
    configs = {
        'co': ModelConfig(
            model_type='lstm',
            model_params={'units': 50, 'dropout': 0.2, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_CO_1.json"),
                'model': os.path.join(base_dir, "final_model_CO_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_CO.pkl")
            },
            pollutant='co'
        ),
        
        'no2': ModelConfig(
            model_type='gru',
            model_params={'units': 60, 'dropout': 0.3, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_NO2_1.json"),
                'model': os.path.join(base_dir, "final_model_NO2_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_NO2.pkl")
            },
            pollutant='no2'
        ),
        
        'o3': ModelConfig(
            model_type='cnn_lstm',
            model_params={'cnn_filters': 64, 'lstm_units': 50, 'dropout': 0.2, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_O3_1.json"),
                'model': os.path.join(base_dir, "final_model_O3_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_O3.pkl")
            },
            pollutant='o3'
        ),
        
        'so2': ModelConfig(
            model_type='deep_lstm',
            model_params={'units': [50, 30], 'dropout': 0.2, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_SO2_1.json"),
                'model': os.path.join(base_dir, "final_model_SO2_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_SO2.pkl")
            },
            pollutant='so2'
        ),
        
        'pm25': ModelConfig(
            model_type='bidirectional_lstm',
            model_params={'units': 50, 'dropout': 0.2, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_PM25_1.json"),
                'model': os.path.join(base_dir, "final_model_PM25_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_PM25.pkl")
            },
            pollutant='pm25'
        ),
        
        'pm10': ModelConfig(
            model_type='dense',
            model_params={'units': [64, 32], 'dropout': 0.2, 'lr': 0.001},
            file_paths={
                'params': os.path.join(base_dir, "best_params_PM10_1.json"),
                'model': os.path.join(base_dir, "final_model_PM10_1.keras"),
                'scaler': os.path.join(base_dir, "scaler_PM10.pkl")
            },
            pollutant='pm10'
        )
    }
    
    return configs


def run_multi_model_forecasting():
    """Run the complete multi-model forecasting pipeline."""
    
    print("🌬️ Multi-Model Air Quality Forecasting System")
    print("=" * 60)
    
    # Create custom model configurations
    print("🔄 Creating model configurations...")
    model_configs = create_custom_model_configs()
    
    # Display model assignments
    print("\n📋 Model Assignments:")
    print("-" * 30)
    for pollutant, config in model_configs.items():
        print(f"{pollutant.upper():>6} → {config.model_type.upper()}")
    
    # Initialize pipeline
    print("\n🔄 Initializing multi-model pipeline...")
    pipeline = MultiModelAirQualityPipeline(
        model_configs=model_configs,
        drive_mount_path="/content/drive"
    )
    
    try:
        # Run complete pipeline
        print("\n🚀 Running complete multi-model forecasting pipeline...")
        df_final, forecasts = pipeline.run_complete_pipeline(
            num_forecast_days=14,
            save_retrained=True
        )
        
        # Display results
        print("\n📊 Multi-Model Forecast Results:")
        print("=" * 50)
        
        # Create results summary
        results_summary = []
        for pollutant, forecast_values in forecasts.items():
            model_type = model_configs[pollutant].model_type
            results_summary.append({
                'Pollutant': pollutant.upper(),
                'Model Type': model_type.upper(),
                'Forecast Days': len(forecast_values),
                'Min Value': round(min(forecast_values), 2),
                'Max Value': round(max(forecast_values), 2),
                'Mean Value': round(np.mean(forecast_values), 2),
                'Std Dev': round(np.std(forecast_values), 2)
            })
        
        # Display summary table
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.index = [f"Day_{i+1}" for i in range(len(forecast_df))]
        
        print(f"\n📈 Forecast Data Shape: {forecast_df.shape}")
        print("\n📋 Sample Forecast Data (First 7 days):")
        print(forecast_df.head(7).round(2))
        
        # Save results
        save_results = input("\n💾 Save forecast results to CSV? (y/n): ").lower().strip()
        if save_results == 'y':
            output_file = "multi_model_air_quality_forecasts.csv"
            forecast_df.to_csv(output_file)
            print(f"✅ Results saved to {output_file}")
            
            # Also save summary
            summary_file = "multi_model_forecast_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary saved to {summary_file}")
        
        return df_final, forecasts, model_configs
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your model files and configurations.")
        return None, None, None


def compare_model_performance(forecasts, model_configs):
    """Compare performance of different model types."""
    
    print("\n🔍 Model Performance Comparison")
    print("=" * 40)
    
    # Calculate statistics for each model type
    model_stats = {}
    
    for pollutant, forecast_values in forecasts.items():
        model_type = model_configs[pollutant].model_type
        
        if model_type not in model_stats:
            model_stats[model_type] = []
        
        model_stats[model_type].extend(forecast_values)
    
    # Display statistics for each model type
    for model_type, values in model_stats.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Total predictions: {len(values)}")
        print(f"  Mean value: {np.mean(values):.2f}")
        print(f"  Std deviation: {np.std(values):.2f}")
        print(f"  Min value: {np.min(values):.2f}")
        print(f"  Max value: {np.max(values):.2f}")


def create_visualization(forecasts, model_configs):
    """Create visualization of forecasts by model type."""
    
    print("\n📊 Creating forecast visualizations...")
    
    # Create subplots for each model type
    model_types = list(set(config.model_type for config in model_configs.values()))
    n_models = len(model_types)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_type in enumerate(model_types):
        if i >= len(axes):
            break
            
        # Get pollutants using this model type
        pollutants = [p for p, config in model_configs.items() if config.model_type == model_type]
        
        # Plot forecasts for each pollutant
        for pollutant in pollutants:
            if pollutant in forecasts:
                axes[i].plot(forecasts[pollutant], label=pollutant.upper(), marker='o', markersize=3)
        
        axes[i].set_title(f'{model_type.upper()} Model Forecasts')
        axes[i].set_xlabel('Days')
        axes[i].set_ylabel('Concentration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(model_types), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('multi_model_forecasts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'multi_model_forecasts.png'")


def run_individual_model_testing():
    """Test individual models separately."""
    
    print("\n🧪 Individual Model Testing")
    print("=" * 40)
    
    # Create model configurations
    model_configs = create_custom_model_configs()
    
    # Initialize pipeline
    pipeline = MultiModelAirQualityPipeline(model_configs, "/content/drive")
    
    # Fetch and preprocess data
    df_final = pipeline.fetch_and_preprocess_data()
    
    # Test each model individually
    for pollutant, config in model_configs.items():
        print(f"\n🔬 Testing {pollutant.upper()} with {config.model_type.upper()} model...")
        
        try:
            # Prepare data
            data_tis = df_final[pollutant].copy()
            df_supervised = pd.DataFrame({'y': data_tis.values})
            
            # Retrain model
            retrained_model, scaler, df_processed = pipeline.model_manager.retrain_model(
                pollutant, df_supervised
            )
            
            # Generate short forecast
            forecasts, _, _, _ = pipeline.forecaster.sequential_forecast_with_retrain(
                pollutant, df_supervised, num_steps=3
            )
            
            print(f"  ✅ {config.model_type.upper()} model for {pollutant.upper()} working correctly")
            print(f"  📊 Sample forecast: {[round(x, 2) for x in forecasts]}")
            
        except Exception as e:
            print(f"  ❌ Error with {config.model_type.upper()} model for {pollutant.upper()}: {str(e)}")


def main():
    """Main function to run all examples."""
    
    print("🚀 Multi-Model Air Quality Forecasting Examples")
    print("=" * 60)
    
    # Run main forecasting pipeline
    df_final, forecasts, model_configs = run_multi_model_forecasting()
    
    if forecasts and model_configs:
        # Compare model performance
        compare_model_performance(forecasts, model_configs)
        
        # Create visualizations
        try:
            create_visualization(forecasts, model_configs)
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
        
        # Run individual model testing
        run_individual_model_testing()
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()