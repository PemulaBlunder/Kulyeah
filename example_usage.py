"""
Example usage of the Air Quality Forecasting OOP implementation.
This script demonstrates how to use the classes to perform air quality forecasting.
"""

from air_quality_forecasting import AirQualityPipeline
import pandas as pd

def main():
    """Main function demonstrating the OOP air quality forecasting pipeline."""
    
    # Configuration
    MODEL_DIRECTORY = "/content/drive/MyDrive/Pengmas/Model"
    DRIVE_MOUNT_PATH = "/content/drive"
    FORECAST_DAYS = 14
    
    print("🌬️ Air Quality Forecasting System")
    print("=" * 50)
    
    # Initialize the pipeline
    print("🔄 Initializing pipeline...")
    pipeline = AirQualityPipeline(
        model_dir=MODEL_DIRECTORY,
        drive_mount_path=DRIVE_MOUNT_PATH
    )
    
    try:
        # Run the complete pipeline
        print("\n🚀 Running complete forecasting pipeline...")
        df_final, forecasts = pipeline.run_complete_pipeline(num_forecast_days=FORECAST_DAYS)
        
        # Display results
        print("\n📊 Results Summary:")
        print("-" * 30)
        print(f"📈 Historical data shape: {df_final.shape}")
        print(f"🔮 Forecast days: {FORECAST_DAYS}")
        
        print("\n📋 Forecast Results by Pollutant:")
        for pollutant, forecast_values in forecasts.items():
            print(f"  {pollutant.upper()}:")
            print(f"    - Forecast values: {len(forecast_values)}")
            print(f"    - Min: {min(forecast_values):.2f}")
            print(f"    - Max: {max(forecast_values):.2f}")
            print(f"    - Mean: {sum(forecast_values)/len(forecast_values):.2f}")
            print(f"    - Sample: {[round(x, 2) for x in forecast_values[:3]]}...")
            print()
        
        # Create a summary DataFrame
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.index = [f"Day_{i+1}" for i in range(len(forecast_df))]
        
        print("📊 Forecast Summary Table:")
        print(forecast_df.head(10))
        
        # Save results (optional)
        save_results = input("\n💾 Save forecast results to CSV? (y/n): ").lower().strip()
        if save_results == 'y':
            output_file = "air_quality_forecasts.csv"
            forecast_df.to_csv(output_file)
            print(f"✅ Results saved to {output_file}")
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your model files and data sources.")


def run_step_by_step():
    """Demonstrate running the pipeline step by step."""
    
    print("\n🔧 Step-by-Step Pipeline Demonstration")
    print("=" * 50)
    
    MODEL_DIRECTORY = "/content/drive/MyDrive/Pengmas/Model"
    pipeline = AirQualityPipeline(MODEL_DIRECTORY, "/content/drive")
    
    try:
        # Step 1: Fetch and preprocess data
        print("Step 1: Fetching and preprocessing data...")
        df_final = pipeline.fetch_and_preprocess_data()
        print(f"✅ Data shape: {df_final.shape}")
        
        # Step 2: Retrain models
        print("\nStep 2: Retraining models...")
        retrained_models = pipeline.retrain_all_models(df_final)
        print(f"✅ Retrained {len(retrained_models)} models")
        
        # Step 3: Generate forecasts
        print("\nStep 3: Generating forecasts...")
        forecasts = pipeline.generate_forecasts(df_final, num_steps=7)
        print(f"✅ Generated forecasts for {len(forecasts)} pollutants")
        
        return df_final, forecasts
        
    except Exception as e:
        print(f"❌ Error in step-by-step execution: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Run the main pipeline
    main()
    
    # Uncomment to run step-by-step demonstration
    # run_step_by_step()