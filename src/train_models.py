import os
import sys
import pandas as pd

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.01_setup_and_fetch import download_dataset, preprocess_data
    from src.02_feature_engineering import feature_engineering, train_baseline_model
except ImportError as e:
    # Try direct import if running from src
    try:
        from 01_setup_and_fetch import download_dataset, preprocess_data
        from 02_feature_engineering import feature_engineering, train_baseline_model
    except ImportError:
        print(f"Import Error: {e}")
        print("Ensure all dependencies are installed (pandas, sklearn, joblib, etc.)")
        sys.exit(1)

def main():
    print("==================================================")
    print("ENERGY DEMAND FORECASTER - LINEAR REGRESSION ONLY")
    print("==================================================\n")

    # Step 1: Data Acquisition
    print("[Step 1/3] Preparing data...")
    csv_file = download_dataset()
    preprocess_data(csv_file)

    # Step 2: Feature Engineering
    print("\n[Step 2/3] Generating features...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, 'data', 'cleaned_energy_dataset.csv')
    
    if not os.path.exists(cleaned_path):
        print(f"Error: {cleaned_path} not found.")
        return

    df = pd.read_csv(cleaned_path, index_col='Datetime', parse_dates=True)
    df = feature_engineering(df)

    # Step 3: Train Linear Regression (Robust Baseline)
    print("\n[Step 3/3] Training Linear Regression...")
    train_baseline_model(df)

    print("\n" + "="*50)
    print("BASELINE MODEL TRAINED AND SAVED SUCCESSFULLY!")
    print("Run 'python api.py' to start the server.")
    print("="*50)

if __name__ == "__main__":
    main()
