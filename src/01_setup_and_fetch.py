import os
import zipfile
import subprocess
import pandas as pd
import sys

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_NAME = 'robikscube/hourly-energy-consumption'
ZIP_PREFIX = 'hourly-energy-consumption'
ZIP_NAME = 'hourly-energy-consumption.zip'
CSV_NAME = 'PJME_hourly.csv'

def download_dataset():
    """Downloads the dataset using Kaggle API if not already present."""
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    csv_path = os.path.join(DATA_DIR, CSV_NAME)

    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}. Skipping download.")
        return csv_path

    print("Downloading dataset using Kaggle API...")
    try:
        # We use Kaggle API directly via subprocess
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', DATASET_NAME, '-p', DATA_DIR],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("\n[!] Error downloading dataset.")
        print("Please ensure your Kaggle API key (kaggle.json) is set up correctly.")
        print("1. Go to kaggle.com -> Settings -> Create New Token")
        print("2. Place the kaggle.json file in your user directory (C:\\Users\\<username>\\.kaggle\\kaggle.json)")
        sys.exit(1)
    except FileNotFoundError:
         print("\n[!] Kaggle CLI tool not found. Make sure you have installed it via 'pip install kaggle'")
         sys.exit(1)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("Download and extraction complete.")
    return csv_path

def preprocess_data(csv_path):
    """Loads and performs initial basic preprocessing on the dataset."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Initial Data Shape:", df.shape)
    
    # 1. Convert timestamp column to datetime
    print("Converting timestamp to datetime...")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 2. Set datetime as index and sort it
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # 3. Handle duplicates (keep the last valid entry for the exact same hour)
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
         print(f"Found {duplicates} duplicate timestamps. Removing...")
         df = df[~df.index.duplicated(keep='last')]
    
    # 4. Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
         print(f"Found {missing} missing values. Forward-filling...")
         df.ffill(inplace=True)
    
    # Save the cleaned baseline data
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_energy_dataset.csv')
    df.to_csv(cleaned_path)
    print(f"\nPreprocessing Complete! Cleaned baseline dataset saved to: {cleaned_path}")
    print("Next step: Feature Engineering (Step 4 & 5)")

if __name__ == "__main__":
    csv_file = download_dataset()
    preprocess_data(csv_file)
