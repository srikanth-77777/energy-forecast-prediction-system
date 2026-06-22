import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load the previously cleaned data."""
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_energy_dataset.csv')
    if not os.path.exists(cleaned_path):
        print(f"Error: {cleaned_path} not found. Run 01_setup_and_fetch.py first.")
        return None
    
    df = pd.read_csv(cleaned_path, index_col='Datetime', parse_dates=True)
    return df

def feature_engineering(df):
    """Generates Time-based and basic Lag features."""
    print("Generating Time-based features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak'] = df['hour'].between(6, 22).astype(int)

    print("Generating basic Lag features...")
    df['lag_1h'] = df['PJME_MW'].shift(1)
    
    # Optional: more lag features as described in Phase 2
    df['lag_24h'] = df['PJME_MW'].shift(24)
    
    # Drop NaNs created by lag features
    before_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before_len - len(df)} rows containing NaNs from lag shifting.")
    
    engineered_path = os.path.join(DATA_DIR, 'engineered_features.csv')
    df.to_csv(engineered_path)
    print(f"Engineered dataset saved to {engineered_path}")
    return df

def train_baseline_model(df):
    """Trains a robust Baseline Linear Regression model."""
    print("\n--- Training Robust Baseline (Linear Regression) ---")
    
    # Define features and target
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak', 'lag_1h', 'lag_24h']
    target = 'PJME_MW'
    
    X = df[features]
    y = df[target]
    
    # Split chronologically (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Dataset: PJME Hourly Energy Consumption")
    print(f"Features: {', '.join(features)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples:  {len(X_test)}")
    
    # Initialize and fit
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n[Linear Regression Performance]")
    print(f"{'Metric':<10} | {'Value':<10}")
    print(f"{'-'*11}|{'-'*11}")
    print(f"{'MAE':<10} | {mae:>10.2f} MW")
    print(f"{'RMSE':<10} | {rmse:>10.2f} MW")
    print(f"{'R-squared':<10} | {r2:>10.4f}")
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, 'baseline_linear_regression.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel successfully exported to: {model_path}")
    
    # Visualization: 1 Week Comparison
    plot_points = 24 * 7
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index[:plot_points], y_test.values[:plot_points], label='Actual Demand', color='#3b82f6', linewidth=2)
    plt.plot(y_test.index[:plot_points], y_pred[:plot_points], label='LR Prediction', color='#ef4444', linestyle='--', linewidth=2)
    plt.title('Linear Regression: 7-Day Forecast Validation', fontsize=14)
    plt.xlabel('Date/Time', fontsize=12)
    plt.ylabel('Energy Demand (MW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'baseline_predictions.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Validation plot generated at: {plot_path}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = feature_engineering(df)
        train_baseline_model(df)
