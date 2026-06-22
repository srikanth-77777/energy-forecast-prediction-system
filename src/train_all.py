"""
Model Training Pipeline — Trains 5 models x 5 regions with GridSearchCV.
Usage: python train_all.py
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DIR, MODELS_DIR, METRICS_DIR, OUTPUTS_DIR,
    REGIONS, FEATURE_COLS, TARGET_COL, TRAIN_RATIO,
    MODEL_NAMES, GRID_PARAMS
)


def load_processed(region_key: str) -> pd.DataFrame:
    """Load a processed regional dataset."""
    path = os.path.join(PROCESSED_DIR, f"{region_key}_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}. Run data_pipeline.py first.")
    df = pd.read_csv(path, index_col='Datetime', parse_dates=True)
    return df


def get_model(name: str):
    """Return an untrained model instance by name."""
    if name == 'Linear Regression':
        return LinearRegression()
    elif name == 'Random Forest':
        return RandomForestRegressor(random_state=42, n_jobs=-1)
    elif name == 'Gradient Boosting':
        return GradientBoostingRegressor(random_state=42)
    elif name == 'XGBoost':
        return xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    elif name == 'LightGBM':
        return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {name}")


def compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'r2': round(float(r2), 6),
        'mape': round(float(mape), 2),
    }


def train_region(region_key: str) -> list:
    """Train all 5 models for one region."""
    print(f"\n{'='*60}")
    print(f"  TRAINING REGION: {region_key}")
    print(f"{'='*60}")
    
    df = load_processed(region_key)
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    # Chronological split
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"  Features: {len(FEATURE_COLS)}")
    
    region_results = []
    tscv = TimeSeriesSplit(n_splits=3)
    
    for model_name in MODEL_NAMES:
        print(f"\n  --- {model_name} ---")
        t0 = time.time()
        
        base_model = get_model(model_name)
        
        if model_name in GRID_PARAMS:
            # GridSearchCV with TimeSeriesSplit
            print(f"  Running GridSearchCV...")
            grid = GridSearchCV(
                base_model,
                GRID_PARAMS[model_name],
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
        else:
            # Linear Regression — no tuning needed
            base_model.fit(X_train, y_train)
            best_model = base_model
        
        # Predict
        y_pred = best_model.predict(X_test)
        elapsed = time.time() - t0
        
        # Metrics
        metrics = compute_metrics(y_test.values, y_pred)
        metrics['train_time_sec'] = round(elapsed, 1)
        
        print(f"  R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:,.0f} | MAE: {metrics['mae']:,.0f} | MAPE: {metrics['mape']:.2f}% | Time: {elapsed:.1f}s")
        
        # Save model
        model_filename = f"{region_key}_{model_name.lower().replace(' ', '_')}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(best_model, model_path)
        
        # Collect result
        result = {
            'region': region_key,
            'model': model_name,
            'metrics': metrics,
            'model_file': model_filename,
        }
        region_results.append(result)
    
    return region_results


def main():
    print("=" * 60)
    print("  ENERGY DEMAND FORECASTING - MODEL TRAINING")
    print("  5 Models x 5 Regions = 25 models with GridSearchCV")
    print("=" * 60)
    
    all_results = []
    total_start = time.time()
    
    for region_key in REGIONS:
        results = train_region(region_key)
        all_results.extend(results)
    
    total_time = time.time() - total_start
    
    # Save all metrics as JSON
    metrics_path = os.path.join(METRICS_DIR, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")
    
    # Also save per-region JSON for fast API lookup
    for region_key in REGIONS:
        region_data = [r for r in all_results if r['region'] == region_key]
        region_path = os.path.join(METRICS_DIR, f"{region_key}_metrics.json")
        with open(region_path, 'w') as f:
            json.dump(region_data, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Region':<10} {'Model':<22} {'R2':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>8}")
    print(f"  {'-'*10} {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    
    for r in all_results:
        m = r['metrics']
        print(f"  {r['region']:<10} {r['model']:<22} {m['r2']:>8.4f} {m['rmse']:>10,.0f} {m['mae']:>10,.0f} {m['mape']:>7.2f}%")
    
    # Best model per region
    print(f"\n  BEST MODEL PER REGION:")
    for region_key in REGIONS:
        region_data = [r for r in all_results if r['region'] == region_key]
        best = max(region_data, key=lambda x: x['metrics']['r2'])
        print(f"  {region_key:<10} -> {best['model']} (R2: {best['metrics']['r2']:.4f})")
    
    print(f"\n  Total training time: {total_time/60:.1f} minutes")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
