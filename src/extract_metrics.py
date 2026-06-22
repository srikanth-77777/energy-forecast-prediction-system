import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = '../data'
MODELS_DIR = '../models'

def load_data():
    engineered_path = os.path.join(DATA_DIR, 'engineered_features.csv')
    df = pd.read_csv(engineered_path, index_col='Datetime', parse_dates=True)
    return df

def get_metrics_for(model_path, X_test, y_test, name):
    if not os.path.exists(model_path):
        print(f"Model {name} not found at {model_path}.")
        return
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n[{name} Metrics]")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

if __name__ == "__main__":
    df = load_data()
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak', 'lag_1h', 'lag_24h']
    target = 'PJME_MW'
    
    split_idx = int(len(df) * 0.8)
    X_test = df[features].iloc[split_idx:]
    y_test = df[target].iloc[split_idx:]
    
    get_metrics_for(os.path.join(MODELS_DIR, 'random_forest.pkl'), X_test, y_test, "Random Forest")
    get_metrics_for(os.path.join(MODELS_DIR, 'xgboost.pkl'), X_test, y_test, "XGBoost")
