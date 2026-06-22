"""
Central configuration for the Energy Demand Forecasting project.
Single source of truth for regions, features, paths, and model parameters.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

for d in [PROCESSED_DIR, MODELS_DIR, METRICS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Regions ─────────────────────────────────────────────────────────
# Maps region key → (csv filename, MW column name)
REGIONS = {
    'PJME':   ('PJME_hourly.csv',   'PJME_MW'),
    'AEP':    ('AEP_hourly.csv',    'AEP_MW'),
    'DAYTON': ('DAYTON_hourly.csv',  'DAYTON_MW'),
    'COMED':  ('COMED_hourly.csv',   'COMED_MW'),
    'DOM':    ('DOM_hourly.csv',     'DOM_MW'),
}

# ─── Feature Configuration ──────────────────────────────────────────
# These are the column names the pipeline will create
FEATURE_COLS = [
    # Basic Temporal
    'hour', 'day_of_week', 'month', 'quarter', 'day_of_year', 'week_of_year',
    # Cyclical Encoding
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Binary Flags
    'is_weekend', 'is_peak', 'is_summer', 'is_winter', 'is_holiday',
    # Lag Features
    'lag_24h', 'lag_168h',
    # Rolling Statistics
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_168h',
    # Interaction
    'hour_x_weekend',
]

TARGET_COL = 'demand_mw'  # Unified target column name after processing

# ─── Train / Test Split ─────────────────────────────────────────────
TRAIN_RATIO = 0.80  # Chronological split

# ─── Model Definitions ──────────────────────────────────────────────
MODEL_NAMES = [
    'Linear Regression',
    'Random Forest',
    'Gradient Boosting',
    'XGBoost',
    'LightGBM',
]

# ─── GridSearchCV Parameters ────────────────────────────────────────
GRID_PARAMS = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, -1],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
    },
}
