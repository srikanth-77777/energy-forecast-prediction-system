"""
FastAPI Backend — Multi-region, multi-model energy demand forecasting API.
Serves 6 endpoints + static frontend.
Usage: python api.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DIR, MODELS_DIR, METRICS_DIR, FRONTEND_DIR,
    REGIONS, FEATURE_COLS, TARGET_COL, MODEL_NAMES
)

app = FastAPI(title="Energy Demand Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data & Model Loading ───────────────────────────────────────────
datasets = {}      # region -> DataFrame
models = {}        # (region, model_name) -> trained model
all_metrics = {}   # region -> [{ model, metrics, ... }]

print("[API] Loading datasets and models...")

for region_key in REGIONS:
    # Load processed data
    proc_path = os.path.join(PROCESSED_DIR, f"{region_key}_processed.csv")
    if os.path.exists(proc_path):
        df = pd.read_csv(proc_path, index_col='Datetime', parse_dates=True)
        datasets[region_key] = df
        print(f"  [OK] {region_key} data: {len(df):,} rows")
    else:
        print(f"  [WARN] {region_key} processed data not found")
    
    # Load metrics
    metrics_path = os.path.join(METRICS_DIR, f"{region_key}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics[region_key] = json.load(f)
    
    # Load models
    for model_name in MODEL_NAMES:
        model_filename = f"{region_key}_{model_name.lower().replace(' ', '_')}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            models[(region_key, model_name)] = joblib.load(model_path)

print(f"[API] Loaded {len(datasets)} regions, {len(models)} models")


# ─── Feature Helper ─────────────────────────────────────────────────
def create_features_for_dt(dt, lag_24h, lag_168h, is_holiday_override=False):
    """Create a feature row for a future datetime."""
    import holidays
    us_holidays = holidays.UnitedStates()
    
    hour = dt.hour
    dow = dt.dayofweek
    month = dt.month
    
    return pd.DataFrame([{
        'hour': hour,
        'day_of_week': dow,
        'month': month,
        'quarter': (month - 1) // 3 + 1,
        'day_of_year': dt.timetuple().tm_yday,
        'week_of_year': dt.isocalendar()[1],
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dow_sin': np.sin(2 * np.pi * dow / 7),
        'dow_cos': np.cos(2 * np.pi * dow / 7),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'is_weekend': int(dow >= 5),
        'is_peak': int(6 <= hour <= 22),
        'is_summer': int(month in [6, 7, 8]),
        'is_winter': int(month in [12, 1, 2]),
        'is_holiday': 1 if is_holiday_override else int(dt.date() in us_holidays),
        'lag_24h': lag_24h,
        'lag_168h': lag_168h,
        'rolling_mean_24h': lag_24h,   # approximated
        'rolling_std_24h': 0,
        'rolling_mean_168h': lag_168h,
        'hour_x_weekend': hour * int(dow >= 5),
    }])[FEATURE_COLS]


# ─── AI Insight Engine ──────────────────────────────────────────────
def generate_insight(hour, is_weekend, actual, predicted, region, simulate_anomaly=False):
    """Generate a context-aware demand narrative."""
    error = predicted - actual
    error_pct = abs(error / actual * 100) if actual else 0
    
    if simulate_anomaly or error_pct > 25:
        narrative = f"🚨 CRITICAL ANOMALY DETECTED: The {region} power grid has experienced a sudden massive deviation ({error_pct:.1f}% error). "
        narrative += f"The AI model expected {predicted:,.0f} MW, but actual load crashed to {actual:,.0f} MW. "
        narrative += "This extreme pattern is characteristic of a major localized blackout, severe infrastructure failure, or an unprecedented grid isolation event. Immediate operator investigation required!"
        return narrative
        
    time_ctx = "overnight low-demand" if hour < 6 else "morning ramp-up" if hour < 10 else "midday peak" if hour < 14 else "afternoon sustained" if hour < 18 else "evening peak" if hour < 22 else "nighttime wind-down"
    day_ctx = "weekend" if is_weekend else "weekday"
    
    if error_pct < 5:
        accuracy_msg = f"The model shows excellent confidence, tracking within {error_pct:.1f}% of the exact load. This suggests stable and predictable operational conditions without major anomalies."
    elif error_pct < 10:
        accuracy_msg = f"The prediction indicates a normal expected variance, deviating by {error_pct:.1f}% from actual readings. Small shifts in weather or human behavior likely account for this minor gap."
    else:
        accuracy_msg = f"A notable deviation of {error_pct:.1f}% has been detected. This could indicate a sudden weather event, localized grid stress, or unexpected industrial activity throwing off standard algorithmic estimates."
    
    direction = "above" if error > 0 else "below"
    
    narrative = f"The {region} power grid is currently entering a {time_ctx} phase on a {day_ctx}. "
    narrative += f"Our active AI pipeline has evaluated historical 24-hour patterns, seasonal shifts, and cyclical human routines to make this calculation. "
    narrative += f"Specifically, the model predicts {predicted:,.0f} MW of demand, which sits {abs(error):,.0f} MW {direction} the true measured load of {actual:,.0f} MW. "
    narrative += f"{accuracy_msg} "
    narrative += f"Grid operators should continue monitoring these metrics, especially as we transition into the next 24-hour predictive horizon shown in the forecasting timeline."
    return narrative


# ─── Request Models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    region: str
    date: str
    hour: int
    model: str
    is_holiday_override: Optional[bool] = False
    simulate_anomaly: Optional[bool] = False

class CompareRequest(BaseModel):
    region: str
    date: str
    hour: int


# ─── Endpoints ──────────────────────────────────────────────────────

@app.get("/api/regions")
async def get_regions():
    """List all available regions with date ranges and stats."""
    result = []
    for region_key, df in datasets.items():
        result.append({
            'key': region_key,
            'name': region_key,
            'rows': len(df),
            'date_min': str(df.index.min().date()),
            'date_max': str(df.index.max().date()),
            'mean_demand': round(float(df[TARGET_COL].mean()), 0),
            'peak_demand': round(float(df[TARGET_COL].max()), 0),
        })
    return result


@app.get("/api/models/{region}")
async def get_models(region: str):
    """List trained models + real metrics for a region."""
    if region not in datasets:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
    
    region_metrics = all_metrics.get(region, [])
    result = []
    for entry in region_metrics:
        result.append({
            'name': entry['model'],
            'metrics': entry['metrics'],
            'available': (region, entry['model']) in models,
        })
    return result


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Predict demand for a given region, date, hour, and model."""
    if req.region not in datasets:
        raise HTTPException(status_code=404, detail=f"Region '{req.region}' not found")
    if (req.region, req.model) not in models:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found for region '{req.region}'")
    
    df = datasets[req.region]
    model = models[(req.region, req.model)]
    
    target_dt = pd.to_datetime(f"{req.date} {req.hour:02d}:00:00")
    
    # Snap to nearest available timestamp
    if target_dt not in df.index:
        idx_loc = df.index.get_indexer([target_dt], method='nearest')[0]
        target_dt = df.index[idx_loc]
    
    idx = df.index.get_loc(target_dt)
    target_features = df.loc[[target_dt]][FEATURE_COLS].copy()
    if req.is_holiday_override:
        target_features['is_holiday'] = 1
        
    actual_val = float(df.loc[target_dt, TARGET_COL])
    if req.simulate_anomaly:
        actual_val = actual_val * 0.65  # Force a massive 35% drop
        
    predicted_val = float(model.predict(target_features)[0])
    
    # History (48h back for richer context)
    start_idx = max(0, idx - 47)
    history = [
        {"time": str(df.index[i]), "val": round(float(df.iloc[i][TARGET_COL]), 2)}
        for i in range(start_idx, idx + 1)
    ]
    
    # 24h forecast
    forecast = []
    
    for i in range(1, 25):
        next_dt = target_dt + timedelta(hours=i)
        
        # Use historical data since we only forecast 24h ahead
        lag_24_dt = next_dt - timedelta(hours=24)
        lag_24 = float(df.loc[lag_24_dt, TARGET_COL]) if lag_24_dt in df.index else actual_val
        
        lag_168_dt = next_dt - timedelta(hours=168)
        lag_168 = float(df.loc[lag_168_dt, TARGET_COL]) if lag_168_dt in df.index else actual_val
        
        f_input = create_features_for_dt(next_dt, lag_24, lag_168, req.is_holiday_override)
        f_pred = float(model.predict(f_input)[0])
        
        forecast.append({"time": str(next_dt), "val": round(f_pred, 2)})
    
    # Get model metrics
    region_metrics = all_metrics.get(req.region, [])
    model_entry = next((e for e in region_metrics if e['model'] == req.model), None)
    model_metrics = model_entry['metrics'] if model_entry else {}
    
    return {
        "target_time": str(target_dt),
        "region": req.region,
        "model": req.model,
        "predicted": round(predicted_val, 2),
        "actual": round(actual_val, 2),
        "history": history,
        "forecast_24h": forecast,
        "ai_insight": generate_insight(
            target_dt.hour, int(target_dt.dayofweek >= 5),
            actual_val, predicted_val, req.region, req.simulate_anomaly
        ),
        "model_metrics": model_metrics,
    }


@app.post("/api/compare")
async def compare_models(req: CompareRequest):
    """Run all available models on same input, return comparison."""
    if req.region not in datasets:
        raise HTTPException(status_code=404, detail=f"Region '{req.region}' not found")
    
    df = datasets[req.region]
    target_dt = pd.to_datetime(f"{req.date} {req.hour:02d}:00:00")
    
    if target_dt not in df.index:
        idx_loc = df.index.get_indexer([target_dt], method='nearest')[0]
        target_dt = df.index[idx_loc]
    
    target_features = df.loc[[target_dt]][FEATURE_COLS]
    actual_val = float(df.loc[target_dt, TARGET_COL])
    
    results = []
    for model_name in MODEL_NAMES:
        if (req.region, model_name) not in models:
            continue
        model = models[(req.region, model_name)]
        pred = float(model.predict(target_features)[0])
        
        region_metrics = all_metrics.get(req.region, [])
        model_entry = next((e for e in region_metrics if e['model'] == model_name), None)
        
        results.append({
            'model': model_name,
            'predicted': round(pred, 2),
            'error': round(pred - actual_val, 2),
            'error_pct': round(abs(pred - actual_val) / actual_val * 100, 2) if actual_val else 0,
            'metrics': model_entry['metrics'] if model_entry else {},
        })
    
    return {
        "target_time": str(target_dt),
        "region": req.region,
        "actual": round(actual_val, 2),
        "comparisons": results,
    }


@app.get("/api/history/{region}")
async def get_history(region: str, days: int = 30):
    """Return historical demand data for EDA charts."""
    if region not in datasets:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
    
    df = datasets[region]
    # Return last N days of data, sampled to avoid huge payloads
    end_dt = df.index.max()
    start_dt = end_dt - timedelta(days=days)
    subset = df.loc[start_dt:end_dt]
    
    # For large ranges, sample every few hours
    if len(subset) > 720:  # more than 30 days of hourly
        step = max(1, len(subset) // 720)
        subset = subset.iloc[::step]
    
    return {
        "region": region,
        "data": [
            {"time": str(idx), "val": round(float(row[TARGET_COL]), 2)}
            for idx, row in subset.iterrows()
        ],
        "stats": {
            "mean": round(float(df[TARGET_COL].mean()), 0),
            "std": round(float(df[TARGET_COL].std()), 0),
            "min": round(float(df[TARGET_COL].min()), 0),
            "max": round(float(df[TARGET_COL].max()), 0),
            "total_rows": len(df),
        }
    }


@app.get("/api/stats/{region}")
async def get_stats(region: str):
    """Return detailed statistics for analytics page."""
    if region not in datasets:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
    
    df = datasets[region]
    
    # Hourly averages
    hourly_avg = df.groupby('hour')[TARGET_COL].mean().round(0).to_dict()
    hourly_avg = {str(k): v for k, v in hourly_avg.items()}
    
    # Monthly averages
    monthly_avg = df.groupby('month')[TARGET_COL].mean().round(0).to_dict()
    monthly_avg = {str(k): v for k, v in monthly_avg.items()}
    
    # Day of week averages
    dow_avg = df.groupby('day_of_week')[TARGET_COL].mean().round(0).to_dict()
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    dow_avg = {dow_names.get(k, str(k)): v for k, v in dow_avg.items()}
    
    # Weekend vs weekday
    weekday_avg = float(df[df['is_weekend'] == 0][TARGET_COL].mean())
    weekend_avg = float(df[df['is_weekend'] == 1][TARGET_COL].mean())
    
    # Peak vs off-peak
    peak_avg = float(df[df['is_peak'] == 1][TARGET_COL].mean())
    offpeak_avg = float(df[df['is_peak'] == 0][TARGET_COL].mean())
    
    # Yearly averages
    yearly_avg = df.groupby(df.index.year)[TARGET_COL].mean().round(0).to_dict()
    yearly_avg = {str(k): v for k, v in yearly_avg.items()}
    
    return {
        "region": region,
        "hourly_avg": hourly_avg,
        "monthly_avg": monthly_avg,
        "dow_avg": dow_avg,
        "weekday_avg": round(weekday_avg, 0),
        "weekend_avg": round(weekend_avg, 0),
        "peak_avg": round(peak_avg, 0),
        "offpeak_avg": round(offpeak_avg, 0),
        "yearly_avg": yearly_avg,
    }


# ─── Serve Frontend ─────────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
