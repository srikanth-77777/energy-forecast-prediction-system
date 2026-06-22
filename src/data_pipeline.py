"""
Data Pipeline — Processes all configured regions with 26-feature engineering.
Usage: python data_pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
import holidays

# Add src to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PROCESSED_DIR, REGIONS, TARGET_COL

# US holidays instance (PJM regions are all US-based)
US_HOLIDAYS = holidays.UnitedStates()


def load_raw(region_key: str) -> pd.DataFrame:
    """Load a raw regional CSV and return a clean DataFrame with unified column names."""
    csv_name, mw_col = REGIONS[region_key]
    path = os.path.join(DATA_DIR, csv_name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Rename MW column to unified name
    df = df.rename(columns={mw_col: TARGET_COL})
    
    # Remove duplicates
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    dupes = before - len(df)
    if dupes > 0:
        print(f"  Removed {dupes} duplicate timestamps")
    
    # Forward-fill missing values (max 3 consecutive hours)
    missing = df[TARGET_COL].isna().sum()
    if missing > 0:
        df[TARGET_COL] = df[TARGET_COL].ffill(limit=3)
        remaining = df[TARGET_COL].isna().sum()
        df = df.dropna(subset=[TARGET_COL])
        print(f"  Filled {missing - remaining} missing values, dropped {remaining} unfillable")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 26 engineered features to the dataframe."""
    demand = df[TARGET_COL]
    idx = df.index
    
    # ── Category 1: Basic Temporal (6) ──
    df['hour'] = idx.hour
    df['day_of_week'] = idx.dayofweek
    df['month'] = idx.month
    df['quarter'] = idx.quarter
    df['day_of_year'] = idx.dayofyear
    df['week_of_year'] = idx.isocalendar().week.astype(int)
    
    # ── Category 2: Cyclical Encoding (6) ──
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ── Category 3: Binary Flags (5 — includes holiday) ──
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_holiday'] = idx.to_series().apply(lambda dt: 1 if dt.date() in US_HOLIDAYS else 0).values
    
    # ── Category 4: Lag Features (2) ──
    df['lag_24h'] = demand.shift(24)
    df['lag_168h'] = demand.shift(168)
    
    # ── Category 5: Rolling Statistics (3) — shifted to avoid data leakage ──
    df['rolling_mean_24h'] = demand.shift(24).rolling(window=24, min_periods=24).mean()
    df['rolling_std_24h'] = demand.shift(24).rolling(window=24, min_periods=24).std()
    df['rolling_mean_168h'] = demand.shift(168).rolling(window=168, min_periods=168).mean()
    
    # ── Category 6: Interaction (1) ──
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    
    # Drop rows with NaN from lag/rolling (first 168 hours)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    print(f"  Dropped {dropped} rows from lag/rolling NaNs")
    
    return df


def process_region(region_key: str) -> dict:
    """Full pipeline for one region: load → clean → engineer → save."""
    print(f"\n{'='*60}")
    print(f"  Processing: {region_key}")
    print(f"{'='*60}")
    
    # Load
    df = load_raw(region_key)
    print(f"  Raw data: {len(df)} rows | {df.index.min().date()} to {df.index.max().date()}")
    
    # Engineer
    df = engineer_features(df)
    print(f"  Engineered: {len(df)} rows | {len(df.columns)} columns")
    
    # Save
    out_path = os.path.join(PROCESSED_DIR, f"{region_key}_processed.csv")
    df.to_csv(out_path)
    print(f"  Saved to: {out_path}")
    
    # Stats
    stats = {
        'region': region_key,
        'rows': len(df),
        'features': len(df.columns) - 1,  # minus target
        'date_min': str(df.index.min().date()),
        'date_max': str(df.index.max().date()),
        'mean_demand': round(df[TARGET_COL].mean(), 2),
        'peak_demand': round(df[TARGET_COL].max(), 2),
        'min_demand': round(df[TARGET_COL].min(), 2),
    }
    return stats


def main():
    print("=" * 60)
    print("  ENERGY DEMAND FORECASTING - DATA PIPELINE")
    print("  Processing 5 regions with 26 engineered features")
    print("=" * 60)
    
    all_stats = []
    for region_key in REGIONS:
        stats = process_region(region_key)
        all_stats.append(stats)
    
    # Summary table
    print("\n\n" + "="*70)
    print("  DATA PIPELINE SUMMARY")
    print("="*70)
    print(f"  {'Region':<10} {'Rows':>10} {'Features':>10} {'Avg MW':>10} {'Peak MW':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s in all_stats:
        print(f"  {s['region']:<10} {s['rows']:>10,} {s['features']:>10} {s['mean_demand']:>10,.0f} {s['peak_demand']:>10,.0f}")
    
    print(f"\n  [OK] All {len(all_stats)} regions processed successfully!")
    print(f"  Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
