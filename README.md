# Energy Demand Forecasting

A FastAPI-based energy demand forecasting project with multi-region model training, feature engineering, and a static frontend.

## What is included

- `src/` — project source code
  - `api.py` — FastAPI backend serving model predictions and analytics
  - `config.py` — central configuration for regions, paths, and model settings
  - `data_pipeline.py` — process raw regional CSV files into engineered datasets
  - `train_all.py` — train 5 models for 5 regions and save metrics
  - `extract_metrics.py` — helper for model evaluation
  - `01_setup_and_fetch.py`, `02_feature_engineering.py`, `train_models.py` — supporting training/demo scripts
- `frontend/` — static UI served by the API
- `requirements.txt` — Python dependencies

## What is intentionally ignored

The repository is configured to ignore raw datasets, generated processed files, model artifacts, and outputs:
- `data/`
- `processed/`
- `models/`
- `metrics/`
- `outputs/`

This keeps the repo GitHub-friendly while preserving the full pipeline.

## Setup

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare data

Place your regional raw CSV files into `data/` using the names configured in `src/config.py`:

- `PJME_hourly.csv`
- `AEP_hourly.csv`
- `DAYTON_hourly.csv`
- `COMED_hourly.csv`
- `DOM_hourly.csv`

Each file should contain a `Datetime` column and its region-specific MW column.

## Run the pipeline

1. Process raw data into engineered features:

```powershell
python src/data_pipeline.py
```

2. Train models and save metrics:

```powershell
python src/train_all.py
```

3. Start the API server:

```powershell
python src/api.py
```

The backend will run on `http://localhost:8000` and serve the frontend automatically.

## API endpoints

- `GET /api/regions`
- `GET /api/models/{region}`
- `POST /api/predict`
- `POST /api/compare`
- `GET /api/history/{region}`
- `GET /api/stats/{region}`

## Notes

- The repo ignores large data and model artifacts. Use the `src/data_pipeline.py` and `src/train_all.py` scripts to generate them after cloning.
- `RUN_PROJECT.bat` launches `src/api.py` from the `src/` directory for Windows users.
