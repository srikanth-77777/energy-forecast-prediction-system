@echo off
TITLE EnvisionAI - Energy Demand Forecaster
echo ==================================================
echo   EnvisionAI - Energy Demand Forecasting Engine
echo   5 Regions ^| 5 Models ^| 26 Features
echo ==================================================
echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
cd src
python api.py
pause
