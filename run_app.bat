@echo off
echo ========================================
echo Sponsored Jobs Analysis Engine
echo ========================================
echo.

REM Check if parquet file exists
if not exist "H1B-Sponsored-Jobs.parquet" (
    echo ERROR: H1B-Sponsored-Jobs.parquet not found!
    echo Please make sure the parquet file is in this directory.
    pause
    exit /b 1
)

echo Data file found: H1B-Sponsored-Jobs.parquet
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo Python is available
echo.

REM Try to import required packages
python -c "import streamlit, pandas, plotly, numpy, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements!
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo All requirements are satisfied
echo.
echo Starting Streamlit application...
echo The app will open in your default web browser
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run app.py

echo.
echo Application stopped.
pause
