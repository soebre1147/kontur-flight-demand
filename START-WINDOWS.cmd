@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto dependencies
where py >nul 2>&1
if errorlevel 1 goto pythonfallback
py -3 -m venv .venv
goto checkvenv
:pythonfallback
python -m venv .venv
:checkvenv
if not exist .venv\Scripts\python.exe (
  echo Python 3.12 or newer is required. Install Python and run this file again.
  pause
  exit /b 1
)
:dependencies
.venv\Scripts\python.exe -c "import streamlit, scipy, pandas, numpy" >nul 2>&1
if not errorlevel 1 goto launch
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
:launch
echo Open http://127.0.0.1:8501 in your browser after the server starts.
echo Keep this window open. Press Ctrl+C to stop the application.
.venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 pause
