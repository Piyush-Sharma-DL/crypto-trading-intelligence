@echo off
title Crypto Trading Intelligence
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"

if not defined PY (
  echo Could not find the virtual environment ^(.venv^).
  echo It should be in this folder or the folder above it.
  pause
  exit /b 1
)

echo Starting the dashboard... your browser will open in a few seconds.
echo Keep this window open while you use it. Close it to stop the dashboard.
echo.
"%PY%" -m streamlit run app.py
pause
