@echo off
title Paper Bot Worker (keeps your bots running, fake money)
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

echo Paper bots are running. Fake money only. Keep this window open, close it to stop.
echo Your computer must stay on, awake and online for the bots to keep working.
echo.
"%PY%" -m src.bot --loop --every 30
pause
