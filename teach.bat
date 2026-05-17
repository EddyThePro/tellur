@echo off
setlocal
cd /d "%~dp0"

if not defined TELLUR_HOME set "TELLUR_HOME=%LOCALAPPDATA%\Tellur"
if not defined TELLUR_VENV set "TELLUR_VENV=%TELLUR_HOME%\.venv"

if not exist "%TELLUR_VENV%\Scripts\python.exe" (
    echo Tellur venv not found. Run run.bat first to set it up.
    pause
    exit /b 1
)

"%TELLUR_VENV%\Scripts\python.exe" teach.py %*
