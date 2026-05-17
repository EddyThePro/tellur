@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Resolve TELLUR_HOME the same way run.bat does: process env, then the
rem user-level Windows env var (where `setx` writes), then per-user default.
if not defined TELLUR_HOME (
    for /f "skip=2 tokens=2,*" %%a in ('reg query "HKCU\Environment" /v TELLUR_HOME 2^>nul') do set "TELLUR_HOME=%%b"
)
if not defined TELLUR_HOME set "TELLUR_HOME=%LOCALAPPDATA%\Tellur"
if not defined TELLUR_VENV set "TELLUR_VENV=%TELLUR_HOME%\.venv"

if not exist "%TELLUR_VENV%\Scripts\python.exe" (
    echo Tellur venv not found. Run run.bat first to set it up.
    pause
    exit /b 1
)

"%TELLUR_VENV%\Scripts\python.exe" teach.py %*
