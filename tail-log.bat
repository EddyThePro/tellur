@echo off
setlocal enabledelayedexpansion

rem Resolve TELLUR_HOME the same way run.bat does: process env, then the
rem user-level Windows env var (where `setx` writes), then per-user default.
if not defined TELLUR_HOME (
    for /f "skip=2 tokens=2,*" %%a in ('reg query "HKCU\Environment" /v TELLUR_HOME 2^>nul') do set "TELLUR_HOME=%%b"
)
if not defined TELLUR_HOME set "TELLUR_HOME=%LOCALAPPDATA%\Tellur"
if not defined TELLUR_LOG_DIR set "TELLUR_LOG_DIR=%TELLUR_HOME%\logs"

set "LOG=%TELLUR_LOG_DIR%\tellur.log"

if not exist "%LOG%" (
    echo Log file not found: %LOG%
    echo The app may not have started yet.
    pause
    exit /b 1
)

powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Wait -Tail 50"
