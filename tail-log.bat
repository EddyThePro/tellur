@echo off
setlocal

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
