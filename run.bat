@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem Tellur launcher.
rem
rem On first run, creates a Python 3.11 venv at %TELLUR_HOME%\.venv and
rem installs dependencies. On subsequent runs, just launches the app via
rem pythonw (no console window).
rem
rem TELLUR_HOME resolution order:
rem   1. The process env var (if set in this shell or inherited).
rem   2. The user-level Windows env var (where `setx TELLUR_HOME ...` writes).
rem   3. Default: %LOCALAPPDATA%\Tellur
rem
rem TELLUR_VENV defaults to %TELLUR_HOME%\.venv.
rem ----------------------------------------------------------------------

if not defined TELLUR_HOME (
    for /f "skip=2 tokens=2,*" %%a in ('reg query "HKCU\Environment" /v TELLUR_HOME 2^>nul') do set "TELLUR_HOME=%%b"
)
if not defined TELLUR_HOME set "TELLUR_HOME=%LOCALAPPDATA%\Tellur"
if not defined TELLUR_VENV set "TELLUR_VENV=%TELLUR_HOME%\.venv"

set "HF_HOME=%TELLUR_HOME%\hf-cache"
set "HF_HUB_CACHE=%TELLUR_HOME%\hf-cache"

if not exist "%TELLUR_VENV%\Scripts\pythonw.exe" (
    echo.
    echo ====================================================================
    echo  First-time setup — Tellur will install to:
    echo      %TELLUR_HOME%
    echo  Venv:        %TELLUR_VENV%        ^(~2.5 GB^)
    echo  Model cache: %TELLUR_HOME%\hf-cache ^(~1.5 GB first run^)
    echo.
    echo  Wrong location? Press Ctrl+C now, then run:
    echo      setx TELLUR_HOME ^<your-path^>
    echo  and reopen this script in a new terminal.
    echo ====================================================================
    echo.
    pause

    where py >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python launcher ^("py"^) not found on PATH.
        echo Install Python 3.11 from https://www.python.org/downloads/
        echo ^(check "Add to PATH" during install^)
        echo.
        pause
        exit /b 1
    )
    rem Probe specifically for Python 3.11 so we can give a clearer message
    rem than the cryptic venv-failure path.
    py -3.11 --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.11 specifically is required, but `py -3.11` failed.
        echo You may have a different version installed ^(3.10, 3.12, etc.^).
        echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3119/
        echo Both versions can coexist; the `py` launcher will pick the right one.
        echo.
        pause
        exit /b 1
    )
    py -3.11 -m venv "%TELLUR_VENV%"
    if errorlevel 1 (
        echo Failed to create venv with Python 3.11.
        pause
        exit /b 1
    )
    echo Installing dependencies into the venv ^(this may take a few minutes^)...
    "%TELLUR_VENV%\Scripts\python.exe" -m pip install --upgrade pip
    "%TELLUR_VENV%\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency install failed. See errors above.
        pause
        exit /b 1
    )
    echo.
    echo Setup complete. Launching Tellur...
)

start "" "%TELLUR_VENV%\Scripts\pythonw.exe" tellur.py %*
