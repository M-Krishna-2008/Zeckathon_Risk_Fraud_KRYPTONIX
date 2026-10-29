@echo off
setlocal enabledelayedexpansion

REM ===================================================================
REM Distress Watch - Early Warning System for Indian Distress
REM Automated Self-Sufficient Setup and Launch Script for Windows
REM ===================================================================

REM 1. Change directory to script folder so it works from any location
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to project root directory: %~dp0
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

REM Parse command line flags
set REBUILD_FLAG=0
set TEST_FLAG=0
for %%A in (%*) do (
    if /i "%%~A"=="--rebuild" set REBUILD_FLAG=1
    if /i "%%~A"=="--test" set TEST_FLAG=1
)

echo ===================================================================
echo Distress Watch - Automated Setup and Launch
echo ===================================================================

REM Ensure required directory structure exists
for %%D in ("models" "data\raw" "data\processed" "reports" "reports\screenshots") do (
    if not exist "%%~D" mkdir "%%~D"
)

REM 2. Detect Python 3.10 - 3.13 (preferring versions with prebuilt wheels)
set PYTHON_CMD=

for %%C in ("py -3.11" "py -3.12" "py -3.10" "python" "py -3.13" "py -3") do (
    if not defined PYTHON_CMD (
        %%~C -c "import sys; v = sys.version_info; sys.exit(0 if v.major == 3 and 10 <= v.minor <= 13 else 1)" >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=%%~C
        )
    )
)

REM If still not found, try installing Python 3.11 via winget without admin rights
if not defined PYTHON_CMD (
    echo [Notice] Python 3.11-3.13 not detected. Attempting to install Python 3.11 via winget...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    
    REM Re-detect after winget installation attempt
    for %%C in ("py -3.11" "py -3.12" "py -3.10" "python" "py -3.13" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe") do (
        if not defined PYTHON_CMD (
            %%~C -c "import sys; v = sys.version_info; sys.exit(0 if v.major == 3 and 10 <= v.minor <= 13 else 1)" >nul 2>&1
            if not errorlevel 1 (
                set PYTHON_CMD=%%~C
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo.
    echo =========================================================================
    echo [ERROR] Python 3.11, 3.12, or 3.13 is required but was not found.
    echo.
    echo Please download and install Python 3.11 manually from:
    echo   https://www.python.org/downloads/release/python-3118/
    echo.
    echo IMPORTANT: During installation, check "Add python.exe to PATH".
    echo =========================================================================
    echo.
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

echo [Setup] Detected compatible Python runtime: %PYTHON_CMD%

REM 3. Create .venv if missing, then activate it
if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating virtual environment at .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment in .venv.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )
)

echo [Setup] Activating virtual environment at .venv ...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment at .venv\Scripts\activate.bat.
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

REM 4. Check requirements hash and install if changed
set REQ_CHANGED=0
python -c "import hashlib, pathlib, sys; f=pathlib.Path('.venv/.req_hash'); rf=pathlib.Path('requirements.txt'); sys.exit(0 if f.exists() and rf.exists() and f.read_text(encoding='utf-8').strip() == hashlib.sha256(rf.read_bytes()).hexdigest() else 1)" >nul 2>&1
if errorlevel 1 (
    set REQ_CHANGED=1
)

if "%REQ_CHANGED%"=="1" (
    echo [Setup] Installing or updating dependencies from requirements.txt ...
    
    set PIP_DEFAULT_TIMEOUT=100
    REM Upgrade pip quietly
    python -m pip install --upgrade pip >nul 2>&1

    REM Execute robust setup installer with optional fallback
    python -m src.setup_requirements
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )

    REM Verify that essential runtime dependencies are installed
    python -c "import fastapi, uvicorn, sklearn, pandas, scipy; print('[Setup] Core dependencies verified.')"
    if errorlevel 1 (
        echo [ERROR] Essential runtime dependencies failed to install.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )

    REM Test for LightGBM and SHAP
    python -c "import lightgbm, shap" >nul 2>&1
    if errorlevel 1 (
        echo [Notice] LightGBM/SHAP unavailable; system will operate with GradientBoosting and heuristic explainer fallback.
    )

    python -c "import hashlib, pathlib; pathlib.Path('.venv/.req_hash').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(), encoding='utf-8')"
    if errorlevel 1 (
        echo [ERROR] Failed to write requirements hash to .venv\.req_hash.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )
    echo [Setup] Dependencies installed successfully.
) else (
    echo [Setup] Dependencies are up to date.
)

REM 5. Model training and pipeline artifacts check
set NEED_TRAIN=0
if not exist "models\benchmark_lightgbm.joblib" set NEED_TRAIN=1
if "%REBUILD_FLAG%"=="1" set NEED_TRAIN=1

if "%NEED_TRAIN%"=="1" (
    if "%REBUILD_FLAG%"=="1" (
        echo [Setup] Rebuild flag specified. Retraining all models from scratch...
        python -m src.pipeline --rebuild
    ) else (
        echo [Setup] Model artifact missing in models directory. Running setup pipeline...
        python -m src.pipeline
    )
    if errorlevel 1 (
        echo [ERROR] Setup pipeline failed during data download, model training, or evaluation.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )
) else (
    echo [Setup] Verified trained model artifact in models directory.
)

REM 6. Copy .env.example to .env if missing
if not exist ".env" (
    if exist ".env.example" (
        echo [Setup] Configuration file .env not found. Copying .env.example to .env ...
        copy /y ".env.example" ".env" >nul
        if errorlevel 1 (
            echo [ERROR] Failed to copy .env.example to .env.
            if not "%NO_PAUSE%"=="1" pause
            exit /b 1
        )
        echo [Setup] Created .env from .env.example.
    )
) else (
    echo [Setup] Found existing .env configuration.
)

REM 7. Handle --test flag if requested
if "%TEST_FLAG%"=="1" (
    echo.
    echo ===================================================================
    echo Running Distress Watch Test Suite - Test Mode
    echo ===================================================================
    python -m unittest discover tests -v
    if errorlevel 1 (
        echo [ERROR] One or more tests failed.
        if not "%NO_PAUSE%"=="1" pause
        exit /b 1
    )
    echo.
    echo ===================================================================
    echo All tests passed successfully!
    echo ===================================================================
    exit /b 0
)

REM 8. Start Uvicorn on 127.0.0.1:8000, wait for HTTP response, and open default browser
echo.
echo ===================================================================
echo Launching Distress Watch application...
echo The default browser will automatically open once the server is live.
echo Press Ctrl+C in this terminal window to stop the server.
echo ===================================================================
echo.

python -m src.serve
if errorlevel 1 (
    echo [ERROR] Server exited with an error.
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)
