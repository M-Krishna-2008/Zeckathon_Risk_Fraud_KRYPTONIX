#!/usr/bin/env bash
# ===================================================================
# Distress Watch - Early Warning System for Indian Distress
# Automated Setup and Launch Script for macOS and Linux
# ===================================================================

set -e

# 1. Change directory to script location
cd "$(cd "$(dirname "$0")" && pwd)"

REBUILD_FLAG=0
TEST_FLAG=0
for arg in "$@"; do
    if [ "$arg" = "--rebuild" ]; then
        REBUILD_FLAG=1
    fi
    if [ "$arg" = "--test" ]; then
        TEST_FLAG=1
    fi
done

echo "==================================================================="
echo "Distress Watch - Automated Setup and Launch (macOS/Linux)"
echo "==================================================================="

# Ensure required directories exist
mkdir -p models data/raw data/processed reports reports/screenshots

# 2. Detect Python 3.11 - 3.13 (or compatible 3.10)
PYTHON_CMD=""

for cmd in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import sys; v = sys.version_info; sys.exit(0 if (v.major == 3 and 10 <= v.minor <= 13) else 1)" >/dev/null 2>&1; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "========================================================================="
    echo "[ERROR] Python 3.11, 3.12, or 3.13 is required but was not found."
    echo ""
    echo "Please install Python 3.11 manually from:"
    echo "  https://www.python.org/downloads/"
    echo "Or using your package manager (e.g., 'brew install python@3.11' on macOS,"
    echo "or 'sudo apt install python3.11 python3.11-venv' on Debian/Ubuntu)."
    echo "========================================================================="
    echo ""
    exit 1
fi

echo "[Setup] Detected compatible Python runtime: $($PYTHON_CMD --version)"

# 3. Create .venv if missing, then activate it
if [ ! -d ".venv" ]; then
    echo "[Setup] Creating virtual environment (.venv) ..."
    "$PYTHON_CMD" -m venv .venv || {
        echo "[ERROR] Failed to create virtual environment in .venv" >&2
        exit 1
    }
fi

echo "[Setup] Activating virtual environment (.venv) ..."
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
else
    echo "[ERROR] Virtual environment activation script not found in .venv" >&2
    exit 1
fi

# 4. Check requirements hash and install if changed
REQ_CHANGED=0
python -c "import hashlib, pathlib, sys; f=pathlib.Path('.venv/.req_hash'); rf=pathlib.Path('requirements.txt'); sys.exit(0 if f.exists() and rf.exists() and f.read_text(encoding='utf-8').strip() == hashlib.sha256(rf.read_bytes()).hexdigest() else 1)" >/dev/null 2>&1 || REQ_CHANGED=1

if [ "$REQ_CHANGED" -eq 1 ]; then
    echo "[Setup] Installing / updating dependencies from requirements.txt ..."
    python -m pip install -r requirements.txt || {
        echo "[Notice] Standard requirements install encountered build issues. Attempting best-effort install..."
        python -c "import subprocess, sys, pathlib; lines=[line.strip() for line in pathlib.Path('requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]; [subprocess.run([sys.executable, '-m', 'pip', 'install', r]) for r in lines if not any(k in r.lower() for k in ['lightgbm', 'shap'])]; [subprocess.run([sys.executable, '-m', 'pip', 'install', r]) for r in lines if any(k in r.lower() for k in ['lightgbm', 'shap'])]"
    }
    python -c "import hashlib, pathlib; pathlib.Path('.venv/.req_hash').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(), encoding='utf-8')" || {
        echo "[ERROR] Failed to write requirements hash to .venv/.req_hash" >&2
        exit 1
    }
    echo "[Setup] Dependencies installed successfully."
else
    echo "[Setup] Dependencies are up to date (.venv/.req_hash matches)."
fi

# 5. Model training & pipeline artifacts check
NEED_TRAIN=0
if [ ! -f "models/benchmark_lightgbm.joblib" ]; then
    NEED_TRAIN=1
fi
if [ "$REBUILD_FLAG" -eq 1 ]; then
    NEED_TRAIN=1
fi

if [ "$NEED_TRAIN" -eq 1 ]; then
    if [ "$REBUILD_FLAG" -eq 1 ]; then
        echo "[Setup] --rebuild flag specified. Retraining all models from scratch..."
        python -m src.pipeline --rebuild || {
            echo "[ERROR] Model rebuild pipeline failed" >&2
            exit 1
        }
    else
        echo "[Setup] Model artifact missing in models/. Running setup pipeline..."
        python -m src.pipeline || {
            echo "[ERROR] Setup pipeline failed" >&2
            exit 1
        }
    fi
else
    echo "[Setup] Verified trained model artifact in models/."
fi

# 6. Copy .env.example to .env if missing
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[Setup] .env configuration not found. Copying .env.example to .env ..."
        cp .env.example .env || {
            echo "[ERROR] Failed to copy .env.example to .env" >&2
            exit 1
        }
        echo "[Setup] Created .env from .env.example."
    fi
else
    echo "[Setup] Found existing .env configuration."
fi

# 7. Handle --test flag
if [ "$TEST_FLAG" -eq 1 ]; then
    echo ""
    echo "==================================================================="
    echo "Running Distress Watch Test Suite (--test)"
    echo "==================================================================="
    python -m unittest discover tests -v
    exit 0
fi

# 8. Start Uvicorn on 127.0.0.1:8000, wait for HTTP response, and open default browser
echo ""
echo "==================================================================="
echo "Launching Distress Watch application..."
echo "The default browser will automatically open once the server is live."
echo "Press Ctrl+C in this terminal window to stop the server."
echo "==================================================================="
echo ""

python -m src.serve || {
    echo "[ERROR] Server exited with an error" >&2
    exit 1
}
