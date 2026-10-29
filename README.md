# Early Warning System for Indian Corporate Distress (EWS-India)

An ML-powered early-warning system designed to rank Indian listed corporate borrowers by default and earnings manipulation risk at least **4 quarters ahead** of distress events. Designed specifically for bank credit officers and risk committees under realistic operational capacity constraints.

---

## 1. Project Goal & Operational Constraints

- **Predictive Horizon**: Minimum **4 quarters** lead time prior to default or insolvency event.
- **Capacity Cap**: Flags strictly the **top 5%** of the monitored universe (e.g., top 40 out of 800 corporate borrowers).
- **Explainability**: Every flagged borrower provides plain-language, credit-officer-readable risk rationales (no raw uninterpreted SHAP values).
- **Decision-Theoretic Cost Model**: Missed default ($C_{FN}$) is penalized at **20x** a false alarm ($C_{FP}$). Cost-optimal probability threshold on well-calibrated probabilities:
  $$\tau^* = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{1}{20 + 1} = \frac{1}{21} \approx 0.0476$$
  The top-5% (40/800) ranking acts as the hard operational capacity budget.

---

## 2. Standing Hard Rules

1. **Strictly Zero Data Leakage**:
   - Features use only information publicly available *before* the event/observation cutoff.
   - Point-in-time filing/publication dates must be used, **never** fiscal year-end dates.
   - Original reported numbers only, **never** restated figures.
2. **Zero Financial Data Fabrication**:
   - If a financial figure is unavailable, leave it blank and mark `source="MISSING"`.
   - Every Indian data record must preserve auditability with explicit `source_url` and `filing_date`.
3. **Environment & Zero-Dep Deployment**:
   - Runs cleanly on a standard Windows PC with internet access.
   - No Docker, no Node.js, no paid API keys, no GPU required.
   - Pure Python 3.11 backend with FastAPI serving vanilla HTML/CSS/JS frontend.
4. **Reproducibility**:
   - Pinned dependency versions in `requirements.txt`.
   - Clean, modular, well-commented code.

---

## 2.1 Peer Selection & Survivorship-Bias Risk

- **Sector-Matched Peer Design**: In `data/india/peers.csv`, each defaulter is paired with 3+ sector-matched corporate peers that were demonstrably solvent and active during the observation period (e.g., IndiGo for Jet Airways, DMart for Future Retail, HDFC Ltd for DHFL).
- **Survivorship-Bias Risk Warning**: Selecting historically verified "healthy" survivors ex-post introduces survivorship bias if models are trained naively without accounting for contemporaneous peer conditions. In live production monitoring, the candidate universe must be established strictly *as of* the observation cutoff date (all active listed borrowers at that quarter), without conditioning on future non-default survival.

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.11
- **Data & Computation**: `pandas`, `numpy`, `scipy`, `openpyxl`
- **Machine Learning & Calibration**: `scikit-learn`, `lightgbm`, `joblib`
- **Model Explainability**: `shap`
- **Web Backend**: `FastAPI`, `uvicorn`, `pydantic`
- **Network & Scraping**: `requests`
- **Frontend**: Plain HTML5, CSS3, JavaScript (served via FastAPI static mount)

---

## 4. Project Structure

```text
ews_india/
│── run.bat                 # One-click startup script for Windows
│── run.sh                  # One-click startup script for macOS / Linux
│── README.md               # Project documentation and guidelines
│── requirements.txt        # Pinned dependencies
│── .env.example            # Environment configuration template
│
├── data/
│   ├── raw/                # UCI Polish bankruptcy ARFF archives
│   ├── processed/          # Cleaned, point-in-time aligned datasets
│   └── india/              # india_panel.csv, financials.csv, events.csv, peers.csv
│
├── src/
│   ├── benchmark/          # Baseline models (Altman Z-Score, LightGBM, SHAP explainer)
│   ├── scores/             # Accounting distress scores (Altman Z/Z'', Beneish M, Piotroski F)
│   ├── india/              # India loaders and panel builder (india_panel.csv)
│   ├── explain/            # Credit officer plain-language explanation generators
│   ├── policy/             # Cost-sensitive evaluation, calibration, and 5% cap ranking
│   ├── pipeline.py         # End-to-end setup and training pipeline
│   └── serve.py            # Uvicorn launcher with auto-health-check and browser opener
│
├── app/
│   ├── server.py           # FastAPI application endpoints
│   └── static/             # Frontend UI (index.html, style.css, app.js)
│
├── models/                 # Serialized model artifacts (.joblib)
├── reports/                # Performance benchmarks, confusion matrices, audit logs
└── tests/                  # Unit and integration test suites
```

---

## 5. Quick start

A fresh machine can be fully configured and launched with a single click.

### Windows (One-Click)

Simply **double-click** `run.bat` in the project root, or execute it from PowerShell / Command Prompt:

```cmd
run.bat
```

What `run.bat` does automatically:
1. **Directory-Safe Execution**: Runs `cd /d "%~dp0"` so it operates reliably from any directory or drive.
2. **Runtime Detection**: Scans for Python 3.10–3.12 via `py -3` or `python`. If missing, attempts silent user-scoped installation via `winget install -e --id Python.Python.3.11 --silent` (no administrator privileges required). If unavailable, displays a clear download link to [python.org](https://www.python.org/downloads/) and pauses.
3. **Virtual Environment**: Automatically provisions a local `.venv` virtual environment if absent and activates it.
4. **Smart Dependency Caching**: Hashes `requirements.txt` into `.venv\.req_hash`. Runs `pip install -r requirements.txt` only when dependencies have actually changed, speeding subsequent launches to under 1 second.
5. **Auto-Model Training**: If `models/benchmark_lightgbm.joblib` is absent:
   - Downloads the UCI Polish Bankruptcy dataset into `data/raw/` (with offline synthetic fallback).
   - Trains the calibrated LightGBM classifier with SHAP explainer.
   - Builds the multi-year Indian corporate feature panel (`data/india/india_panel.csv`).
   - Runs 800-borrower portfolio scoring and generates recall reports (`reports/india_recall.json`).
6. **Environment Configuration**: Copies `.env.example` to `.env` if `.env` does not exist.
7. **Server & Browser Launch**: Starts Uvicorn on `127.0.0.1:8000`, verifies an HTTP 200 health check response, and opens the dashboard in your default browser.
8. **Error Handling**: Verifies `errorlevel` after every critical stage and provides human-readable diagnostic messages with `pause`.

### macOS / Linux

Make the script executable and run:

```bash
chmod +x run.sh
./run.sh
```

Follows the identical sequence (Python detection, `.venv` isolation, hash-based `pip` caching, automated model training, environment generation, and browser launch).

### Retraining From Scratch (`--rebuild`)

To force retraining all models, re-downloading source datasets, and regenerating all evaluation reports from scratch, pass the `--rebuild` flag:

```cmd
# Windows
run.bat --rebuild

# macOS / Linux
./run.sh --rebuild
```

---

## 6. Manual Setup & Execution (Alternative)

If you prefer running commands manually:

```cmd
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # On Windows
# source .venv/bin/activate     # On macOS/Linux

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Copy environment configuration
cp .env.example .env

# 4. Run end-to-end training and evaluation pipeline
python -m src.pipeline

# 5. Start development server
python -m src.serve
# Or directly via uvicorn:
# uvicorn app.server:app --reload --host 127.0.0.1 --port 8000
```
