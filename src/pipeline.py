"""
End-to-End Setup and Training Pipeline.

Orchestrates:
1. Downloading UCI Polish Companies Bankruptcy dataset (into data/raw).
2. Training and serializing the LightGBM benchmark model (models/benchmark_lightgbm.joblib).
3. Building the Indian corporate panel features dataset (data/india/india_panel.csv).
4. Running the 800-borrower portfolio scoring evaluation and generating reports (reports/india_recall.json).

Can be executed standalone:
  python -m src.pipeline
  python -m src.pipeline --rebuild
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Silence noisy parallelization warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_FILE = MODELS_DIR / "benchmark_lightgbm.joblib"
UCI_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"


def step_download_uci(force: bool = False) -> None:
    """Step 1: Download and extract UCI Polish Bankruptcy dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    arff_files = list(RAW_DIR.glob("*year.arff"))

    if arff_files and not force:
        print(f"  -> UCI dataset already present in {RAW_DIR} ({len(arff_files)} files found). Skipping download.")
        return

    print(f"  -> Downloading UCI dataset from {UCI_URL} ...")
    try:
        req = urllib.request.Request(
            UCI_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DistressWatch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        print("  -> Download complete. Extracting ARFF archives...")
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(RAW_DIR)
        extracted = list(RAW_DIR.glob("*year.arff"))
        print(f"  -> Extracted {len(extracted)} ARFF files to {RAW_DIR}.")
    except Exception as e:
        print(f"  -> [WARNING] Network download failed: {e}")
        print("  -> Continuing with offline empirical fallback distributions.")


def step_train_benchmark(force: bool = False) -> None:
    """Step 2: Train and serialize benchmark LightGBM model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_FILE.exists() and not force:
        print(f"  -> Benchmark model already exists at {MODEL_FILE}. Skipping training.")
        return

    print("  -> Fitting calibrated LightGBM benchmark model on Polish Attr ratios...")
    from src.benchmark.model import BenchmarkModel

    model = BenchmarkModel(model_path=MODEL_FILE)
    model.train_and_save(raw_dir=RAW_DIR)
    print(f"  -> Model trained and saved to {MODEL_FILE}.")


def step_build_indian_panel(force: bool = False) -> None:
    """Step 3: Construct Indian corporate panel features."""
    panel_file = BASE_DIR / "data" / "india" / "india_panel.csv"
    if panel_file.exists() and not force:
        print(f"  -> Indian panel features already exist at {panel_file}. Skipping build.")
        return

    print("  -> Constructing multi-year Indian corporate panel features...")
    from src.india.panel_builder import build_indian_panel

    out = build_indian_panel(output_file=panel_file)
    print(f"  -> Indian corporate panel features saved to {out}.")


def step_run_scoring_and_reports() -> None:
    """Step 4: Score 800-borrower portfolio and generate evaluation reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("  -> Simulating 800-borrower portfolio under 5% capacity cap (4Q & 8Q lead times)...")
    from src.evaluation import run_full_evaluation_and_save

    rep = run_full_evaluation_and_save()
    r4q = rep["lead_time_4_quarters"]["recall_at_top_5pct_cap"] * 100
    r8q = rep["lead_time_8_quarters"]["recall_at_top_5pct_cap"] * 100
    print(f"  -> Reports saved to {REPORTS_DIR / 'india_recall.json'}")
    print(f"  -> Defaulter Recall @ 4 Quarters ahead: {r4q:.1f}%")
    print(f"  -> Defaulter Recall @ 8 Quarters ahead: {r8q:.1f}%")


def run_pipeline(rebuild: bool = False) -> None:
    """Execute the full setup/training pipeline."""
    print("===================================================================")
    if rebuild:
        print("Running Distress Watch Pipeline [REBUILD MODE - Retraining All]")
    else:
        print("Running Distress Watch Setup Pipeline")
    print("===================================================================")

    print("\n[1/4] Downloading UCI Polish Bankruptcy data...")
    step_download_uci(force=rebuild)

    print("\n[2/4] Training benchmark ML model...")
    step_train_benchmark(force=rebuild)

    print("\n[3/4] Building Indian corporate panel features...")
    step_build_indian_panel(force=rebuild)

    print("\n[4/4] Running portfolio scoring and generating evaluation reports...")
    step_run_scoring_and_reports()

    print("\n===================================================================")
    print("Pipeline completed successfully!")
    print("===================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distress Watch Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Force retrain and rebuild all artifacts from scratch")
    args = parser.parse_args()

    run_pipeline(rebuild=args.rebuild)

