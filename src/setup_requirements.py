"""
Robust requirements installer with best-effort fallback for optional ML accelerators.

Installs core dependencies strictly, then attempts optional packages (LightGBM/SHAP)
best-effort so that environments without C++ build wheels (such as Python 3.13)
continue running seamlessly with fallback estimators.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements(req_path: Path) -> None:
    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file not found at {req_path}")

    is_py313_or_higher = sys.version_info >= (3, 13)

    if not is_py313_or_higher:
        # Try standard pinned installation first
        print("[Setup] Running pip install from requirements.txt...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", "--default-timeout=100", "-r", str(req_path)])
        if res.returncode == 0:
            return

    # Best-effort fallback (e.g. Python 3.13 or missing wheels)
    print("\n[Notice] Standard install encountered package build issues. Attempting best-effort install...")
    lines = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    core_packages = [r for r in lines if not any(k in r.lower() for k in ["lightgbm", "shap"])]
    optional_packages = [r for r in lines if any(k in r.lower() for k in ["lightgbm", "shap"])]

    # On Python 3.13+, unpin older C-extension libraries that require >=2.0 wheels on 3.13
    if is_py313_or_higher:
        core_packages = [pkg.split("==")[0].split(">=")[0].strip() for pkg in core_packages]

    print("[Setup] Installing core runtime packages...")
    for pkg in core_packages:
        print(f"  -> Installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--default-timeout=100", pkg], check=True)

    print("[Setup] Attempting best-effort install of optional ML packages (LightGBM / SHAP)...")
    for pkg in optional_packages:
        name = pkg.split("==")[0].split(">=")[0].strip()
        pkg_to_try = name if is_py313_or_higher else pkg
        print(f"  -> Attempting {pkg_to_try} ...")
        res_opt = subprocess.run([sys.executable, "-m", "pip", "install", "--default-timeout=100", pkg_to_try])
        if res_opt.returncode != 0:
            print(f"  -> [Notice] {pkg_to_try} could not be installed; using built-in scikit-learn fallback.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    install_requirements(base_dir / "requirements.txt")
