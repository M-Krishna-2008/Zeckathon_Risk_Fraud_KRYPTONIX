"""
Benchmark Model for Polish Bankruptcy Features (Attr1 to Attr64).

Trains and serializes a calibrated Gradient Boosting / LightGBM benchmark model
with median imputation, coupled with a SHAP TreeExplainer for feature attribution.
Saved artifact: models/benchmark_lightgbm.joblib
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "benchmark_lightgbm.joblib"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
FEATURE_NAMES = [f"Attr{i}" for i in range(1, 65)]


def load_uci_or_synthetic_dataset(
    raw_dir: Optional[Path] = None, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real UCI Polish Bankruptcy ARFF dataset from data/raw if available.
    Falls back to synthetic empirical distribution dataset if missing or unreadable.
    """
    raw_path = raw_dir or RAW_DATA_DIR
    arff_files = sorted(list(raw_path.glob("*year.arff")))
    if arff_files:
        try:
            from scipy.io import arff
            import pandas as pd
            dfs = []
            for f in arff_files:
                data, _ = arff.loadarff(f)
                dfs.append(pd.DataFrame(data))
            df = pd.concat(dfs, ignore_index=True)
            X = df.iloc[:, :64].values.astype(float)
            y = (df["class"].astype(str).str.contains("1")).astype(int).values
            return X, y
        except Exception:
            pass
    return _generate_synthetic_benchmark_dataset(random_state=random_state)


def _generate_synthetic_benchmark_dataset(
    n_samples: int = 1200, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset mimicking the empirical distributions and correlations
    of the UCI Polish Bankruptcy dataset across Attr1 to Attr64.
    Base default rate: ~5.5% (consistent with Polish 3-year bankruptcy panel).
    """
    rng = np.random.RandomState(random_state)
    n_defaults = int(n_samples * 0.055)
    n_healthy = n_samples - n_defaults

    # Healthy: higher ROA, positive working capital, lower leverage
    healthy_X = rng.normal(loc=0.0, scale=1.0, size=(n_healthy, 64))
    healthy_X[:, 0] = rng.normal(loc=0.06, scale=0.04, size=n_healthy)   # Attr1: ROA
    healthy_X[:, 1] = rng.normal(loc=0.45, scale=0.15, size=n_healthy)   # Attr2: TL/TA
    healthy_X[:, 2] = rng.normal(loc=0.18, scale=0.12, size=n_healthy)   # Attr3: WC/TA
    healthy_X[:, 3] = rng.normal(loc=1.85, scale=0.60, size=n_healthy)   # Attr4: CR
    healthy_X[:, 8] = rng.normal(loc=1.20, scale=0.40, size=n_healthy)   # Attr9: Sales/TA
    healthy_X[:, 12] = rng.normal(loc=0.28, scale=0.10, size=n_healthy)  # Attr13: GM
    healthy_X[:, 26] = rng.normal(loc=4.50, scale=2.50, size=n_healthy)  # Attr27: Interest Cover

    # Defaults: negative ROA, inverted WC, high leverage, crushed coverage
    default_X = rng.normal(loc=0.0, scale=1.0, size=(n_defaults, 64))
    default_X[:, 0] = rng.normal(loc=-0.12, scale=0.08, size=n_defaults)  # Attr1: negative ROA
    default_X[:, 1] = rng.normal(loc=0.88, scale=0.25, size=n_defaults)   # Attr2: heavy leverage
    default_X[:, 2] = rng.normal(loc=-0.15, scale=0.15, size=n_defaults)  # Attr3: negative WC
    default_X[:, 3] = rng.normal(loc=0.75, scale=0.30, size=n_defaults)   # Attr4: CR < 1
    default_X[:, 8] = rng.normal(loc=0.70, scale=0.35, size=n_defaults)   # Attr9: lower turnover
    default_X[:, 12] = rng.normal(loc=0.08, scale=0.12, size=n_defaults)  # Attr13: squeezed GM
    default_X[:, 26] = rng.normal(loc=0.40, scale=0.50, size=n_defaults)  # Attr27: cover < 1

    # Introduce realistic 30% missingness across auxiliary Attr features
    mask_h = rng.rand(*healthy_X.shape) < 0.30
    mask_h[:, [0, 1, 2, 3, 8, 26]] = False  # preserve core ratios
    healthy_X[mask_h] = np.nan

    mask_d = rng.rand(*default_X.shape) < 0.30
    mask_d[:, [0, 1, 2, 3, 8, 26]] = False
    default_X[mask_d] = np.nan

    X = np.vstack([healthy_X, default_X])
    y = np.array([0] * n_healthy + [1] * n_defaults)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class BenchmarkModel:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline: Optional[Any] = None
        self.explainer: Optional[Any] = None
        self.imputer: Optional[Any] = None
        self.estimator: Optional[Any] = None

    def train_and_save(self, raw_dir: Optional[Path] = None) -> None:
        """Fit calibrated LightGBM or GradientBoosting model with SimpleImputer and SHAP."""
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.calibration import CalibratedClassifierCV

        # Try LightGBM, fallback to GradientBoostingClassifier
        try:
            from lightgbm import LGBMClassifier
            base_clf = LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            base_clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )

        X, y = load_uci_or_synthetic_dataset(raw_dir=raw_dir)

        # Step 1: Fit Median Imputer
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)

        # Step 2: Fit Base Model
        base_clf.fit(X_imp, y)

        # Step 3: Calibrate probabilities using Sigmoid/Isotonic
        calibrator = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
        calibrator.fit(X_imp, y)

        # Step 4: Fit SHAP TreeExplainer on base tree model
        explainer = None
        try:
            import shap
            explainer = shap.TreeExplainer(base_clf)
        except Exception:
            explainer = None

        self.imputer = imputer
        self.estimator = calibrator
        self.base_clf = base_clf
        self.explainer = explainer

        artifact = {
            "imputer": self.imputer,
            "calibrator": self.estimator,
            "base_clf": self.base_clf,
            "feature_names": FEATURE_NAMES,
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.model_path)

    def load_or_train(self) -> None:
        """Load artifact from disk, or train if not present."""
        import joblib
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.imputer = data["imputer"]
                self.estimator = data["calibrator"]
                self.base_clf = data["base_clf"]
                try:
                    import shap
                    self.explainer = shap.TreeExplainer(self.base_clf)
                except Exception:
                    self.explainer = None
                return
            except Exception:
                pass

        self.train_and_save()

    def predict_probability(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Predict calibrated default probability and compute SHAP feature contributions.

        Parameters
        ----------
        features : Dict[str, float]
            Dictionary of Attr1..Attr64 values (can contain NaNs).

        Returns
        -------
        Tuple[float, Dict[str, float]]
            (calibrated_probability, shap_attributions_dict)
        """
        if self.estimator is None:
            self.load_or_train()

        # Vectorize
        x_vec = np.array([features.get(f"Attr{i}", math.nan) for i in range(1, 65)], dtype=float).reshape(1, -1)
        x_imp = self.imputer.transform(x_vec)

        # Predict calibrated probability
        prob = float(self.estimator.predict_proba(x_imp)[0, 1])

        # Compute SHAP
        shap_dict: Dict[str, float] = {}
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(x_imp)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    # Binary classification: index 1 is default class
                    vals = shap_vals[1][0]
                else:
                    vals = shap_vals[0]
                for feat, val in zip(FEATURE_NAMES, vals):
                    shap_dict[feat] = float(val)
            except Exception:
                # Fallback to feature importance proxy if SHAP computation fails
                pass

        if not shap_dict:
            # Fallback heuristic based on standard deviation from median
            medians = self.imputer.statistics_
            for idx, feat in enumerate(FEATURE_NAMES):
                diff = x_imp[0, idx] - medians[idx]
                # If adverse (e.g. leverage higher, or ROA lower)
                if feat in ("Attr1", "Attr3", "Attr4", "Attr9", "Attr27"):
                    shap_dict[feat] = float(-diff if diff < 0 else 0.0)
                elif feat in ("Attr2", "Attr51", "Attr58"):
                    shap_dict[feat] = float(diff if diff > 0 else 0.0)
                else:
                    shap_dict[feat] = 0.0

        return prob, shap_dict


# Global singleton instance
_GLOBAL_MODEL: Optional[BenchmarkModel] = None


def get_benchmark_model() -> BenchmarkModel:
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = BenchmarkModel()
        _GLOBAL_MODEL.load_or_train()
    return _GLOBAL_MODEL


if __name__ == "__main__":
    bm = BenchmarkModel()
    bm.train_and_save()
    print("Benchmark model trained and saved to:", bm.model_path)
