"""
Unit tests for Policy Cost Model, Explainability, and Feature Mapping.
"""

import math
import unittest

from src.policy.cost import (
    compute_cost_at_threshold,
    generate_cost_table,
    find_optimal_threshold,
)
from src.explain.explainer import explain_flag, FEATURE_TEMPLATES, SOFT_SIGNAL_TEMPLATES
from src.benchmark.mapper import map_indian_to_polish_features


class TestPolicyCostModel(unittest.TestCase):
    def test_cost_calculation(self):
        """Verify 20*FN + 1*FP cost formula."""
        y_true = [1, 1, 0, 0, 0]
        y_prob = [0.9, 0.02, 0.8, 0.01, 0.03]
        # At threshold 0.05:
        # y_true=1, prob=0.9 -> TP
        # y_true=1, prob=0.02 -> FN (1 missed default)
        # y_true=0, prob=0.8 -> FP (1 false alarm)
        # y_true=0, prob=0.01 -> TN
        # y_true=0, prob=0.03 -> TN
        res = compute_cost_at_threshold(y_true, y_prob, threshold=0.05, cost_fn=20.0, cost_fp=1.0)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["tn"], 2)
        # Cost = 20*1 + 1*1 = 21
        self.assertEqual(res["total_cost"], 21.0)

    def test_optimal_threshold_near_one_over_twenty_one(self):
        """
        Verify that in calibrated probabilities with 20x cost asymmetry,
        the cost-minimizing threshold is near 1/21 ~= 0.0476.
        """
        table = generate_cost_table()
        optimal_tau = 1.0 / 21.0  # 0.047619

        min_cost_row = min(table, key=lambda x: x["total_cost"])
        # The best threshold in the standard table should be within proximity of 1/21 (e.g. 0.03 to 0.06)
        self.assertAlmostEqual(min_cost_row["threshold"], optimal_tau, delta=0.025)


class TestExplainability(unittest.TestCase):
    def test_plain_language_synthesis(self):
        """Ensure explain_flag outputs credit officer plain-language sentences."""
        shap_vals = {"Attr27": 0.35, "Attr2": 0.25, "Attr1": -0.10}
        features = {"Attr27": 0.8, "Attr2": 82.5, "Attr1": -5.2}
        scores_summary = {
            "altman_z": {"zone": "distress", "score": 0.45, "model_used": "manufacturing"},
            "beneish_m": {"is_manipulator": True, "score": -1.25},
        }
        soft_signals = [
            {
                "signal_type": "auditor_resignation",
                "value": "Statutory auditor resigned citing lack of audit evidence",
            }
        ]

        drivers = explain_flag(
            company="TestCorp",
            shap_values=shap_vals,
            feature_values=features,
            scores_summary=scores_summary,
            soft_signals=soft_signals,
            top_n=3,
        )

        self.assertEqual(len(drivers), 3)
        # Top driver should be the acute soft signal (auditor resignation)
        self.assertIn("Auditor abruptly resigned", drivers[0])
        # Other drivers should include Beneish or Altman distress or Attr27
        all_text = " ".join(drivers)
        self.assertTrue(
            "Beneish M-Score" in all_text or "Interest cover" in all_text or "Altman" in all_text
        )


class TestFeatureMapper(unittest.TestCase):
    def test_ratio_computations(self):
        fin = {
            "revenue": 1000.0,
            "net_income": 80.0,
            "total_assets": 2000.0,
            "current_assets": 800.0,
            "current_liabilities": 400.0,
            "ebit": 150.0,
            "interest_expense": 50.0,
            "cogs": 600.0,
            "equity": 900.0,
        }
        attrs = map_indian_to_polish_features(fin)

        self.assertEqual(len(attrs), 64)
        # Attr1 = NI / TA = 80 / 2000 = 0.04
        self.assertAlmostEqual(attrs["Attr1"], 0.04, places=4)
        # Attr3 = WC / TA = (800 - 400) / 2000 = 0.20
        self.assertAlmostEqual(attrs["Attr3"], 0.20, places=4)
        # Attr4 = CA / CL = 800 / 400 = 2.0
        self.assertAlmostEqual(attrs["Attr4"], 2.0, places=4)
        # Attr27 = EBIT / Interest = 150 / 50 = 3.0
        self.assertAlmostEqual(attrs["Attr27"], 3.0, places=4)
        # Unmapped attributes must remain NaN
        self.assertTrue(math.isnan(attrs["Attr7"]))
        self.assertTrue(math.isnan(attrs["Attr64"]))


if __name__ == "__main__":
    unittest.main()
