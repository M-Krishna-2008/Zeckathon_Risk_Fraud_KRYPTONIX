"""
Unit tests for distress, manipulation, and fundamental scoring functions in src/scores.

Tests verify:
1. Altman Z (manufacturing) and Z'' (emerging/non-manufacturing) with hand-computed values.
2. Distress, grey, and safe classifications.
3. Beneish 8-variable M-Score with hand-computed ratios and manipulation threshold (> -1.78).
4. Piotroski 9-signal F-Score with full pass and partial fail conditions.
5. Missing input handling (never guessing, returns None with reason).
6. NBFC / Bank handling (skips corporate scores, returns NBFC proxies where data present).
"""

import math
import unittest
from src.scores.altman_z import altman_z
from src.scores.beneish_m import beneish_m
from src.scores.piotroski_f import piotroski_f
from src.scores.nbfc_proxies import calculate_nbfc_proxies


class TestAltmanZ(unittest.TestCase):
    def test_hand_computed_manufacturing_safe(self):
        """
        Hand-computed manufacturing Altman Z:
        total_assets = 1000
        working_capital = 200 -> X1 = 0.20
        retained_earnings = 300 -> X2 = 0.30
        ebit = 150 -> X3 = 0.15
        market_value_equity = 800, total_liabilities = 400 -> X4 = 2.0
        sales = 1200 -> X5 = 1.20
        Z = 1.2*0.2 + 1.4*0.3 + 3.3*0.15 + 0.6*2.0 + 0.999*1.2
          = 0.24 + 0.42 + 0.495 + 1.2 + 1.1988 = 3.5538
        Zone: safe (> 2.99)
        """
        curr = {
            "total_assets": 1000.0,
            "working_capital": 200.0,
            "retained_earnings": 300.0,
            "ebit": 150.0,
            "market_value_equity": 800.0,
            "total_liabilities": 400.0,
            "sales": 1200.0,
        }
        res = altman_z(curr, model_type="manufacturing")
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["score"], 3.5538, places=3)
        self.assertEqual(res["zone"], "safe")
        self.assertEqual(res["z_original_zone"], "safe")

    def test_hand_computed_manufacturing_distress(self):
        """
        Distress manufacturing test:
        total_assets = 1000
        working_capital = -100 -> X1 = -0.10
        retained_earnings = -200 -> X2 = -0.20
        ebit = -50 -> X3 = -0.05
        market_value_equity = 90, total_liabilities = 900 -> X4 = 0.10
        sales = 400 -> X5 = 0.40
        Z = 1.2*(-0.1) + 1.4*(-0.2) + 3.3*(-0.05) + 0.6*(0.1) + 0.999*(0.4)
          = -0.12 - 0.28 - 0.165 + 0.06 + 0.3996 = -0.1054
        Zone: distress (< 1.81)
        """
        curr = {
            "total_assets": 1000.0,
            "working_capital": -100.0,
            "retained_earnings": -200.0,
            "ebit": -50.0,
            "market_value_equity": 90.0,
            "total_liabilities": 900.0,
            "sales": 400.0,
        }
        res = altman_z(curr, model_type="manufacturing")
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["score"], -0.1054, places=3)
        self.assertEqual(res["zone"], "distress")

    def test_hand_computed_emerging_z_double_prime_grey(self):
        """
        Emerging Market Z'' (Non-manufacturing):
        total_assets = 1000
        working_capital = 100 -> X1 = 0.10
        retained_earnings = 150 -> X2 = 0.15
        ebit = 80 -> X3 = 0.08
        book_value_equity = 400, total_liabilities = 600 -> X4 = 400/600 = 0.66667
        Z'' = 6.56*0.1 + 3.26*0.15 + 6.72*0.08 + 1.05*(400/600)
            = 0.656 + 0.489 + 0.5376 + 0.70 = 2.3826
        Zone: grey (1.10 <= 2.3826 <= 2.60)
        """
        curr = {
            "total_assets": 1000.0,
            "working_capital": 100.0,
            "retained_earnings": 150.0,
            "ebit": 80.0,
            "book_value_equity": 400.0,
            "total_liabilities": 600.0,
        }
        res = altman_z(curr, model_type="emerging")
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["score"], 2.3826, places=3)
        self.assertEqual(res["zone"], "grey")
        self.assertEqual(res["z_double_prime_zone"], "grey")

    def test_missing_data_returns_none_never_guesses(self):
        curr = {
            "total_assets": 1000.0,
            "working_capital": 100.0,
            # Missing retained_earnings, ebit, total_liabilities
        }
        res = altman_z(curr)
        self.assertIsNone(res["score"])
        self.assertIsNone(res["zone"])
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIn("Missing required field", res["reason"])

    def test_nbfc_skips_corporate_altman_and_returns_proxies(self):
        nbfc_curr = {
            "equity": 200.0,
            "total_debt": 1000.0,
            "crar": 16.5,
            "gross_npa": 30.0,
            "gross_advances": 1000.0,
            "short_term_borrowings": 150.0,
            "liquid_assets": 200.0,
        }
        nbfc_prior = {
            "gross_npa": 40.0,
            "gross_advances": 900.0,
        }
        res = altman_z(nbfc_curr, prior=nbfc_prior, sector_type="nbfc")
        self.assertIsNone(res["score"])
        self.assertIsNone(res["zone"])
        self.assertEqual(res["status"], "INVALID_SECTOR")
        self.assertIn("not valid for banks and NBFCs", res["reason"])
        self.assertIsNotNone(res["nbfc_proxies"])
        self.assertEqual(res["nbfc_proxies"]["leverage_debt_to_equity"], 5.0)
        self.assertEqual(res["nbfc_proxies"]["crar_percent"], 16.5)
        self.assertEqual(res["nbfc_proxies"]["gnpa_ratio"], 3.0)
        # prior GNPA was 40/900 = 4.4444% -> trend = 3.0 - 4.4444 = -1.4444%
        self.assertAlmostEqual(res["nbfc_proxies"]["gnpa_trend"], -1.4444, places=3)
        self.assertEqual(res["nbfc_proxies"]["alm_short_term_gap_ratio"], 0.75)


class TestBeneishM(unittest.TestCase):
    def test_hand_computed_beneish_non_manipulator(self):
        """
        Controlled clean financial series yielding M < -1.78.
        """
        curr = {
            "sales": 1000.0,
            "receivables": 200.0,
            "cogs": 600.0,
            "total_assets": 2000.0,
            "current_assets": 800.0,
            "ppe": 600.0,
            "securities": 100.0,
            "depreciation": 50.0,
            "sga": 150.0,
            "current_liabilities": 300.0,
            "long_term_debt": 500.0,
            "net_income": 80.0,
            "cfo": 60.0,
        }
        prior = {
            "sales": 900.0,
            "receivables": 150.0,
            "cogs": 500.0,
            "total_assets": 1800.0,
            "current_assets": 700.0,
            "ppe": 550.0,
            "securities": 50.0,
            "depreciation": 45.0,
            "sga": 130.0,
            "current_liabilities": 250.0,
            "long_term_debt": 450.0,
            "net_income": 70.0,
            "cfo": 55.0,
        }
        res = beneish_m(curr, prior)
        self.assertEqual(res["status"], "OK")
        self.assertIsNotNone(res["score"])
        # Verify individual index values
        vars_ = res["variables"]
        self.assertAlmostEqual(vars_["DSRI"], (200 / 1000) / (150 / 900), places=3)
        self.assertAlmostEqual(vars_["SGI"], 1000 / 900, places=3)
        self.assertAlmostEqual(vars_["TATA"], (80 - 60) / 2000, places=3)
        # Check that score is well below -1.78
        self.assertLess(res["score"], -1.78)
        self.assertFalse(res["is_manipulator"])

    def test_beneish_flagged_manipulator(self):
        """
        Distorted financials with surging receivables, crashing margins,
        and massive non-cash accruals (cfo << net_income) -> M > -1.78.
        """
        curr = {
            "sales": 1500.0,
            "receivables": 800.0,  # massive receivable jump
            "cogs": 1200.0,       # margin compressed
            "total_assets": 3000.0,
            "current_assets": 1500.0,
            "ppe": 500.0,
            "securities": 0.0,
            "depreciation": 10.0,  # lowered depreciation
            "sga": 100.0,
            "current_liabilities": 800.0,
            "long_term_debt": 1200.0,
            "net_income": 400.0,
            "cfo": -200.0,         # large positive accruals (NI - CFO = 600)
        }
        prior = {
            "sales": 800.0,
            "receivables": 100.0,
            "cogs": 400.0,
            "total_assets": 2000.0,
            "current_assets": 600.0,
            "ppe": 500.0,
            "securities": 0.0,
            "depreciation": 50.0,
            "sga": 120.0,
            "current_liabilities": 300.0,
            "long_term_debt": 400.0,
            "net_income": 100.0,
            "cfo": 120.0,
        }
        res = beneish_m(curr, prior)
        self.assertEqual(res["status"], "OK")
        self.assertGreater(res["score"], -1.78)
        self.assertTrue(res["is_manipulator"])

    def test_missing_data_returns_none(self):
        curr = {"sales": 1000.0}
        prior = {"sales": 900.0}
        res = beneish_m(curr, prior)
        self.assertIsNone(res["score"])
        self.assertIsNone(res["is_manipulator"])
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIn("Missing required fields", res["reason"])

    def test_nbfc_skips_beneish(self):
        curr = {"sales": 1000.0}
        prior = {"sales": 900.0}
        res = beneish_m(curr, prior, sector_type="nbfc")
        self.assertIsNone(res["score"])
        self.assertIsNone(res["is_manipulator"])
        self.assertEqual(res["status"], "INVALID_SECTOR")
        self.assertIsNotNone(res["nbfc_proxies"])


class TestPiotroskiF(unittest.TestCase):
    def test_perfect_score_9_of_9(self):
        """
        Firm meeting all 9 criteria:
        1. Net Income > 0 (100 > 0)
        2. CFO > 0 (150 > 0)
        3. Delta ROA > 0 (100/1000 = 0.10 > 50/900 = 0.0556)
        4. Accrual: CFO > Net Income (150 > 100)
        5. Leverage: LTD/Assets decreased (200/1000 = 0.20 <= 250/900 = 0.2778)
        6. Liquidity: Current ratio increased (400/200 = 2.0 > 300/200 = 1.5)
        7. No dilution: Shares <= prior shares (1000 <= 1000)
        8. Gross margin increased (500/1000 = 0.50 > 350/800 = 0.4375)
        9. Asset turnover increased (1000/1000 = 1.0 > 800/900 = 0.8889)
        """
        curr = {
            "total_assets": 1000.0,
            "net_income": 100.0,
            "cfo": 150.0,
            "long_term_debt": 200.0,
            "current_assets": 400.0,
            "current_liabilities": 200.0,
            "shares_outstanding": 1000.0,
            "sales": 1000.0,
            "cogs": 500.0,
        }
        prior = {
            "total_assets": 900.0,
            "net_income": 50.0,
            "cfo": 40.0,
            "long_term_debt": 250.0,
            "current_assets": 300.0,
            "current_liabilities": 200.0,
            "shares_outstanding": 1000.0,
            "sales": 800.0,
            "cogs": 450.0,
        }
        res = piotroski_f(curr, prior)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["score"], 9)
        self.assertEqual(res["strength"], "strong")
        self.assertEqual(len(res["failing_tests"]), 0)
        self.assertEqual(len(res["passed_tests"]), 9)

    def test_failing_tests_identified(self):
        """
        Construct a deteriorating firm that fails specific tests:
        - Negative net income (fails F_ROA, F_DELTA_ROA)
        - CFO < 0 (fails F_CFO)
        - Shares issued (fails F_EQ_OFFER)
        """
        curr = {
            "total_assets": 1000.0,
            "net_income": -50.0,    # Fails F_ROA & F_DELTA_ROA
            "cfo": -30.0,           # Fails F_CFO
            "long_term_debt": 200.0,
            "current_assets": 400.0,
            "current_liabilities": 200.0,
            "shares_outstanding": 1200.0,  # Fails F_EQ_OFFER (dilution)
            "sales": 1000.0,
            "cogs": 500.0,
        }
        prior = {
            "total_assets": 900.0,
            "net_income": 50.0,
            "cfo": 40.0,
            "long_term_debt": 250.0,
            "current_assets": 300.0,
            "current_liabilities": 200.0,
            "shares_outstanding": 1000.0,
            "sales": 800.0,
            "cogs": 450.0,
        }
        res = piotroski_f(curr, prior)
        self.assertEqual(res["status"], "OK")
        self.assertIn("F_ROA", res["failing_tests"])
        self.assertIn("F_CFO", res["failing_tests"])
        self.assertIn("F_DELTA_ROA", res["failing_tests"])
        self.assertIn("F_EQ_OFFER", res["failing_tests"])
        # Should pass F_ACCRUAL since -30 > -50
        self.assertIn("F_ACCRUAL", res["passed_tests"])
        self.assertEqual(res["score"], 9 - len(res["failing_tests"]))

    def test_missing_data_returns_none(self):
        curr = {"total_assets": 1000.0}
        prior = {"total_assets": 900.0}
        res = piotroski_f(curr, prior)
        self.assertIsNone(res["score"])
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIn("Missing required fields", res["reason"])

    def test_nbfc_skips_piotroski(self):
        curr = {"total_assets": 1000.0}
        prior = {"total_assets": 900.0}
        res = piotroski_f(curr, prior, sector_type="nbfc")
        self.assertIsNone(res["score"])
        self.assertEqual(res["status"], "INVALID_SECTOR")
        self.assertIn("not valid for banks and NBFCs", res["reason"])


if __name__ == "__main__":
    unittest.main()
