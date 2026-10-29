"""
Unit and integration tests for the full scoring pipeline and portfolio simulation.
"""

from datetime import date
import unittest

from src.scoring_pipeline import score_company
from src.evaluation import run_portfolio_simulation


class TestScoringPipeline(unittest.TestCase):
    def test_score_jet_airways_4q_ahead(self):
        """Jet Airways evaluated at 2018-06-20 (4 quarters prior to 2019-06-20 default)."""
        res = score_company("Jet Airways", "2018-06-20")
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["company"], "Jet Airways")
        # Must have used FY17 filing date (2017-05-31), NOT FY18 (2018-08-09 post-cutoff)
        self.assertEqual(res["latest_fiscal_year"], "FY17")
        self.assertLess(date.fromisoformat(res["latest_filing_used"]), date(2018, 6, 20))
        # Composite score must be higher than healthy prior
        self.assertGreater(res["composite_risk_score"], 0.10)
        self.assertEqual(len(res["top_drivers"]), 3)

    def test_score_future_retail_4q_ahead(self):
        """Future Retail evaluated 4 quarters prior to 2022-07-20."""
        res = score_company("Future Retail", "2021-07-20")
        self.assertEqual(res["status"], "OK")
        self.assertGreater(res["composite_risk_score"], 0.50)
        self.assertEqual(len(res["top_drivers"]), 3)

    def test_portfolio_simulation_structure(self):
        """Test portfolio simulation output schema and capacity cap ranking."""
        sim = run_portfolio_simulation(quarters_ahead=4, portfolio_size=800, top_capacity_cap=40)
        self.assertEqual(sim["portfolio_size"], 800)
        self.assertEqual(sim["capacity_cap"], 40)
        self.assertGreater(sim["recall_at_top_5pct_cap"], 0.50)
        self.assertIn("defaulter_details", sim)
        self.assertEqual(len(sim["defaulter_details"]), 11)


if __name__ == "__main__":
    unittest.main()
