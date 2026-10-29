"""
Unit tests for Indian panel loader and strict point-in-time anti-leakage enforcement.
"""

from datetime import date
from pathlib import Path
import unittest

from src.india.loader import (
    load_events,
    get_company_event,
    load_peers,
    load_point_in_time_financials,
    load_point_in_time_financials_wide,
    load_point_in_time_soft_signals,
)


class TestIndiaLoader(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "data" / "india"

    def test_events_and_peers_coverage(self):
        """Verify events file has all 11 defaulters and peers has 30+ matched rows."""
        events = load_events(self.data_dir)
        self.assertEqual(len(events), 11)

        defaulter_names = {e["company"] for e in events}
        expected = {
            "IL&FS",
            "DHFL",
            "Jet Airways",
            "Reliance Capital",
            "Future Retail",
            "Coffee Day Enterprises",
            "Reliance Communications",
            "Videocon Industries",
            "Jaypee Infratech",
            "Educomp Solutions",
            "Amtek Auto",
        }
        self.assertEqual(defaulter_names, expected)

        peers = load_peers(data_dir=self.data_dir)
        self.assertGreaterEqual(len(peers), 30)

        # Every defaulter must have at least 3 peers
        for d in expected:
            d_peers = load_peers(defaulter=d, data_dir=self.data_dir)
            self.assertGreaterEqual(
                len(d_peers),
                3,
                f"Defaulter {d} has fewer than 3 peers: {len(d_peers)}",
            )

    def test_anti_leakage_strictly_before_as_of_date(self):
        """
        Verify that financials filed on or after as_of_date are NEVER returned.
        Jet Airways filing dates in dataset:
          - FY16: 2016-05-18
          - FY17: 2017-05-31
          - FY18: 2018-08-09
        """
        # Case A: As of 2017-01-01 -> Only FY16 allowed
        rows_early = load_point_in_time_financials("Jet Airways", "2017-01-01", self.data_dir)
        years_early = {r["fiscal_year"] for r in rows_early}
        self.assertIn("FY16", years_early)
        self.assertNotIn("FY17", years_early)
        self.assertNotIn("FY18", years_early)

        # Case B: As of 2018-01-01 -> FY16 & FY17 allowed, FY18 strictly excluded
        rows_mid = load_point_in_time_financials("Jet Airways", "2018-01-01", self.data_dir)
        years_mid = {r["fiscal_year"] for r in rows_mid}
        self.assertIn("FY16", years_mid)
        self.assertIn("FY17", years_mid)
        self.assertNotIn("FY18", years_mid)

        # Verify that for every returned row, filing_date is strictly < as_of_date
        for r in rows_mid:
            filing_d = date.fromisoformat(r["filing_date"])
            self.assertLess(filing_d, date(2018, 1, 1))

    def test_boundary_leakage_condition(self):
        """
        Prove that a row filed EXACTLY ON the as_of_date is NOT returned (strict '<' rule).
        """
        # FY17 was filed on 2017-05-31. If as_of_date == 2017-05-31, FY17 must be excluded.
        rows_boundary = load_point_in_time_financials("Jet Airways", "2017-05-31", self.data_dir)
        years = {r["fiscal_year"] for r in rows_boundary}
        self.assertNotIn("FY17", years)
        self.assertIn("FY16", years)

        # If as_of_date == 2017-06-01 (day after), FY17 is now validly included
        rows_next_day = load_point_in_time_financials("Jet Airways", "2017-06-01", self.data_dir)
        years_next = {r["fiscal_year"] for r in rows_next_day}
        self.assertIn("FY17", years_next)

    def test_post_event_data_is_never_returned(self):
        """
        Prove that when querying a company at least 4 quarters before default,
        post-cutoff and post-event information is strictly unreachable.
        Jet Airways default event: 2019-06-20.
        4 quarters prior cutoff: 2018-06-20.
        FY18 was filed on 2018-08-09 (post-cutoff).
        """
        event = get_company_event("Jet Airways", self.data_dir)
        self.assertIsNotNone(event)
        self.assertEqual(event["event_date"], "2019-06-20")

        # 4 quarters prior (365 days)
        cutoff_date = "2018-06-20"
        rows = load_point_in_time_financials("Jet Airways", cutoff_date, self.data_dir)
        for r in rows:
            filing_d = date.fromisoformat(r["filing_date"])
            self.assertLess(
                filing_d,
                date(2018, 6, 20),
                f"LEAKAGE DETECTED: Row with filing date {r['filing_date']} returned for cutoff {cutoff_date}!",
            )

        # Even querying exactly on the event date (2019-06-20), any post-event date is blocked
        future_rows = load_point_in_time_financials("Jet Airways", "2019-06-20", self.data_dir)
        for r in future_rows:
            self.assertLess(date.fromisoformat(r["filing_date"]), date(2019, 6, 20))

    def test_soft_signals_point_in_time(self):
        """
        Verify soft signals filter strictly by announcement date.
        Reliance Capital PwC resignation date: 2019-06-11.
        """
        # Day before announcement: must NOT return the signal
        sigs_before = load_point_in_time_soft_signals(
            "Reliance Capital", "2019-06-11", self.data_dir
        )
        types_before = [s["signal_type"] for s in sigs_before]
        self.assertNotIn("auditor_resignation", types_before)

        # Day after announcement: must return the signal
        sigs_after = load_point_in_time_soft_signals(
            "Reliance Capital", "2019-06-12", self.data_dir
        )
        types_after = [s["signal_type"] for s in sigs_after]
        self.assertIn("auditor_resignation", types_after)

    def test_wide_format_pivoting(self):
        """Test wide format aggregation and metadata preservation."""
        wide = load_point_in_time_financials_wide("Jet Airways", "2018-01-01", self.data_dir)
        self.assertIn("FY16", wide)
        self.assertIn("FY17", wide)
        self.assertNotIn("FY18", wide)

        # Check values
        self.assertEqual(wide["FY16"]["revenue"], 22207.0)
        self.assertEqual(wide["FY16"]["total_assets"], 12365.0)
        self.assertEqual(wide["FY16"]["_filing_date"], "2016-05-18")
        self.assertIsInstance(wide["FY16"]["_sources"], list)
        self.assertGreater(len(wide["FY16"]["_sources"]), 0)


if __name__ == "__main__":
    unittest.main()
