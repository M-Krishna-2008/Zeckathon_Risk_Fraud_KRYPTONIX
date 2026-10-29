"""
Integration tests for FastAPI endpoints in app/server.py.
"""

import unittest
from fastapi.testclient import TestClient
from app.server import app


class TestServerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_watchlist_endpoint(self):
        resp = self.client.get("/api/watchlist")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["portfolio_size"], 800)
        self.assertEqual(data["capacity_cap"], 40)
        self.assertIn("top_40_borrowers", data)
        self.assertIn("caveat", data)

    def test_get_company_endpoint(self):
        resp = self.client.get("/api/company/Jet Airways")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["company"], "Jet Airways")
        self.assertIn("composite_risk_score", data)
        self.assertIn("top_drivers", data)
        self.assertEqual(len(data["top_drivers"]), 3)

    def test_get_metrics_endpoint(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("evaluation", data)
        self.assertIn("cost_curve", data)
        self.assertEqual(data["benchmark_pr_auc"], 0.842)

    def test_chat_intent_watchlist(self):
        resp = self.client.post("/api/chat", json={"message": "show watch-list"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "watchlist")
        self.assertIn("Top 40 Flagged Borrowers", data["response"])

    def test_chat_intent_company_investigation(self):
        resp = self.client.post("/api/chat", json={"message": "why was Jet Airways flagged?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "company_investigation")
        self.assertIn("Jet Airways", data["response"])
        self.assertIn("Top 3 Risk Drivers", data["response"])

    def test_chat_intent_performance(self):
        resp = self.client.post("/api/chat", json={"message": "recall and model performance"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "performance_metrics")
        self.assertIn("Recall", data["response"])

    def test_chat_intent_policy(self):
        resp = self.client.post("/api/chat", json={"message": "policy and cost threshold"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "policy_cost")
        self.assertIn("0.0476", data["response"])

    def test_chat_intent_compare(self):
        resp = self.client.post("/api/chat", json={"message": "compare Jet Airways and IndiGo"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "company_comparison")
        self.assertIn("Jet Airways vs. InterGlobe Aviation", data["response"])

    def test_chat_intent_fallback(self):
        resp = self.client.post("/api/chat", json={"message": "hello what can you do?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "fallback")
        self.assertIn("Distress Watch", data["response"])

    def test_serve_static_index(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Distress Watch", resp.text)
        self.assertIn("chat-input", resp.text)


if __name__ == "__main__":
    unittest.main()
