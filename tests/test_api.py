"""
Test Suite for FastAPI REST Endpoints
"""

import unittest
from fastapi.testclient import TestClient
import main


class TestFastAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_root_endpoint(self):
        """GET / endpointini test eder."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "Online")
        self.assertIn("OncoVision", data.get("service", ""))

    def test_health_endpoint(self):
        """GET /health endpointini test eder."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("models_loaded"))

    def test_model_info_endpoint(self):
        """GET /model-info endpointini test eder."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 3)

    def test_benchmarks_endpoint(self):
        """GET /benchmarks endpointini test eder."""
        response = self.client.get("/benchmarks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_zoo_benchmarks", data)

    def test_advanced_metrics_endpoint(self):
        """GET /metrics/advanced endpointini test eder."""
        response = self.client.get("/metrics/advanced")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("macro_summary", data)
        self.assertIn("class_diagnostic_metrics", data)
        self.assertIn("cost_sensitive_evaluation", data)

    def test_table_one_endpoint(self):
        """GET /metrics/table-one endpointini test eder."""
        response = self.client.get("/metrics/table-one")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("table_one", data)
        self.assertGreaterEqual(len(data["table_one"]), 5)

    def test_dca_endpoint(self):
        """GET /metrics/dca endpointini test eder."""
        response = self.client.get("/metrics/dca?target_class=AML")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("target_class"), "AML")
        self.assertIn("net_benefit_curve", data)
        self.assertGreater(len(data["net_benefit_curve"]), 10)

    def test_predict_endpoint(self):
        """POST /predict endpointini test eder."""
        payload = {
            "age": 45,
            "runx1": 6.5,
            "flt3": 6.5,
            "wt1": 6.5,
            "pax5": 6.5,
            "cd19": 6.5,
            "ikzf1": 6.5,
            "mpo": 6.5,
            "cd34": 6.5,
            "wbc": 6.8,
            "hb": 14.2,
            "plt_val": 260.0,
            "blasts": 0.5,
            "ldh": 180.0,
            "cytogenetics": "Normal Karyotip (Orta Risk)"
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("final_clinical_class"), "Healthy")
        self.assertIn("class_probabilities", data)

    def test_simulate_endpoint(self):
        """POST /simulate endpointini test eder."""
        payload = {
            "patient": {
                "age": 55,
                "runx1": 10.5,
                "flt3": 11.2,
                "wt1": 9.5,
                "pax5": 6.0,
                "cd19": 5.5,
                "ikzf1": 5.0,
                "mpo": 8.0,
                "cd34": 7.0,
                "wbc": 75.0,
                "hb": 7.2,
                "plt_val": 25.0,
                "blasts": 65.0,
                "ldh": 900.0,
                "cytogenetics": "Karmaşık Karyotip (>=3 Anomali, Adverse)"
            },
            "target_intervention": "FLT3 İnhibisyonu (Gilteritinib / Midostaurin)",
            "intensity_pct": 50.0
        }
        response = self.client.post("/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("post_intervention_probabilities", data)
        self.assertIn("healthy_convergence_delta_pct", data)

    def test_export_fhir_endpoint(self):
        """POST /report/fhir endpointini test eder."""
        payload = {
            "age": 45,
            "runx1": 6.5,
            "flt3": 6.5,
            "wt1": 6.5,
            "pax5": 6.5,
            "cd19": 6.5,
            "ikzf1": 6.5,
            "mpo": 6.5,
            "cd34": 6.5,
            "wbc": 6.8,
            "hb": 14.2,
            "plt_val": 260.0,
            "blasts": 0.5,
            "ldh": 180.0,
            "cytogenetics": "Normal Karyotip (Orta Risk)"
        }
        response = self.client.post("/report/fhir", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("resourceType"), "Bundle")

    def test_export_pdf_endpoint(self):
        """POST /report/pdf endpointini test eder."""
        payload = {
            "age": 45,
            "runx1": 6.5,
            "flt3": 6.5,
            "wt1": 6.5,
            "pax5": 6.5,
            "cd19": 6.5,
            "ikzf1": 6.5,
            "mpo": 6.5,
            "cd34": 6.5,
            "wbc": 6.8,
            "hb": 14.2,
            "plt_val": 260.0,
            "blasts": 0.5,
            "ldh": 180.0,
            "cytogenetics": "Normal Karyotip (Orta Risk)"
        }
        response = self.client.post("/report/pdf", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(response.content), 1000)


if __name__ == "__main__":
    unittest.main()
