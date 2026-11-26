"""
Test Suite for Core Reporting (PDF & FHIR R4 JSON)
"""

import os
import json
import unittest
from core.reporting import generate_clinical_pdf_report, generate_fhir_bundle


class TestReporting(unittest.TestCase):
    def test_fhir_bundle_generation(self):
        """HL7 FHIR R4 Bundle JSON üretimini doğrular."""
        test_path = "test_fhir_bundle.json"
        try:
            res_path = generate_fhir_bundle(
                pred_class="AML",
                confidence=94.5,
                risk_strat="ELN Yüksek Risk",
                cytogenetics="t(15;17) PML::RARA",
                conformal_set_str="AML",
                ood_status="Uyumlu",
                wbc=45.0,
                hb=8.0,
                plt_val=22.0,
                blasts=65.0,
                ldh=800.0,
                age=40,
                output_path=test_path
            )
            self.assertTrue(os.path.exists(res_path))
            with open(res_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data.get("resourceType"), "Bundle")
            self.assertEqual(data.get("type"), "document")
            self.assertIn("entry", data)
            self.assertEqual(len(data["entry"]), 1)
            report = data["entry"][0]["resource"]
            self.assertEqual(report.get("resourceType"), "DiagnosticReport")
            self.assertIn("AML", report.get("conclusion", ""))
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_pdf_report_generation(self):
        """Klinik epikriz PDF raporunun hatasız oluşturulduğunu doğrular."""
        test_pdf = "test_report.pdf"
        try:
            res_path = generate_clinical_pdf_report(
                pred_class="AML",
                confidence=92.0,
                risk_strat="Yüksek Risk",
                cytogenetics="Normal Karyotip",
                conformal_set_str="AML",
                ood_status="Uyumlu",
                uncertainty_str="Varyans: 0.012 (Kararlı)",
                early_dx_badge="Akut Lösemi Evresi",
                is_safety_override=False,
                shap_summary_text="Pozitif Sürücüler: RUNX1, FLT3",
                wbc=65.0,
                hb=7.8,
                plt_val=35.0,
                blasts=45.0,
                ldh=600.0,
                age=55,
                dic_alert=False,
                output_path=test_pdf
            )
            self.assertTrue(os.path.exists(res_path))
            # Dosyanın boş olmadığını kontrol et
            self.assertGreater(os.path.getsize(res_path), 1000)
        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)


if __name__ == "__main__":
    unittest.main()
