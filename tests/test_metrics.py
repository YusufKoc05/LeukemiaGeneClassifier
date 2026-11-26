"""
Test Suite for Core Biostatistical Metrics, DCA, and Table 1
"""

import unittest
import numpy as np
import pandas as pd
from core.models import get_model_manager
from core.metrics import (
    calculate_comprehensive_metrics,
    calculate_decision_curve_analysis,
    generate_clinical_table_one,
    calculate_cost_sensitive_loss,
    generate_latex_table
)


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm = get_model_manager()
        cls.preds = cls.mm.rf_model.predict(cls.mm.X_test_sc)
        cls.probs = cls.mm.rf_model.predict_proba(cls.mm.X_test_sc)

    def test_comprehensive_metrics_shape_and_keys(self):
        """Kapsamlı tanısal metriklerin doğruluğunu ve anahtarlarını test eder."""
        df_metrics, macro_summary = calculate_comprehensive_metrics(
            self.mm.y_test, self.preds, self.probs, self.mm.classes
        )
        self.assertEqual(len(df_metrics), len(self.mm.classes))
        self.assertIn("Duyarlılık (Sensitivity)", df_metrics.columns)
        self.assertIn("Özgüllük (Specificity)", df_metrics.columns)
        self.assertIn("PPV (Kesinlik)", df_metrics.columns)
        self.assertIn("NPV (Neg. Tahmin)", df_metrics.columns)
        self.assertIn("AUPRC", df_metrics.columns)
        self.assertIn("Brier Skoru", df_metrics.columns)

        self.assertIn("Dengeli Doğruluk (Balanced Acc)", macro_summary)
        self.assertIn("Matthews Korelasyon Katsayısı (MCC)", macro_summary)
        self.assertIn("Cohen's Kappa (κ)", macro_summary)

    def test_decision_curve_analysis(self):
        """Decision Curve Analysis (DCA) Net Benefit hesaplamalarını test eder."""
        dca_df = calculate_decision_curve_analysis(
            self.mm.y_test, self.probs, self.mm.classes, target_class="AML"
        )
        self.assertGreater(len(dca_df), 10)
        self.assertIn("OncoVision AI Net Fayda", dca_df.columns)
        self.assertIn("Tümünü Tedavi Et (Treat All)", dca_df.columns)
        self.assertIn("Hiçbirini Tedavi Etme (Treat None)", dca_df.columns)

        # Net Benefit makul aralıklarda olmalıdır ([-1, 1])
        self.assertTrue((dca_df["OncoVision AI Net Fayda"] <= 1.0).all())
        self.assertTrue((dca_df["OncoVision AI Net Fayda"] >= -1.0).all())

    def test_clinical_table_one(self):
        """Klinik Table 1 ve hipotez testleri p-değerlerini test eder."""
        t1_df = generate_clinical_table_one(self.mm.df)
        self.assertGreaterEqual(len(t1_df), 5)
        self.assertIn("Klinik Parametre", t1_df.columns)
        self.assertTrue(any("Genel Kohort" in c for c in t1_df.columns))
        self.assertIn("ANOVA p-değeri", t1_df.columns)
        self.assertIn("Kruskal-Wallis p-değeri", t1_df.columns)

    def test_cost_sensitive_loss(self):
        """Asimetrik klinik maliyet analizini test eder."""
        cost_loss = calculate_cost_sensitive_loss(self.mm.y_test, self.preds)
        self.assertIn("Hayati Yanlış Negatifler (Kanser Kaçırma)", cost_loss)
        self.assertIn("Ağırlıklı Klinik Maliyet Skoru", cost_loss)
        self.assertGreaterEqual(cost_loss["Ağırlıklı Klinik Maliyet Skoru"], 0.0)

    def test_latex_table_generation(self):
        """LaTeX tablo kodunun doğru üretildiğini test eder."""
        df_metrics, _ = calculate_comprehensive_metrics(
            self.mm.y_test, self.preds, self.probs, self.mm.classes
        )
        latex_str = generate_latex_table(df_metrics)
        self.assertIn("\\begin{table}", latex_str)
        self.assertIn("\\end{table}", latex_str)


if __name__ == "__main__":
    unittest.main()
