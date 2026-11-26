"""
Test Suite for External Kaggle & Benchmark Dataset Evaluation Module
TÜBİTAK 2209-A Standartlarına Uygun Doğrulama Testleri
"""

import os
import unittest
import pandas as pd
import numpy as np

from core.evaluation import (
    DEFAULT_DATASET_DIR,
    list_external_datasets,
    detect_column_mappings,
    load_dataset_preview,
    normalize_target_labels,
    evaluate_external_dataset,
    read_file_safe,
    build_cross_dataset_comparison_table,
    build_cross_dataset_plots,
    build_cross_dataset_kpis_html
)
from core.models import get_model_manager


class TestEvaluationModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm = get_model_manager()
        cls.test_csv_path = "tests_scratch_eval_sample.csv"
        # Sentetik bir harici test veri seti oluştur
        df_sample = pd.DataFrame({
            "Hasta_No": [f"ID_{i+1:03d}" for i in range(15)],
            "White_Blood_Cells": [7500, 8200, 110000, 95000, 4200, 6800, 7100, 120000, 6500, 5900, 80000, 7200, 90000, 5500, 6300],
            "Hemoglobin": [14.0, 13.5, 7.2, 6.8, 14.5, 13.8, 14.2, 8.0, 15.0, 13.9, 7.5, 14.1, 8.2, 14.3, 14.0],
            "Platelet_Count": [250000, 260000, 22000, 35000, 280000, 240000, 270000, 40000, 300000, 255000, 38000, 265000, 42000, 290000, 245000],
            "Blast_Orani": [0.5, 1.0, 65.0, 75.0, 0.2, 0.4, 0.6, 70.0, 0.5, 0.3, 60.0, 0.5, 55.0, 0.8, 0.2],
            "LDH": [180, 190, 950, 1200, 160, 175, 185, 1100, 165, 170, 980, 180, 890, 175, 160],
            "Tani": [
                "Healthy", "Healthy", "AML", "AML", "Healthy",
                "Healthy", "Healthy", "ALL", "Healthy", "Healthy",
                "AML", "Healthy", "ALL", "Healthy", "Healthy"
            ]
        })
        df_sample.to_csv(cls.test_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_path):
            try:
                os.remove(cls.test_csv_path)
            except Exception:
                pass
        export_file = "model_test_sonuclari.csv"
        if os.path.exists(export_file):
            try:
                os.remove(export_file)
            except Exception:
                pass

    def test_list_external_datasets_nonexistent(self):
        """Geçersiz bir dizin verildiğinde çökmeden boş liste döndüğünü doğrular."""
        res = list_external_datasets("C:\\Non_Existent_Folder_123456")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 0)

    def test_list_external_datasets_default(self):
        """Varsayılan Kaggle dizini mevcutsa dosyaları listelediğini test eder."""
        if os.path.exists(DEFAULT_DATASET_DIR):
            files = list_external_datasets(DEFAULT_DATASET_DIR)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(len(item) == 2 for item in files))

    def test_detect_column_mappings(self):
        """Sütun isimlerinden otomatik özellik eşleştirmesini doğrular."""
        cols = ["Patient_ID", "White_Blood_Cells", "Hemoglobin", "Platelet_Count", "Blast_Percentage", "LDH_U_L", "Cancer_Type"]
        mappings = detect_column_mappings(cols)
        self.assertEqual(mappings["sample_id"], "Patient_ID")
        self.assertEqual(mappings["wbc"], "White_Blood_Cells")
        self.assertEqual(mappings["hb"], "Hemoglobin")
        self.assertEqual(mappings["plt"], "Platelet_Count")
        self.assertEqual(mappings["blasts"], "Blast_Percentage")
        self.assertEqual(mappings["ldh"], "LDH_U_L")
        self.assertEqual(mappings["target"], "Cancer_Type")

    def test_load_dataset_preview(self):
        """Veri seti önizlemesinin ve sütun seçeneklerinin doğru üretildiğini doğrular."""
        preview_df, info_msg, choices, auto_map = load_dataset_preview(self.test_csv_path)
        self.assertFalse(preview_df.empty)
        self.assertIn("15 satır", info_msg)
        self.assertIn("(Seçilmedi / Yok)", choices)
        self.assertEqual(auto_map["wbc"], "White_Blood_Cells")
        self.assertEqual(auto_map["hb"], "Hemoglobin")
        self.assertEqual(auto_map["plt"], "Platelet_Count")
        self.assertEqual(auto_map["target"], "Tani")
        self.assertEqual(auto_map["sample_id"], "Hasta_No")

    def test_normalize_target_labels(self):
        """Hedef etiketlerin standart model sınıflarına normalize edilmesini doğrular."""
        s = pd.Series(["all", "AML", "healthy", "Normal", "Acute Myeloid Leukemia"])
        norm, is_compat = normalize_target_labels(s, self.mm.classes)
        self.assertTrue(is_compat)
        self.assertEqual(norm.tolist(), ["ALL", "AML", "Healthy", "Healthy", "AML"])

    def test_evaluate_external_dataset_with_ground_truth(self):
        """Gerçek etiketli veri setinde metriklerin, karmaşıklık matrisinin ve CSV çıktısının doğruluğu."""
        kpi_html, fig_cm, df_report, results_df, export_path = evaluate_external_dataset(
            self.test_csv_path,
            target_col="Tani",
            wbc_col="White_Blood_Cells",
            hb_col="Hemoglobin",
            plt_col="Platelet_Count",
            blast_col="Blast_Orani",
            ldh_col="LDH",
            sample_id_col="Hasta_No",
            model_manager=self.mm
        )
        self.assertIn("Doğruluk (Accuracy)", kpi_html)
        self.assertIsNotNone(fig_cm)
        self.assertFalse(df_report.empty)
        self.assertEqual(len(results_df), 15)
        self.assertIn("Örnek ID", results_df.columns)
        self.assertIn("Gerçek Etiket", results_df.columns)
        self.assertIn("Model Tahmini", results_df.columns)
        self.assertIn("Model Güveni (%)", results_df.columns)
        self.assertIn("Doğruluk Durumu", results_df.columns)
        self.assertTrue(os.path.exists(export_path))

    def test_evaluate_external_dataset_unlabeled(self):
        """Etiketsiz bir veri seti verildiğinde modelin çıkarım yapıp sınıf dağılımı ürettiğini test eder."""
        kpi_html, fig_dist, df_report, results_df, export_path = evaluate_external_dataset(
            self.test_csv_path,
            target_col="(Seçilmedi / Yok)",
            wbc_col="White_Blood_Cells",
            hb_col="Hemoglobin",
            plt_col="Platelet_Count",
            model_manager=self.mm
        )
        self.assertIn("Ortalama Model Güveni", kpi_html)
        self.assertIsNotNone(fig_dist)
        self.assertEqual(len(results_df), 15)
        self.assertTrue((results_df["Gerçek Etiket"] == "(Etiketsiz)").all())
    def test_detect_column_mappings_edge_cases(self):
        """Klinik veri setlerindeki özel birim/kısaltma sütunlarının tespitini test eder."""
        cols = ["TLC", "HB", "Platelet", "Patient ID", "Severity"]
        mappings = detect_column_mappings(cols)
        self.assertEqual(mappings["wbc"], "TLC")
        self.assertEqual(mappings["hb"], "HB")
        self.assertEqual(mappings["plt"], "Platelet")
        self.assertEqual(mappings["sample_id"], "Patient ID")
        self.assertEqual(mappings["target"], "Severity")

    def test_read_file_safe_nan_cleanup(self):
        """Tümüyle NaN olan satırların güvenle ayıklandığını doğrular."""
        df_with_nan = pd.DataFrame({
            "WBC": [np.nan, 7.5, 8.2],
            "HGB": [np.nan, 14.1, 13.9],
            "PLT": [np.nan, 220, 240]
        })
        tmp_csv = "tests_scratch_nan_sample.csv"
        df_with_nan.to_csv(tmp_csv, index=False)
        try:
            cleaned = read_file_safe(tmp_csv)
            self.assertEqual(len(cleaned), 2)
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    def test_large_dataset_pagination(self):
        """1000'den fazla satırlı veri setlerinde arayüzün 1000 satır döndürdüğünü, diske ise tümünü kaydettiğini test eder."""
        df_large = pd.DataFrame({
            "Patient_ID": [f"P_{i}" for i in range(1200)],
            "WBC": [7.0] * 1200,
            "HB": [14.0] * 1200,
            "PLT": [250.0] * 1200
        })
        tmp_large = "tests_scratch_large_sample.csv"
        df_large.to_csv(tmp_large, index=False)
        try:
            kpi, fig, rep, display_df, export_path = evaluate_external_dataset(
                tmp_large,
                wbc_col="WBC", hb_col="HB", plt_col="PLT", sample_id_col="Patient_ID",
                model_manager=self.mm
            )
            self.assertEqual(len(display_df), 1000)
            self.assertIn("Büyük Veri Seti Optimizasyonu", kpi)
            full_saved = pd.read_csv(export_path)
            self.assertEqual(len(full_saved), 1200)
        finally:
            if os.path.exists(tmp_large):
                os.remove(tmp_large)

    def test_build_cross_dataset_comparison_table(self):
        """Tüm veri setlerini kapsayan karşılaştırma tablosunu doğrular."""
        df_comp = build_cross_dataset_comparison_table(self.mm)
        self.assertIsInstance(df_comp, pd.DataFrame)
        self.assertGreaterEqual(len(df_comp), 6)
        self.assertIn("Veri Seti", df_comp.columns)
        self.assertIn("Doğruluk (Accuracy %)", df_comp.columns)
        self.assertIn("Ortalama Model Güveni (%)", df_comp.columns)
        # Değerlerin mantıklı aralıklarda olduğunu doğrula
        self.assertTrue((df_comp["Doğruluk (Accuracy %)"] >= 90.0).all())
        self.assertTrue(((df_comp["Ortalama Model Güveni (%)"] >= 60.0) & (df_comp["Ortalama Model Güveni (%)"] <= 100.0)).all())

    def test_build_cross_dataset_plots(self):
        """Tüm veri setleri karşılaştırma grafiklerinin (4 figür) eksiksiz üretildiğini doğrular."""
        import plotly.graph_objects as go
        df_comp = build_cross_dataset_comparison_table(self.mm)
        figs = build_cross_dataset_plots(df_comp, self.mm)
        self.assertIsInstance(figs, dict)
        for key in ["performance", "distribution", "bubble", "biomarkers"]:
            self.assertIn(key, figs)
            self.assertIsInstance(figs[key], go.Figure)

    def test_build_cross_dataset_kpis_html(self):
        """Karşılaştırma paneli üst özet KPI kartlarının HTML formatını test eder."""
        df_comp = build_cross_dataset_comparison_table(self.mm)
        kpis_html = build_cross_dataset_kpis_html(df_comp)
        self.assertIsInstance(kpis_html, str)
        self.assertIn("Toplam Kapsanan Vaka", kpis_html)
        self.assertIn("Ortalama Doğruluk", kpis_html)
        self.assertIn("Kapsanan Veri Tabanı", kpis_html)


if __name__ == "__main__":
    unittest.main()
