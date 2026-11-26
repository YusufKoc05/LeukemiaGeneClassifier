"""
Test Suite for OncoVision AI Advanced Research Modules
TÜBİTAK 2209-A İleri Düzey Doğrulama Testleri:
- Golub Microarray Benchmark Parser
- Ensemble / Stacking Meta-Classifier (RF + LGBM + XGB)
- NCBI GEO Data Bridge
- Pharmacogenomics (PGx) Dosing & Toxicity
- WHO 2022 / ICC Molecular Classification
- Case Comparison View
- Leukocyte Vision Classifier
"""

import unittest
import numpy as np
import pandas as pd

from core import (
    parse_and_unify_golub,
    get_ensemble_classifier,
    fetch_geo_cohort,
    list_supported_geo_studies,
    evaluate_pharmacogenomic_profile,
    classify_who_2022_subtypes,
    build_patient_comparison_view,
    classify_and_count_blood_smear
)


class TestAdvancedModules(unittest.TestCase):

    def test_ensemble_classifier(self):
        """RF + LightGBM + XGBoost Ensemble Meta-Sınıflandırıcısının doğrulaması."""
        ens = get_ensemble_classifier()
        df_models = ens.evaluate_all_models()
        self.assertGreaterEqual(len(df_models), 4)
        self.assertTrue((df_models["Test Doğruluğu (%)"] >= 95.0).all())

        # Tekil çıkarım
        from core import get_model_manager
        mm = get_model_manager()
        sample_x = mm.X_test_sc[0:1]
        det = ens.get_prediction_detail(sample_x)
        self.assertIn("ensemble_pred", det)
        self.assertIn("model_agreement_ratio", det)
        self.assertIn("Hemfikir", det["model_agreement_ratio"])

    def test_geo_bridge(self):
        """NCBI GEO veri tabanı köprüsünün çalışma listesi ve kohort entegrasyonu."""
        studies = list_supported_geo_studies()
        self.assertGreaterEqual(len(studies), 3)
        self.assertTrue(any(s["Accession"] == "GSE13159" for s in studies))

        df_geo, msg = fetch_geo_cohort("GSE13159", sample_size=15)
        self.assertEqual(len(df_geo), 15)
        self.assertIn("WBC_x10k", df_geo.columns)
        self.assertIn("FLT3", df_geo.columns)

    def test_pharmacogenomics_poor_metabolizer(self):
        """TPMT zayıf metabolizörde %90 doz azaltımı ve hayati toksisite uyarısı üretimi."""
        res = evaluate_pharmacogenomic_profile(
            tpmt_status="Zayıf Metabolizör (*3A/*3A veya *3A/*3C - Aktivite Yok / Kritik)",
            nudt15_status="Normal Metabolizör (*1/*1)",
            dpyd_status="Normal DPYD Aktivitesi (Tam Fonksiyon)",
            planned_regimen="ALL İdame (6-MP + MTX)"
        )
        self.assertIn("HAYATİ TEHLİKE", res["risk_level"])
        self.assertEqual(res["card_color"], "#7a1f2c")
        dosing_df = res["dosing_df"]
        self.assertTrue(any("%90 Doz Azaltımı" in str(d) for d in dosing_df["Önerilen Doz"]))

    def test_pharmacogenomics_normal_metabolizer(self):
        """Tüm genleri normal olan hastada %100 standart protokol dozu önerisi."""
        res = evaluate_pharmacogenomic_profile(
            tpmt_status="Normal Metabolizör (*1/*1 - Standart Aktivite)",
            nudt15_status="Normal Metabolizör (*1/*1)",
            dpyd_status="Normal DPYD Aktivitesi (Tam Fonksiyon)"
        )
        self.assertIn("Standart", res["risk_level"])
        dosing_df = res["dosing_df"]
        self.assertTrue(any("%100 Standart" in str(d) for d in dosing_df["Önerilen Doz"]))

    def test_who_2022_subtypes(self):
        """DSÖ 2022 ve ICC moleküler alt tip sınıflandırma kurallarının doğruluğu."""
        # APL vaka
        apl = classify_who_2022_subtypes("AML", "t(15;17) PML::RARA (APL)", 6.5, 6.5, 6.5, 6.5, 35)
        self.assertIn("Promiyelositik", apl["who_entity"])
        self.assertIn("ATRA", apl["first_line_therapy"])

        # Ph+ ALL vaka
        ph_all = classify_who_2022_subtypes("ALL", "t(9;22) BCR::ABL1 (Ph+ Yüksek Risk)", 6.5, 12.0, 11.5, 6.5, 50)
        self.assertIn("BCR::ABL1", ph_all["who_entity"])
        self.assertTrue(ph_all["allo_hsct_candidate"])

        # Sağlıklı
        h = classify_who_2022_subtypes("Healthy", "Normal Karyotip (Orta Risk)", 6.5, 6.5, 6.5, 6.5, 25)
        self.assertIn("Fizyolojik", h["who_entity"])

    def test_case_comparison(self):
        """İki vaka arası split-screen kıyaslama radarı ve fark tablosu üretimi."""
        p_a = {"name": "Hasta A", "wbc": 6.8, "hb": 14.0, "plt_val": 250.0, "blasts": 0.5, "ldh": 180.0, "flt3": 6.5, "cd19": 6.5, "pred_class": "Healthy"}
        p_b = {"name": "Hasta B", "wbc": 95.0, "hb": 7.0, "plt_val": 22.0, "blasts": 65.0, "ldh": 1100.0, "flt3": 12.5, "cd19": 5.5, "pred_class": "AML"}
        fig, diff_df, html = build_patient_comparison_view(p_a, p_b)
        self.assertIsNotNone(fig)
        self.assertEqual(len(diff_df), 8)
        self.assertIn("Hasta A", html)
        self.assertIn("Hasta B", html)

    def test_vision_classifier(self):
        """Morfolojik periferik yayma hücre sınıflandırıcısı ve blast yüzdesi hesabı."""
        img = np.full((200, 300, 3), 235, dtype=np.uint8)
        # 1 adet blast çekirdeği ekle
        img[30:80, 30:80] = [75, 25, 110]
        # 1 adet lenfosit ekle
        img[120:140, 150:170] = [80, 30, 95]
        
        annotated_img, blast_pct, summary_html, df_cells = classify_and_count_blood_smear(img, current_blast_val=0.5)
        self.assertIsNotNone(annotated_img)
        self.assertGreater(len(df_cells), 0)
        self.assertIn("Dijital Morfoloji", summary_html)


if __name__ == "__main__":
    unittest.main()
