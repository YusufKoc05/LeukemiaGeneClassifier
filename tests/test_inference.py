"""
Test Suite for Core Inference Engine & Clinical Safety Nets
"""

import unittest
from core.inference import get_inference_engine


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_inference_engine()

    def test_healthy_case_prediction(self):
        """Sağlıklı referans profilin 'Healthy' sınıflandırıldığını doğrular."""
        res = self.engine.predict_patient(
            age=30, runx1=6.5, flt3=6.5, pax5=6.5, ikzf1=6.5, cd19=6.5, wt1=6.5,
            wbc=6.8, hb=14.2, plt_val=260.0, blasts=0.5, ldh=180.0,
            cytogenetics="Normal Karyotip (Orta Risk)"
        )
        self.assertEqual(res["final_clinical_class"], "Healthy")
        self.assertFalse(res["safety_net_triggered"])
        self.assertFalse(res["dic_coagulopathy_alert"])
        self.assertIn("Healthy", res["conformal_prediction_set"])

    def test_who_acute_blast_safety_net_override(self):
        """
        KRİTİK TEST: Gen ekspresyonu bazal düzeyde olsa bile blast >= %20 ise
        modelin 'Healthy' tahmini geçersiz kılınmalı ve güvenlik ağı tetiklenmelidir.
        """
        res = self.engine.predict_patient(
            age=45, runx1=6.5, flt3=6.5, pax5=6.5, ikzf1=6.5, cd19=6.5, wt1=6.5,
            wbc=6.8, hb=14.2, plt_val=260.0, blasts=25.0, ldh=180.0,
            cytogenetics="Normal Karyotip (Orta Risk)"
        )
        # Ham model çıktısı Healthy olsa bile:
        self.assertTrue(res["safety_net_triggered"])
        self.assertNotEqual(res["final_clinical_class"], "Healthy")
        self.assertIn("Şüpheli", res["final_clinical_class"])
        self.assertIn("Blast", res["clinical_warning"])
        self.assertIn("%20", res["clinical_warning"])
        self.assertIn("WHO Akut Lösemi Kriteri", res["clinical_warning"])

    def test_apl_dic_coagulopathy_alert(self):
        """
        KRİTİK TEST: t(15;17) PML::RARA ve derin trombositopeni (PLT < 30) durumunda
        Acil DIC uyarısının tetiklendiğini doğrular.
        """
        res = self.engine.predict_patient(
            age=35, runx1=8.0, flt3=6.5, pax5=6.0, ikzf1=6.5, cd19=6.0, wt1=7.5,
            wbc=20.0, hb=7.5, plt_val=18.0, blasts=60.0, ldh=700.0,
            cytogenetics="t(15;17) PML::RARA (APL)"
        )
        self.assertTrue(res["dic_coagulopathy_alert"])
        self.assertIn("DIC", res["actionable_protocol"])

    def test_counterfactual_simulation(self):
        """What-if simülasyonunun geçerli olasılık dağılımları ürettiğini test eder."""
        orig_p, sim_p, delta_healthy = self.engine.run_counterfactual_simulation(
            base_age=50, base_runx1=10.0, base_flt3=11.5, base_pax5=6.0,
            base_ikzf1=5.0, base_cd19=6.0, base_wt1=10.0, base_wbc=80.0,
            base_hb=7.0, base_plt=25.0, base_blasts=60.0, base_ldh=900.0,
            cytogenetics="Karmaşık Karyotip (>=3 Anomali, Adverse)",
            target_intervention="FLT3 İnhibisyonu (Gilteritinib / Midostaurin)",
            intensity_pct=50.0
        )
        self.assertEqual(len(orig_p), 3)
        self.assertEqual(len(sim_p), 3)
        self.assertAlmostEqual(sum(orig_p), 1.0, places=2)
        self.assertAlmostEqual(sum(sim_p), 1.0, places=2)

    def test_batch_screening_rejects_unrelated_csv(self):
        """Alakasız CSV dosyalarının (uyumsuz şema) tespit edilip reddedildiğini doğrular."""
        import pandas as pd
        bad_df = pd.DataFrame({"City": ["Ankara", "Izmir"], "Temp": [22.0, 26.5], "Score": [88, 92]})
        with self.assertRaises(ValueError):
            self.engine.run_batch_screening(bad_df)

    def test_batch_screening_processes_valid_csv(self):
        """Geçerli şablon kohort CSV dosyasının başarıyla işlendiğini doğrular."""
        import pandas as pd
        good_df = pd.read_csv("ornek_kohort_sablonu.csv")
        res_df, df_drift, triage = self.engine.run_batch_screening(good_df)
        self.assertEqual(len(res_df), len(good_df))
        self.assertIn("Ön Tanı", res_df.columns)
        self.assertIn("Triyaj", res_df.columns)
        self.assertTrue(len(df_drift) > 0)


    def test_blood_smear_qc_rejects_unrelated_image(self):
        """Doğal sahne / belge gibi alakasız görsellerin kan yayması QC testinde elendiğini doğrular."""
        import numpy as np
        from core.visualizations import validate_and_detect_blood_smear
        # Doğa/manzara renkli görsel (yeşil ve açık mavi)
        arr_nature = np.zeros((150, 150, 3), dtype=np.uint8)
        arr_nature[:75, :] = [100, 180, 240]  # açık mavi
        arr_nature[75:, :] = [40, 160, 40]    # yeşil
        out_img, b_val, msg = validate_and_detect_blood_smear(arr_nature, 0.5)
        self.assertEqual(b_val, 0.5)
        self.assertIn("Başarısız", msg)

    def test_blood_smear_qc_accepts_valid_smear(self):
        """Wright-Giemsa boyalı sentetik kan yaymasının başarıyla analiz edildiğini doğrular."""
        import numpy as np
        from core.visualizations import validate_and_detect_blood_smear
        arr_smear = np.ones((200, 200, 3), dtype=np.uint8) * 230
        arr_smear[:, :, 1] = 220
        # Eritrositler
        for cx, cy in [(40, 40), (80, 50), (120, 60), (50, 120), (100, 140), (150, 120), (160, 50), (40, 160)]:
            y, x = np.ogrid[:200, :200]
            mask = (x - cx)**2 + (y - cy)**2 <= 10**2
            arr_smear[mask] = [210, 140, 160]
        # 1 adet atipik blast hücresi
        y, x = np.ogrid[:200, :200]
        mask_cell = (x - 100)**2 + (y - 100)**2 <= 22**2
        mask_nuc = (x - 100)**2 + (y - 100)**2 <= 16**2
        arr_smear[mask_cell] = [170, 130, 190]
        arr_smear[mask_nuc] = [90, 50, 130]

        out_img, b_val, msg = validate_and_detect_blood_smear(arr_smear, 0.5)
        self.assertGreater(b_val, 20.0)
        self.assertIn("Başarılı", msg)


if __name__ == "__main__":
    unittest.main()
