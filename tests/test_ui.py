"""
Test Suite for OncoVision AI UI Architecture and Components
TÜBİTAK 2209-A Standartlarına Uygun Arayüz Doğrulama Testleri
"""

import unittest
import gradio as gr
from ui.styles import (
    build_disclaimer_banner_html,
    build_masthead_html,
    build_kpi_cards_html,
    build_xai_summary_card_html,
    build_academic_kpi_html,
    build_cost_summary_html,
    build_cf_summary_html,
    build_review_status_html
)
from ui.static_cache import get_static_resources
from ui.handlers import (
    handle_scan_directory,
    handle_case_comparison_ui,
    handle_pgx_calc,
    PREDEFINED_CASES
)
import app


class TestUIModule(unittest.TestCase):
    def test_html_style_builders(self):
        """HTML ve kart üreticilerinin doğru etiket ve içerik ürettiğini test eder."""
        banner = build_disclaimer_banner_html()
        self.assertIn("AKADEMİK ARAŞTIRMA UYARISI", banner)
        self.assertIn("disclaimer-banner", banner)

        masthead = build_masthead_html()
        self.assertIn("OncoVision AI", masthead)
        self.assertIn("oc-masthead", masthead)

        kpis = build_kpi_cards_html("AML", False, "['AML']", "Stabil Normal", "Adverse")
        self.assertIn("AML", kpis)
        self.assertIn("Erişkin", kpis)
        self.assertIn("Conformal Küme", kpis)

        xai = build_xai_summary_card_html("FLT3 pozitif katkı", 0.12, "AML with FLT3", "Midostaurin")
        self.assertIn("FLT3 pozitif katkı", xai)
        self.assertIn("Midostaurin", xai)

        academic = build_academic_kpi_html(
            {"Dengeli Doğruluk (Balanced Acc)": "%98.5", "Matthews Korelasyon Katsayısı (MCC)": "0.96", "Cohen's Kappa (κ)": "0.95"},
            {"Hayati Yanlış Negatifler (Kanser Kaçırma)": 0}
        )
        self.assertIn("%98.5", academic)

        cost = build_cost_summary_html({"Maliyet Cezası Çarpanı": "10x", "Hayati Yanlış Negatifler (Kanser Kaçırma)": 0, "Yanlış Pozitifler (Gereksiz Biyopsi)": 1, "Ağırlıklı Klinik Maliyet Skoru": 1.0})
        self.assertIn("10x", cost)

        cf = build_cf_summary_html(50, 24.5)
        self.assertIn("%50 Baskılama", cf)
        self.assertIn("+24.5%", cf)

        review = build_review_status_html("DR-999", "Onaylandı")
        self.assertIn("DR-999", review)
        self.assertIn("Onaylandı", review)

    def test_static_resource_manager(self):
        """Statik kaynak yöneticisinin tekil çalıştığını ve grafikleri ürettiğini test eder."""
        sr = get_static_resources()
        self.assertIsNotNone(sr.mm)
        self.assertIsNotNone(sr.engine)
        self.assertIsNotNone(sr.pca)
        self.assertEqual(len(sr.exp_var), 3)
        self.assertFalse(sr.df_benchmarks.empty)
        self.assertFalse(sr.df_table_one.empty)
        self.assertIsNotNone(sr.fig_roc)
        self.assertIsNotNone(sr.fig_dca)

    def test_ui_handlers(self):
        """Arayüz olay yöneticilerinin beklendiği gibi çalıştığını test eder."""
        upd, file_map = handle_scan_directory(".")
        self.assertIsInstance(file_map, dict)

        fig, diff_df, card = handle_case_comparison_ui(
            list(PREDEFINED_CASES.keys())[0],
            list(PREDEFINED_CASES.keys())[1]
        )
        self.assertIsNotNone(fig)
        self.assertFalse(diff_df.empty)

        card_html, dosing_df = handle_pgx_calc("Normal Metabolizer (*1/*1)", "Normal Metabolizer (*1/*1)", "Normal Metabolizer (*1/*1)", "ALL İdame (6-MP + MTX)")
        self.assertIn("Fizyolojik Toksisite", card_html)
        self.assertFalse(dosing_df.empty)

    def test_app_blocks_layout(self):
        """Ana uygulamanın (demo) geçerli bir Blocks örneği olduğunu test eder."""
        self.assertIsInstance(app.demo, gr.Blocks)
        self.assertEqual(app.demo.title, "OncoVision AI — TÜBİTAK 2209-A")


if __name__ == "__main__":
    unittest.main()
