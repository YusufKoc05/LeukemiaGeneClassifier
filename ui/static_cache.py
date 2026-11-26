"""
OncoVision AI — Statik Kaynak ve Biyoistatistik Önbellek Yöneticisi
TÜBİTAK 2209-A Akademik Raporlama ve Görselleştirme Hazırlığı
"""

from typing import Dict, Any, Tuple
from functools import lru_cache
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

from core import (
    get_model_manager,
    get_inference_engine,
    get_ensemble_classifier,
    build_static_evaluation_plots,
    calculate_comprehensive_metrics,
    calculate_decision_curve_analysis,
    generate_clinical_table_one,
    calculate_cost_sensitive_loss,
    generate_latex_table,
    plot_decision_curve_analysis,
    plot_precision_recall_curves,
    build_cross_dataset_comparison_table,
    build_cross_dataset_plots,
    build_cross_dataset_kpis_html
)


class StaticResourceManager:
    """
    Model eğitim sonrası istatistiksel raporları, biyo-parametreleri
    ve 3D PCA projeksiyon matrislerini tek seferlik hesaplar ve önbelleğe alır.
    """
    _instance = None

    def __init__(self):
        self.mm = get_model_manager()
        self.engine = get_inference_engine()

        # 3D PCA Projeksiyon Matrisi ve Temel Grafik (Tek seferlik çizim ile hızlandırılmıştır)
        import plotly.express as px
        COLOR_MAP = {'Healthy': '#047857', 'AML': '#7a1f2c', 'ALL': '#475569'}
        self.pca = PCA(n_components=3)
        X_pca = self.pca.fit_transform(self.mm.X_all_sc)
        self.exp_var = self.pca.explained_variance_ratio_ * 100
        self.pca_base_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        self.pca_base_df['Alt Tip'] = self.mm.y.values
        self.pca_base_df['Örnek ID'] = self.mm.df.index

        self.fig_pca_base = px.scatter_3d(
            self.pca_base_df, x='PC1', y='PC2', z='PC3', color='Alt Tip',
            color_discrete_map=COLOR_MAP, opacity=0.40
        )
        self.fig_pca_base.update_layout(
            title=f"3D PCA: Kohort Projeksiyonu (%{sum(self.exp_var):.1f} Varyans)",
            template="plotly_white", margin=dict(l=0, r=0, b=0, t=30),
            font=dict(family="IBM Plex Sans", size=11)
        )

        # Gen Isı Haritası Kohort Ortalamaları (Önbellek)
        self.gene_panel = ['RUNX1', 'FLT3', 'WT1', 'PAX5', 'CD19', 'IKZF1']
        self.cohort_heatmap_means = self.mm.df.groupby('Leukemia_Subtype')[self.gene_panel].mean()

        # Hematoloji Referans Kohort Kutu Grafiği Şablonu (Precomputed Base CBC)
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from core.visualizations import apply_plot_theme

        cbc_cols_meta = [
            ('WBC_x10k', 'WBC (x10³/µL)', 'x10³/µL'),
            ('Hemoglobin_g_dL', 'Hb (g/dL)', 'g/dL'),
            ('PLT_x10k', 'PLT (x10³/µL)', 'x10³/µL'),
            ('Blast_Percentage', 'Blast (%)', '%'),
            ('LDH_U_L', 'LDH (U/L)', 'U/L')
        ]
        self.cbc_meta = cbc_cols_meta
        self.cbc_stats = {}
        self.fig_cbc_base = make_subplots(
            rows=1, cols=5,
            subplot_titles=[p[1] for p in cbc_cols_meta],
            horizontal_spacing=0.06
        )
        for idx, (col_name, title_text, unit_str) in enumerate(cbc_cols_meta, start=1):
            col_series = self.mm.df[col_name].dropna()
            min_c = float(col_series.min()) if not col_series.empty else 0.0
            max_c = float(col_series.max()) if not col_series.empty else 100.0
            self.cbc_stats[col_name] = (min_c, max_c)
            for sub_type, color in COLOR_MAP.items():
                sub_vals = self.mm.df[self.mm.df['Leukemia_Subtype'] == sub_type][col_name].dropna()
                self.fig_cbc_base.add_trace(go.Box(
                    y=sub_vals, name=sub_type, marker_color=color, showlegend=(idx == 1),
                    boxpoints='outliers',
                    hovertemplate=f"<b>{sub_type}</b><br>{title_text}: %{{y:.2f}} {unit_str}<extra></extra>"
                ), row=1, col=idx)
        self.fig_cbc_base = apply_plot_theme(self.fig_cbc_base, "Hematoloji Dağılımları (CBC Referans Kohortu vs Aktif Hasta)", height=290)
        self.fig_cbc_base.update_layout(
            margin=dict(l=35, r=25, b=25, t=45),
            legend=dict(orientation="h", y=1.22, x=0.5, xanchor="center", font=dict(size=10))
        )

        # Çapraz Doğrulama ve Ensemble Benchmark Tabloları
        self.df_benchmarks = self.mm.get_benchmarks_df()
        try:
            self.df_ensemble_benchmarks = get_ensemble_classifier().evaluate_all_models()
        except Exception:
            self.df_ensemble_benchmarks = pd.DataFrame()

        # Table 1: Klinik Kohort Temel Karakteristikleri
        self.df_table_one = generate_clinical_table_one(self.mm.df)

        # Çok Değişkenli Biyoistatistik Metrikleri
        y_pred = self.mm.rf_model.predict(self.mm.X_test_sc)
        y_prob = self.mm.rf_model.predict_proba(self.mm.X_test_sc)
        self.df_comprehensive_metrics, self.macro_summary = calculate_comprehensive_metrics(
            self.mm.y_test, y_pred, y_prob, self.mm.classes
        )

        # Karar Eğrisi Analizi (DCA)
        self.df_dca = calculate_decision_curve_analysis(
            self.mm.y_test, y_prob, self.mm.classes, target_class="AML"
        )
        self.fig_dca = plot_decision_curve_analysis(self.df_dca)

        # PR Eğrileri ve Maliyet Kaybı
        y_test_bin = label_binarize(self.mm.y_test, classes=self.mm.classes)
        self.fig_pr = plot_precision_recall_curves(y_test_bin, y_prob, self.mm.classes)
        self.cost_loss_dict = calculate_cost_sensitive_loss(self.mm.y_test, y_pred)
        self.latex_report_code = generate_latex_table(
            self.df_comprehensive_metrics,
            caption="OncoVision AI Model Tanısal Performans Metrikleri (TÜBİTAK 2209-A)"
        )

        # Statik Değerlendirme Grafikleri (ROC, Calib, CM, Volcano, Bio)
        (
            self.fig_roc,
            self.fig_calib,
            self.fig_cm,
            self.fig_volcano,
            self.fig_bio_global
        ) = build_static_evaluation_plots(
            rf_model=self.mm.rf_model,
            X_test_sc=self.mm.X_test_sc,
            y_test=self.mm.y_test,
            X_all_sc=self.mm.X_all_sc,
            df=self.mm.df,
            GENE_COLS=self.mm.gene_cols,
            model_classes=self.mm.classes
        )

        # Tüm Kaggle ve Benchmark Veri Setleri Karşılaştırma Modülü (Önbellek)
        self.df_cross_comparison = build_cross_dataset_comparison_table(self.mm)
        self.cross_figs = build_cross_dataset_plots(self.df_cross_comparison, self.mm)
        self.cross_kpis_html = build_cross_dataset_kpis_html(self.df_cross_comparison)

    @classmethod
    def get_instance(cls) -> "StaticResourceManager":
        if cls._instance is None:
            cls._instance = StaticResourceManager()
        return cls._instance


def get_static_resources() -> StaticResourceManager:
    """Tekil (Singleton) statik kaynak yöneticisini döndürür."""
    return StaticResourceManager.get_instance()
