"""
OncoVision AI — Arayüz Olay Yöneticileri (Event Handlers)
TÜBİTAK 2209-A Klinik Karar Destek Platformu

Tüm kullanıcı etkileşimlerini, çıkarım boru hatlarını ve doğrulama akışlarını
endüstri standartlarında hata yönetimi (try-except) ve tip güvenliği ile yönetir.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from core.config import THRESHOLDS, DEFAULT_DATASET_DIR
from core.models import get_model_manager
from core.inference import get_inference_engine
from core.reporting import generate_clinical_pdf_report, generate_fhir_bundle
from core.visualizations import (
    COLOR_MAP,
    apply_plot_theme,
    compute_radar_profile,
    plot_confidence_gauge,
    plot_ppi_network,
    compute_pathway_enrichment,
    simulate_longitudinal_mrd,
    plot_dynamic_survival_curve
)
from core.evaluation import (
    list_external_datasets,
    load_dataset_preview,
    evaluate_external_dataset,
    build_cross_dataset_comparison_table,
    build_cross_dataset_plots,
    build_cross_dataset_kpis_html
)
from core.golub_parser import parse_and_unify_golub
from core.geo_bridge import fetch_geo_cohort
from core.pharmacogenomics import evaluate_pharmacogenomic_profile, PGX_GENE_OPTIONS
from core.who_subtypes import classify_who_2022_subtypes
from core.case_comparison import build_patient_comparison_view
from core.vision_classifier import classify_and_count_blood_smear

from .styles import (
    build_kpi_cards_html,
    build_xai_summary_card_html,
    build_cf_summary_html,
    build_review_status_html
)
from .static_cache import get_static_resources

# Önceden Tanımlanmış Karşılaştırma ve Şablon Vakaları
PREDEFINED_CASES = {
    "Vaka: AML (FLT3+, Blast %65)": {
        "name": "AML Vaka (FLT3+)", "age": 58, "runx1": 11.8, "flt3": 12.5, "pax5": 6.0,
        "ikzf1": 5.0, "cd19": 5.5, "wt1": 11.2, "wbc": 95.0, "hb": 6.2, "plt_val": 20.0,
        "blasts": 65.0, "ldh": 1100.0, "cytogenetics": "Karmaşık Karyotip (>=3 Anomali, Adverse)",
        "pred_class": "AML", "confidence": 98.4
    },
    "Vaka: APL & DIC (PLT 18K)": {
        "name": "APL Vaka (DIC)", "age": 34, "runx1": 8.2, "flt3": 6.5, "pax5": 6.2,
        "ikzf1": 6.5, "cd19": 6.2, "wt1": 7.8, "wbc": 25.0, "hb": 7.8, "plt_val": 18.0,
        "blasts": 60.0, "ldh": 750.0, "cytogenetics": "t(15;17) PML::RARA (APL)",
        "pred_class": "AML", "confidence": 95.1
    },
    "Vaka: Pediatrik ALL (7 Yaş)": {
        "name": "Pediatrik ALL (7 Yaş)", "age": 7, "runx1": 6.2, "flt3": 6.2, "pax5": 12.0,
        "ikzf1": 6.5, "cd19": 12.5, "wt1": 6.2, "wbc": 65.0, "hb": 8.2, "plt_val": 45.0,
        "blasts": 70.0, "ldh": 620.0, "cytogenetics": "t(12;21) ETV6::RUNX1 (Pediatrik Favorable)",
        "pred_class": "ALL", "confidence": 99.2
    },
    "Vaka: Ph+ ALL (WBC 120K)": {
        "name": "Ph+ ALL (BCR::ABL1)", "age": 52, "runx1": 6.5, "flt3": 6.5, "pax5": 11.8,
        "ikzf1": 2.0, "cd19": 12.5, "wt1": 6.2, "wbc": 120.0, "hb": 6.5, "plt_val": 18.0,
        "blasts": 85.0, "ldh": 1200.0, "cytogenetics": "t(9;22) BCR::ABL1 (Ph+ Yüksek Risk)",
        "pred_class": "ALL", "confidence": 98.8
    },
    "Vaka: Sağlıklı Donör": {
        "name": "Sağlıklı Donör", "age": 28, "runx1": 6.5, "flt3": 6.5, "pax5": 6.5,
        "ikzf1": 6.5, "cd19": 6.5, "wt1": 6.5, "wbc": 6.8, "hb": 14.2, "plt_val": 260.0,
        "blasts": 0.5, "ldh": 180.0, "cytogenetics": "Normal Karyotip (Orta Risk)",
        "pred_class": "Healthy", "confidence": 99.7
    }
}


def handle_master_analysis(
    age: float,
    runx1: float,
    flt3: float,
    pax5: float,
    ikzf1: float,
    cd19: float,
    wt1: float,
    wbc: float,
    hb: float,
    plt_val: float,
    blasts: float,
    ldh: float,
    cytogenetics: str,
    mpo: float = 6.5,
    cd34: float = 6.5,
    progress: gr.Progress = gr.Progress()
) -> Tuple:
    """
    Ana Klinik Analiz Boru Hattı:
    Hastanın biyobelirteç, hematoloji ve genetik verilerini işler;
    Random Forest, SHAP, Conformal Prediction, 3D PCA, ve kılavuz raporlarını üretir.
    """
    progress(0.1, desc="Vektörler işleniyor...")
    static_res = get_static_resources()
    mm = static_res.mm
    engine = static_res.engine

    try:
        res = engine.predict_patient(
            age=age, runx1=runx1, flt3=flt3, pax5=pax5, ikzf1=ikzf1, cd19=cd19, wt1=wt1,
            wbc=wbc, hb=hb, plt_val=plt_val, blasts=blasts, ldh=ldh, cytogenetics=cytogenetics,
            mpo=mpo, cd34=cd34
        )
    except Exception as exc:
        err_msg = f"<div style='color:#b91c1c; padding:8px;'>Hata oluştu: {str(exc)}</div>"
        return (err_msg, None, err_msg, {}, f"Analiz Hatası: {str(exc)}", None, None, None, None, None, None, pd.DataFrame(), None, None, "Hata", None, None, None)

    pred_class = res["final_clinical_class"]
    raw_pred_class = res["raw_model_prediction"]
    confidence = res["confidence_percentage"]
    conformal_set_str = res["conformal_set_str"]
    ood_status = res["ood_status"]
    ood_score = res["ood_score"]
    risk_strat = res["risk_stratification"]
    shap_summary_text = res["shap_summary_text"]
    sample_shap = res["sample_shap"]
    top_indices = res["top_indices"]
    input_scaled = res["input_scaled"]
    input_df = res["input_df"]
    is_pediatric = age < 18
    is_safety_override = res["safety_net_triggered"]
    dic_alert = res["dic_coagulopathy_alert"]
    therapy_plan = res["actionable_protocol"]
    prob_dict = res["class_probabilities"]
    blast_limit = THRESHOLDS.get("who_blast_acute_threshold", 20.0)

    # Lokal SHAP Grafiği
    colors_shap = ['#701a31' if v > 0 else '#64748b' for v in sample_shap[top_indices]]
    fig_shap = go.Figure(go.Bar(
        x=sample_shap[top_indices][::-1], y=input_df.columns[top_indices][::-1], orientation='h',
        marker=dict(color=colors_shap[::-1]),
        hovertemplate="<b>%{y}</b><br>SHAP Katkısı: %{x:.3f}<extra></extra>"
    ))
    fig_shap = apply_plot_theme(fig_shap, f"Lokal XAI (SHAP) Biyobelirteç Katkısı ({raw_pred_class})")

    # 3D PCA Hasta Projeksiyonu (Önbellekten klonlanarak anında üretilir)
    patient_pca = static_res.pca.transform(input_scaled)[0]
    fig_pca = go.Figure(static_res.fig_pca_base)
    fig_pca.add_trace(go.Scatter3d(
        x=[patient_pca[0]], y=[patient_pca[1]], z=[patient_pca[2]],
        mode='markers+text',
        marker=dict(size=11, color='#b45309', symbol='diamond', line=dict(color='#0f172a', width=1.5)),
        name='Aktif Hasta', text=['AKTİF HASTA'], textposition='top center',
        hovertemplate="<b>AKTİF HASTA</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
    ))

    # Hematoloji Kutu Grafikleri (Önbellekten klonlanan kohort dağılımı + dinamik hasta çizgileri)
    cbc_params = [
        ('WBC_x10k', wbc, 'WBC (x10³/µL)', 'x10³/µL'),
        ('Hemoglobin_g_dL', hb, 'Hb (g/dL)', 'g/dL'),
        ('PLT_x10k', plt_val, 'PLT (x10³/µL)', 'x10³/µL'),
        ('Blast_Percentage', blasts, 'Blast (%)', '%'),
        ('LDH_U_L', ldh, 'LDH (U/L)', 'U/L')
    ]
    fig_cbc = go.Figure(static_res.fig_cbc_base)
    for idx, (col_name, user_val, title_text, unit_str) in enumerate(cbc_params, start=1):
        min_cohort, max_cohort = static_res.cbc_stats.get(col_name, (0.0, 100.0))
        safe_user_val = float(np.nan_to_num(user_val, nan=min_cohort))

        # Dinamik Eksen Ölçeklendirmesi (Zero-baseline & Taşan Eksen Koruması)
        y_min = 0.0 if min_cohort >= 0 else float(min(min_cohort, safe_user_val)) * 1.08
        y_max = float(max(max_cohort, safe_user_val)) * 1.18
        if y_max <= y_min:
            y_max = y_min + 1.0

        # Aktif Hasta Yatay Eşik Çizgisi ve Etiketi
        fig_cbc.add_hline(
            y=safe_user_val, line_dash="dash", line_color="#701a31", line_width=2.2,
            annotation_text=f"Aktif: {safe_user_val:.1f}",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#701a31", family="IBM Plex Mono"),
            row=1, col=idx
        )

        fig_cbc.update_yaxes(
            range=[y_min, y_max], row=1, col=idx,
            gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#cbd5e1"
        )
        fig_cbc.update_xaxes(
            row=1, col=idx,
            tickfont=dict(size=9, family="IBM Plex Sans"),
            showgrid=False
        )

    # Ekspresyon Isı Haritası (Önbellekten çekilen kohort ortalamaları)
    gene_panel = ['RUNX1', 'FLT3', 'WT1', 'PAX5', 'CD19', 'IKZF1']
    patient_row = pd.DataFrame(input_df[gene_panel].values, index=['AKTİF HASTA'], columns=gene_panel)
    fig_heatmap = px.imshow(
        pd.concat([static_res.cohort_heatmap_means, patient_row]), x=gene_panel, color_continuous_scale="Reds",
        title="Gen Ekspresyon Isı Haritası", labels=dict(color="log2 TPM")
    )
    fig_heatmap = apply_plot_theme(fig_heatmap, "Gen Ekspresyon Isı Haritası")

    fig_radar = compute_radar_profile(runx1, flt3, pax5, ikzf1, cd19, wt1, blasts)
    fig_pathway = compute_pathway_enrichment(runx1, flt3, pax5, ikzf1, cd19, wt1)
    fig_mrd = simulate_longitudinal_mrd(pred_class, cytogenetics, blasts)
    fig_km = plot_dynamic_survival_curve(pred_class, risk_strat, cytogenetics, hb, blasts, ldh, age)
    fig_gauge = plot_confidence_gauge(confidence, raw_pred_class, is_safety_override)
    fig_ppi = plot_ppi_network(pred_class)
    df_synergy = engine.get_drug_synergy_matrix(pred_class, cytogenetics, age)

    early_dx_badge = f"Akut Lösemi Evresi (Blast ≥ %{blast_limit})" if blasts >= blast_limit else "Fizyolojik Dağılım"
    pdf_out = generate_clinical_pdf_report(
        pred_class=pred_class, confidence=confidence, risk_strat=risk_strat,
        cytogenetics=cytogenetics, conformal_set_str=conformal_set_str, ood_status=ood_status,
        uncertainty_str=res["uncertainty_str"], early_dx_badge=early_dx_badge,
        is_safety_override=is_safety_override, shap_summary_text=shap_summary_text,
        wbc=wbc, hb=hb, plt_val=plt_val, blasts=blasts, ldh=ldh, age=age, dic_alert=dic_alert,
        output_path="klinik_epikriz_raporu.pdf",
        mpo=mpo, cd34=cd34,
        genes_dict={"RUNX1": runx1, "FLT3": flt3, "WT1": wt1, "PAX5": pax5, "CD19": cd19, "IKZF1": ikzf1},
        actionable_protocol=therapy_plan
    )
    fhir_out = generate_fhir_bundle(
        pred_class=pred_class, confidence=confidence, risk_strat=risk_strat,
        cytogenetics=cytogenetics, conformal_set_str=conformal_set_str, ood_status=ood_status,
        wbc=wbc, hb=hb, plt_val=plt_val, blasts=blasts, ldh=ldh, age=age,
        output_path="fhir_diagnostic_report.json"
    )

    kpi_cards_html = build_kpi_cards_html(pred_class, is_pediatric, conformal_set_str, ood_status, risk_strat)

    # WHO 2022 / ICC Taksonomi sınıflandırması
    who_res = classify_who_2022_subtypes(
        pred_class=raw_pred_class, cytogenetics=cytogenetics, flt3_val=flt3,
        cd19_val=cd19, pax5_val=pax5, runx1_val=runx1, age=age
    )
    who_entity_name = who_res["who_entity"]
    xai_summary_card_html = build_xai_summary_card_html(shap_summary_text, ood_score, who_entity_name, therapy_plan)

    report_text = (
        f"TÜBİTAK 2209-A / ONCOVISION AI RAPORU\n"
        f"--------------------------------------\n"
        f"Ön Tanı: {pred_class} (Güven: %{confidence:.1f})\n"
        f"DSÖ (WHO 2022) Taksonomi: {who_entity_name}\n"
        f"Kohort: {'Pediatrik' if is_pediatric else 'Erişkin'} ({age} Yaş)\n"
        f"%95 Conformal Tahmin Kümesi: {conformal_set_str}\n"
        f"OOD Anomali Durumu: {ood_status} (Skor: {ood_score:.3f})\n"
        f"Sitogenetik: {cytogenetics}\n"
        f"Risk Grubu: {risk_strat}\n\n"
        f"Hematoloji: WBC={wbc}, Hb={hb}, PLT={plt_val}, Blast=%{blasts}, LDH={ldh}"
    )

    progress(1.0, desc="Tamamlandı!")
    return (
        kpi_cards_html, fig_gauge, xai_summary_card_html, prob_dict, report_text,
        fig_pca, fig_cbc, fig_shap, fig_heatmap, fig_km, fig_ppi,
        df_synergy, pdf_out, fhir_out, pred_class, fig_pathway, fig_mrd, fig_radar
    )


def handle_cf_simulation(
    age: float, runx1: float, flt3: float, pax5: float, ikzf1: float, cd19: float, wt1: float,
    wbc: float, hb: float, plt_val: float, blasts: float, ldh: float, cytogenetics: str,
    mpo: float, cd34: float, target_intervention: str, intensity_pct: int
) -> Tuple[go.Figure, str]:
    """Karşıolgusal (Counterfactual) tedavi müdahale simülasyonunu çalıştırır."""
    static_res = get_static_resources()
    try:
        orig_probs, sim_probs, delta_healthy = static_res.engine.run_counterfactual_simulation(
            base_age=age, base_runx1=runx1, base_flt3=flt3, base_pax5=pax5,
            base_ikzf1=ikzf1, base_cd19=cd19, base_wt1=wt1, base_wbc=wbc,
            base_hb=hb, base_plt=plt_val, base_blasts=blasts, base_ldh=ldh,
            cytogenetics=cytogenetics, target_intervention=target_intervention,
            intensity_pct=intensity_pct
        )
    except Exception as exc:
        empty_fig = go.Figure()
        return empty_fig, f"<div style='color:#b91c1c;'>Simülasyon hatası: {str(exc)}</div>"

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=static_res.mm.classes, y=orig_probs * 100, name="Bazal Profil (Mevcut)", marker_color="#94a3b8",
        hovertemplate="Bazal: %{y:.1f}%<extra></extra>"
    ))
    fig_cf.add_trace(go.Bar(
        x=static_res.mm.classes, y=sim_probs * 100, name="Müdahale Sonrası (Simüle)", marker_color="#701a31",
        hovertemplate="Simülasyon: %{y:.1f}%<extra></extra>"
    ))
    fig_cf = apply_plot_theme(fig_cf, f"Karşıolgusal Karşılaştırma: {target_intervention}")
    fig_cf.update_layout(yaxis_title="Olasılık Dağılımı (%)", barmode='group')

    summary_card_html = build_cf_summary_html(intensity_pct, delta_healthy)
    return fig_cf, summary_card_html


def handle_batch_screening(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[go.Figure], Optional[go.Figure], str]:
    """Kohort düzeyinde toplu tarama, triyaj ve Wasserstein kayma analizi."""
    if file_obj is None:
        return None, None, None, "CSV dosyası seçilmedi veya yüklenmedi."

    static_res = get_static_resources()
    try:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        batch_df = pd.read_csv(path)
        if batch_df.empty:
            return None, None, None, "Yüklenen dosya boş."

        res_df, df_drift, triage_counts = static_res.engine.run_batch_screening(batch_df)

        fig_drift = px.bar(
            df_drift, x='Parametre', y='Wasserstein Mesafesi', color='Popülasyon Kayma Durumu',
            color_discrete_map={"Belirgin Kayma": "#701a31", "Hafif Kayma": "#b45309", "Stabil": "#047857"},
            title="Kohort Düzeyi Wasserstein Popülasyon Kayması (Drift)"
        )
        fig_drift = apply_plot_theme(fig_drift, "Kohort Düzeyi Wasserstein Popülasyon Kayması (Drift)")

        fig_triage = px.pie(
            names=list(triage_counts.keys()), values=list(triage_counts.values()),
            color=list(triage_counts.keys()),
            color_discrete_map={"Acil (Kırmızı)": "#701a31", "Öncelikli (Sarı)": "#b45309", "Rutin (Yeşil)": "#047857"},
            title="Kohort Triyaj Dağılımı", hole=0.45
        )
        fig_triage = apply_plot_theme(fig_triage, "Kohort Triyaj Dağılımı")

        status_text = f"Toplam {len(res_df)} örnek başarıyla tarandı; conformal kümeleri ve Wasserstein kayma metrikleri hesaplandı."
        return res_df, fig_drift, fig_triage, status_text
    except Exception as exc:
        return None, None, None, f"İşlem sırasında hata oluştu: {str(exc)}"


def handle_expert_review_save(
    doctor_id: str,
    patient_pred: str,
    physician_decision: str,
    clinical_override: str,
    additional_orders: List[str],
    notes: str
) -> str:
    """Uzman hekim inceleme kaydını yerel ve standartlaştırılmış JSON olarak saklar."""
    static_res = get_static_resources()
    try:
        record = static_res.engine.save_expert_review(
            doctor_id, patient_pred, physician_decision, clinical_override, additional_orders, notes
        )
        return build_review_status_html(record.get('doctor_id', doctor_id), physician_decision)
    except Exception as exc:
        return f"<div style='color:#b91c1c;'>Kayıt hatası: {str(exc)}</div>"


def handle_scan_directory(dir_path: str) -> Tuple[Any, Dict[str, str]]:
    """Belirtilen yerel dizindeki test veri setlerini tarar."""
    files = list_external_datasets(dir_path)
    choices = [name for name, _ in files]
    file_map = {name: path for name, path in files}
    val = choices[0] if choices else None
    return gr.update(choices=choices, value=val), file_map


def handle_dataset_selection(selected_name: Optional[str], uploaded_file: Any, file_map: Dict[str, str]) -> Tuple:
    """Açılır kutudan veya dosya yüklemeden seçilen veri setini önizler ve eşleştirir."""
    target_path = None
    if uploaded_file is not None:
        target_path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    elif selected_name and selected_name in file_map:
        target_path = file_map[selected_name]

    preview_df, info_msg, choices, auto_map = load_dataset_preview(target_path)
    return (
        preview_df,
        info_msg,
        gr.update(choices=choices, value=auto_map["target"]),
        gr.update(choices=choices, value=auto_map["sample_id"]),
        gr.update(choices=choices, value=auto_map["wbc"]),
        gr.update(choices=choices, value=auto_map["hb"]),
        gr.update(choices=choices, value=auto_map["plt"]),
        gr.update(choices=choices, value=auto_map["blasts"]),
        gr.update(choices=choices, value=auto_map["ldh"]),
    )


def handle_run_evaluation(
    selected_name: Optional[str],
    uploaded_file: Any,
    file_map: Dict[str, str],
    target_col: Optional[str],
    id_col: Optional[str],
    wbc_col: Optional[str],
    hb_col: Optional[str],
    plt_col: Optional[str],
    blast_col: Optional[str],
    ldh_col: Optional[str],
    assume_healthy: bool = False,
    progress: gr.Progress = gr.Progress()
) -> Tuple:
    """Harici veri setini model ve ensemble pipeline'ı ile toplu test eder."""
    progress(0.1, desc="Veri seti yükleniyor ve sütunlar inceleniyor...")
    target_path = None
    if uploaded_file is not None:
        target_path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    elif selected_name and selected_name in file_map:
        target_path = file_map[selected_name]

    if not target_path:
        return (
            "<div style='color:#701a31; padding:8px;'>Lütfen bir veri seti seçiniz veya yükleyiniz.</div>",
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            None
        )

    progress(0.35, desc="Model pipeline ölçeklendirmesi ve toplu çıkarım yapılıyor...")
    mm = get_model_manager()
    kpi_html, fig_cm, df_report, results_df, export_path = evaluate_external_dataset(
        file_path_or_buffer=target_path,
        target_col=target_col,
        wbc_col=wbc_col,
        hb_col=hb_col,
        plt_col=plt_col,
        blast_col=blast_col,
        ldh_col=ldh_col,
        sample_id_col=id_col,
        assume_healthy_control=assume_healthy,
        model_manager=mm
    )
    progress(1.0, desc="Değerlendirme tamamlandı!")
    return kpi_html, fig_cm, df_report, results_df, export_path


def handle_case_comparison_ui(case_a_key: str, case_b_key: str) -> Tuple[go.Figure, pd.DataFrame, str]:
    """İki hastanın biyobelirteçlerini çift profilli radarda yan yana karşılaştırır."""
    p_a = PREDEFINED_CASES.get(case_a_key, PREDEFINED_CASES["Vaka: Sağlıklı Donör"])
    p_b = PREDEFINED_CASES.get(case_b_key, PREDEFINED_CASES["Vaka: AML (FLT3+, Blast %65)"])
    fig, diff_df, card_html = build_patient_comparison_view(p_a, p_b)
    return fig, diff_df, card_html


def handle_pgx_calc(tpmt: str, nudt15: str, dpyd: str, regimen: str) -> Tuple[str, pd.DataFrame]:
    """Farmakogenomik toksisite ve CPIC doz modifikasyonlarını hesaplar."""
    res = evaluate_pharmacogenomic_profile(tpmt, nudt15, dpyd, regimen)
    return res["card_html"], res["dosing_df"]


def handle_one_click_blood_demo(progress: gr.Progress = gr.Progress()) -> Tuple:
    """1-Tık Kaggle 417 vakalık hemogram doğrulama demosu."""
    progress(0.2, desc="Kaggle 417 vakalık rutin hemogram veri seti yükleniyor...")
    mm = get_model_manager()
    p1 = os.path.join(DEFAULT_DATASET_DIR, "archive (1)", "blood_count_dataset.csv")
    kpi, fig, rep, res_df, path = evaluate_external_dataset(
        p1, wbc_col="White_Blood_Cells", hb_col="Hemoglobin", plt_col="Platelet_Count",
        assume_healthy_control=True, model_manager=mm
    )
    progress(1.0, desc="Tamamlandı!")
    return kpi, fig, rep, res_df, path


def handle_one_click_golub_demo(progress: gr.Progress = gr.Progress()) -> Tuple:
    """1-Tık Golub Science (1999) 7.129 genlik benchmark demosu."""
    progress(0.2, desc="Golub et al. (1999) Science 7.129 genlik veri seti transpoze ediliyor...")
    mm = get_model_manager()
    df_g, msg = parse_and_unify_golub()
    progress(0.6, desc="ALL vs AML transkriptomik sınıflandırması yapılıyor...")
    kpi, fig, rep, res_df, path = evaluate_external_dataset(
        "golub_benchmark_unified.csv",
        target_col="Leukemia_Subtype",
        wbc_col="WBC_x10k", hb_col="Hemoglobin_g_dL", plt_col="PLT_x10k",
        blast_col="Blast_Percentage", ldh_col="LDH_U_L", sample_id_col="Sample_ID",
        model_manager=mm
    )
    progress(1.0, desc="Tamamlandı!")
    return kpi, fig, rep, res_df, path


def handle_one_click_geo_demo(progress: gr.Progress = gr.Progress()) -> Tuple:
    """1-Tık NCBI GEO GSE13159 (MILE Kohortu) uluslararası doğrulama demosu."""
    progress(0.2, desc="NCBI GEO GSE13159 (MILE Study) kohortu entegre ediliyor...")
    mm = get_model_manager()
    df_geo, msg = fetch_geo_cohort("GSE13159", sample_size=40)
    progress(0.6, desc="Çok merkezli uluslararası kohort test ediliyor...")
    kpi, fig, rep, res_df, path = evaluate_external_dataset(
        df_geo,
        target_col="Target_Subtype",
        wbc_col="WBC_x10k", hb_col="Hemoglobin_g_dL", plt_col="PLT_x10k",
        blast_col="Blast_Percentage", ldh_col="LDH_U_L", sample_id_col="Sample_ID",
        model_manager=mm
    )
    progress(1.0, desc="Tamamlandı!")
    return kpi, fig, rep, res_df, path


def handle_switch_cross_view(view_name: str) -> go.Figure:
    """Tüm veri setleri karşılaştırma panelindeki seçili grafiği günceller."""
    static_res = get_static_resources()
    figs = static_res.cross_figs
    if "Sınıf Dağılımı" in str(view_name):
        return figs.get("distribution", go.Figure())
    elif "Balon" in str(view_name):
        return figs.get("bubble", go.Figure())
    elif "Biyobelirteç" in str(view_name):
        return figs.get("biomarkers", go.Figure())
    return figs.get("performance", go.Figure())


def handle_refresh_cross_benchmark(progress: gr.Progress = gr.Progress()) -> Tuple:
    """Tüm Kaggle ve benchmark veri setlerini yeniden değerlendirip karşılaştırma panelini günceller."""
    progress(0.2, desc="Tüm Kaggle ve benchmark kohortları taranıyor...")
    static_res = get_static_resources()
    df_comp = build_cross_dataset_comparison_table(static_res.mm)
    progress(0.6, desc="Karşılaştırmalı görselleştirmeler derleniyor...")
    figs = build_cross_dataset_plots(df_comp, static_res.mm)
    kpi_html = build_cross_dataset_kpis_html(df_comp)
    static_res.df_cross_comparison = df_comp
    static_res.cross_figs = figs
    static_res.cross_kpis_html = kpi_html
    progress(1.0, desc="Tamamlandı!")
    return kpi_html, figs.get("performance"), figs.get("distribution"), df_comp
