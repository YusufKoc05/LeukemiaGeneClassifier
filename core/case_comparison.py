"""
OncoVision AI — Klinik Vaka Karşılaştırma Modülü (Split-Screen Kıyaslama)
TÜBİTAK 2209-A Karşılaştırmalı Biyoistatistiksel Hasta Analizi

İki farklı hastanın hematolojik parametrelerini, transkriptomik belirteçlerini
ve moleküler risk profillerini yan yana koyup çift eksenli radar ve delta fark analizi yapar.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def build_patient_comparison_view(patient_a: dict, patient_b: dict) -> tuple[go.Figure, pd.DataFrame, str]:
    """
    İki hastanın parametrelerini kıyaslayan çift profilli Radar Grafiği,
    fark (delta) tablosu ve özet klinik HTML kartı üretir.
    """
    labels = ["Lökosit (WBC)", "Hemoglobin (Hb)", "Trombosit (PLT)", "Blast Oranı (%)", "LDH Düzeyi", "FLT3 (AML)", "CD19 (ALL)"]
    
    # Normalize edilmiş radar değerleri (0-100 ölçeği)
    def to_radar_scale(p):
        wbc = min(100, (p.get("wbc", 6.8) / 150.0) * 100)
        hb = min(100, (p.get("hb", 14.0) / 18.0) * 100)
        plt_v = min(100, (p.get("plt_val", 250.0) / 500.0) * 100)
        blast = min(100, p.get("blasts", 0.5))
        ldh = min(100, (p.get("ldh", 180.0) / 1500.0) * 100)
        flt3 = min(100, (p.get("flt3", 6.5) / 15.0) * 100)
        cd19 = min(100, (p.get("cd19", 6.5) / 15.0) * 100)
        return [wbc, hb, plt_v, blast, ldh, flt3, cd19]

    vals_a = to_radar_scale(patient_a)
    vals_b = to_radar_scale(patient_b)
    
    # Radar grafiği kapatmak için ilk elemanı sona ekle
    labels_closed = labels + [labels[0]]
    vals_a_closed = vals_a + [vals_a[0]]
    vals_b_closed = vals_b + [vals_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a_closed, theta=labels_closed, fill='toself',
        name=patient_a.get("name", "Vaka A"),
        line_color="#2563eb", fillcolor="rgba(37, 99, 235, 0.2)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b_closed, theta=labels_closed, fill='toself',
        name=patient_b.get("name", "Vaka B"),
        line_color="#7a1f2c", fillcolor="rgba(122, 31, 44, 0.2)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            bgcolor="#ffffff"
        ),
        paper_bgcolor="#ffffff",
        title={"text": "<b>İki Vaka Arası Moleküler & Hematolojik Profil Kıyaslaması</b>", "font": {"size": 13, "color": "#17202b"}},
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", y=-0.1),
        height=380
    )

    # Parametre Fark Tablosu
    diff_rows = [
        {"Parametre": "WBC (x10³/µL)", "Hasta A": patient_a.get("wbc", 6.8), "Hasta B": patient_b.get("wbc", 6.8)},
        {"Parametre": "Hemoglobin (g/dL)", "Hasta A": patient_a.get("hb", 14.0), "Hasta B": patient_b.get("hb", 14.0)},
        {"Parametre": "Trombosit (x10³/µL)", "Hasta A": patient_a.get("plt_val", 250.0), "Hasta B": patient_b.get("plt_val", 250.0)},
        {"Parametre": "Blast Hücre Oranı (%)", "Hasta A": f"%{patient_a.get('blasts', 0.5):.1f}", "Hasta B": f"%{patient_b.get('blasts', 0.5):.1f}"},
        {"Parametre": "LDH (U/L)", "Hasta A": patient_a.get("ldh", 180.0), "Hasta B": patient_b.get("ldh", 180.0)},
        {"Parametre": "FLT3 Düzeyi", "Hasta A": patient_a.get("flt3", 6.5), "Hasta B": patient_b.get("flt3", 6.5)},
        {"Parametre": "CD19 Düzeyi", "Hasta A": patient_a.get("cd19", 6.5), "Hasta B": patient_b.get("cd19", 6.5)},
        {"Parametre": "Sitogenetik", "Hasta A": patient_a.get("cytogenetics", "-"), "Hasta B": patient_b.get("cytogenetics", "-")}
    ]
    diff_df = pd.DataFrame(diff_rows)

    card_html = f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px;'>
        <div style='background: #ffffff; border: 1px solid #dde1e7; border-left: 3px solid #2563eb; border-radius: 4px; padding: 10px 14px;'>
            <div style='font-size: 10px; font-weight: 600; color: #5c6675; font-family: "IBM Plex Mono", monospace;'>VAKA A ({patient_a.get("name", "Vaka 1")})</div>
            <div style='font-size: 15px; font-weight: 700; color: #2563eb; margin-top: 2px;'>{patient_a.get("pred_class", "Değerlendirilmedi")}</div>
            <div style='font-size: 11px; color: #5c6675; margin-top: 2px;'>Yaş: {patient_a.get("age", 45)} | Güven: %{patient_a.get("confidence", 95.0):.1f}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #dde1e7; border-left: 3px solid #7a1f2c; border-radius: 4px; padding: 10px 14px;'>
            <div style='font-size: 10px; font-weight: 600; color: #5c6675; font-family: "IBM Plex Mono", monospace;'>VAKA B ({patient_b.get("name", "Vaka 2")})</div>
            <div style='font-size: 15px; font-weight: 700; color: #7a1f2c; margin-top: 2px;'>{patient_b.get("pred_class", "Değerlendirilmedi")}</div>
            <div style='font-size: 11px; color: #5c6675; margin-top: 2px;'>Yaş: {patient_b.get("age", 45)} | Güven: %{patient_b.get("confidence", 95.0):.1f}</div>
        </div>
    </div>
    """

    return fig, diff_df, card_html
