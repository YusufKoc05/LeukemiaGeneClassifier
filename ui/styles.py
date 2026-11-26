"""
OncoVision AI — Arayüz Tasarım Sistemi, Stiller ve HTML Kart Üreticileri
TÜBİTAK 2209-A Klinik Karar Destek Platformu

Kurumsal ve medikal slate tasarım standartlarına uygun CSS ve HTML bileşenleri.
"""

from typing import Dict, Any, Optional
from core.config import DISCLAIMER_TR

# Medikal Slate Teması (CSS)
CUSTOM_SLATE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    color-scheme: light !important;
    --oc-bg: #f8fafc;
    --oc-panel: #ffffff;
    --oc-border: #e2e8f0;
    --oc-border-focus: #94a3b8;
    --oc-ink: #0f172a;
    --oc-muted: #64748b;
    --oc-subtle: #94a3b8;
    --oc-accent: #881337;
    --oc-accent-dark: #701a31;
    --oc-accent-subtle: #fdf2f4;
    --oc-danger: #b91c1c;
    --oc-danger-bg: #fef2f2;
    --oc-success: #047857;
    --oc-success-bg: #ecfdf5;
    --oc-warning: #b45309;
    --oc-warning-bg: #fffbeb;
}

body, .gradio-container {
    background-color: var(--oc-bg) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--oc-ink) !important;
    line-height: 1.5 !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    letter-spacing: -0.015em !important;
    color: var(--oc-ink) !important;
    font-weight: 600 !important;
}

/* Masthead & Başlık Alanı */
.oc-masthead {
    background: var(--oc-panel);
    border: 1px solid var(--oc-border);
    border-top: 3px solid var(--oc-accent-dark);
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}

.disclaimer-banner {
    background-color: var(--oc-danger-bg);
    border-left: 3px solid var(--oc-danger);
    padding: 6px 12px;
    font-size: 11.5px;
    color: var(--oc-danger);
    font-weight: 500;
    margin-bottom: 10px;
    border-radius: 3px;
    letter-spacing: 0.01em;
}

/* Tab Yapısı */
.tabs > .tab-nav {
    border-bottom: 1px solid var(--oc-border) !important;
    background: transparent !important;
    gap: 4px !important;
}

.tabs button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--oc-muted) !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease-in-out !important;
}

.tabs button:hover {
    color: var(--oc-ink) !important;
    background: #f1f5f9 !important;
}

.tabs button.selected {
    color: var(--oc-accent-dark) !important;
    border-bottom: 2px solid var(--oc-accent-dark) !important;
    background: #ffffff !important;
    font-weight: 600 !important;
}

/* Düğmeler */
.gr-button-primary, .gradio-container button.primary {
    background: var(--oc-accent-dark) !important;
    color: #ffffff !important;
    border-radius: 5px !important;
    border: none !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: background 0.15s ease !important;
}

.gr-button-primary:hover, .gradio-container button.primary:hover {
    background: var(--oc-accent) !important;
}

.gr-button-secondary, .gradio-container button.secondary {
    background: #ffffff !important;
    border: 1px solid var(--oc-border) !important;
    color: var(--oc-ink) !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
}

.gr-button-secondary:hover, .gradio-container button.secondary:hover {
    background: #f8fafc !important;
    border-color: var(--oc-border-focus) !important;
}

/* Akordeon ve Kartlar */
.gr-accordion {
    border: 1px solid var(--oc-border) !important;
    border-radius: 6px !important;
    background: #ffffff !important;
    margin-bottom: 8px !important;
}

.gr-accordion > .label-wrap {
    padding: 8px 12px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: var(--oc-ink) !important;
}
"""


def build_disclaimer_banner_html() -> str:
    """Akademik yasal uyarı şeridini üretir."""
    return f"<div class='disclaimer-banner'>AKADEMİK ARAŞTIRMA UYARISI — {DISCLAIMER_TR}</div>"


def build_masthead_html() -> str:
    """Üst bilgi başlık kartını üretir (Sade ve akademik tasarım)."""
    return """
    <div class="oc-masthead" style="border-top: none; padding: 12px 18px; margin-bottom: 10px;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; text-transform: uppercase; color: #475569; font-weight: 600; letter-spacing: 0.04em;">TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projesi</div>
        <h1 style="font-size: 19px; margin: 3px 0 2px 0; color: #0f172a; font-weight: 600;">OncoVision AI — Lösemi Gen Ekspresyonu ve Klinik Biyobelirteç Sınıflandırma Sistemi</h1>
        <div style="font-size: 12px; color: #64748b; line-height: 1.4;">Transkriptomik ve Hematolojik Verilerle Akut Lösemi (ALL / AML) Tanı ve Karar Destek Modeli</div>
    </div>
    """


def build_kpi_cards_html(
    pred_class: str,
    is_pediatric: bool,
    conformal_set_str: str,
    ood_status: str,
    risk_strat: str
) -> str:
    """Çıkarım sonuçlarını özetleyen 4'lü üst KPI kartını üretir."""
    cohort_badge = "Pediatrik" if is_pediatric else "Erişkin"
    
    # Renk atamaları
    ood_color = "#047857" if "Stabil" in ood_status or "Güvenli" in ood_status or "Normal" in ood_status else "#b91c1c"
    
    return f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; margin-bottom: 8px;'>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Klinik Ön Tanı ({cohort_badge})</div>
            <div style='font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 2px;'>{pred_class}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>%95 Conformal Küme</div>
            <div style='font-size: 13px; font-weight: 600; color: #0f172a; margin-top: 2px;'>{conformal_set_str}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid {ood_color}; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>OOD Anomali Kontrolü</div>
            <div style='font-size: 12px; font-weight: 600; color: #0f172a; margin-top: 2px;'>{ood_status}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Sitogenetik & Risk</div>
            <div style='font-size: 12px; font-weight: 600; color: #0f172a; margin-top: 2px;'>{risk_strat}</div>
        </div>
    </div>
    """


def build_xai_summary_card_html(
    shap_summary_text: str,
    ood_score: float,
    who_entity_name: str,
    therapy_plan: str
) -> str:
    """XAI açıklanabilirlik, WHO taksonomisi ve kılavuz protokol özet kartı üretir."""
    return f"""
    <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 10px 14px; margin-bottom: 10px;'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: center;'>
            <div>
                <div style='font-size: 10px; font-weight: 700; color: #701a31; font-family: "IBM Plex Mono", monospace; text-transform: uppercase; letter-spacing: 0.05em;'>XAI Karar Çıkarımı & İstatistiksel Uyumluluk</div>
                <div style='font-size: 12px; color: #0f172a; margin-top: 3px; line-height: 1.4;'>
                    {shap_summary_text} <span style='color:#64748b; margin-left:4px;'>(OOD Skoru: <code>{ood_score:.3f}</code>)</span>
                </div>
                <div style='font-size: 11px; color: #64748b; margin-top: 4px;'>DSÖ (WHO 2022) / ICC: <b style='color:#0f172a;'>{who_entity_name}</b></div>
            </div>
            <div style='border-left: 1px solid #f1f5f9; padding-left: 14px;'>
                <div style='font-size: 10px; font-weight: 700; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase; letter-spacing: 0.05em;'>Kılavuz Tedavi & Takip Protokolü</div>
                <div style='font-size: 12px; font-weight: 600; color: #0f172a; margin-top: 3px; line-height: 1.4;'>
                    <code>{therapy_plan}</code>
                </div>
            </div>
        </div>
    </div>
    """


def build_academic_kpi_html(macro_summary: Dict[str, Any], cost_loss_dict: Dict[str, Any]) -> str:
    """Akademik biyoistatistiksel performans gösterge kartları."""
    return f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-bottom: 10px;'>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Dengeli Doğruluk</div>
            <div style='font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 2px;'>{macro_summary.get('Dengeli Doğruluk (Balanced Acc)', 'N/A')}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Matthews Korelasyon (MCC)</div>
            <div style='font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 2px;'>{macro_summary.get('Matthews Korelasyon Katsayısı (MCC)', 'N/A')}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #701a31; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Cohen's Kappa (κ)</div>
            <div style='font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 2px;'>{macro_summary.get("Cohen's Kappa (κ)", 'N/A')}</div>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #047857; border-radius: 5px; padding: 8px 12px;'>
            <div style='font-size: 10px; font-weight: 600; color: #64748b; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>Hayati Kaçırma (FN)</div>
            <div style='font-size: 16px; font-weight: 700; color: #047857; margin-top: 2px;'>{cost_loss_dict.get('Hayati Yanlış Negatifler (Kanser Kaçırma)', 0)} Vaka</div>
        </div>
    </div>
    """


def build_cost_summary_html(cost_loss_dict: Dict[str, Any]) -> str:
    """Asimetrik klinik kayıp özeti kartı."""
    return f"""
    <div style='background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid #b45309; border-radius:5px; padding:10px 14px; margin-bottom:10px;'>
        <div style='font-size:12px; color:#0f172a; line-height:1.4;'>
            <b>Asimetrik Klinik Maliyet Prensibi:</b> Tıpta Yanlış Negatif (FN: Kanserli hastaya sağlıklı demek) hayatı tehdit ettiği için <b>10 kat ({cost_loss_dict.get('Maliyet Cezası Çarpanı', '10x')})</b> cezalandırılmıştır.<br/>
            • Hayati Yanlış Negatif: <b>{cost_loss_dict.get('Hayati Yanlış Negatifler (Kanser Kaçırma)', 0)}</b> | Yanlış Pozitif (Gereksiz Biyopsi): <b>{cost_loss_dict.get('Yanlış Pozitifler (Gereksiz Biyopsi)', 0)}</b> | Ağırlıklı Klinik Zarar Skoru: <b>{cost_loss_dict.get('Ağırlıklı Klinik Maliyet Skoru', 0)}</b>
        </div>
    </div>
    """


def build_cf_summary_html(intensity_pct: int, delta_healthy: float) -> str:
    """Karşıolgusal simülasyon özet kartı."""
    delta_sign = '+' if delta_healthy >= 0 else ''
    color = "#047857" if delta_healthy >= 0 else "#b91c1c"
    return f"""
    <div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:8px;'>
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:5px; padding:8px 12px;'>
            <div style='font-size:10px; color:#64748b; font-family:"IBM Plex Mono", monospace; text-transform:uppercase; font-weight:600;'>Müdahale Düzeyi</div>
            <div style='font-size:15px; font-weight:700; color:#0f172a; margin-top:2px;'>%{intensity_pct} Baskılama</div>
        </div>
        <div style='background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid {color}; border-radius:5px; padding:8px 12px;'>
            <div style='font-size:10px; color:#64748b; font-family:"IBM Plex Mono", monospace; text-transform:uppercase; font-weight:600;'>Fizyolojik Sınıfa Yakınsama</div>
            <div style='font-size:15px; font-weight:700; color:{color}; margin-top:2px;'>{delta_sign}{delta_healthy:.1f}%</div>
        </div>
    </div>
    """


def build_review_status_html(doctor_id: str, physician_decision: str) -> str:
    """Hekim inceleme kaydı onay bildirimi."""
    return f"""
    <div style='background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid #047857; border-radius:5px; padding:8px 12px; color:#0f172a;'>
        <b>Konsültasyon Kaydedildi:</b> Protokol kaydı tamamlandı. Hekim: <code>{doctor_id}</code> | Karar: <code>{physician_decision}</code>
    </div>
    """
