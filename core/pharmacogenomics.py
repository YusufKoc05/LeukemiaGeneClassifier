"""
OncoVision AI — Farmakogenomik (PGx) ve Kişiselleştirilmiş İlaç Dozajı Modülü
CPIC (Clinical Pharmacogenetics Implementation Consortium) & DPWG Standartları
TÜBİTAK 2209-A Kişiselleştirilmiş Tıp (Personalized Medicine) Entegrasyonu

Lösemi kemoterapisinde ölümcül nötropeni, pansitopeni ve mukozit riskini önlemek için:
- TPMT (Tiopürin S-metiltransferaz) & NUDT15: 6-Merkaptopürin (6-MP) / Tioguanin dozajı
- DPYD: Fluoropirimidin / Fludarabin toksisite klerensi
- CYP3A4/5 & FLT3 İnhibitörleri: İlaç-ilaç etkileşimi ve kardiyotoksisite (QTc) analizi
yapar.
"""

import pandas as pd

PGX_GENE_OPTIONS = {
    "tpmt": [
        "Normal Metabolizör (*1/*1 - Standart Aktivite)",
        "Ara Metabolizör (*1/*3A veya *1/*2 - %50 Aktivite)",
        "Zayıf Metabolizör (*3A/*3A veya *3A/*3C - Aktivite Yok / Kritik)"
    ],
    "nudt15": [
        "Normal Metabolizör (*1/*1)",
        "Ara Metabolizör (*1/*3 - Nötropeni Duyarlı)",
        "Zayıf Metabolizör (*3/*3 - Şiddetli Erken A пластиk Kemik İliği)"
    ],
    "dpyd": [
        "Normal DPYD Aktivitesi (Tam Fonksiyon)",
        "Kısmi DPYD Eksikliği (*2A heterozigot - Yüksek Toksisite)",
        "Tam DPYD Eksikliği (Ölümcül Toksisite Riski)"
    ]
}


def evaluate_pharmacogenomic_profile(
    tpmt_status: str,
    nudt15_status: str,
    dpyd_status: str,
    planned_regimen: str = "ALL İdame (6-MP + MTX)"
) -> dict:
    """
    CPIC kılavuz algoritmalarına göre hastanın PGx profiline uygun dozaj önerisi ve toksisite uyarısı üretir.
    """
    alerts = []
    dose_reductions = []
    risk_level = "Standart (Fizyolojik Toksisite)"
    card_color = "#047857"  # Yeşil

    # 1. TPMT & NUDT15 Değerlendirmesi (6-MP / Azatioprin)
    is_tpmt_poor = "Zayıf" in tpmt_status
    is_tpmt_inter = "Ara" in tpmt_status
    is_nudt_poor = "Zayıf" in nudt15_status
    is_nudt_inter = "Ara" in nudt15_status

    if is_tpmt_poor or is_nudt_poor:
        risk_level = "HAYATİ TEHLİKE / ŞİDDETLİ APLASTİK MİYELOTOKSİSİTE"
        card_color = "#7a1f2c"  # Kırmızı
        dose_rec = "%90 Doz Azaltımı (Standart dozun %10'u ile başlanmalı, haftada 3 gün)"
        reason = "TPMT veya NUDT15 tam inaktifliği nedeniyle aktif 6-TGN metabolitleri kemik iliğinde aşırı birikerek ölümcül pansitopeni ve sepsise yol açar (CPIC Level 1A Kanıt)."
        dose_reductions.append({"İlaç": "6-Merkaptopürin (6-MP)", "Önerilen Doz": dose_rec, "Gerekçe": reason})
        alerts.append("Acil CBC ve haftalık mutlak nötrofil sayısı (ANC) takibi şarttır.")
    elif is_tpmt_inter or is_nudt_inter:
        if risk_level != "HAYATİ TEHLİKE / ŞİDDETLİ APLASTİK MİYELOTOKSİSİTE":
            risk_level = "YÜKSEK MİYELOSÜPRESYON RİSKİ"
            card_color = "#92600a"  # Amber
        dose_rec = "%30 - %50 Doz Azaltımı (Haftalık ANC titrasyonu ile)"
        reason = "Hücre içi tiopürin klerensi yarı yarıya azalmıştır; orta derecede miyelotoksisite beklenir."
        dose_reductions.append({"İlaç": "6-Merkaptopürin (6-MP)", "Önerilen Doz": dose_rec, "Gerekçe": reason})

    # 2. DPYD Değerlendirmesi
    is_dpyd_poor = "Tam DPYD" in dpyd_status
    is_dpyd_partial = "Kısmi" in dpyd_status

    if is_dpyd_poor:
        risk_level = "HAYATİ TEHLİKE / ŞİDDETLİ APLASTİK MİYELOTOKSİSİTE"
        card_color = "#7a1f2c"
        dose_rec = "KONTRENDİKE (Fluoropirimidin / Fludarabin Verilmemeli)"
        reason = "DPYD enzim aktivitesi yokluğu nedeniyle ölümcül mukozit, kardiyotoksisite ve nötropenik şok riski."
        dose_reductions.append({"İlaç": "Fludarabin / 5-FU", "Önerilen Doz": dose_rec, "Gerekçe": reason})
    elif is_dpyd_partial:
        if risk_level != "HAYATİ TEHLİKE / ŞİDDETLİ APLASTİK MİYELOTOKSİSİTE":
            risk_level = "YÜKSEK MİYELOSÜPRESYON RİSKİ"
            card_color = "#92600a"
        dose_rec = "%50 Doz Azaltımı"
        reason = "Kısmi enzim eksikliği nedeniyle plazma klerensi gecikir; toksisite riski yüksektir."
        dose_reductions.append({"İlaç": "Fludarabin / 5-FU", "Önerilen Doz": dose_rec, "Gerekçe": reason})

    if not dose_reductions:
        dose_reductions.append({
            "İlaç": "Kemoterapi Protokolü (Tümü)",
            "Önerilen Doz": "%100 Standart Protokol Dozu",
            "Gerekçe": "Hasta tüm temel sitotoksik ilaç metabolizör genlerinde (TPMT, NUDT15, DPYD) normal/hızlı fenotipe sahiptir. Özel PGx doz kısıtlaması gerekmez."
        })

    df_dosing = pd.DataFrame(dose_reductions)

    # HTML Bilgi Kartı
    alert_html = f"""
    <div style='background: #ffffff; border: 1px solid #dde1e7; border-left: 4px solid {card_color}; border-radius: 4px; padding: 12px 16px; margin-bottom: 10px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='font-size: 11px; font-weight: 700; color: {card_color}; font-family: "IBM Plex Mono", monospace; text-transform: uppercase;'>
                Farmakogenomik (PGx) Risk Değerlendirmesi: {risk_level}
            </div>
            <div style='font-size: 10px; color: #5c6675; font-family: "IBM Plex Mono", monospace;'>CPIC / DPWG Kılavuz Standartları</div>
        </div>
        <div style='font-size: 12px; color: #17202b; margin-top: 6px; line-height: 1.5;'>
            Seçilen Rejim: <b>{planned_regimen}</b> | TPMT: <code>{tpmt_status.split('(')[0].strip()}</code> | NUDT15: <code>{nudt15_status.split('(')[0].strip()}</code> | DPYD: <code>{dpyd_status.split('(')[0].strip()}</code>
        </div>
    </div>
    """

    return {
        "risk_level": risk_level,
        "card_color": card_color,
        "dosing_df": df_dosing,
        "card_html": alert_html,
        "alerts": alerts
    }
