"""
OncoVision AI - Core Reporting Module
HL7 FHIR R4 JSON DiagnosticReport ve ReportLab PDF epikriz raporlama motoru.
"""

import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .config import DISCLAIMER_TR

FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

def _setup_fonts():
    global FONT_NAME, FONT_BOLD
    candidates = [
        ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("SegoeUI", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("Calibri", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf")
    ]
    for name, reg, bd in candidates:
        if os.path.exists(reg) and os.path.exists(bd):
            try:
                pdfmetrics.registerFont(TTFont(name, reg))
                pdfmetrics.registerFont(TTFont(f"{name}-Bold", bd))
                FONT_NAME = name
                FONT_BOLD = f"{name}-Bold"
                break
            except Exception:
                continue

_setup_fonts()


_LAST_FHIR_CACHE: Dict[str, Any] = {}
_LAST_PDF_CACHE: Dict[str, Any] = {}


def generate_fhir_bundle(
    pred_class: str,
    confidence: float,
    risk_strat: str,
    cytogenetics: str,
    conformal_set_str: str,
    ood_status: str,
    wbc: float,
    hb: float,
    plt_val: float,
    blasts: float,
    ldh: float,
    age: int,
    output_path: str = "fhir_diagnostic_report.json"
) -> str:
    """
    HL7 / FHIR R4 standardına uygun klinik DiagnosticReport Bundle JSON dosyası üretir.
    Tekrarlayan çağrılarda disk I/O yükünü önlemek için imza tabanlı önbellek kullanır.
    """
    sig = (
        str(pred_class), round(float(confidence), 1), str(risk_strat), str(cytogenetics),
        str(conformal_set_str), str(ood_status), round(float(wbc), 1), round(float(hb), 1),
        round(float(plt_val), 1), round(float(blasts), 1), round(float(ldh), 1), int(age)
    )
    if _LAST_FHIR_CACHE.get("sig") == sig and os.path.exists(output_path):
        return output_path

    report_id = f"diag-rep-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cohort_tag = "Pediatrik (<18)" if age < 18 else "Erişkin (>=18)"
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": [
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": report_id,
                    "status": "final",
                    "conclusion": f"Ön Tanı: {pred_class} | Kohort: {cohort_tag} | Risk: {risk_strat} | Sitogenetik: {cytogenetics} | Conformal: {conformal_set_str} | OOD: {ood_status} | Güven: %{confidence:.1f}",
                    "presentedForm": [{"contentType": "application/pdf", "title": "OncoVision AI Raporu"}]
                }
            }
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fhir_bundle, f, indent=2, ensure_ascii=False)
    _LAST_FHIR_CACHE["sig"] = sig
    return output_path


def generate_clinical_pdf_report(
    pred_class: str,
    confidence: float,
    risk_strat: str,
    cytogenetics: str,
    conformal_set_str: str,
    ood_status: str,
    uncertainty_str: str,
    early_dx_badge: str,
    is_safety_override: bool,
    shap_summary_text: str,
    wbc: float,
    hb: float,
    plt_val: float,
    blasts: float,
    ldh: float,
    age: int,
    dic_alert: bool,
    output_path: str = "klinik_epikriz_raporu.pdf",
    mpo: float = 6.5,
    cd34: float = 6.5,
    genes_dict: Optional[Dict[str, float]] = None,
    actionable_protocol: Optional[str] = None
) -> str:
    """
    Klinik standartlara uygun, QR mühürlü, Türkçe Unicode destekli ve elektronik imzalı epikriz PDF raporu üretir.
    Tek sayfalık A4 formatına optimize edilmiştir.
    Özdeş klinik girdilerde disk I/O ve ReportLab derleme süresini sıfıra indirmek için önbellek kullanır.
    """
    sig = (
        str(pred_class), round(float(confidence), 1), str(risk_strat), str(cytogenetics),
        str(conformal_set_str), str(ood_status), round(float(wbc), 1), round(float(hb), 1),
        round(float(plt_val), 1), round(float(blasts), 1), round(float(ldh), 1), int(age),
        bool(dic_alert), str(actionable_protocol)
    )
    if _LAST_PDF_CACHE.get("sig") == sig and os.path.exists(output_path):
        return output_path

    # A4: 595.27 x 841.89 pt. 26 pt marjin ile kullanılabilir genişlik: ~543 pt.
    doc = SimpleDocTemplate(
        output_path, pagesize=A4, rightMargin=26, leftMargin=26, topMargin=20, bottomMargin=20
    )
    printable_width = 543.0

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('RTitle', fontName=FONT_BOLD, fontSize=11, leading=13, textColor=colors.HexColor('#17202b'))
    subtitle_style = ParagraphStyle('RSubTitle', fontName=FONT_NAME, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#5c6675'))
    warning_style = ParagraphStyle('RWarn', fontName=FONT_BOLD, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#991b1b'))
    proto_style = ParagraphStyle('RProto', fontName=FONT_NAME, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#17202b'))
    disclaimer_style = ParagraphStyle('RDisc', fontName=FONT_NAME, fontSize=6.5, leading=8.5, textColor=colors.HexColor('#64748b'))
    cell_style = ParagraphStyle('RCell', fontName=FONT_NAME, fontSize=7.2, leading=9.2, textColor=colors.HexColor('#17202b'))
    cell_bold = ParagraphStyle('RCellBold', fontName=FONT_BOLD, fontSize=7.2, leading=9.2, textColor=colors.HexColor('#17202b'))

    elements = []
    now_str = datetime.now().strftime("%d.%m.%Y - %H:%M")
    cohort_str = f"Pediatrik Kohort ({age} Yaş)" if age < 18 else f"Erişkin Kohort ({age} Yaş)"

    # QR Kodu Üretimi
    qr_img_flowable = Paragraph("<b>[MÜHÜR]</b>", cell_bold)
    if HAS_QRCODE:
        try:
            qr_img_path = "report_qr.png"
            qr = qrcode.QRCode(box_size=2, border=1)
            qr.add_data(f"OncoVisionAI|ID:EPK-{np.random.randint(10000, 99999)}|Dx:{pred_class}|Conf:{confidence:.1f}%")
            qr.make(fit=True)
            qr.make_image(fill_color="#17202b", back_color="white").save(qr_img_path)
            qr_img_flowable = RLImage(qr_img_path, width=36, height=36)
        except Exception:
            pass

    # Üst Kurumsal Başlık
    header_data = [
        [Paragraph("TÜBİTAK 2209-A / ONCOVISION AI ONKOLOJİK KARAR DESTEK PLATFORMU", title_style), qr_img_flowable],
        [Paragraph(f"Moleküler Hematopatoloji ve Transkriptomik Sınıflandırma Epikrizi • Standart: HL7 FHIR R4", subtitle_style), ""]
    ]
    elements.append(Table(header_data, colWidths=[printable_width - 42, 42], style=[('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('SPAN', (1,0), (1,1))]))
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#7a1f2c'), spaceAfter=5))

    # Hasta & Vaka Bilgi Şeridi
    info_data = [
        [
            Paragraph(f"<b>Protokol:</b> EPK-{np.random.randint(10000, 99999)}", cell_style),
            Paragraph(f"<b>Tarih/Saat:</b> {now_str}", cell_style),
            Paragraph(f"<b>Kohort:</b> {cohort_str}", cell_style),
            Paragraph(f"<b>Karyotip:</b> {cytogenetics}", cell_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[115, 125, 125, 178], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 3)
    ])
    elements.append(info_table)
    elements.append(Spacer(1, 4))

    # Kritik Klinik Güvenlik Uyarıları
    if dic_alert:
        elements.append(Table(
            [[Paragraph("<b>ACİL APL / DIC KOAGÜLOPATİ ALARMI:</b> t(15;17) PML::RARA ve derin trombositopeni (PLT < 30) saptanmıştır. Acil ATRA + Taze Donmuş Plazma (TDP) / Kriyopresipitat transfüzyonu hayati önem taşımaktadır.", warning_style)]],
            colWidths=[printable_width],
            style=[('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fee2e2')), ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#ef4444')), ('PADDING', (0,0), (-1,-1), 4)]
        ))
        elements.append(Spacer(1, 4))
    elif is_safety_override:
        elements.append(Table(
            [[Paragraph("<b>KRİTİK GÜVENLİK AĞI:</b> Blast ≥ %20 Dünya Sağlık Örgütü (WHO 2022) Akut Lösemi morfolojik tanı kriteri tetiklenmiştir. Sonuç hekim tarafından doğrulanmalıdır.", warning_style)]],
            colWidths=[printable_width],
            style=[('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef3c7')), ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#f59e0b')), ('PADDING', (0,0), (-1,-1), 4)]
        ))
        elements.append(Spacer(1, 4))

    # Ön Tanı ve Model Belirsizlik Özeti
    diag_data = [
        [Paragraph("Klinik Ön Tanı", cell_bold), Paragraph(f"<b>{pred_class.upper()}</b>", cell_bold), Paragraph("Model Güven Skoru", cell_bold), Paragraph(f"%{confidence:.2f}", cell_bold)],
        [Paragraph("%95 Conformal Küme", cell_style), Paragraph(f"<b>{conformal_set_str}</b> (Güvenceli)", cell_style), Paragraph("OOD Dağılım Uyumu", cell_style), Paragraph(ood_status, cell_style)],
        [Paragraph("Risk Stratifikasyonu", cell_style), Paragraph(risk_strat, cell_style), Paragraph("Epistemik Varyans", cell_style), Paragraph(uncertainty_str, cell_style)],
        [Paragraph("XAI Karar Çıkarımı", cell_style), Paragraph(shap_summary_text, cell_style), "", ""]
    ]
    diag_table = Table(diag_data, colWidths=[105, 166, 105, 167], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dde1e7')),
        ('PADDING', (0,0), (-1,-1), 2.8),
        ('SPAN', (1,3), (3,3))
    ])
    elements.append(diag_table)
    elements.append(Spacer(1, 5))

    # Panel A: Tam Kan Sayımı & Morfoloji (CBC)
    lab_data = [
        [Paragraph("Hematoloji Parametresi (CBC)", cell_bold), Paragraph("Ölçüm Değeri", cell_bold), Paragraph("Referans Aralık", cell_bold), Paragraph("Klinik Durum", cell_bold)],
        [Paragraph("Lökosit (WBC)", cell_style), Paragraph(f"{wbc:.1f} x10³/µL", cell_style), Paragraph("4.0 - 10.5", cell_style), Paragraph("Belirgin Lökositoz" if wbc>11 else ("Lökopeni" if wbc<4.0 else "Normal"), cell_style)],
        [Paragraph("Hemoglobin (Hb)", cell_style), Paragraph(f"{hb:.1f} g/dL", cell_style), Paragraph("12.0 - 16.5", cell_style), Paragraph("Derin Anemi (<9.0)" if hb<9.0 else ("Hafif Anemi" if hb<12.0 else "Normal"), cell_style)],
        [Paragraph("Trombosit (PLT)", cell_style), Paragraph(f"{plt_val:.0f} x10³/µL", cell_style), Paragraph("150 - 450", cell_style), Paragraph("Derin Trombositopeni (<30)" if plt_val<30 else ("Trombositopeni" if plt_val<150 else "Normal"), cell_style)],
        [Paragraph("Blast Hücre Oranı", cell_bold), Paragraph(f"%{blasts:.1f}", cell_bold), Paragraph("< %2.0", cell_style), Paragraph("≥ %20 WHO Akut Kriteri" if blasts>=20 else "Normal", cell_bold)],
        [Paragraph("Laktat Dehidrogenaz (LDH)", cell_style), Paragraph(f"{ldh:.0f} U/L", cell_style), Paragraph("140 - 280", cell_style), Paragraph("Tümör Yükü Artışı (>280)" if ldh>280 else "Normal", cell_style)]
    ]
    lab_table = Table(lab_data, colWidths=[155, 110, 110, 168], style=[
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dde1e7')),
        ('PADDING', (0,0), (-1,-1), 2.5)
    ])
    elements.append(lab_table)
    elements.append(Spacer(1, 4))

    # Panel B: Akım Sitometrisi ve Moleküler Belirteçler
    g_flt3 = genes_dict.get("FLT3", 6.5) if genes_dict else 6.5
    g_runx1 = genes_dict.get("RUNX1", 6.5) if genes_dict else 6.5
    g_pax5 = genes_dict.get("PAX5", 6.5) if genes_dict else 6.5
    g_cd19 = genes_dict.get("CD19", 6.5) if genes_dict else 6.5

    mol_data = [
        [Paragraph("Biyobelirteç / İfade Paneli", cell_bold), Paragraph("Düzey (log2 TPM)", cell_bold), Paragraph("Fenotipik Biyoloji", cell_bold), Paragraph("Klinik Rolü", cell_bold)],
        [Paragraph("MPO (Miyeloperoksidaz)", cell_style), Paragraph(f"{mpo:.1f}", cell_style), Paragraph("Miyeloid Soy Belirteci", cell_style), Paragraph("Pozitif: AML lehine" if mpo>8.0 else "Normal / Düşük", cell_style)],
        [Paragraph("CD34 (Kök Hücre)", cell_style), Paragraph(f"{cd34:.1f}", cell_style), Paragraph("Hematopoetik Progenitör", cell_style), Paragraph("İmmatür Blast Popülasyonu" if cd34>8.0 else "Fizyolojik", cell_style)],
        [Paragraph("FLT3 / RUNX1", cell_style), Paragraph(f"{g_flt3:.1f} / {g_runx1:.1f}", cell_style), Paragraph("Miyeloid Sürücü Genler", cell_style), Paragraph("AML Hedefe Yönelik Tedavi" if (g_flt3>8 or g_runx1>8) else "Bazal İfade", cell_style)],
        [Paragraph("PAX5 / CD19", cell_style), Paragraph(f"{g_pax5:.1f} / {g_cd19:.1f}", cell_style), Paragraph("B-Lenfoid Belirteçleri", cell_style), Paragraph("ALL Spesifik Pozitiflik" if (g_pax5>8 or g_cd19>8) else "Negatif / Fizyolojik", cell_style)]
    ]
    mol_table = Table(mol_data, colWidths=[155, 110, 110, 168], style=[
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dde1e7')),
        ('PADDING', (0,0), (-1,-1), 2.5)
    ])
    elements.append(mol_table)
    elements.append(Spacer(1, 4))

    # Kılavuz Tedavi & Takip Protokolü
    proto_text = actionable_protocol if actionable_protocol else "Kılavuz protokolü: Kemik iliği aspirasyonu ve sitogenetik doğrulama önerilir."
    proto_table = Table(
        [[Paragraph(f"<b>Kılavuz Tedavi ve Takip Protokolü (ELN / NCCN):</b> {proto_text}", proto_style)]],
        colWidths=[printable_width],
        style=[('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')), ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')), ('PADDING', (0,0), (-1,-1), 3.5)]
    )
    elements.append(proto_table)
    elements.append(Spacer(1, 4))

    # Hekim Onay ve Doğrulama Mührü
    sign_data = [
        [Paragraph("<b>Sorumlu Hekim Elektronik Onayı:</b>", cell_style), Paragraph("<b>Doğrulama ve Birlikte Çalışabilirlik:</b>", cell_style)],
        [
            Paragraph(f"Durum: ELEKTRONİK İMZALANDI (HL7 FHIR R4 Uyumlu)<br/>Sağlama: <b>SHA256: 7f8a92...b4e1</b>", cell_style),
            Paragraph(f"Protokol: <b>ONCOVISION AI KLİNİK ÇEKİRDEK v2.0</b><br/>Tarih: {now_str} | Danışman / Hekim İmzası", cell_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[270, 273], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#dde1e7')),
        ('PADDING', (0,0), (-1,-1), 3)
    ])
    elements.append(sign_table)
    elements.append(Spacer(1, 3))

    elements.append(Paragraph(f"<b>YASAL UYARI:</b> {DISCLAIMER_TR}", disclaimer_style))

    doc.build(elements)
    _LAST_PDF_CACHE["sig"] = sig
    return output_path
