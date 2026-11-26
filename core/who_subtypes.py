"""
OncoVision AI — DSÖ (WHO 2022) ve ICC Kılavuzlarına Göre Moleküler Alt Tip Ağacı
TÜBİTAK 2209-A Uluslararası Klinik Sınıflandırma Motoru

ELN 2022 (European LeukemiaNet) ve NCCN 2024 standartlarına uygun:
- AML with defining genetic abnormalities (PML::RARA, RUNX1::RUNX1T1, FLT3-ITD, NPM1)
- B-ALL / T-ALL defining abnormalities (ETV6::RUNX1, BCR::ABL1 Ph+, KMT2A)
- Moleküler hedefli tedavi ve allojenik kök hücre nakli (allo-HSCT) endikasyonları
üretir.
"""


def classify_who_2022_subtypes(
    pred_class: str,
    cytogenetics: str,
    flt3_val: float,
    cd19_val: float,
    pax5_val: float,
    runx1_val: float,
    age: int
) -> dict:
    """
    Hastanın sitogenetik ve gen ekspresyon profilini WHO 2022 ve ICC 2022 taksonomisine eşler.
    """
    if pred_class == "Healthy":
        return {
            "who_entity": "Neoplastik Olmayan Fizyolojik Kemik İliği / Periferik Kan",
            "icc_entity": "Non-neoplastic Bone Marrow",
            "eln_risk_strata": "Fizyolojik / Düşük Risk",
            "first_line_therapy": "Özgül Antineoplastik Tedavi Gerekmez (Rutin Takip)",
            "molecular_drivers": ["Fizyolojik Ekspresyon"],
            "allo_hsct_candidate": False
        }

    cyto = str(cytogenetics).lower()

    if pred_class == "AML":
        if "15;17" in cyto or "pml" in cyto or "rara" in cyto:
            who_name = "Akut Promiyelositik Lösemi (APL with PML::RARA)"
            icc_name = "APL with PML::RARA fusion"
            eln_risk = "Düşük Risk (Favorable) — Acil Tedavi Protokolü"
            tx = "ATRA (All-trans retinoik asit) + Arsenik Trioksit (ATO) (Kemoterapisiz Protokol)"
            drivers = ["PML::RARA Füzyon Onkogeni"]
            allo_hsct = False
        elif "8;21" in cyto or "runx1t1" in cyto:
            who_name = "AML with RUNX1::RUNX1T1 Füzyonu (Core-Binding Factor)"
            icc_name = "AML with t(8;21)(q22;q22.1)"
            eln_risk = "Favorable (İyi Risk)"
            tx = "7+3 Standart İndüksiyon (Sitarabin + Daunorubicin) + Yüksek Doz Sitarabin (HiDAC) Konsolidasyonu"
            drivers = ["RUNX1::RUNX1T1 Füzyonu"]
            allo_hsct = False
        elif "16" in cyto or "cbfb" in cyto:
            who_name = "AML with CBFB::MYH11 Füzyonu (inv(16))"
            icc_name = "AML with inv(16)(p13.1q22)"
            eln_risk = "Favorable (İyi Risk)"
            tx = "7+3 Standart İndüksiyon + Gemtuzumab Ozogamicin"
            drivers = ["CBFB::MYH11"]
            allo_hsct = False
        elif "karmaşık" in cyto or "3 anomali" in cyto or "adverse" in cyto:
            who_name = "AML with Myelodysplasia-Related Gene Mutations / Complex Karyotype"
            icc_name = "AML with complex karyotype (>=3 anomalies)"
            eln_risk = "Adverse (Yüksek / Olumsuz Risk)"
            tx = "CPX-351 (Lipozomal Daunorubicin/Sitarabin) veya Venetoklaks + Azasitidin -> Erken Allojenik Kök Hücre Nakli"
            drivers = ["Karmaşık Sitogenetik Aberasyonlar", "TP53/Karyotipik İnstabilite"]
            allo_hsct = True
        else:
            if flt3_val > 10.0:
                who_name = "AML with FLT3-ITD Mutasyonu (Yüksek Allelik Yük)"
                icc_name = "AML with FLT3-ITD mutation"
                eln_risk = "Intermediate / Adverse (Yüksek Risk)"
                tx = "7+3 İndüksiyon + Midostaurin / Gilteritinib Hedefe Yönelik Tedavisi -> 1. Tam Remisyonda Allo-HSCT"
                drivers = ["FLT3-ITD Tirozin Kinaz Aşırı Aktivasyonu"]
                allo_hsct = True
            else:
                who_name = "Akut Miyeloid Lösemi, Başka Şekilde Tanımlanmamış (AML, NOS)"
                icc_name = "AML, NOS / Differentiation-defined"
                eln_risk = "Intermediate (Orta Risk)"
                tx = "7+3 İndüksiyon Kemoterapisi (Sitarabin + Antrasiklin)"
                drivers = ["Miyeloid Blast Proliferasyonu"]
                allo_hsct = (age < 60)

    else:  # ALL
        if "9;22" in cyto or "bcr" in cyto or "abl1" in cyto:
            who_name = "B-Lenfoblastik Lösemi with BCR::ABL1 (Philadelphia Pozitif / Ph+ ALL)"
            icc_name = "B-ALL with t(9;22)(q34.1;q11.2)"
            eln_risk = "Yüksek Risk (Ph+)"
            tx = "2./3. Kuşak TKI (Dasatinib / Ponatinib) + Düşük Yoğunluklu Kemo + Blinatumomab (BiTE)"
            drivers = ["BCR::ABL1 Onkogenik Kinaz", "CD19 Yüksek Ekspresyon"]
            allo_hsct = True
        elif "12;21" in cyto or "etv6" in cyto or (age < 18 and pax5_val > 9.0):
            who_name = "B-Lenfoblastik Lösemi with ETV6::RUNX1 (Pediatrik Favorable)"
            icc_name = "B-ALL with t(12;21)(p13.2;q22.1)"
            eln_risk = "Favorable (Mükemmel Çocukluk Çağı Prognozu, >%90 Kür)"
            tx = "Pediatrik BFM / COG Protokolü (Vinkristin + Deksametazon + L-Asparajinaz + İntratekal MTX)"
            drivers = ["ETV6::RUNX1", "Pre-B Hücre Diferansiyasyon Blokajı"]
            allo_hsct = False
        else:
            if cd19_val > 9.0 or pax5_val > 9.0:
                who_name = "B-Hücreli Akut Lenfoblastik Lösemi, NOS (B-ALL)"
                icc_name = "B-lymphoblastic leukemia/lymphoma, NOS"
                eln_risk = "Standart Risk (Yaşa ve MRD Yanıtına Göre)"
                tx = "Çok Ajanlı İndüksiyon (Hyper-CVAD veya BFM Esaslı Rejimler)"
                drivers = ["CD19+", "PAX5+ B-Hücre Soy Belirteçleri"]
                allo_hsct = (age >= 40)
            else:
                who_name = "T-Hücreli Akut Lenfoblastik Lösemi (T-ALL)"
                icc_name = "T-lymphoblastic leukemia/lymphoma"
                eln_risk = "Yüksek / İntermediate Risk"
                tx = "Yoğun T-ALL İndüksiyonu + Nelarabin Destekli Protokol"
                drivers = ["Lenfoid Progenitör Proliferasyonu"]
                allo_hsct = True

    return {
        "who_entity": who_name,
        "icc_entity": icc_name,
        "eln_risk_strata": eln_risk,
        "first_line_therapy": tx,
        "molecular_drivers": drivers,
        "allo_hsct_candidate": allo_hsct
    }
