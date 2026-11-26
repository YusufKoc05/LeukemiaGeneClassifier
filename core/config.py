"""
OncoVision AI - Core Configuration Module
Klinik konfigürasyon, eşik değerleri, referanslar ve yasal uyarılar.
"""

import os
import json
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "OncoVision AI",
    "project_version": "2.5-TÜBİTAK-2209A",
    "disclaimer_tr": "YASAL UYARI & SORUMLULUK REDDİ: Bu platform sentetik ve simüle edilmiş biyolojik verilerle eğitilmiş bir TÜBİTAK 2209-A üniversite araştırma prototipidir. Gerçek hasta teşhisinde veya klinik karar süreçlerinde tek başına kullanılamaz.",
    "disclaimer_en": "DISCLAIMER: This research platform was trained on synthetic biological data as part of a TUBITAK 2209-A student project. It is strictly not approved for clinical diagnostics or real patient care.",
    "thresholds": {
        "who_blast_acute_threshold": 20.0,
        "early_stage_blast_threshold": 5.0,
        "confidence_low_threshold": 60.0,
        "tree_variance_warning_threshold": 0.040,
        "tree_variance_stable_threshold": 0.015,
        "platelet_severe_threshold": 30.0,
        "flt3_adverse_threshold": 8.5,
        "runx1_adverse_threshold": 9.0,
        "wt1_intermediate_threshold": 8.0,
        "ikzf1_adverse_deletion_threshold": 4.0,
        "cd19_preb_threshold": 8.0,
        "ldh_upper_limit": 280.0
    },
    "references": [
        {
            "title": "WHO Classification of Haematolymphoid Tumours (5th Edition, 2022)",
            "link": "https://tumourclassification.iarc.who.int/",
            "notes": "Akut lösemi kesin morfolojik tanı eşiği (Blast >= %20)"
        },
        {
            "title": "ELN 2022 Recommendations for AML Diagnosis and Management",
            "link": "https://doi.org/10.1182/blood.2022016867",
            "notes": "FLT3-ITD, RUNX1, WT1 mutasyonel ve ekspresyonel risk katmanlandırması"
        },
        {
            "title": "NCCN Guidelines Version 1.2024 Acute Lymphoblastic Leukemia",
            "link": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1410",
            "notes": "IKZF1 delesyonu, CD19 hedefe yönelik BiTE/CAR-T endikasyonları"
        },
        {
            "title": "Microarray Innovations in LEukemia (MILE Study - GSE13159)",
            "link": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159",
            "notes": "Transkriptomik panel gen profili referans seti"
        }
    ]
}


def load_clinical_config(config_path: str = "clinical_config.json") -> Dict[str, Any]:
    """
    clinical_config.json dosyasını yükler, bulunamazsa varsayılan konfigürasyonu döndürür.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                config = DEFAULT_CONFIG.copy()
                config.update(loaded)
                # İç içe thresholds birleştirme
                if "thresholds" in loaded:
                    merged_thresh = DEFAULT_CONFIG["thresholds"].copy()
                    merged_thresh.update(loaded["thresholds"])
                    config["thresholds"] = merged_thresh
                return config
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


CONFIG = load_clinical_config()
THRESHOLDS = CONFIG.get("thresholds", DEFAULT_CONFIG["thresholds"])
DISCLAIMER_TR = CONFIG.get("disclaimer_tr", DEFAULT_CONFIG["disclaimer_tr"])
DEFAULT_DATASET_DIR = r"C:\Users\ykoc0\OneDrive\Yeni klasör\verisetleri"

