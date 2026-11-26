"""
OncoVision AI - Core Inference Engine
Klinik tahmin, güvenlik ağları, XAI (SHAP), belirsizlik, simülasyon ve triyaj motoru.
"""

import os
import json
import numpy as np
import pandas as pd
import shap
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from scipy.stats import wasserstein_distance

from .config import THRESHOLDS, DISCLAIMER_TR
from .models import get_model_manager

FEEDBACK_FILE = "clinical_expert_feedback.json"


class InferenceEngine:
    """
    Klinik karar motorunu ve XAI çıkarımlarını yöneten tekil servis.
    """
    def __init__(self):
        self.mm = get_model_manager()
        self.explainer = shap.TreeExplainer(self.mm.rf_model)

    def predict_patient(
        self,
        age: int,
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
        mpo: Optional[float] = None,
        cd34: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Bireysel hasta için uçtan uca klinik tahmin, güvenlik ağı ve XAI analizi yürütür.
        """
        sample = self.mm.X.mean().copy()
        sample['RUNX1'] = runx1
        sample['FLT3'] = flt3
        sample['PAX5'] = pax5
        sample['IKZF1'] = ikzf1
        sample['CD19'] = cd19
        sample['WT1'] = wt1
        sample['WBC_x10k'] = wbc
        sample['Hemoglobin_g_dL'] = hb
        sample['PLT_x10k'] = plt_val
        sample['Blast_Percentage'] = blasts
        sample['LDH_U_L'] = ldh

        if 'MPO' in sample and mpo is not None:
            sample['MPO'] = mpo
        if 'CD34' in sample and cd34 is not None:
            sample['CD34'] = cd34

        input_df = pd.DataFrame([sample])
        input_scaled = self.mm.scaler.transform(input_df)

        # 1. Model Çıktıları ve Olasılıklar
        probs = self.mm.rf_model.predict_proba(input_scaled)[0]
        prob_dict = {cls: float(round(prob, 4)) for cls, prob in zip(self.mm.classes, probs)}
        pred_idx = int(np.argmax(probs))
        raw_pred_class = self.mm.classes[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        # 2. İstatistiksel Güvenceler (Conformal & OOD)
        conformal_set = self.mm.compute_conformal_set(probs)
        conformal_set_str = ", ".join(conformal_set)
        ood_status, ood_score, ood_badge_color = self.mm.compute_ood_score(input_scaled)

        # 3. Epistemik Belirsizlik (Ağaç Varyansı)
        tree_preds = [tree.predict_proba(input_scaled)[0][pred_idx] for tree in self.mm.rf_model.estimators_]
        tree_variance = float(np.var(tree_preds))
        v_warn = THRESHOLDS.get("tree_variance_warning_threshold", 0.040)
        v_stable = THRESHOLDS.get("tree_variance_stable_threshold", 0.015)
        uncertainty_level = "Kararlı" if tree_variance < v_stable else ("Orta" if tree_variance < v_warn else "Yüksek")
        uncertainty_str = f"Varyans: {tree_variance:.4f} ({uncertainty_level})"

        # 4. Klinik Güvenlik Ağları (Safety-Nets)
        blast_limit = THRESHOLDS.get("who_blast_acute_threshold", 20.0)
        plt_severe_limit = THRESHOLDS.get("platelet_severe_threshold", 30.0)
        is_pediatric = age < 18
        dic_alert = False

        if "t(15;17)" in cytogenetics and plt_val < plt_severe_limit:
            dic_alert = True

        is_safety_override = False
        warning_message = "Normal fizyolojik parametreler."

        blast_limit_val = float(blast_limit)
        blast_limit_str = f"{int(blast_limit_val)}" if blast_limit_val.is_integer() else f"{blast_limit_val}"

        if blasts >= blast_limit and raw_pred_class == 'Healthy':
            is_safety_override = True
            final_class = "Şüpheli / İleri Tetkik Gerekli"
            risk_strat = f"KRİTİK UYARI: Blast ≥ %{blast_limit_str} (WHO Kriteri)"
            risk_color = "#dc2626"
            warning_message = f"KRİTİK GÜVENLİK AĞI: Blast oranı ≥ %{blast_limit_str} (WHO Akut Lösemi Kriteri). Model tahmini güvenlik protokolüyle geçersiz kılınmıştır."
            therapy_plan = "Acil Kemik İliği Biyopsisi ve Flow Sitometri Önerilir."
        else:
            final_class = raw_pred_class
            if "t(15;17)" in cytogenetics:
                risk_strat, risk_color = "Uluslararası Kılavuz: APL (PML-RARA - Küratif)", "#7a1f2c"
                therapy_plan = "Acil ATRA + Arsenik Trioksit (ATO) Protokolü."
            elif final_class == 'Healthy':
                risk_strat, risk_color = "Normal / Düşük Risk", "#047857"
                therapy_plan = "Tedavi endikasyonu bulunmamaktadır. Rutin hematolojik takip önerilir."
            elif final_class == 'AML':
                if "Karmaşık" in cytogenetics or flt3 > THRESHOLDS.get("flt3_adverse_threshold", 8.5):
                    risk_strat, risk_color = "ELN Rehberi: Yüksek Risk (Adverse)", "#dc2626"
                    therapy_plan = "Gilteritinib + Standart 7+3 İndüksiyon + Erken Allogeneik KİT"
                else:
                    risk_strat, risk_color = "ELN Rehberi: Standart / İyi Risk (Favorable)", "#047857"
                    therapy_plan = "Standart 7+3 İndüksiyon + HiDAC Konsolidasyonu"
            else:  # ALL
                if "t(9;22)" in cytogenetics:
                    risk_strat, risk_color = "NCCN Rehberi: Ph+ ALL (BCR-ABL1 - Yüksek Risk)", "#dc2626"
                    therapy_plan = "2./3. Kuşak TKI (Ponatinib/Dasatinib) + Blinatumomab"
                elif is_pediatric:
                    risk_strat, risk_color = "Pediatrik ALL Protokolü (Yüksek Kemoduyarlılık)", "#047857"
                    therapy_plan = "BFM-Inspired Pediatrik Protokol (Peg-Asparaginaz + Vinkristin + Metotreksat)"
                else:
                    risk_strat, risk_color = "NCCN Rehberi: Erişkin Standart Risk ALL", "#7a1f2c"
                    therapy_plan = "Çoklu Ajan Kemoterapi + MRD Takibi ile Blinatumomab Yönlendirmesi"

            if dic_alert:
                therapy_plan += " [ACİL DIC KORUMASI: TDP/Kriyopresipitat]"

            if blasts >= blast_limit:
                warning_message = f"WHO Akut Lösemi Kriteri Karşılandı (Blast ≥ %{blast_limit_str})."

        # 5. XAI (SHAP) Açıklanabilirlik
        shap_vals = self.explainer.shap_values(input_scaled)
        sample_shap = shap_vals[pred_idx][0] if isinstance(shap_vals, list) else shap_vals[0, :, pred_idx]
        top_indices = np.argsort(np.abs(sample_shap))[::-1][:8]

        clean_feature_name = lambda f: f.replace('_', ' ').replace('x10k', '(x10^3/uL)').replace('g dL', '(g/dL)').replace('Percentage', '(%)').replace('U L', '(U/L)')
        top_pos = [clean_feature_name(input_df.columns[i]) for i in top_indices if sample_shap[i] > 0][:2]
        top_neg = [clean_feature_name(input_df.columns[i]) for i in top_indices if sample_shap[i] < 0][:1]

        pos_str = ', '.join(top_pos) if top_pos else "Fizyolojik bazal değerler"
        neg_str = f" | Baskılayan: {', '.join(top_neg)}" if top_neg else ""
        shap_summary_text = f"Pozitif Sürücüler: {pos_str}{neg_str}"

        return {
            "raw_model_prediction": raw_pred_class,
            "final_clinical_class": final_class,
            "confidence_percentage": round(confidence, 2),
            "epistemic_variance": round(tree_variance, 4),
            "uncertainty_level": uncertainty_level,
            "uncertainty_str": uncertainty_str,
            "safety_net_triggered": is_safety_override,
            "dic_coagulopathy_alert": dic_alert,
            "clinical_warning": warning_message,
            "actionable_protocol": therapy_plan,
            "risk_stratification": risk_strat,
            "risk_color": risk_color,
            "class_probabilities": prob_dict,
            "conformal_prediction_set": conformal_set,
            "conformal_set_str": conformal_set_str,
            "ood_status": ood_status,
            "ood_score": round(ood_score, 4),
            "ood_badge_color": ood_badge_color,
            "shap_summary_text": shap_summary_text,
            "sample_shap": sample_shap,
            "top_indices": top_indices,
            "input_scaled": input_scaled,
            "input_df": input_df,
            "disclaimer": DISCLAIMER_TR
        }

    def run_counterfactual_simulation(
        self,
        base_age: int,
        base_runx1: float,
        base_flt3: float,
        base_pax5: float,
        base_ikzf1: float,
        base_cd19: float,
        base_wt1: float,
        base_wbc: float,
        base_hb: float,
        base_plt: float,
        base_blasts: float,
        base_ldh: float,
        cytogenetics: str,
        target_intervention: str,
        intensity_pct: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        What-If karşıolgusal tedavi simülasyonunu yürütür.
        """
        sample = self.mm.X.mean().copy()
        sample['RUNX1'] = base_runx1
        sample['FLT3'] = base_flt3
        sample['PAX5'] = base_pax5
        sample['IKZF1'] = base_ikzf1
        sample['CD19'] = base_cd19
        sample['WT1'] = base_wt1
        sample['WBC_x10k'] = base_wbc
        sample['Hemoglobin_g_dL'] = base_hb
        sample['PLT_x10k'] = base_plt
        sample['Blast_Percentage'] = base_blasts
        sample['LDH_U_L'] = base_ldh

        sim_sample = sample.copy()
        factor = (1.0 - (intensity_pct / 100.0))

        if target_intervention == "FLT3 İnhibisyonu (Gilteritinib / Midostaurin)":
            sim_sample['FLT3'] = max(4.0, sim_sample['FLT3'] * factor)
        elif target_intervention == "Anti-CD19 / BiTE Simülasyonu (Blinatumomab)":
            sim_sample['CD19'] = max(4.0, sim_sample['CD19'] * factor)
        elif target_intervention == "Kemoterapi Blast Redüksiyonu":
            sim_sample['Blast_Percentage'] = max(0.0, sim_sample['Blast_Percentage'] * factor)
            sim_sample['LDH_U_L'] = max(180.0, sim_sample['LDH_U_L'] * factor)

        orig_probs = self.mm.rf_model.predict_proba(self.mm.scaler.transform(pd.DataFrame([sample])))[0]
        sim_probs = self.mm.rf_model.predict_proba(self.mm.scaler.transform(pd.DataFrame([sim_sample])))[0]

        healthy_idx = self.mm.classes.index('Healthy')
        delta_healthy = (sim_probs[healthy_idx] - orig_probs[healthy_idx]) * 100

        return orig_probs, sim_probs, delta_healthy

    def run_batch_screening(self, batch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """
        Kohort CSV dosyası üzerinden toplu tarama, triyaj ve Wasserstein drift hesaplar.
        Uyumsuz/alakasız veri setlerini tespit eder, OOD anomali koruması uygular.
        """
        if batch_df is None or len(batch_df) == 0:
            raise ValueError("Yüklenen CSV dosyası boş veya okunamadı.")

        df_work = batch_df.copy()

        # Olası sütun eşleşmeleri ve normalizasyon
        col_map = {}
        for c in df_work.columns:
            c_clean = str(c).strip().lower()
            if c_clean in ["wbc", "wbc_x10k", "lökosit", "lokosit", "white_blood_cells"]:
                col_map[c] = "WBC_x10k"
            elif c_clean in ["hb", "hemoglobin", "hemoglobin_g_dl", "hgb"]:
                col_map[c] = "Hemoglobin_g_dL"
            elif c_clean in ["plt", "plt_x10k", "trombosit", "platelets", "platelet"]:
                col_map[c] = "PLT_x10k"
            elif c_clean in ["blast", "blasts", "blast_percentage", "blast_orani", "blast_yuzdesi"]:
                col_map[c] = "Blast_Percentage"
            elif c_clean in ["ldh", "ldh_u_l", "ldh_val"]:
                col_map[c] = "LDH_U_L"
            elif c_clean in ["sample_id", "hasta_id", "id", "patient_id"]:
                col_map[c] = "Sample_ID"

        if col_map:
            df_work.rename(columns=col_map, inplace=True)

        # Geçerli lösemi özniteliklerini kontrol et
        core_clinical = {"WBC_x10k", "Hemoglobin_g_dL", "PLT_x10k", "Blast_Percentage", "LDH_U_L"}
        key_genes = {"RUNX1", "FLT3", "PAX5", "IKZF1", "CD19", "WT1", "NPM1", "CEBPA", "MPO", "CD34"}

        matched_clinical = [c for c in df_work.columns if c in core_clinical]
        matched_genes = [c for c in df_work.columns if (c in key_genes or str(c).startswith("GENE_"))]
        total_matched = len(matched_clinical) + len(matched_genes)

        # Eğer yüklenen dosyada neredeyse hiçbir tanıdık parametre yoksa reddet!
        if total_matched < 2:
            raise ValueError(
                "Uyumsuz veya Geçersiz Veri Seti! "
                "Yüklenen CSV dosyasında lösemi klinik veya transkriptomik parametreleri (WBC, Blast %, PLT, Hb, LDH veya RUNX1, FLT3, PAX5 vb.) tespit edilemedi. "
                "Lütfen sistemin kabul ettiği kohort CSV formatını yükleyiniz veya arayüzdeki 'Örnek Şablon CSV İndir' butonundan şablonu temin ediniz."
            )

        # Eksik öznitelikleri kohort ortalamalarıyla güvenli ve tek seferde tamamla
        missing_cols = {col: self.mm.X[col].mean() for col in self.mm.X.columns if col not in df_work.columns}
        if missing_cols:
            df_missing = pd.DataFrame(missing_cols, index=df_work.index)
            df_work = pd.concat([df_work, df_missing], axis=1)

        features_sc = self.mm.scaler.transform(df_work[self.mm.X.columns])
        probs = self.mm.rf_model.predict_proba(features_sc)
        preds = self.mm.rf_model.predict(features_sc)
        ood_preds = self.mm.iso_forest.predict(features_sc)

        results = []
        triage_counts = {"Acil (Kırmızı)": 0, "Öncelikli (Sarı)": 0, "Rutin (Yeşil)": 0, "İnceleme Gerekli": 0}

        for pos_idx, (df_idx, row) in enumerate(df_work.iterrows()):
            p_class = preds[pos_idx]
            conf = float(np.max(probs[pos_idx]) * 100)
            blast_v = float(row.get('Blast_Percentage', 0.0))
            p_set = self.mm.compute_conformal_set(probs[pos_idx])
            is_anomaly = "OOD" if ood_preds[pos_idx] == -1 else "Normal"

            # OOD veya Aşırı Anormal Değer Koruması
            if is_anomaly == "OOD":
                display_pred = "Dağılım Dışı / Anomali (OOD)"
                display_conf = "%0.0"
                display_conformal = "Tanımlanamayan / Belirsiz"
                triage = "İnceleme Gerekli"
            elif blast_v >= 20.0 and p_class == 'Healthy':
                # WHO Blast Eşiği Güvenlik Ağı
                display_pred = "Şüpheli / İleri Tetkik Gerekli (Blast ≥ %20)"
                display_conf = f"%{conf:.1f}"
                display_conformal = ", ".join(p_set)
                triage = "Acil (Kırmızı)"
            elif blast_v >= 20.0 or (p_class != 'Healthy' and conf > 80):
                display_pred = p_class
                display_conf = f"%{conf:.1f}"
                display_conformal = ", ".join(p_set)
                triage = "Acil (Kırmızı)"
            elif p_class != 'Healthy':
                display_pred = p_class
                display_conf = f"%{conf:.1f}"
                display_conformal = ", ".join(p_set)
                triage = "Öncelikli (Sarı)"
            else:
                display_pred = p_class
                display_conf = f"%{conf:.1f}"
                display_conformal = ", ".join(p_set)
                triage = "Rutin (Yeşil)"

            triage_counts[triage] = triage_counts.get(triage, 0) + 1
            results.append({
                "Hasta ID": str(row.get('Sample_ID', f"Örnek_{pos_idx+1}")),
                "Ön Tanı": display_pred,
                "Model Güveni": display_conf,
                "%95 Conformal Küme": display_conformal,
                "OOD Durumu": is_anomaly,
                "Triyaj": triage
            })

        res_df = pd.DataFrame(results)

        active_triage = {k: v for k, v in triage_counts.items() if v > 0}
        if not active_triage:
            active_triage = {"Rutin (Yeşil)": 0}

        # Wasserstein Popülasyon Kayması (Drift)
        drift_metrics = []
        for col in ['WBC_x10k', 'Hemoglobin_g_dL', 'PLT_x10k', 'Blast_Percentage', 'LDH_U_L']:
            if col in df_work.columns:
                w_dist = float(wasserstein_distance(self.mm.df[col], df_work[col].dropna()))
                drift_status = "Belirgin Kayma" if w_dist > 5.0 else ("Hafif Kayma" if w_dist > 1.5 else "Stabil")
                drift_metrics.append({
                    "Parametre": col,
                    "Wasserstein Mesafesi": round(w_dist, 3),
                    "Popülasyon Kayma Durumu": drift_status
                })
        df_drift = pd.DataFrame(drift_metrics)

        return res_df, df_drift, active_triage

    def get_drug_synergy_matrix(self, pred_class: str, cytogenetics: str, age: int) -> pd.DataFrame:
        """Sitogenetik ve yaşa duyarlı ilaç sinerji tablosunu döndürür."""
        is_pediatric = age < 18
        if "AML" in pred_class:
            if "t(15;17)" in cytogenetics:
                data = [{"Hedef Onkogen": "PML::RARA", "Kombinasyon Rejimi": "ATRA + Arsenik Trioksit (ATO)", "Yanıt": "%96.5", "Sinerji": "2.10 (Küratif)"}]
            else:
                data = [
                    {"Hedef Onkogen": "FLT3 / ITD", "Kombinasyon Rejimi": "Gilteritinib + Standart 7+3", "Yanıt": "%84.2", "Sinerji": "1.42 (Yüksek Sinerji)"},
                    {"Hedef Onkogen": "BCL-2 / WT1", "Kombinasyon Rejimi": "Venetoclax + Azasitidin" if not is_pediatric else "Doz-Modifiye Venetoclax", "Yanıt": "%78.5", "Sinerji": "1.65 (Güçlü Sinerji)"}
                ]
        elif "ALL" in pred_class:
            if "t(9;22)" in cytogenetics:
                data = [{"Hedef Onkogen": "BCR::ABL1 (Ph+)", "Kombinasyon Rejimi": "Ponatinib + Blinatumomab", "Yanıt": "%92.0", "Sinerji": "1.90 (Güçlü Sinerji)"}]
            elif is_pediatric:
                data = [
                    {"Hedef Onkogen": "ETV6::RUNX1 / Pre-B", "Kombinasyon Rejimi": "BFM-Inspired Yoğunlaştırılmış Protokol", "Yanıt": "%94.0", "Sinerji": "1.85 (Yüksek Yanıt)"},
                    {"Hedef Onkogen": "CD19 / MRD+", "Kombinasyon Rejimi": "Blinatumomab Konsolidasyonu", "Yanıt": "%90.2", "Sinerji": "1.60 (Hedefe Yönelik)"}
                ]
            else:
                data = [
                    {"Hedef Onkogen": "CD19 / CD22", "Kombinasyon Rejimi": "Blinatumomab + Kemoterapi", "Yanıt": "%88.6", "Sinerji": "1.58 (Yüksek Sinerji)"},
                    {"Hedef Onkogen": "CD22", "Kombinasyon Rejimi": "Inotuzumab Ozogamicin", "Yanıt": "%74.1", "Sinerji": "1.30 (Orta Sinerji)"}
                ]
        else:
            data = [{"Hedef Onkogen": "Fizyolojik", "Kombinasyon Rejimi": "Hematolojik Takip (Tedavi Endikasyonu Yok)", "Yanıt": "N/A", "Sinerji": "N/A"}]
        return pd.DataFrame(data)

    def save_expert_review(
        self,
        doctor_id: str,
        patient_pred: str,
        physician_decision: str,
        clinical_override: str,
        additional_orders: List[str],
        notes: str
    ) -> Dict[str, Any]:
        """Uzman hekim doğrulamasını ve epikriz notunu JSON dosyasına kaydeder."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "doctor_id": doctor_id.strip() if doctor_id and doctor_id.strip() else "DR-ANON",
            "model_pred": patient_pred,
            "physician_decision": physician_decision,
            "clinical_override_diagnosis": clinical_override,
            "additional_tests_ordered": additional_orders,
            "expert_notes": notes
        }
        existing = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(record)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return record


_engine_instance: Optional[InferenceEngine] = None

def get_inference_engine() -> InferenceEngine:
    """Tekil (Singleton) InferenceEngine örneğini döndürür."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance
