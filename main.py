"""
OncoVision AI - Klinik Karar Destek REST API (FastAPI)
WHO 2022 Kriterleri, Split Conformal Prediction, XAI (SHAP) ve Güvenlik Ağları Destekli Servis.
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

from core import (
    CONFIG,
    THRESHOLDS,
    DISCLAIMER_TR,
    get_model_manager,
    get_inference_engine,
    generate_clinical_pdf_report,
    generate_fhir_bundle,
    calculate_comprehensive_metrics,
    calculate_decision_curve_analysis,
    generate_clinical_table_one,
    calculate_cost_sensitive_loss
)

app = FastAPI(
    title="OncoVision AI — Klinik Karar Destek REST API",
    description=(
        "RNA-Seq transkriptomik profili, akım sitometrisi ve tam kan sayımı (CBC) parametreleri üzerinden "
        "lösemi alt tipi sınıflandırması yapan, WHO 2022 blast eşiği, DIC koagülopati koruması, "
        "Split Conformal Prediction ve XAI (SHAP) barındıran klinik karar destek API'si."
    ),
    version="2.5"
)

# Model ve Çıkarım Motorları
mm = get_model_manager()
engine = get_inference_engine()


class PatientDiagnosticInput(BaseModel):
    age: int = Field(45, ge=1, le=100, description="Hasta Yaşı (Pediatrik / Erişkin Eşiği: 18)")
    runx1: float = Field(6.5, description="RUNX1 (log2 TPM)")
    flt3: float = Field(6.5, description="FLT3 (log2 TPM)")
    wt1: float = Field(6.5, description="WT1 (log2 TPM)")
    pax5: float = Field(6.5, description="PAX5 (log2 TPM)")
    cd19: float = Field(6.5, description="CD19 (log2 TPM)")
    ikzf1: float = Field(6.5, description="IKZF1 (log2 TPM)")
    mpo: float = Field(6.5, description="Miyeloperoksidaz (MPO) ekspresyonu")
    cd34: float = Field(6.5, description="Kök hücre belirteci CD34")
    wbc: float = Field(6.8, description="Lökosit sayısı (x10^3/uL)")
    hb: float = Field(14.2, description="Hemoglobin düzeyi (g/dL)")
    plt_val: float = Field(260.0, description="Trombosit sayısı (x10^3/uL)")
    blasts: float = Field(0.5, description="Blast hücre yüzdesi (%)")
    ldh: float = Field(180.0, description="LDH (U/L)")
    cytogenetics: str = Field("Normal Karyotip (Orta Risk)", description="Sitogenetik bulgular")


class SimulationInput(BaseModel):
    patient: PatientDiagnosticInput
    target_intervention: str = Field(
        "FLT3 İnhibisyonu (Gilteritinib / Midostaurin)",
        description="Hedef Müdahale: FLT3 İnhibisyonu (Gilteritinib / Midostaurin), Anti-CD19 / BiTE Simülasyonu (Blinatumomab), veya Kemoterapi Blast Redüksiyonu"
    )
    intensity_pct: float = Field(50.0, ge=0.0, le=100.0, description="Müdahale Şiddeti (%)")


@app.get("/")
def root():
    return {
        "status": "Online",
        "service": "OncoVision AI Karar Destek API",
        "version": "2.5 (TÜBİTAK 2209-A Standart)",
        "disclaimer": DISCLAIMER_TR
    }


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "models_loaded": mm.rf_model is not None,
        "classes": mm.classes,
        "num_features": mm.X.shape[1]
    }


@app.get("/model-info")
def get_model_info():
    return {
        "model_type": "Random Forest Classifier (XAI Entegre)",
        "n_estimators": 150,
        "classes": mm.classes,
        "num_features": mm.X.shape[1],
        "thresholds": THRESHOLDS,
        "conformal_q_threshold": round(mm.q_conformal, 4)
    }


@app.get("/benchmarks")
def get_benchmarks():
    return {
        "model_zoo_benchmarks": mm.benchmarks
    }


@app.get("/metrics/advanced")
def get_advanced_metrics():
    df_metrics, macro_summary = calculate_comprehensive_metrics(
        mm.y_test, mm.rf_model.predict(mm.X_test_sc), mm.rf_model.predict_proba(mm.X_test_sc), mm.classes
    )
    cost_loss = calculate_cost_sensitive_loss(mm.y_test, mm.rf_model.predict(mm.X_test_sc))
    return {
        "macro_summary": macro_summary,
        "class_diagnostic_metrics": df_metrics.to_dict(orient="records"),
        "cost_sensitive_evaluation": cost_loss
    }


@app.get("/metrics/table-one")
def get_table_one():
    df_table_one = generate_clinical_table_one(mm.df)
    return {
        "table_one": df_table_one.to_dict(orient="records")
    }


@app.get("/metrics/dca")
def get_decision_curve_analysis(target_class: str = "AML"):
    if target_class not in mm.classes:
        raise HTTPException(status_code=400, detail=f"Geçersiz hedef sınıf: {target_class}. Mevcut sınıflar: {mm.classes}")
    df_dca = calculate_decision_curve_analysis(
        mm.y_test, mm.rf_model.predict_proba(mm.X_test_sc), mm.classes, target_class=target_class
    )
    return {
        "target_class": target_class,
        "net_benefit_curve": df_dca.to_dict(orient="records")
    }


@app.post("/predict")
def predict_patient(data: PatientDiagnosticInput):
    res = engine.predict_patient(
        age=data.age,
        runx1=data.runx1,
        flt3=data.flt3,
        pax5=data.pax5,
        ikzf1=data.ikzf1,
        cd19=data.cd19,
        wt1=data.wt1,
        wbc=data.wbc,
        hb=data.hb,
        plt_val=data.plt_val,
        blasts=data.blasts,
        ldh=data.ldh,
        cytogenetics=data.cytogenetics,
        mpo=data.mpo,
        cd34=data.cd34
    )

    return {
        "raw_model_prediction": res["raw_model_prediction"],
        "final_clinical_class": res["final_clinical_class"],
        "confidence_percentage": res["confidence_percentage"],
        "epistemic_variance": res["epistemic_variance"],
        "uncertainty_level": res["uncertainty_level"],
        "safety_net_triggered": res["safety_net_triggered"],
        "dic_coagulopathy_alert": res["dic_coagulopathy_alert"],
        "clinical_warning": res["clinical_warning"],
        "actionable_protocol": res["actionable_protocol"],
        "risk_stratification": res["risk_stratification"],
        "conformal_prediction_set": res["conformal_prediction_set"],
        "ood_status": res["ood_status"],
        "ood_score": res["ood_score"],
        "shap_summary": res["shap_summary_text"],
        "class_probabilities": res["class_probabilities"],
        "disclaimer": res["disclaimer"]
    }


@app.post("/simulate")
def simulate_intervention(sim_input: SimulationInput):
    p = sim_input.patient
    orig_probs, sim_probs, delta_healthy = engine.run_counterfactual_simulation(
        base_age=p.age,
        base_runx1=p.runx1,
        base_flt3=p.flt3,
        base_pax5=p.pax5,
        base_ikzf1=p.ikzf1,
        base_cd19=p.cd19,
        base_wt1=p.wt1,
        base_wbc=p.wbc,
        base_hb=p.hb,
        base_plt=p.plt_val,
        base_blasts=p.blasts,
        base_ldh=p.ldh,
        cytogenetics=p.cytogenetics,
        target_intervention=sim_input.target_intervention,
        intensity_pct=sim_input.intensity_pct
    )

    return {
        "intervention": sim_input.target_intervention,
        "intensity_pct": sim_input.intensity_pct,
        "baseline_probabilities": {cls: round(float(prob * 100), 2) for cls, prob in zip(mm.classes, orig_probs)},
        "post_intervention_probabilities": {cls: round(float(prob * 100), 2) for cls, prob in zip(mm.classes, sim_probs)},
        "healthy_convergence_delta_pct": round(float(delta_healthy), 2)
    }


@app.post("/batch-predict")
async def batch_screening(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Yalnızca CSV dosyaları desteklenmektedir.")

    try:
        df_batch = pd.read_csv(file.file)
        res_df, df_drift, triage_counts = engine.run_batch_screening(df_batch)
        return {
            "total_screened": len(res_df),
            "triage_summary": triage_counts,
            "drift_analysis": df_drift.to_dict(orient="records"),
            "patient_results": res_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toplu tarama hatası: {str(e)}")


@app.post("/report/pdf")
def export_pdf_report(data: PatientDiagnosticInput):
    res = engine.predict_patient(
        age=data.age, runx1=data.runx1, flt3=data.flt3, pax5=data.pax5,
        ikzf1=data.ikzf1, cd19=data.cd19, wt1=data.wt1, wbc=data.wbc,
        hb=data.hb, plt_val=data.plt_val, blasts=data.blasts, ldh=data.ldh,
        cytogenetics=data.cytogenetics, mpo=data.mpo, cd34=data.cd34
    )

    blast_limit = THRESHOLDS.get("who_blast_acute_threshold", 20.0)
    early_dx_badge = f"Akut Lösemi Evresi (Blast ≥ %{blast_limit})" if data.blasts >= blast_limit else "Fizyolojik Dağılım"

    pdf_path = generate_clinical_pdf_report(
        pred_class=res["final_clinical_class"],
        confidence=res["confidence_percentage"],
        risk_strat=res["risk_stratification"],
        cytogenetics=data.cytogenetics,
        conformal_set_str=res["conformal_set_str"],
        ood_status=res["ood_status"],
        uncertainty_str=res["uncertainty_str"],
        early_dx_badge=early_dx_badge,
        is_safety_override=res["safety_net_triggered"],
        shap_summary_text=res["shap_summary_text"],
        wbc=data.wbc,
        hb=data.hb,
        plt_val=data.plt_val,
        blasts=data.blasts,
        ldh=data.ldh,
        age=data.age,
        dic_alert=res["dic_coagulopathy_alert"],
        output_path="klinik_epikriz_raporu.pdf",
        mpo=data.mpo,
        cd34=data.cd34,
        genes_dict={"RUNX1": data.runx1, "FLT3": data.flt3, "WT1": data.wt1, "PAX5": data.pax5, "CD19": data.cd19, "IKZF1": data.ikzf1},
        actionable_protocol=res["actionable_protocol"]
    )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"OncoVision_Rapor_{data.age}yas_{res['final_clinical_class']}.pdf"
    )


@app.post("/report/fhir")
def export_fhir_bundle(data: PatientDiagnosticInput):
    res = engine.predict_patient(
        age=data.age, runx1=data.runx1, flt3=data.flt3, pax5=data.pax5,
        ikzf1=data.ikzf1, cd19=data.cd19, wt1=data.wt1, wbc=data.wbc,
        hb=data.hb, plt_val=data.plt_val, blasts=data.blasts, ldh=data.ldh,
        cytogenetics=data.cytogenetics, mpo=data.mpo, cd34=data.cd34
    )

    fhir_path = generate_fhir_bundle(
        pred_class=res["final_clinical_class"],
        confidence=res["confidence_percentage"],
        risk_strat=res["risk_stratification"],
        cytogenetics=data.cytogenetics,
        conformal_set_str=res["conformal_set_str"],
        ood_status=res["ood_status"],
        wbc=data.wbc,
        hb=data.hb,
        plt_val=data.plt_val,
        blasts=data.blasts,
        ldh=data.ldh,
        age=data.age,
        output_path="fhir_diagnostic_report.json"
    )

    with open(fhir_path, "r", encoding="utf-8") as f:
        import json
        content = json.load(f)
    return content