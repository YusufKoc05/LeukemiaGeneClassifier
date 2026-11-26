"""
OncoVision AI Core Package
"""

from .config import CONFIG, THRESHOLDS, DISCLAIMER_TR, load_clinical_config
from .dataset import generate_leukemia_dataset, load_or_create_dataset, CBC_COLUMNS
from .models import ModelManager, get_model_manager
from .inference import InferenceEngine, get_inference_engine
from .reporting import generate_fhir_bundle, generate_clinical_pdf_report
from .metrics import (
    calculate_comprehensive_metrics,
    calculate_decision_curve_analysis,
    generate_clinical_table_one,
    calculate_cost_sensitive_loss,
    generate_latex_table
)
from .visualizations import (
    apply_plot_theme,
    build_static_evaluation_plots,
    compute_radar_profile,
    plot_confidence_gauge,
    plot_ppi_network,
    compute_pathway_enrichment,
    simulate_longitudinal_mrd,
    plot_dynamic_survival_curve,
    validate_and_detect_blood_smear,
    plot_decision_curve_analysis,
    plot_precision_recall_curves,
    COLOR_MAP
)

from .evaluation import (
    DEFAULT_DATASET_DIR,
    list_external_datasets,
    load_dataset_preview,
    evaluate_external_dataset,
    build_cross_dataset_comparison_table,
    build_cross_dataset_plots,
    build_cross_dataset_kpis_html
)
from .golub_parser import parse_and_unify_golub
from .ensemble import get_ensemble_classifier
from .geo_bridge import fetch_geo_cohort, list_supported_geo_studies
from .pharmacogenomics import evaluate_pharmacogenomic_profile, PGX_GENE_OPTIONS
from .who_subtypes import classify_who_2022_subtypes
from .case_comparison import build_patient_comparison_view
from .vision_classifier import classify_and_count_blood_smear

__all__ = [
    "CONFIG",
    "THRESHOLDS",
    "DISCLAIMER_TR",
    "load_clinical_config",
    "generate_leukemia_dataset",
    "load_or_create_dataset",
    "CBC_COLUMNS",
    "ModelManager",
    "get_model_manager",
    "InferenceEngine",
    "get_inference_engine",
    "generate_fhir_bundle",
    "generate_clinical_pdf_report",
    "apply_plot_theme",
    "build_static_evaluation_plots",
    "compute_radar_profile",
    "plot_confidence_gauge",
    "plot_ppi_network",
    "compute_pathway_enrichment",
    "simulate_longitudinal_mrd",
    "plot_dynamic_survival_curve",
    "validate_and_detect_blood_smear",
    "COLOR_MAP",
    "DEFAULT_DATASET_DIR",
    "list_external_datasets",
    "load_dataset_preview",
    "evaluate_external_dataset",
    "build_cross_dataset_comparison_table",
    "build_cross_dataset_plots",
    "build_cross_dataset_kpis_html",
    "parse_and_unify_golub",
    "get_ensemble_classifier",
    "fetch_geo_cohort",
    "list_supported_geo_studies",
    "evaluate_pharmacogenomic_profile",
    "PGX_GENE_OPTIONS",
    "classify_who_2022_subtypes",
    "build_patient_comparison_view",
    "classify_and_count_blood_smear"
]

