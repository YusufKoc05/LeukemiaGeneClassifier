"""
OncoVision AI — Arayüz (UI) Paketi
TÜBİTAK 2209-A Klinik Karar Destek Platformu
"""

from .styles import (
    CUSTOM_SLATE_CSS,
    build_disclaimer_banner_html,
    build_masthead_html,
    build_kpi_cards_html,
    build_xai_summary_card_html,
    build_academic_kpi_html,
    build_cost_summary_html,
    build_cf_summary_html,
    build_review_status_html
)

from .static_cache import (
    StaticResourceManager,
    get_static_resources
)

from .handlers import (
    PREDEFINED_CASES,
    handle_master_analysis,
    handle_cf_simulation,
    handle_batch_screening,
    handle_expert_review_save,
    handle_scan_directory,
    handle_dataset_selection,
    handle_run_evaluation,
    handle_case_comparison_ui,
    handle_pgx_calc,
    handle_one_click_blood_demo,
    handle_one_click_golub_demo,
    handle_one_click_geo_demo,
    handle_switch_cross_view,
    handle_refresh_cross_benchmark
)

from .components import (
    build_sidebar_controls,
    build_tab_diagnosis,
    build_tab_treatment,
    build_tab_metrics,
    build_tab_screening,
    build_tab_external_evaluation
)

__all__ = [
    "CUSTOM_SLATE_CSS",
    "build_disclaimer_banner_html",
    "build_masthead_html",
    "build_kpi_cards_html",
    "build_xai_summary_card_html",
    "build_academic_kpi_html",
    "build_cost_summary_html",
    "build_cf_summary_html",
    "build_review_status_html",
    "StaticResourceManager",
    "get_static_resources",
    "PREDEFINED_CASES",
    "handle_master_analysis",
    "handle_cf_simulation",
    "handle_batch_screening",
    "handle_expert_review_save",
    "handle_scan_directory",
    "handle_dataset_selection",
    "handle_run_evaluation",
    "handle_case_comparison_ui",
    "handle_pgx_calc",
    "handle_one_click_blood_demo",
    "handle_one_click_golub_demo",
    "handle_one_click_geo_demo",
    "handle_switch_cross_view",
    "handle_refresh_cross_benchmark",
    "build_sidebar_controls",
    "build_tab_diagnosis",
    "build_tab_treatment",
    "build_tab_metrics",
    "build_tab_screening",
    "build_tab_external_evaluation"
]
