"""
OncoVision AI — Onkolojik Karar Destek Platformu (Gradio Arayüzü)
TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Kapsamında Geliştirilmiştir.

Modüler, endüstri standardı mimari ile refaktör edilmiş ana uygulama giriş noktası.
"""

import gradio as gr
from core.config import DEFAULT_DATASET_DIR
from core.evaluation import list_external_datasets, load_dataset_preview
from core.vision_classifier import classify_and_count_blood_smear

from ui import (
    CUSTOM_SLATE_CSS,
    build_disclaimer_banner_html,
    build_masthead_html,
    get_static_resources,
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
    handle_refresh_cross_benchmark,
    build_sidebar_controls,
    build_tab_diagnosis,
    build_tab_treatment,
    build_tab_metrics,
    build_tab_screening,
    build_tab_external_evaluation
)

# 1. Statik Modeller ve İstatistikleri Hazırla
static_res = get_static_resources()

# 2. Harici Veri Setleri Başlangıç Taraması
initial_external_files = list_external_datasets(DEFAULT_DATASET_DIR)
initial_file_map = {name: path for name, path in initial_external_files}
if initial_external_files:
    initial_preview_df, initial_preview_info, initial_col_choices, initial_auto_map = load_dataset_preview(initial_external_files[0][1])
else:
    initial_preview_df, initial_preview_info, initial_col_choices, initial_auto_map = load_dataset_preview(None)


# 3. Gradio Uygulama Blokları
with gr.Blocks(title="OncoVision AI — TÜBİTAK 2209-A") as demo:

    gr.HTML(build_disclaimer_banner_html())
    gr.HTML(build_masthead_html())

    kpi_cards = gr.HTML()

    with gr.Row():
        # Sol Panel: Laboratuvar Giriş İstasyonu
        with gr.Column(scale=4):
            sidebar = build_sidebar_controls()

        # Sağ Panel: Sekmeli Analiz ve Rapor Konsolu
        with gr.Column(scale=8):
            out_xai_summary = gr.HTML()

            with gr.Tabs():
                with gr.Tab("Teşhis ve XAI"):
                    tab_diag = build_tab_diagnosis(sidebar["blasts"])

                with gr.Tab("Tedavi ve Biyoloji"):
                    tab_treat = build_tab_treatment()

                with gr.Tab("Model ve İstatistik"):
                    tab_stat = build_tab_metrics(static_res)

                with gr.Tab("Tarama ve Rapor"):
                    tab_screen = build_tab_screening()

                with gr.Tab("Harici Veri Seti Değerlendirme"):
                    tab_eval = build_tab_external_evaluation(
                        initial_files=initial_external_files,
                        initial_file_map=initial_file_map,
                        initial_preview_df=initial_preview_df,
                        initial_preview_info=initial_preview_info,
                        initial_col_choices=initial_col_choices,
                        initial_auto_map=initial_auto_map,
                        static_res=static_res
                    )

    # 4. Girdi ve Çıktı Vektörlerinin Eşleştirilmesi
    all_inputs = [
        sidebar["age"], sidebar["runx1"], sidebar["flt3"], sidebar["pax5"],
        sidebar["ikzf1"], sidebar["cd19"], sidebar["wt1"], sidebar["wbc"],
        sidebar["hb"], sidebar["plt_val"], sidebar["blasts"], sidebar["ldh"],
        sidebar["cytogenetics"], sidebar["mpo"], sidebar["cd34"]
    ]

    dynamic_outputs = [
        kpi_cards, sidebar["plot_gauge"], out_xai_summary, sidebar["out_label"],
        tab_screen["out_report"], tab_diag["plot_live_pca"], tab_diag["plot_live_cbc"],
        tab_diag["plot_live_shap"], tab_treat["plot_live_heatmap"], tab_diag["plot_km"],
        tab_treat["plot_ppi"], tab_treat["df_synergy"], tab_screen["out_pdf"],
        tab_screen["out_fhir"], tab_screen["txt_active_pred"], tab_treat["plot_live_pathway"],
        tab_treat["plot_live_mrd"], tab_diag["plot_live_radar"]
    ]

    # 5. Olay Dinleyicileri (Event Listeners)

    # Şablon Vaka Düğmeleri (DRY Prensibi)
    def create_case_loader(args_tuple):
        def _loader():
            results = handle_master_analysis(*args_tuple)
            return args_tuple + results
        return _loader

    sidebar["btn_aml"].click(
        fn=create_case_loader((58, 11.8, 12.5, 6.0, 5.0, 5.5, 11.2, 95.0, 6.2, 20.0, 65.0, 1100.0, "Karmaşık Karyotip (>=3 Anomali, Adverse)", 10.5, 9.8)),
        outputs=all_inputs + dynamic_outputs
    )
    sidebar["btn_apl"].click(
        fn=create_case_loader((34, 8.2, 6.5, 6.2, 6.5, 6.2, 7.8, 25.0, 7.8, 18.0, 60.0, 750.0, "t(15;17) PML::RARA (APL)", 14.2, 4.5)),
        outputs=all_inputs + dynamic_outputs
    )
    sidebar["btn_ped_all"].click(
        fn=create_case_loader((7, 6.2, 6.2, 12.0, 6.5, 12.5, 6.2, 65.0, 8.2, 45.0, 70.0, 620.0, "t(12;21) ETV6::RUNX1 (Pediatrik Favorable)", 3.8, 11.0)),
        outputs=all_inputs + dynamic_outputs
    )
    sidebar["btn_ph_all"].click(
        fn=create_case_loader((52, 6.5, 6.5, 11.8, 2.0, 12.5, 6.2, 120.0, 6.5, 18.0, 85.0, 1200.0, "t(9;22) BCR::ABL1 (Ph+ Yüksek Risk)", 4.8, 10.2)),
        outputs=all_inputs + dynamic_outputs
    )
    sidebar["btn_h"].click(
        fn=create_case_loader((28, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.8, 14.2, 260.0, 0.5, 180.0, "Normal Karyotip (Orta Risk)", 6.5, 6.5)),
        outputs=all_inputs + dynamic_outputs
    )

    # Ana Çıkarım Boru Hattı
    sidebar["btn_run"].click(
        fn=handle_master_analysis,
        inputs=all_inputs,
        outputs=dynamic_outputs
    )

    # Kan Yayması QC & Hücre Sınıflandırma
    tab_diag["btn_vision"].click(
        fn=classify_and_count_blood_smear,
        inputs=[tab_diag["img_input"], sidebar["blasts"]],
        outputs=[tab_diag["img_output"], sidebar["blasts"], tab_diag["vision_status"], tab_diag["df_vision_cells"]]
    )

    # Karşıolgusal Simülasyon
    tab_treat["btn_sim"].click(
        fn=handle_cf_simulation,
        inputs=all_inputs + [tab_treat["sim_target"], tab_treat["sim_intensity"]],
        outputs=[tab_treat["plot_cf"], tab_treat["out_sim_cards"]]
    )

    # Farmakogenomik (PGx) Dozaj ve Toksisite
    tab_treat["btn_pgx"].click(
        fn=handle_pgx_calc,
        inputs=[tab_treat["dd_tpmt"], tab_treat["dd_nudt15"], tab_treat["dd_dpyd"], tab_treat["regimen_choice"]],
        outputs=[tab_treat["out_pgx_card"], tab_treat["df_pgx_dosing"]]
    )

    # Vaka Karşılaştırma Modülü
    tab_treat["btn_compare_cases"].click(
        fn=handle_case_comparison_ui,
        inputs=[tab_treat["vaka_a_select"], tab_treat["vaka_b_select"]],
        outputs=[tab_treat["plot_compare_radar"], tab_treat["df_compare_diff"], tab_treat["out_compare_card"]]
    )

    # Toplu Tarama ve Kohort Triyajı
    tab_screen["btn_batch"].click(
        fn=handle_batch_screening,
        inputs=[tab_screen["batch_file"]],
        outputs=[tab_screen["batch_table"], tab_screen["batch_drift_plot"], tab_screen["batch_triage_plot"], tab_screen["batch_status"]]
    )

    # Uzman Hekim İnceleme Kaydı
    tab_screen["btn_save_review"].click(
        fn=handle_expert_review_save,
        inputs=[
            tab_screen["txt_doctor_id"], tab_screen["txt_active_pred"],
            tab_screen["physician_choice"], tab_screen["clinical_override"],
            tab_screen["additional_orders"], tab_screen["expert_note"]
        ],
        outputs=[tab_screen["lbl_save_status"]]
    )

    # Harici Veri Seti Toplu Test & Doğrulama
    tab_eval["btn_scan_dir"].click(
        fn=handle_scan_directory,
        inputs=[tab_eval["txt_dataset_dir"]],
        outputs=[tab_eval["dd_dataset_file"], tab_eval["state_file_map"]]
    )

    mapping_outputs = [
        tab_eval["df_dataset_preview"], tab_eval["lbl_dataset_info"],
        tab_eval["dd_target_col"], tab_eval["dd_id_col"], tab_eval["dd_wbc_col"],
        tab_eval["dd_hb_col"], tab_eval["dd_plt_col"], tab_eval["dd_blasts_col"],
        tab_eval["dd_ldh_col"]
    ]

    tab_eval["dd_dataset_file"].change(
        fn=handle_dataset_selection,
        inputs=[tab_eval["dd_dataset_file"], tab_eval["upload_dataset_file"], tab_eval["state_file_map"]],
        outputs=mapping_outputs
    )

    tab_eval["upload_dataset_file"].change(
        fn=handle_dataset_selection,
        inputs=[tab_eval["dd_dataset_file"], tab_eval["upload_dataset_file"], tab_eval["state_file_map"]],
        outputs=mapping_outputs
    )

    tab_eval["btn_run_eval"].click(
        fn=handle_run_evaluation,
        inputs=[
            tab_eval["dd_dataset_file"], tab_eval["upload_dataset_file"], tab_eval["state_file_map"],
            tab_eval["dd_target_col"], tab_eval["dd_id_col"], tab_eval["dd_wbc_col"],
            tab_eval["dd_hb_col"], tab_eval["dd_plt_col"], tab_eval["dd_blasts_col"],
            tab_eval["dd_ldh_col"], tab_eval["cb_assume_healthy"]
        ],
        outputs=[
            tab_eval["out_eval_kpis"], tab_eval["plot_eval_cm"],
            tab_eval["df_eval_class_metrics"], tab_eval["df_eval_results"],
            tab_eval["file_export_csv"]
        ]
    )

    eval_demo_outputs = [
        tab_eval["out_eval_kpis"], tab_eval["plot_eval_cm"],
        tab_eval["df_eval_class_metrics"], tab_eval["df_eval_results"],
        tab_eval["file_export_csv"]
    ]
    tab_eval["btn_demo_blood"].click(fn=handle_one_click_blood_demo, outputs=eval_demo_outputs)
    tab_eval["btn_demo_golub"].click(fn=handle_one_click_golub_demo, outputs=eval_demo_outputs)
    tab_eval["btn_demo_geo"].click(fn=handle_one_click_geo_demo, outputs=eval_demo_outputs)

    # Tüm Kaggle & Benchmark Veri Setleri Karşılaştırma Olay Dinleyicileri
    tab_eval["rd_cross_view"].change(
        fn=handle_switch_cross_view,
        inputs=[tab_eval["rd_cross_view"]],
        outputs=[tab_eval["plot_cross_main"]]
    )
    tab_eval["btn_refresh_cross"].click(
        fn=handle_refresh_cross_benchmark,
        outputs=[
            tab_eval["out_cross_kpis"],
            tab_eval["plot_cross_main"],
            tab_eval["plot_cross_secondary"],
            tab_eval["df_cross_table"]
        ]
    )

    # Başlangıçta Varsayılan Profil ile Yükleme
    demo.load(fn=handle_master_analysis, inputs=all_inputs, outputs=dynamic_outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"), css=CUSTOM_SLATE_CSS)