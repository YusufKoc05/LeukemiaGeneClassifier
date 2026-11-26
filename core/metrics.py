"""
OncoVision AI - Core Biostatistical & Academic Metrics Module
TÜBİTAK 2209-A ve tıp literatürü standartlarında tanısal metrikler,
Decision Curve Analysis (DCA), Klinik Table 1 ve Asimetrik Maliyet Analizi.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import f_oneway, kruskal
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    auc,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize


def calculate_comprehensive_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    TÜBİTAK ve biyoenformatik literatürüne uygun kapsamlı tanısal metrik matrisi hesaplar:
    Duyarlılık (Sensitivity), Özgüllük (Specificity), PPV, NPV, Dengeli Doğruluk,
    Matthews Korelasyon Katsayısı (MCC), Cohen's Kappa (κ), Brier Skoru ve AUPRC.
    Null, sıfıra bölme ve float hassasiyeti kontrolleri eklenmiştir.
    """
    # 1. Null / Undefined ve uyumsuzluk temizliği
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_pred_arr = np.array(y_pred)
    
    valid_mask = pd.notna(y_true_s) & pd.notna(y_pred_arr)
    if y_prob is not None:
        valid_mask = valid_mask & (~np.isnan(np.array(y_prob)).any(axis=1))

    y_true_clean = y_true_s[valid_mask].values
    y_pred_clean = y_pred_arr[valid_mask]
    y_prob_clean = np.array(y_prob)[valid_mask] if y_prob is not None else None

    if len(y_true_clean) == 0:
        empty_df = pd.DataFrame(columns=["Klinik Sınıf", "Duyarlılık (Sensitivity)", "Özgüllük (Specificity)", "PPV (Kesinlik)", "NPV (Neg. Tahmin)", "F1-Skoru", "AUPRC", "Brier Skoru"])
        return empty_df, {"Dengeli Doğruluk (Balanced Acc)": "%0.00", "Matthews Korelasyon Katsayısı (MCC)": "0.0000", "Cohen's Kappa (κ)": "0.0000", "Cross-Entropy / Log-Loss": "0.0000"}

    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=classes)
    total_samples = int(np.sum(cm))
    y_true_bin = label_binarize(y_true_clean, classes=classes)
    if len(classes) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    class_metrics = []

    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - (tp + fp + fn))

        # Standart istatistiksel oranlar ve sıfıra bölme (division by zero) koruması
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float((2.0 * ppv * sens) / (ppv + sens)) if (ppv + sens) > 0 else 0.0

        # AUPRC (Precision-Recall Area)
        auprc_val = 0.0
        if y_prob_clean is not None and i < y_prob_clean.shape[1] and np.sum(y_true_bin[:, i]) > 0:
            try:
                prec_arr, rec_arr, _ = precision_recall_curve(y_true_bin[:, i], y_prob_clean[:, i])
                calc_auc = auc(rec_arr, prec_arr)
                auprc_val = float(calc_auc) if not np.isnan(calc_auc) else 0.0
            except Exception:
                auprc_val = 0.0

        # Sınıf bazlı Brier skoru (0 = mükemmel, 1 = kötü)
        brier_val = 0.0
        if y_prob_clean is not None and i < y_prob_clean.shape[1]:
            try:
                b_score = brier_score_loss(y_true_bin[:, i], y_prob_clean[:, i])
                brier_val = float(b_score) if not np.isnan(b_score) else 0.0
            except Exception:
                brier_val = 0.0

        class_metrics.append({
            "Klinik Sınıf": str(cls),
            "Duyarlılık (Sensitivity)": f"%{sens * 100.0:.2f}",
            "Özgüllük (Specificity)": f"%{spec * 100.0:.2f}",
            "PPV (Kesinlik)": f"%{ppv * 100.0:.2f}",
            "NPV (Neg. Tahmin)": f"%{npv * 100.0:.2f}",
            "F1-Skoru": f"{f1:.4f}",
            "AUPRC": f"{auprc_val:.4f}",
            "Brier Skoru": f"{brier_val:.4f}"
        })

    # Genel / Makro Düzey Özet (Ağırlıklı ve dengeli metrikler)
    bal_acc = float(balanced_accuracy_score(y_true_clean, y_pred_clean)) if total_samples > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true_clean, y_pred_clean)) if total_samples > 0 else 0.0
    kappa = float(cohen_kappa_score(y_true_clean, y_pred_clean)) if total_samples > 0 else 0.0
    
    loss_val = 0.0
    if y_prob_clean is not None:
        try:
            loss_calc = log_loss(y_true_clean, y_prob_clean, labels=classes)
            loss_val = float(loss_calc) if not np.isnan(loss_calc) else 0.0
        except Exception:
            loss_val = 0.0

    summary_df = pd.DataFrame(class_metrics)
    return summary_df, {
        "Dengeli Doğruluk (Balanced Acc)": f"%{bal_acc * 100.0:.2f}",
        "Matthews Korelasyon Katsayısı (MCC)": f"{mcc:.4f}",
        "Cohen's Kappa (κ)": f"{kappa:.4f}",
        "Cross-Entropy / Log-Loss": f"{loss_val:.4f}"
    }


def calculate_decision_curve_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    classes: List[str],
    target_class: str = "AML",
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Klinik Karar Eğrisi Analizi (Decision Curve Analysis - DCA).
    Vickers et al. (Memorial Sloan Kettering Cancer Center) metodolojisiyle
    farklı klinik eşik olasılıklarındaki (Threshold Probability) Net Faydayı (Net Benefit) hesaplar.
    Division by zero ve sınır değerler (pt -> 1.0) matematiksel olarak korunmuştur.
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.95, 50)

    if target_class not in classes:
        target_idx = 0
    else:
        target_idx = classes.index(target_class)

    y_binary = (pd.Series(y_true).astype(str) == str(target_class)).astype(int).values
    probs = np.array(y_prob)[:, target_idx]

    n = len(y_binary)
    if n == 0:
        return pd.DataFrame()

    prevalence = float(np.sum(y_binary)) / n

    dca_records = []
    for pt in thresholds:
        pt_clamped = float(np.clip(pt, 0.001, 0.999))
        y_pred = (probs >= pt_clamped).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_binary == 1)))
        fp = int(np.sum((y_pred == 1) & (y_binary == 0)))

        # Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        weight = pt_clamped / max(1.0 - pt_clamped, 1e-7)
        net_benefit_model = (tp / n) - (fp / n) * weight

        # "Tümünü Tedavi Et" (Treat All) Stratejisi
        net_benefit_all = prevalence - (1.0 - prevalence) * weight

        # "Hiçbirini Tedavi Etme" (Treat None) Stratejisi
        net_benefit_none = 0.0

        dca_records.append({
            "Eşik Olasılık (Threshold)": round(pt_clamped, 3),
            "Eşik (%)": round(pt_clamped * 100.0, 1),
            "OncoVision AI Net Fayda": round(float(net_benefit_model), 4),
            "Tümünü Tedavi Et (Treat All)": round(float(net_benefit_all), 4),
            "Hiçbirini Tedavi Etme (Treat None)": round(float(net_benefit_none), 4)
        })

    return pd.DataFrame(dca_records)


def generate_clinical_table_one(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uluslararası klinik makalelerin (NEJM, Blood, Lancet) standart açılış tablosu:
    'Table 1: Baseline Demographic & Hematologic Characteristics'.
    Sağlıklı, AML ve ALL grupları için Ortalama ± SS, Medyan [IQR] ve
    One-Way ANOVA & Kruskal-Wallis p-değerlerini dinamik örneklem boyutuyla hesaplar.
    Çift sayma (double-counting) ve hardcoded N değerleri engellenmiştir.
    """
    params = [
        ('WBC_x10k', 'Lökosit (WBC)', 'x10³/µL'),
        ('Hemoglobin_g_dL', 'Hemoglobin (Hb)', 'g/dL'),
        ('PLT_x10k', 'Trombosit (PLT)', 'x10³/µL'),
        ('Blast_Percentage', 'Blast Hücre Oranı', '%'),
        ('LDH_U_L', 'Laktat Dehidrogenaz (LDH)', 'U/L')
    ]

    groups = ['Healthy', 'AML', 'ALL']
    rows = []

    for col, display_name, unit in params:
        if col not in df.columns:
            continue

        clean_col = df[col].dropna()
        n_total = len(clean_col)
        overall_mean = float(clean_col.mean()) if n_total > 0 else 0.0
        overall_std = float(clean_col.std(ddof=1)) if n_total > 1 else 0.0
        overall_med = float(clean_col.median()) if n_total > 0 else 0.0

        row_data = {
            "Klinik Parametre": f"{display_name} ({unit})",
            f"Genel Kohort (N={n_total})": f"{overall_mean:.1f} ± {overall_std:.1f} [{overall_med:.1f}]"
        }

        group_arrays = []
        for grp in groups:
            sub = df[df['Leukemia_Subtype'] == grp][col].dropna()
            n_grp = len(sub)
            group_arrays.append(sub.values)
            m = float(sub.mean()) if n_grp > 0 else 0.0
            s = float(sub.std(ddof=1)) if n_grp > 1 else 0.0
            med = float(sub.median()) if n_grp > 0 else 0.0
            row_data[f"{grp} (N={n_grp})"] = f"{m:.1f} ± {s:.1f} [{med:.1f}]"

        # Hipotez Testleri: One-Way ANOVA ve Kruskal-Wallis
        try:
            valid_arrays = [arr for arr in group_arrays if len(arr) > 1 and np.var(arr) > 0]
            if len(valid_arrays) == len(group_arrays):
                _, anova_p = f_oneway(*group_arrays)
                p_anova_str = "< 0.001" if anova_p < 0.001 else f"{anova_p:.4f}"
            else:
                p_anova_str = "N/A"
        except Exception:
            p_anova_str = "N/A"

        try:
            valid_arrays = [arr for arr in group_arrays if len(arr) > 1]
            if len(valid_arrays) == len(group_arrays):
                _, kw_p = kruskal(*group_arrays)
                p_kw_str = "< 0.001" if kw_p < 0.001 else f"{kw_p:.4f}"
            else:
                p_kw_str = "N/A"
        except Exception:
            p_kw_str = "N/A"

        row_data["ANOVA p-değeri"] = p_anova_str
        row_data["Kruskal-Wallis p-değeri"] = p_kw_str
        rows.append(row_data)

    return pd.DataFrame(rows)


def calculate_cost_sensitive_loss(
    y_true: pd.Series,
    y_pred: np.ndarray,
    fn_weight: float = 10.0,
    fp_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Asimetrik Klinik Maliyet Analizi:
    Bir lösemi hastasını 'Sağlıklı' olarak yanlış sınıflandırmanın maliyeti (FN) = 10x
    Bir sağlıklı hastayı 'Lösemi' olarak yanlış sınıflandırmanın maliyeti (FP) = 1x
    """
    classes = ['Healthy', 'AML', 'ALL']
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Sağlıklı indeks 0
    # Yanlış Negatifler: Gerçek AML veya ALL iken modelin Healthy demesi
    fn_aml = int(cm[1, 0])  # Gerçek AML, Tahmin Healthy
    fn_all = int(cm[2, 0])  # Gerçek ALL, Tahmin Healthy
    total_lethal_fn = fn_aml + fn_all

    # Yanlış Pozitifler: Gerçek Healthy iken modelin AML veya ALL demesi
    fp_healthy = int(cm[0, 1] + cm[0, 2])

    # Toplam Asimetrik Klinik Maliyet Skoru
    clinical_loss = float((total_lethal_fn * fn_weight) + (fp_healthy * fp_weight))

    return {
        "Hayati Yanlış Negatifler (Kanser Kaçırma)": int(total_lethal_fn),
        "Yanlış Pozitifler (Gereksiz Biyopsi)": int(fp_healthy),
        "Ağırlıklı Klinik Maliyet Skoru": round(clinical_loss, 2),
        "Maliyet Cezası Çarpanı": f"FN = {fn_weight}x, FP = {fp_weight}x"
    }


def generate_latex_table(df_metrics: pd.DataFrame, caption: str = "Model Tanısal Performans Metrikleri") -> str:
    """
    TÜBİTAK 2209-A proje raporuna veya akademik makaleye doğrudan eklenebilir LaTeX tablo kodu üretir.
    """
    latex_code = df_metrics.to_latex(index=False, caption=caption, label="tab:model_metrics", column_format="l" + "c" * (len(df_metrics.columns) - 1))
    return latex_code
