"""
OncoVision AI - Core Dataset Module
Sentetik ve biyolojik olarak temellendirilmiş RNA-Seq + CBC kohort üreteci ve veri yükleyicisi.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from .config import THRESHOLDS

CBC_COLUMNS = ['WBC_x10k', 'Hemoglobin_g_dL', 'PLT_x10k', 'Blast_Percentage', 'LDH_U_L']
AML_MYELOID_PANEL = ['RUNX1', 'FLT3', 'NPM1', 'CEBPA', 'DNMT3A', 'KIT', 'WT1', 'IDH1', 'IDH2', 'TET2', 'MPO', 'CD33']
ALL_LYMPHOID_PANEL = ['PAX5', 'IKZF1', 'NOTCH1', 'EBF1', 'CD19', 'CD22', 'CD3', 'TDT', 'IL7R', 'CRLF2', 'TCF3', 'ETV6']
STEM_CELL_PANEL = ['CD34']
HOUSEKEEPING_GENES = ['ACTB', 'GAPDH', 'B2M', 'RPLP0']


def generate_leukemia_dataset(num_samples: int = 450, num_genes: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    WHO 2022 / ICC ve GSE13159 (MILE Study) transkriptomik profiline,
    akım sitometrisi (Flow Cytometry CD belirteçleri) ve CBC değerlerine dayalı
    kapsamlı sentetik lösemi kohort üreteci.
    """
    np.random.seed(seed)
    n_per_class = num_samples // 3
    labels = np.array(['Healthy'] * n_per_class + ['AML'] * n_per_class + ['ALL'] * n_per_class)

    # 1. Gen Panelleri (Transkriptomik & İmmünolojik)
    known_genes = AML_MYELOID_PANEL + ALL_LYMPHOID_PANEL + STEM_CELL_PANEL + HOUSEKEEPING_GENES
    remaining_count = num_genes - len(known_genes)
    other_genes = [f"GENE_{i+1:04d}" for i in range(remaining_count)]
    all_gene_names = known_genes + other_genes

    # 2. Bazal Ekspresyon Matrisi (log2 TPM)
    data = np.random.normal(loc=6.5, scale=1.0, size=(num_samples, num_genes))
    gene_map = {name: idx for idx, name in enumerate(all_gene_names)}

    # Housekeeping genleri
    for hk in HOUSEKEEPING_GENES:
        data[:, gene_map[hk]] = np.random.normal(12.5, 0.35, size=num_samples)

    aml_idx = np.where(labels == 'AML')[0]
    all_idx = np.where(labels == 'ALL')[0]
    healthy_idx = np.where(labels == 'Healthy')[0]

    # 3. Biyolojik Sinyal Simülasyonu
    # AML Grubu
    aml_latent = np.random.normal(3.5, 0.7, size=len(aml_idx))
    data[aml_idx, gene_map['RUNX1']] += aml_latent * 0.95 + np.random.normal(0, 0.3, len(aml_idx))
    data[aml_idx, gene_map['FLT3']]  += aml_latent * 0.90 + np.random.normal(0, 0.35, len(aml_idx))
    data[aml_idx, gene_map['NPM1']]  += np.random.normal(2.6, 0.5, size=len(aml_idx))
    data[aml_idx, gene_map['WT1']]   += np.random.normal(3.9, 0.8, size=len(aml_idx))
    data[aml_idx, gene_map['MPO']]   += np.random.normal(4.2, 0.7, size=len(aml_idx))  # AML Patognomonik
    data[aml_idx, gene_map['CD33']]  += np.random.normal(3.8, 0.6, size=len(aml_idx))
    data[aml_idx, gene_map['CD34']]  += np.random.normal(2.5, 0.9, size=len(aml_idx))
    data[aml_idx, gene_map['PAX5']]  -= np.random.normal(1.8, 0.4, size=len(aml_idx))
    data[aml_idx, gene_map['CD19']]  -= np.random.normal(1.6, 0.4, size=len(aml_idx))

    # ALL Grubu
    all_latent = np.random.normal(3.6, 0.8, size=len(all_idx))
    data[all_idx, gene_map['PAX5']]  += all_latent * 0.98 + np.random.normal(0, 0.25, len(all_idx))
    data[all_idx, gene_map['CD19']]  += all_latent * 0.92 + np.random.normal(0, 0.3, len(all_idx))
    data[all_idx, gene_map['CD22']]  += np.random.normal(3.3, 0.6, size=len(all_idx))
    data[all_idx, gene_map['TDT']]   += np.random.normal(3.7, 0.7, size=len(all_idx))
    data[all_idx, gene_map['NOTCH1']]+= np.random.normal(2.8, 0.6, size=len(all_idx))
    data[all_idx, gene_map['CD34']]  += np.random.normal(2.8, 0.8, size=len(all_idx))
    data[all_idx, gene_map['IKZF1']] -= np.random.normal(3.0, 0.5, size=len(all_idx))
    data[all_idx, gene_map['MPO']]   -= np.random.normal(1.8, 0.3, size=len(all_idx))

    # Sınır Vakalar (Edge Cases - Erken/Atipik Fenotipler)
    aml_edge_count = max(2, int(len(aml_idx) * 0.10))
    all_edge_count = max(2, int(len(all_idx) * 0.10))
    aml_edge_idx = aml_idx[:aml_edge_count]
    all_edge_idx = all_idx[:all_edge_count]

    for g in ['RUNX1', 'FLT3', 'MPO', 'WT1']:
        data[aml_edge_idx, gene_map[g]] = np.random.normal(6.5, 0.4, aml_edge_count)
    for g in ['PAX5', 'CD19', 'TDT', 'IKZF1']:
        data[all_edge_idx, gene_map[g]] = np.random.normal(6.5, 0.4, all_edge_count)

    # Ölçüm gürültüsü
    data += np.random.normal(0, 0.20, size=data.shape)

    # 4. Hematolojik Parametreler (CBC)
    wbc = np.zeros(num_samples)
    hb = np.zeros(num_samples)
    plt = np.zeros(num_samples)
    blasts = np.zeros(num_samples)
    ldh = np.zeros(num_samples)

    # Kontrol Grubu
    wbc[healthy_idx] = np.random.normal(6.8, 1.3, len(healthy_idx))
    hb[healthy_idx] = np.random.normal(14.2, 1.0, len(healthy_idx))
    plt[healthy_idx] = np.random.normal(260.0, 42.0, len(healthy_idx))
    blasts[healthy_idx] = np.random.uniform(0.0, 1.8, len(healthy_idx))
    ldh[healthy_idx] = np.random.normal(180.0, 22.0, len(healthy_idx))

    # AML Olguları (WHO blast >= %20)
    wbc[aml_idx] = np.random.lognormal(mean=3.7, sigma=0.55, size=len(aml_idx))
    hb[aml_idx] = np.random.normal(7.4, 0.85, len(aml_idx))
    plt[aml_idx] = np.random.gamma(shape=3.5, scale=8.0, size=len(aml_idx))
    blasts[aml_idx] = np.random.uniform(22.0, 88.0, len(aml_idx))
    ldh[aml_idx] = np.random.normal(680.0, 150.0, len(aml_idx))

    # ALL Olguları
    wbc[all_idx] = np.random.lognormal(mean=4.1, sigma=0.6, size=len(all_idx))
    hb[all_idx] = np.random.normal(8.1, 0.95, len(all_idx))
    plt[all_idx] = np.random.gamma(shape=4.0, scale=8.5, size=len(all_idx))
    blasts[all_idx] = np.random.uniform(30.0, 92.0, len(all_idx))
    ldh[all_idx] = np.random.normal(850.0, 190.0, len(all_idx))

    # Sınır vakaların blastlarını yüksek tutarak modelin morfolojik ağırlığı kavraması sağlanır
    blasts[aml_edge_idx] = np.random.uniform(35.0, 85.0, aml_edge_count)
    blasts[all_edge_idx] = np.random.uniform(40.0, 90.0, all_edge_count)

    sample_ids = [f"GSM_{i+2001:05d}" for i in range(num_samples)]
    df = pd.DataFrame(data, index=sample_ids, columns=all_gene_names)

    df['WBC_x10k'] = np.clip(wbc, 1.0, 180.0).round(2)
    df['Hemoglobin_g_dL'] = np.clip(hb, 3.5, 18.0).round(2)
    df['PLT_x10k'] = np.clip(plt, 4.0, 550.0).round(1)
    df['Blast_Percentage'] = np.clip(blasts, 0.0, 98.0).round(1)
    df['LDH_U_L'] = np.clip(ldh, 100.0, 1800.0).round(0)
    df['Leukemia_Subtype'] = labels

    return df


def load_or_create_dataset(data_file: str = "leukemia_rna_seq.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Mevcut veri dosyasını okur veya yoksa üretir.
    Döndürür: (df, X, y, gene_cols)
    """
    if not os.path.exists(data_file):
        df = generate_leukemia_dataset()
        df.to_csv(data_file)
    else:
        df = pd.read_csv(data_file, index_col=0)

    X = df.drop(columns=['Leukemia_Subtype'])
    y = df['Leukemia_Subtype']
    gene_cols = [c for c in X.columns if c not in CBC_COLUMNS]

    return df, X, y, gene_cols
