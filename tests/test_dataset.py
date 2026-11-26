"""
Test Suite for Core Dataset Generation & Loading
"""

import unittest
import pandas as pd
from core.dataset import (
    generate_leukemia_dataset,
    load_or_create_dataset,
    CBC_COLUMNS,
    AML_MYELOID_PANEL,
    ALL_LYMPHOID_PANEL
)


class TestDataset(unittest.TestCase):
    def test_generate_dataset_shape_and_columns(self):
        """Sentetik veri setinin boyutlarını ve kolonlarını test eder."""
        df = generate_leukemia_dataset(num_samples=30, num_genes=50, seed=42)
        self.assertEqual(len(df), 30)
        self.assertIn('Leukemia_Subtype', df.columns)

        # CBC kolonlarının varlığı
        for col in CBC_COLUMNS:
            self.assertIn(col, df.columns)

        # Bilinen panel genlerinin varlığı
        for g in ['RUNX1', 'FLT3', 'PAX5', 'CD19', 'WT1', 'IKZF1']:
            self.assertIn(g, df.columns)

        # Sınıf dağılımının dengeli olması
        counts = df['Leukemia_Subtype'].value_counts()
        self.assertEqual(counts['Healthy'], 10)
        self.assertEqual(counts['AML'], 10)
        self.assertEqual(counts['ALL'], 10)

    def test_no_null_values(self):
        """Veri setinde hiçbir eksik (NaN) değer olmamalıdır."""
        df = generate_leukemia_dataset(num_samples=30, num_genes=50, seed=42)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_cbc_value_ranges(self):
        """CBC parametrelerinin fizyolojik sınırlar içinde üretildiğini kontrol eder."""
        df = generate_leukemia_dataset(num_samples=60, num_genes=50, seed=42)
        self.assertTrue((df['Blast_Percentage'] >= 0.0).all())
        self.assertTrue((df['Blast_Percentage'] <= 100.0).all())
        self.assertTrue((df['Hemoglobin_g_dL'] >= 2.0).all())
        self.assertTrue((df['WBC_x10k'] >= 0.5).all())


if __name__ == "__main__":
    unittest.main()
