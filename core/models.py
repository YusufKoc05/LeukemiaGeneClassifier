"""
OncoVision AI - Core Models Module
Model zoo eğitimi, Split Conformal Prediction, Isolation Forest OOD ve Benchmark önbellekleme.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple, Optional
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


MODEL_FILE = "leukemia_rf_model.pkl"
SCALER_FILE = "scaler.pkl"
MODELS_BUNDLE_FILE = "trained_models_zoo.pkl"
ISO_FOREST_FILE = "iso_forest_ood.pkl"
BENCHMARKS_FILE = "model_benchmarks.json"


class ModelManager:
    """
    Tüm modelleri, ölçekleyiciyi, OOD tespit edicisini ve conformal kalibrasyonunu
    yöneten merkezi yönetici sınıfı.
    """
    def __init__(self, data_file: str = "leukemia_rna_seq.csv"):
        self.data_file = data_file
        self.scaler: Optional[StandardScaler] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.trained_models: Dict[str, Any] = {}
        self.iso_forest: Optional[IsolationForest] = None
        self.benchmarks: List[Dict[str, Any]] = []
        self.q_conformal: float = 0.15
        self.classes: List[str] = ['ALL', 'AML', 'Healthy']

        # Veri setini yükle ve hazırla
        from .dataset import load_or_create_dataset
        self.df, self.X, self.y, self.gene_cols = load_or_create_dataset(self.data_file)

        # Train / Calib / Test ayrımı
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=42, stratify=self.y
        )
        self.X_train, self.X_calib, self.y_train, self.y_calib = train_test_split(
            self.X_train_full, self.y_train_full, test_size=0.25, random_state=42, stratify=self.y_train_full
        )

        self._initialize_or_load()

    def _initialize_or_load(self):
        """Modeller mevcutsa yükler, yoksa eğitir ve kaydeder."""
        all_exist = (
            os.path.exists(MODEL_FILE) and
            os.path.exists(SCALER_FILE) and
            os.path.exists(MODELS_BUNDLE_FILE) and
            os.path.exists(ISO_FOREST_FILE)
        )

        if all_exist:
            self.scaler = joblib.load(SCALER_FILE)
            self.rf_model = joblib.load(MODEL_FILE)
            self.trained_models = joblib.load(MODELS_BUNDLE_FILE)
            self.iso_forest = joblib.load(ISO_FOREST_FILE)

            self.X_train_sc = self.scaler.transform(self.X_train)
            self.X_calib_sc = self.scaler.transform(self.X_calib)
            self.X_test_sc = self.scaler.transform(self.X_test)
            self.X_all_sc = self.scaler.transform(self.X)
            self.classes = list(self.rf_model.classes_)
        else:
            self.train_and_save_all()

        # Conformal Prediction kalibrasyon eşiğini hesapla
        self._calibrate_conformal()

        # Benchmarkları yükle veya hesapla
        self._load_or_compute_benchmarks()

    def train_and_save_all(self):
        """Tüm modelleri ve ölçekleyiciyi sıfırdan eğitir ve diske kaydeder."""
        self.scaler = StandardScaler()
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_calib_sc = self.scaler.transform(self.X_calib)
        self.X_test_sc = self.scaler.transform(self.X_test)
        self.X_all_sc = self.scaler.transform(self.X)

        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.iso_forest.fit(self.X_train_sc)

        # Scikit-Learn 1.9+ için uyarı vermeyen Calibrated SVC kullanımı
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated_svc = CalibratedClassifierCV(SVC(random_state=42), ensemble=False)

        models_dict = {
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
            "SVM (RBF Kernel)": calibrated_svc,
            "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        }

        if HAS_XGB:
            models_dict["XGBoost"] = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss'
            )
        if HAS_LGB:
            models_dict["LightGBM"] = LGBMClassifier(
                n_estimators=100, max_depth=5, random_state=42, verbose=-1
            )

        self.trained_models = {}
        for name, m in models_dict.items():
            if name == "XGBoost":
                m.fit(self.X_train_sc, self.y_train.map({'ALL': 0, 'AML': 1, 'Healthy': 2}))
            else:
                m.fit(self.X_train_sc, self.y_train)
            self.trained_models[name] = m

        self.rf_model = self.trained_models["Random Forest"]
        self.classes = list(self.rf_model.classes_)

        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.rf_model, MODEL_FILE)
        joblib.dump(self.trained_models, MODELS_BUNDLE_FILE)
        joblib.dump(self.iso_forest, ISO_FOREST_FILE)

    def _calibrate_conformal(self, alpha_level: float = 0.05):
        """Split Conformal Prediction için non-conformity skorları üzerinden eşik belirler."""
        calib_probs = self.rf_model.predict_proba(self.X_calib_sc)
        calib_y_idx = [self.classes.index(label) for label in self.y_calib]
        calib_scores = 1.0 - calib_probs[np.arange(len(self.y_calib)), calib_y_idx]
        n_cal = len(calib_scores)
        self.q_conformal = float(np.quantile(calib_scores, np.ceil((n_cal + 1) * (1 - alpha_level)) / n_cal))

    def _load_or_compute_benchmarks(self):
        """Önceden hesaplanmış benchmarkları yükler, yoksa hesaplar ve kaydeder."""
        if os.path.exists(BENCHMARKS_FILE):
            try:
                with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
                    self.benchmarks = json.load(f)
                    return
            except Exception:
                pass

        # Hesaplanması gerekiyorsa:
        benchmark_results = []
        for name, m in self.trained_models.items():
            if name == "XGBoost":
                preds_num = m.predict(self.X_test_sc)
                inv_map = {0: 'ALL', 1: 'AML', 2: 'Healthy'}
                preds = [inv_map[p] for p in preds_num]
                # Doğru model ile CV
                y_mapped = self.y.map({'ALL': 0, 'AML': 1, 'Healthy': 2})
                cv_s = cross_val_score(m, self.X_all_sc, y_mapped, cv=5)
            else:
                preds = m.predict(self.X_test_sc)
                cv_s = cross_val_score(m, self.X_all_sc, self.y, cv=5)

            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds, average='weighted')
            benchmark_results.append({
                "Model Mimarisi": name,
                "Test Doğruluğu": f"%{acc*100:.2f}",
                "Ağırlıklı F1-Skoru": f"{f1:.4f}",
                "5-Katlı Çapraz Doğrulama": f"%{cv_s.mean()*100:.1f} (±%{cv_s.std()*100:.1f})"
            })

        self.benchmarks = benchmark_results
        try:
            with open(BENCHMARKS_FILE, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_benchmarks_df(self) -> pd.DataFrame:
        """Benchmark tablosunu DataFrame olarak döndürür."""
        return pd.DataFrame(self.benchmarks)

    def compute_conformal_set(self, probs: np.ndarray) -> List[str]:
        """Verilen sınıf olasılıkları için %95 conformal tahmin kümesini üretir."""
        pred_set = [self.classes[idx] for idx in range(len(self.classes)) if (1.0 - probs[idx]) <= self.q_conformal]
        return pred_set if pred_set else [self.classes[int(np.argmax(probs))]]

    def compute_ood_score(self, input_scaled: np.ndarray) -> Tuple[str, float, str]:
        """Isolation Forest kullanarak OOD (Dağılım Dışı) anomali skoru hesaplar."""
        score = float(self.iso_forest.decision_function(input_scaled)[0])
        is_ood = bool(self.iso_forest.predict(input_scaled)[0] == -1)
        status = "Dağılım Dışı (OOD)" if is_ood else "Uyumlu (İç Kohort)"
        badge_color = "#dc2626" if is_ood else "#047857"
        return status, score, badge_color


_model_manager_instance: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Tekil (Singleton) ModelManager örneğini döndürür."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
