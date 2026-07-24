"""
utils/helpers.py
================
Fungsi bersama untuk Streamlit app.
"""

import os
import sys
import json
import numpy as np
import streamlit as st
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Konstanta ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['EarlyPreB', 'PreB', 'ProB', 'benign']
RANDOM_SEED = 88

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR  = os.path.join(ROOT_DIR, 'models')

# ── Hanya dua model ────────────────────────────────────────────────────────────
MODEL_FILES = {
    'EfficientNetB0 Baseline': os.path.join(MODELS_DIR, 'efficientnetb0_baseline_best.h5'),
    'EfficientNetB0 + CBAM'  : os.path.join(MODELS_DIR, 'efficientnetb0_cbam_best.h5'),
}

METRICS_FILES = {
    'EfficientNetB0 Baseline': os.path.join(RESULTS_DIR, 'efficientnetb0_baseline_metrics.json'),
    'EfficientNetB0 + CBAM'  : os.path.join(RESULTS_DIR, 'efficientnetb0_cbam_metrics.json'),
}

HISTORY_FILES = {
    'EfficientNetB0 Baseline': os.path.join(RESULTS_DIR, 'efficientnetb0_baseline_history.npz'),
    'EfficientNetB0 + CBAM'  : os.path.join(RESULTS_DIR, 'efficientnetb0_cbam_history.npz'),
}

RESULTS_FILES = {
    'EfficientNetB0 Baseline': os.path.join(RESULTS_DIR, 'efficientnetb0_baseline_results.npz'),
    'EfficientNetB0 + CBAM'  : os.path.join(RESULTS_DIR, 'efficientnetb0_cbam_results.npz'),
}

PLOT_FILES = {
    'EfficientNetB0 Baseline': {
        'confusion_normalized': os.path.join(RESULTS_DIR, 'EfficientNetB0 Baseline_confusion_matrix_normalized.png'),
        'confusion_raw'       : os.path.join(RESULTS_DIR, 'EfficientNetB0 Baseline_confusion_matrix_raw.png'),
        'training_history'    : os.path.join(RESULTS_DIR, 'EfficientNetB0 Baseline_training_history.png'),
        'sample_predictions'  : os.path.join(RESULTS_DIR, 'EfficientNetB0 Baseline_sample_predictions.png'),
    },
    'EfficientNetB0 + CBAM': {
        'confusion_normalized': os.path.join(RESULTS_DIR, 'EfficientNetB0 + CBAM_confusion_matrix_normalized.png'),
        'confusion_raw'       : os.path.join(RESULTS_DIR, 'EfficientNetB0 + CBAM_confusion_matrix_raw.png'),
        'training_history'    : os.path.join(RESULTS_DIR, 'EfficientNetB0 + CBAM_training_history.png'),
        'sample_predictions'  : os.path.join(RESULTS_DIR, 'EfficientNetB0 + CBAM_sample_predictions.png'),
    },
}

CLASS_DESCRIPTIONS = {
    'EarlyPreB': {
        'label'      : 'Early Precursor B-cell',
        'description': 'Stadium paling awal perkembangan sel B di sumsum tulang.',
        'ciri'       : [
            'Nukleus besar dengan kromatin halus',
            'Rasio nukleus-sitoplasma sangat tinggi',
            'Terlihat nukleolus',
            'Indikator ALL stadium awal',
        ],
        'warna': '#f87171',
    },
    'PreB': {
        'label'      : 'Precursor B-cell',
        'description': 'Stadium menengah perkembangan sel B.',
        'ciri'       : [
            'Lebih matang dari EarlyPreB',
            'Sitoplasma mulai terlihat',
            'Ukuran nukleus sedang',
            'Sering ditemukan pada kasus ALL',
        ],
        'warna': '#fb923c',
    },
    'ProB': {
        'label'      : 'Pro-B-cell',
        'description': 'Stadium paling awal sel B yang dapat diidentifikasi.',
        'ciri'       : [
            'Morfologi khas dengan bentuk nukleus yang khas',
            'Kondensasi kromatin sedang',
            'Dapat mengindikasikan leukemia stadium awal',
        ],
        'warna': '#fbbf24',
    },
    'benign': {
        'label'      : 'Sel Normal / Benign',
        'description': 'Sel darah sehat, tidak terkait leukemia.',
        'ciri'       : [
            'Morfologi sel normal',
            'Rasio nukleus-sitoplasma seimbang',
            'Kromatin padat dan teratur',
            'Tidak ada penanda leukemik',
        ],
        'warna': '#4ade80',
    },
}


# ── Cache: load model ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Memuat model…')
def load_model(model_name: str = 'EfficientNetB0 + CBAM'):
    import tensorflow as tf
    path = MODEL_FILES.get(model_name)
    
    if not path:
        st.error(f'Model tidak dikenal: {model_name}')
        return None
    
    if not os.path.exists(path):
        st.error(f'❌ File model tidak ditemukan: {path}')
        st.info('📌 Pastikan file `.h5` sudah tersedia di folder `models/`')
        return None
    
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None


# ── Cache: load metrics JSON ───────────────────────────────────────────────────
@st.cache_data
def load_metrics(model_name: str):
    path = METRICS_FILES.get(model_name)
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


# ── Cache: load history .npz ──────────────────────────────────────────────────
@st.cache_data
def load_history(model_name: str):
    path = HISTORY_FILES.get(model_name)
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k].tolist() for k in data.files}


# ── Cache: load results .npz ──────────────────────────────────────────────────
@st.cache_data
def load_results(model_name: str):
    path = RESULTS_FILES.get(model_name)
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ── Preprocess gambar ──────────────────────────────────────────────────────────
def preprocess_for_prediction(pil_image: Image.Image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_resized = pil_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    img_array   = np.expand_dims(img_array, axis=0)
    return img_array, img_resized


# ── Interpret confidence ───────────────────────────────────────────────────────
def interpret_confidence(confidence: float):
    if confidence >= 0.90:
        return '✅ Sangat Yakin', 'success'
    elif confidence >= 0.75:
        return '✓ Cukup Yakin', 'info'
    elif confidence >= 0.60:
        return '⚠️ Kurang Yakin', 'warning'
    else:
        return '❌ Tidak Yakin', 'error'