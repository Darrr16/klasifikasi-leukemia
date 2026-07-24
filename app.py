"""
app.py
======
Entry point Streamlit — navigasi & halaman Home.
"""

import os
import sys
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import load_metrics, MODEL_FILES

# ── Page config (dipanggil SATU kali di sini, tidak di page files) ─────────────
st.set_page_config(
    page_title='Klasifikasi Citra Leukimia',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Global dark-theme CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Metric card ─────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background : #1e293b;
        border-radius : 0.6rem;
        padding : 0.8rem 1rem;
        border-left : 4px solid #60a5fa;
    }

    /* ── Hero banner ─────────────────────────────────────────────────────── */
    .hero {
        background : linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border     : 1px solid #2d4a6b;
        color      : #f1f5f9;
        padding    : 2.4rem 2rem;
        border-radius : 1rem;
        margin-bottom : 1.8rem;
    }
    .hero h1 { color: #f1f5f9; margin: 0 0 0.5rem; font-size: 1.9rem; }
    .hero p  { color: #94a3b8;  margin: 0; font-size: 1rem; }

    /* ── Feature card ────────────────────────────────────────────────────── */
    .feat-card {
        background    : #1e293b;
        border        : 1px solid #334155;
        border-radius : 0.7rem;
        padding       : 1.2rem 1.4rem;
        height        : 100%;
    }
    .feat-card h4 { margin: 0 0 0.5rem; color: #60a5fa; }
    .feat-card p  { margin: 0; color: #94a3b8; font-size: 0.9rem; }

    /* ── Section label ───────────────────────────────────────────────────── */
    .sec-label {
        font-size     : 0.75rem;
        font-weight   : 700;
        letter-spacing: 0.1em;
        color         : #475569;
        text-transform: uppercase;
        margin        : 1.4rem 0 0.5rem;
    }

    /* ── Sidebar brand ───────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Home page (didefinisikan sebagai fungsi) ───────────────────────────────────
def home():
    st.markdown("""
<div class="hero">
  <h1>🔬 Sistem Klasifikasi Sel Darah</h1>
  <p>Deteksi Otomatis Acute Lymphoblastic Leukemia (ALL)<br>
     menggunakan <strong>EfficientNetB0 + CBAM</strong> &nbsp;|&nbsp;
     Dataset: Blood Cell Cancer [ALL] — Kaggle</p>
</div>
""", unsafe_allow_html=True)

    # ── Ringkasan metrik ───────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">Perbandingan Performa Model</p>',
                unsafe_allow_html=True)

    m_bl   = load_metrics('EfficientNetB0 Baseline')
    m_cbam = load_metrics('EfficientNetB0 + CBAM')

    col_bl, col_cbam, _, col_kelas = st.columns([2, 2, 0.1, 3])

    with col_bl:
        st.markdown("**EfficientNetB0 Baseline**")
        if m_bl:
            st.metric('Accuracy',  f"{m_bl.get('test_accuracy',0)*100:.2f}%")
            st.metric('Test Loss', f"{m_bl.get('test_loss',0):.4f}")
            st.metric('MAE',       f"{m_bl.get('mae',0):.4f}")
        else:
            st.info('Metrics belum tersedia.')

    with col_cbam:
        st.markdown("**EfficientNetB0 + CBAM** ⭐")
        if m_cbam:
            delta = None
            if m_bl:
                delta = f"+{(m_cbam.get('test_accuracy',0) - m_bl.get('test_accuracy',0))*100:.2f}%"
            st.metric('Accuracy',     f"{m_cbam.get('test_accuracy',0)*100:.2f}%", delta=delta)
            st.metric('Test Loss',    f"{m_cbam.get('test_loss',0):.4f}")
            st.metric('Best Val Acc', f"{m_cbam.get('best_val_acc', m_cbam.get('test_accuracy',0))*100:.2f}%")
        else:
            st.info('Metrics belum tersedia.')

    with col_kelas:
        st.markdown("**Kelas Klasifikasi**")
        st.markdown("""
| Kelas | Keterangan |
|-------|-----------|
| `EarlyPreB` | Early Precursor B-cell (ALL+) |
| `PreB` | Precursor B-cell (ALL+) |
| `ProB` | Pro-B-cell (ALL+) |
| `benign` | Sel normal / sehat |
""")
        st.caption("Dataset split: 90% Train (+ 20% val) | 10% Test — stratified")

    # ── Quick-start ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sec-label">Mulai dari sini</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    items = [
        ("🔍 Prediksi", "Upload gambar sel darah dan dapatkan prediksi kelas beserta confidence score."),
        ("📊 Performa", "Metrik lengkap: confusion matrix, kurva training, classification report."),
        ("📈 Data Explorer", "Distribusi kelas, galeri gambar per kelas, dan contoh augmentasi."),
        ("🏗️ Arsitektur", "Detail EfficientNetB0 + CBAM dan seluruh konfigurasi training."),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(f"""
<div class="feat-card">
<h4>{title}</h4>
<p>{desc}</p>
</div>""", unsafe_allow_html=True)

    # ── About ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Tentang Proyek"):
        st.markdown("""
**Judul:** Klasifikasi Citra Sel Darah untuk Deteksi Acute Lymphoblastic Leukemia (ALL)

**Model yang dibangun:**
- *EfficientNetB0 Baseline* — backbone EfficientNetB0 + classification head standar
- *EfficientNetB0 + CBAM* — backbone + Convolutional Block Attention Module

**Dataset:** [Blood Cell Cancer [ALL] - Kaggle](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)

> ⚠️ Hanya untuk keperluan **penelitian dan akademik** — bukan alat diagnosis klinis.
""")

    st.markdown("""
<hr style="border:none;border-top:1px solid #1e293b;margin-top:2rem"/>
<p style="text-align:center;color:#334155;font-size:0.8rem">
Klasifikasi Citra Leukimia &nbsp;|&nbsp; EfficientNetB0 + CBAM &nbsp;|&nbsp; Proyek Akhir PI
</p>
""", unsafe_allow_html=True)


# ── Navigasi — label sidebar dikontrol dari sini ───────────────────────────────
pg = st.navigation([
    st.Page(home,                      title='Home',         icon='🏠', default=True),
    st.Page('pages/prediksi.py',       title='Prediksi',     icon='🔍'),
    st.Page('pages/performa.py',      title='Performa',     icon='📊'),
    st.Page('pages/data_explorer.py',  title='Data Explorer',icon='📈'),
    st.Page('pages/arsitektur.py',     title='Arsitektur',   icon='🏗️'),
    st.Page('pages/dokumentasi.py',    title='Dokumentasi',  icon='📚'),
])
pg.run()