"""
pages/dokumentasi.py
=====================
Panduan penggunaan, penjelasan kelas, referensi.
"""

import os
import sys
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import CLASS_DESCRIPTIONS

st.title('📚 Dokumentasi & Panduan')

tab1, tab2, tab3, tab4 = st.tabs([
    'Panduan Penggunaan',
    'Penjelasan Kelas',
    'Referensi',
    'Tentang Proyek',
])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Panduan Penggunaan")

    with st.expander("Prediksi Gambar", expanded=True):
        st.markdown("""
1. Buka **Prediksi** di sidebar
2. *(Opsional)* Pilih model di sidebar — default sudah ke CBAM (terbaik)
3. Upload gambar JPG/PNG/BMP
4. Tunggu preprocessing dan prediksi (< 5 detik)
5. Lihat: kartu prediksi, confidence, bar chart, interpretasi

**Tips:** Resolusi ≥ 256×256 px, sel terlihat jelas, tidak blur.
""")

    with st.expander("Performa Model"):
        st.markdown("""
1. Buka **Performa** di sidebar
2. Pilih model (Baseline atau CBAM) via radio button di sidebar
3. Tab yang tersedia:
   - **Confusion Matrix** — heatmap normalized & raw, insight otomatis
   - **Kurva Training** — loss & accuracy per epoch
   - **Classification Report** — precision, recall, F1, specificity per kelas
   - **Sample Prediksi** — grid 5×5 dengan warna kode kode prediksi
   - **Raw JSON** — file metrics lengkap + tombol download
""")

    with st.expander("Data Explorer"):
        st.markdown("""
1. Buka **Data Explorer**
2. Lihat distribusi kelas raw vs setelah oversample
3. Pilih kelas dan sumber untuk menjelajahi galeri gambar
4. Lihat demonstrasi 5 jenis augmentasi preprocessing
""")

    with st.expander("Arsitektur"):
        st.markdown("""
1. Tab **Arsitektur** — diagram kode Baseline vs CBAM
2. Tab **Konfigurasi Training** — hyperparameter, callbacks, augmentasi
3. Tab **Perbandingan Model** — tabel + gambar perbandingan
4. Tab **Model Summary** — tampilkan Keras summary model terpilih
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Penjelasan Kelas Klasifikasi")

    for cls_name, info in CLASS_DESCRIPTIONS.items():
        warna = info['warna']
        with st.expander(f"{info['label']}  (`{cls_name}`)"):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.markdown(f"""
<div style="width:56px;height:56px;background:{warna}33;border:2px solid {warna};
            border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:1.4rem">🔬</div>
""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Deskripsi:** {info['description']}")
                st.markdown("**Karakteristik:**")
                for c in info['ciri']:
                    st.markdown(f"- {c}")

    st.markdown('---')
    st.markdown("""
| Kelas | Status |
|-------|--------|
| `EarlyPreB` | 🔴 ALL-positif |
| `PreB` | 🔴 ALL-positif |
| `ProB` | 🔴 ALL-positif |
| `benign` | 🟢 Normal / sehat |
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Referensi")
    st.markdown("""
1. **Tan & Le (2019)** — *EfficientNet: Rethinking Model Scaling for CNNs*
   ICML 2019 · [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

2. **Woo et al. (2018)** — *CBAM: Convolutional Block Attention Module*
   ECCV 2018 · [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)

3. **Fajrina et al. (2024)** — resize + rescale tanpa filter → 98.48% accuracy

4. **Dataset** — Blood Cell Cancer [ALL] — Kaggle  
   [Lihat dataset](https://www.kaggle.com/datasets/nikhilsharma00/blood-cell-cancer-all-leukaemia)

**Library yang digunakan:**

| Library | Versi | Fungsi |
|---------|-------|--------|
| TensorFlow | 2.13.0 | Deep learning |
| Streamlit | ≥1.36 | Web app |
| OpenCV | 4.8.0 | Image processing |
| scikit-learn | 1.3.0 | Metrics |
| NumPy | 1.24.3 | Array |
| Pandas | 2.0.3 | Data |
| Matplotlib / Seaborn | 3.7.2 / 0.12.2 | Visualisasi |
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Tentang Proyek")
    st.markdown("""
**Judul:**  
Klasifikasi Citra Sel Darah untuk Deteksi Acute Lymphoblastic Leukemia (ALL)
Menggunakan EfficientNetB0 + CBAM

**Tujuan:**  
Membangun sistem klasifikasi otomatis berbasis deep learning yang membedakan
sel darah normal (benign) dari tiga subtipe sel leukemia ALL (EarlyPreB, PreB, ProB).

---

**Kontribusi Utama:**

| Komponen | Keterangan |
|----------|-----------|
| `src/data_preprocessing.py` | Pipeline preprocessing + oversampling target-based |
| `src/models.py` | Implementasi CBAM + EfficientNetB0 |
| `src/training_utils.py` | Training, evaluasi, visualisasi |
| `app.py` + `pages/` | Streamlit multi-page app |

---

**Struktur Project:**
```
KLASIFIKASICITRALEUKIMIA(PI)/
├── data/
│   ├── Blood cell Cancer [ALL]/  ← dataset raw
│   └── tmp/                      ← data setelah preprocessing
├── models/                       ← file model .h5
├── results/                      ← metrics, history, plots
├── src/                          ← source code utama
├── utils/                        ← helpers Streamlit
├── pages/                        ← halaman Streamlit
├── app.py                        ← entry point
└── requirements.txt
```

> ⚠️ Hanya untuk keperluan **penelitian dan tugas akhir (PI)** —
> bukan alat diagnosis medis klinis.
""")