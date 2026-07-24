"""
pages/data_explorer.py
=======================
Eksplorasi Dataset Ringkasan - informasi preprocessing tanpa akses file dataset.
(Dataset lokal ~2GB tidak disertakan di GitHub)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from helpers import CLASS_NAMES, IMG_SIZE

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor' : '#334155', 'axes.labelcolor': '#94a3b8',
    'xtick.color'    : '#94a3b8', 'ytick.color'    : '#94a3b8',
    'text.color'     : '#f1f5f9', 'grid.color'     : '#334155',
})

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('📈 Eksplorasi Data')
st.caption('Ringkasan dataset, distribusi kelas, dan preprocessing pipeline.')

# ── Dataset Summary (Hardcoded) ────────────────────────────────────────────────
DATASET_SUMMARY = {
    'benign'    : {'raw': 100,  'train': 512, 'test': 11},
    'EarlyPreB' : {'raw': 249,  'train': 512, 'test': 28},
    'PreB'      : {'raw': 256,  'train': 512, 'test': 28},
    'ProB'      : {'raw': 391,  'train': 784, 'test': 44},
}

def get_distribution():
    """Return hardcoded distribution data (no file I/O needed)"""
    raw_counts  = {cls: DATASET_SUMMARY[cls]['raw']   for cls in CLASS_NAMES}
    prep_counts = {cls: DATASET_SUMMARY[cls]['train'] for cls in CLASS_NAMES}
    test_counts = {cls: DATASET_SUMMARY[cls]['test']  for cls in CLASS_NAMES}
    return raw_counts, prep_counts, test_counts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Distribusi Kelas
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('### Distribusi Kelas')

raw_counts, prep_counts, test_counts = get_distribution()

c1, c2, c3 = st.columns(3)
with c1: st.metric('Total Raw Dataset',              f"{sum(raw_counts.values()):,}",  'Sebelum split')
with c2: st.metric('Training (setelah oversample)',  f"{sum(prep_counts.values()):,}", 'Termasuk augmentasi')
with c3: st.metric('Test Set',                       f"{sum(test_counts.values()):,}", '10% dari raw')

if raw_counts:
    df_dist = pd.DataFrame({
        'Kelas'              : CLASS_NAMES,
        'Raw (asli)'         : [raw_counts.get(c, 0)  for c in CLASS_NAMES],
        'Train (oversample)' : [prep_counts.get(c, 0) for c in CLASS_NAMES],
        'Test'               : [test_counts.get(c, 0) for c in CLASS_NAMES],
    })
    df_dist['Oversample +N'] = df_dist['Train (oversample)'] - df_dist['Raw (asli)']
    st.dataframe(df_dist, hide_index=True, use_container_width=True)

    colors = ['#f87171', '#fb923c', '#fbbf24', '#4ade80']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals, title in [
        (axes[0], [raw_counts.get(c,0)  for c in CLASS_NAMES], 'Raw Dataset'),
        (axes[1], [prep_counts.get(c,0) for c in CLASS_NAMES], 'Setelah Oversampling (Train)'),
    ]:
        ax.bar(CLASS_NAMES, vals, color=colors)
        ax.set_title(title); ax.set_ylabel('Jumlah Gambar'); ax.grid(axis='y', alpha=.2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*0.01, str(v), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander('Mengapa K-Means segmentation + oversampling?'):
        st.markdown("""
Kelas `benign` memiliki sampel lebih sedikit dibanding kelas ALL → distribusi tidak seimbang.

**Strategi yang digunakan:**
1. **K-Means Segmentation (k=3)** — setiap gambar asli menghasilkan twin hasil segmentasi
   warna, sehingga struktur sel (inti, sitoplasma, latar belakang) lebih kontras.
   Ini sekaligus menggandakan jumlah data dasar (2× per kelas).
2. **Top-up flip/rotate** — kelas yang basisnya (2×n) masih di bawah target
   ditambah dengan augmentasi flip horizontal, flip vertikal, rotasi 90°/180°
   yang dipilih secara acak hingga mencapai target.
3. Kelas besar tidak dipotong.

**Class weights tidak digunakan** — data sudah relatif seimbang setelah langkah di atas.
""")
else:
    st.info('Folder `data/Blood cell Cancer [ALL]/` tidak ditemukan. Pastikan dataset sudah diunduh.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Info Kelas & Karakteristik
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.markdown('### Karakteristik Kelas Blood Cancer [ALL]')

class_info = {
    'benign': {
        'desc': 'Sel darah putih normal',
        'karakteristik': 'Nukleus kecil, sitoplasma sehat, bentuk regular'
    },
    'EarlyPreB': {
        'desc': 'Leukemia Limfoblastik Akut (early precursor)',
        'karakteristik': 'Nukleus besar, sitoplasma sedikit, bentuk irregular'
    },
    'PreB': {
        'desc': 'Leukemia Limfoblastik Akut (pre-cursor B)',
        'karakteristik': 'Nukleus dominan, sitoplasma minimal, chromatin coarse'
    },
    'ProB': {
        'desc': 'Leukemia Limfoblastik Akut (progenitor B)',
        'karakteristik': 'Sel immature, nukleus sangat besar, sitoplasma sedikit'
    },
}

col1, col2, col3, col4 = st.columns(4)
for col, cls in zip([col1, col2, col3, col4], CLASS_NAMES):
    with col:
        st.subheader(f"🔵 {cls}")
        st.caption(f"*{class_info[cls]['desc']}*")
        st.write(f"**Ciri:** {class_info[cls]['karakteristik']}")
        raw = DATASET_SUMMARY[cls]['raw']
        st.write(f"📊 Raw: **{raw}** img")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Augmentasi Demo (Dummy Image)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.markdown('### Contoh Augmentasi Preprocessing')

st.info('📌 **Catatan:** Fitur galeri gambar memerlukan dataset lokal (~2GB). '
        'Di bawah ini adalah demonstrasi menggunakan dummy image.')

# Create dummy image for demo
np.random.seed(42)
dummy_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

# Augmentation examples
aug_dict = {
    'Original': dummy_img,
    'Flip Horizontal': np.fliplr(dummy_img),
    'Flip Vertikal': np.flipud(dummy_img),
    'Rotasi 90°': np.rot90(dummy_img),
}

cols = st.columns(len(aug_dict))
for col, (label, img) in zip(cols, aug_dict.items()):
    with col:
        st.image(img, caption=label, use_container_width=True)

st.caption(
    '✅ **Production:** Setiap gambar asli → K-Means segmentation twin (2×data) + top-up flip/rotate untuk balance'
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Pipeline Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.markdown('### Pipeline Preprocessing Lengkap')

st.markdown("""
| Langkah | Keterangan |
|---------|-----------|
| **1. Train/Test Split** | 90% train (20% → val), 10% test — stratified |
| **2. Resize** | Semua gambar ke **224 × 224** px |
| **3. K-Means Segmentation** | Setiap gambar asli → twin segmentasi K-Means (k=3) = **2× data dasar per kelas** |
| **4. Top-up Oversampling** | Kelas yang basis-nya (2×n) < target → flip/rotate acak sampai seimbang |
| **5. Normalisasi** | `rescale=1/255` → piksel `[0,255] → [0,1]` |
| **6. ImageDataGenerator** | Training: flip + rotate + shift + zoom + brightness |

**Bilateral Filter Dihapus:**
> Filter ini dibuang karena menghaluskan tepi sel yang merupakan fitur diagnostik kritis.
> Referensi: Fajrina et al. (2024) — 98.48% hanya dengan resize + rescale.

**Class Weights tidak digunakan** (`class_weight=None`) — penyeimbangan sudah cukup
dilakukan di tahap preprocessing via K-Means duplikasi + top-up flip/rotate.

**Konfigurasi ImageDataGenerator training:**
```python
ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.15, height_shift_range=0.15,
    zoom_range=0.15, shear_range=0.1,
    brightness_range=[0.85, 1.15], fill_mode='nearest'
)
```
""")