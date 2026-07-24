"""
pages/data_explorer.py
=======================
Eksplorasi dataset: distribusi kelas, galeri gambar, augmentasi.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import CLASS_NAMES, IMG_SIZE

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor' : '#334155', 'axes.labelcolor': '#94a3b8',
    'xtick.color'    : '#94a3b8', 'ytick.color'    : '#94a3b8',
    'text.color'     : '#f1f5f9', 'grid.color'     : '#334155',
})

# ── Path ───────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(ROOT_DIR, 'data', 'Blood cell Cancer [ALL]')
DATA_PREP = os.path.join(ROOT_DIR, 'data', 'tmp', 'prepared_data')
DATA_TEST = os.path.join(ROOT_DIR, 'data', 'tmp', 'prepared_test')

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('📈 Eksplorasi Data')
st.caption('Distribusi kelas, contoh gambar, dan ilustrasi augmentasi preprocessing.')


def count_images(base_dir: str) -> dict:
    counts = {}
    if not os.path.isdir(base_dir):
        return counts
    for cls in CLASS_NAMES:
        cls_path = os.path.join(base_dir, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            counts[cls] = n
    return counts


@st.cache_data
def get_distribution():
    return count_images(DATA_RAW), count_images(DATA_PREP), count_images(DATA_TEST)


@st.cache_data
def get_sample_images(base_dir: str, cls: str, n: int = 6) -> list:
    cls_path = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_path):
        return []
    files = [f for f in os.listdir(cls_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.seed(42)
    return [os.path.join(cls_path, f) for f in random.sample(files, min(n, len(files)))]


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
# SECTION 2: Galeri Gambar
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.markdown('### Galeri Gambar per Kelas')

src_option = st.radio('Sumber:', ['Data Asli (raw)', 'Data Siap Latih (prepared)'], horizontal=True)
src_dir     = DATA_RAW if src_option == 'Data Asli (raw)' else DATA_PREP
sel_cls     = st.selectbox('Kelas:', CLASS_NAMES)
n_show      = st.slider('Jumlah gambar:', 3, 12, 6)

sample_paths = get_sample_images(src_dir, sel_cls, n_show)

if sample_paths:
    n_col = min(n_show, 6)
    cols  = st.columns(n_col)
    for i, path in enumerate(sample_paths):
        with cols[i % n_col]:
            st.image(Image.open(path),
                     caption=os.path.basename(path)[:22],
                     use_container_width=True)
else:
    st.info(f'Tidak ada gambar untuk kelas `{sel_cls}` di sumber yang dipilih.')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Contoh Augmentasi
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('---')
st.markdown('### Contoh Augmentasi Preprocessing')

aug_paths = get_sample_images(DATA_RAW, CLASS_NAMES[0], 1)

if aug_paths:
    try:
        import cv2
        orig_pil = Image.open(aug_paths[0]).resize(IMG_SIZE)
        orig     = np.array(orig_pil)
        
        # K-Means segmentation (k=3)
        def kmeans_segment(img_rgb, k=3):
            flat = img_rgb.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, labels, centers = cv2.kmeans(flat, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.astype(np.uint8)
            result  = centers[labels.flatten()].reshape(img_rgb.shape)
            return result

        aug_dict = {
            'Original'            : orig,
            'K-Means Segmentasi'  : kmeans_segment(orig, k=3),
            'Flip Horizontal'     : np.fliplr(orig),
            'Flip Vertikal'       : np.flipud(orig),
            'Rotasi 90°'          : np.rot90(orig),
            'Rotasi 180°'         : np.rot90(orig, 2),
        }
        cols = st.columns(len(aug_dict))
        for col, (label, img) in zip(cols, aug_dict.items()):
            with col:
                st.image(img, caption=label, use_container_width=True)
        st.caption(
            'Setiap gambar latih menghasilkan **2 output dasar**: '
            'gambar asli + twin K-Means segmentation (k=3). '
            'Augmentasi flip/rotate dipakai untuk top-up kelas yang belum mencapai target.'
        )
    except ImportError:
        st.warning('⚠️ OpenCV (cv2) tidak tersedia. Menampilkan contoh dasar saja.')
        orig_pil = Image.open(aug_paths[0]).resize(IMG_SIZE)
        orig = np.array(orig_pil)
        
        aug_dict = {
            'Original': orig,
            'Flip Horizontal': np.fliplr(orig),
            'Flip Vertikal': np.flipud(orig),
        }
        cols = st.columns(len(aug_dict))
        for col, (label, img) in zip(cols, aug_dict.items()):
            with col:
                st.image(img, caption=label, use_container_width=True)
else:
    st.info('Contoh augmentasi tidak dapat ditampilkan karena folder data belum tersedia.')

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