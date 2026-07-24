"""
pages/arsitektur.py
====================
Arsitektur model dan konfigurasi training.
"""

import os
import sys
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import load_model, MODEL_FILES, RESULTS_DIR

st.title('🏗️ Arsitektur Model')
st.caption('Detail EfficientNetB0 Baseline dan EfficientNetB0 + CBAM beserta konfigurasi training.')

t1, t2, t3, t4 = st.tabs(['Arsitektur', 'Konfigurasi Training', 'Perbandingan Model', 'Model Summary'])

# ═════════════════════════════════════════════════════
# TAB 1: ARSITEKTUR
# ═════════════════════════════════════════════════════
with t1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### EfficientNetB0 Baseline")
        st.code("""
Input: (224, 224, 3)
    ↓
Rescaling(×255.0)
    ↓
EfficientNetB0 backbone (ImageNet, frozen sepenuhnya)
  └─ Conv2D Stem
  └─ MBConv Block × 16
  └─ Conv2D Head
  Output: (7, 7, 1280)
    ↓
GlobalAveragePooling2D → (1280,)
    ↓
BatchNormalization
Dense(256, ReLU)
Dropout(0.3)
    ↓
Dense(4, Softmax)  ← 4 kelas
""", language=None)

    with col2:
        st.markdown("#### EfficientNetB0 + CBAM ⭐")
        st.code("""
Input: (224, 224, 3)
    ↓
Rescaling(×255.0)
    ↓
EfficientNetB0 backbone (Phase 1: frozen | Phase 2: top-60 un-frozen)
  └─ MBConv Block × 16
     Output: (7, 7, 1280)
    ↓
CBAM Block
  ├─ Channel Attention
  │   GAP + GMP → Shared MLP → Sigmoid → scale
  └─ Spatial Attention
      avg+max along C → Conv(7×7) → Sigmoid → scale
    ↓
GlobalAveragePooling2D → (1280,)
    ↓
BatchNormalization
Dense(256, ReLU)
Dropout(0.3)
    ↓
Dense(4, Softmax)  ← 4 kelas
""", language=None)
        st.success("CBAM memfokuskan model pada region dan fitur sel yang diagnostik.")

    st.markdown('---')
    st.markdown("#### Mengapa EfficientNetB0?")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, desc in zip([c1,c2,c3,c4], [
        'Transfer Learning', 'Efisiensi Parameter', 'Kompatibel CBAM', 'Inference Cepat'
    ], [
        'Pre-trained ImageNet → fitur awal berkualitas tanpa scratch',
        'Scaling depth/width/resolution seimbang → akurasi tinggi',
        'Arsitektur modular → mudah sisipkan attention di MBConv',
        '~4M parameter → prediksi < 1 detik untuk deployment',
    ]):
        with col:
            st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:0.7rem;
            padding:1rem;height:100%">
<h5 style="color:#60a5fa;margin:0 0 0.4rem">{title}</h5>
<p style="color:#94a3b8;font-size:0.85rem;margin:0">{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<br>
<div style="background:#1e293b;border:1px solid #334155;border-radius:0.7rem;padding:1rem">
<strong style="color:#f1f5f9">Catatan Preprocessing di Model:</strong>
<ul style="color:#94a3b8;margin:0.5rem 0 0;font-size:0.9rem">
<li>Generator menghasilkan <code>[0,1]</code> via <code>rescale=1/255</code></li>
<li>Model punya layer <code>Rescaling(255.0)</code> yang mengembalikan ke <code>[0,255]</code></li>
<li>EfficientNetB0 internal preprocessing menangani normalisasi akhir</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════
# TAB 2: KONFIGURASI TRAINING
# ═════════════════════════════════════════════════════
with t2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Hyperparameter")
        st.markdown("""
| Parameter | Baseline | CBAM |
|-----------|---------|------|
| Optimizer | Adam | Adam |
| Learning Rate | 1e-4 | Phase 1: 1e-3 → Phase 2: 1e-4 |
| Batch Size | 32 | 32 |
| Max Epochs | 50 | Phase 1: 20 / Phase 2: 60 |
| Early Stopping | patience=7 | patience=7 |
| Loss | Categorical Focal Loss (γ=2) | Categorical Focal Loss (γ=2) |
| Dropout (head) | 0.3 | 0.3 |
| Workers | 2 | 2 |
""")

        st.markdown("#### Strategi Penyeimbangan Kelas")
        st.markdown("""
Class weighting **tidak digunakan** (`class_weight=None`).

Penyeimbangan kelas dilakukan sepenuhnya pada tahap preprocessing:
- Setiap gambar latih → **twin K-Means segmentation** (k=3) = 2× data dasar
- Kelas yang basis-nya masih di bawah target → **top-up flip/rotate** hingga seimbang

Penambahan class weight di atas data yang sudah seimbang berisiko *over-correction*.
""")

    with col2:
        st.markdown("#### Augmentasi (Training Generator)")
        st.code("""ImageDataGenerator(
    rescale           = 1./255,
    horizontal_flip   = True,
    vertical_flip     = True,
    rotation_range    = 20,
    width_shift_range = 0.15,
    height_shift_range= 0.15,
    zoom_range        = 0.15,
    shear_range       = 0.1,
    brightness_range  = [0.85, 1.15],
    fill_mode         = 'nearest'
)""", language='python')

        st.warning("Augmentasi kuat = regularisasi implisit. Mengurangi augmentasi + dropout "
                   "sekaligus menyebabkan overfitting (val acc ~80%).")

        st.markdown("#### Callbacks")
        st.code("""callbacks = [
    ModelCheckpoint(
        filepath='models/best.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, patience=5
    )
]""", language='python')

    st.markdown('---')
    st.markdown("#### Fine-Tuning Strategy")
    st.markdown("**Baseline** — pure transfer learning, tanpa fine-tuning:")
    st.markdown("""
| Phase | Backbone | Epochs | LR | Keterangan |
|-------|----------|--------|-----|-----------|
| **Single Phase** | Fully Frozen | max 50 (early stopping) | 1e-4 | Hanya train classification head |
""")
    st.markdown("**CBAM** — dua fase:")
    st.markdown("""
| Phase | Backbone | Epochs | LR | Keterangan |
|-------|----------|--------|-----|-----------|
| **Phase 1** | Fully Frozen | max 20 | 1e-3 | Train CBAM block + head saja |
| **Phase 2** | Top-60 layers un-frozen | max 60 (early stopping) | 1e-4 | Fine-tune backbone + CBAM + head |
""")

# ═════════════════════════════════════════════════════
# TAB 3: PERBANDINGAN MODEL
# ═════════════════════════════════════════════════════
with t3:
    st.markdown("""
| Aspek | EfficientNetB0 Baseline | EfficientNetB0 + CBAM |
|-------|------------------------|----------------------|
| Backbone | EfficientNetB0 | EfficientNetB0 |
| Attention | Tidak ada | CBAM (Channel + Spatial) |
| Parameter tambahan | — | < 5% overhead |
| Backbone training | Fully frozen | Phase 2: top-60 layers un-frozen |
| Loss function | Categorical Focal Loss (γ=2) | Categorical Focal Loss (γ=2) |
| Dropout head | 0.3 | 0.3 |
| Fokus fitur | Global (ImageNet) | Adaptif (attention-guided) |
| File model | `efficientnetb0_baseline_best.h5` | `efficientnetb0_cbam_best.h5` |
""")

    cmp = os.path.join(RESULTS_DIR, 'models_comparison.png')
    if os.path.exists(cmp):
        st.image(cmp, caption='Grafik perbandingan performa', use_container_width=True)
    else:
        st.info('Gambar `results/models_comparison.png` tidak ditemukan.')

# ═════════════════════════════════════════════════════
# TAB 4: MODEL SUMMARY
# ═════════════════════════════════════════════════════
with t4:
    model_choice = st.selectbox('Model:', list(MODEL_FILES.keys()), index=1)
    if st.button('Tampilkan Model Summary', type='primary'):
        model = load_model(model_choice)
        if model is not None:
            import io
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + '\n'))
            st.code(buf.getvalue(), language=None)
            st.metric('Total Parameters', f'{model.count_params():,}')
        else:
            st.error('Model gagal dimuat.')
    else:
        st.info('Pilih model lalu klik tombol di atas.')
        st.markdown("""
**Estimasi statistik model:**
| | Baseline | CBAM |
|--|---------|------|
| Total params | ~5.3M | ~5.6M |
| Trainable params | ~1.1M | ~1.4M |
| Ukuran .h5 | ~20 MB | ~21 MB |
""")