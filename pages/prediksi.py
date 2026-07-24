"""
pages/prediksi.py
=================
Prediksi satu gambar sel darah.
Catatan: set_page_config TIDAK dipanggil di sini (sudah di app.py).
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import (
    load_model, preprocess_for_prediction,
    interpret_confidence, CLASS_NAMES, CLASS_DESCRIPTIONS, MODEL_FILES,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('🔍 Prediksi Gambar Sel Darah')
st.caption('Upload satu gambar sel darah untuk mendapatkan hasil klasifikasi otomatis.')

# ── Sidebar: pilih model (hanya 2) ────────────────────────────────────────────
st.sidebar.header('Pengaturan')
model_choice = st.sidebar.selectbox(
    'Pilih Model:',
    options=list(MODEL_FILES.keys()),   # ['EfficientNetB0 Baseline', 'EfficientNetB0 + CBAM']
    index=1,
    help='EfficientNetB0 + CBAM adalah model dengan akurasi tertinggi.',
)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Gambar")

col_up, col_tip = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        'Pilih gambar:',
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility='collapsed',
    )
with col_tip:
    st.info("**Format:** JPG, PNG, BMP  \n**Resolusi ideal:** ≥ 256×256 px  \n**Maks ukuran:** 10 MB")

# ── Proses ─────────────────────────────────────────────────────────────────────
if uploaded is not None:
    pil_img = Image.open(uploaded)
    w, h    = pil_img.size

    if w < 64 or h < 64:
        st.warning('⚠️ Gambar terlalu kecil (< 64×64 px). Hasil prediksi mungkin tidak akurat.')

    # ── Preprocessing preview ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Preprocessing Preview")

    img_array, img_resized = preprocess_for_prediction(pil_img)

    c1, c_arrow, c2 = st.columns([2, 0.3, 2])
    with c1:
        st.image(pil_img, caption=f'Original — {w}×{h} px', use_container_width=True)
    with c_arrow:
        st.markdown("<div style='text-align:center;padding-top:5rem;font-size:2rem'>→</div>",
                    unsafe_allow_html=True)
    with c2:
        st.image(img_resized, caption='Input model — 224×224 px (rescale /255)',
                 use_container_width=True)

    with st.expander('Langkah preprocessing yang diterapkan'):
        st.markdown("""
1. Konversi ke **RGB** (jika belum)
2. **Resize** ke 224×224 px (LANCZOS)
3. **Normalisasi** piksel `[0, 255] → [0, 1]`
4. Tambah dimensi batch → shape `(1, 224, 224, 3)`

> *Bilateral filter tidak digunakan* — menghaluskan tepi sel yang merupakan fitur diagnostik kritis.
""")

    # ── Prediksi ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Hasil Prediksi")

    model = load_model(model_choice)

    if model is not None:
        with st.spinner('Memproses prediksi…'):
            preds      = model.predict(img_array, verbose=0)
            proba      = preds[0]
            pred_idx   = int(np.argmax(proba))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(proba[pred_idx])

        class_info              = CLASS_DESCRIPTIONS[pred_class]
        conf_label, conf_type   = interpret_confidence(confidence)
        warna                   = class_info['warna']

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"""
<div style="background:linear-gradient(135deg,{warna}33,{warna}18);
            border:1px solid {warna}55;
            color:#f1f5f9;padding:1.8rem;border-radius:1rem;text-align:center;">
  <p style="margin:0;font-size:0.85rem;color:#94a3b8">Kelas Prediksi</p>
  <h2 style="margin:0.3rem 0 0;color:{warna}">{class_info['label']}</h2>
  <code style="background:rgba(0,0,0,0.35);color:#e2e8f0;padding:0.2rem 0.7rem;
               border-radius:0.3rem;font-size:0.85rem">{pred_class}</code>
  <hr style="border-color:#334155;margin:1rem 0">
  <p style="margin:0;font-size:0.85rem;color:#94a3b8">Confidence Score</p>
  <h1 style="margin:0.2rem 0 0;color:{warna}">{confidence*100:.1f}%</h1>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"**{conf_label}**")
            getattr(st, conf_type)(
                {
                    'success': 'Model sangat yakin dengan prediksi ini.',
                    'info'   : 'Model cukup yakin. Disarankan validasi oleh pakar.',
                    'warning': 'Model kurang yakin. Gambar mungkin ambigu atau berkualitas rendah.',
                    'error'  : 'Confidence rendah. Coba gambar lain yang lebih jelas.',
                }[conf_type]
            )

            st.markdown(f"**Tentang kelas *{pred_class}*:**")
            st.write(class_info['description'])
            st.markdown("**Karakteristik morfologi:**")
            for c in class_info['ciri']:
                st.markdown(f"- {c}")

        with res_col2:
            st.markdown("**Distribusi Confidence Semua Kelas:**")
            df_proba = pd.DataFrame({
                'Kelas'     : CLASS_NAMES,
                'Confidence': proba * 100,
            }).sort_values('Confidence', ascending=True)
            st.bar_chart(df_proba.set_index('Kelas'), height=260)

            st.markdown("**Tabel Probabilitas:**")
            df_tbl = pd.DataFrame({
                'Kelas'   : CLASS_NAMES,
                'Prob (%)': [f"{p*100:.3f}" for p in proba],
            }).sort_values('Prob (%)', ascending=False)
            st.dataframe(df_tbl, hide_index=True, use_container_width=True)

        # ── Interpretasi klinis ────────────────────────────────────────────────
        st.markdown("---")
        if pred_class == 'benign':
            st.success("🟢 Model memrediksi sel ini sebagai **NORMAL (Benign)** — tidak terdeteksi ciri leukemia.")
        else:
            st.error(f"🔴 Model memrediksi sel ini sebagai **{pred_class}** — termasuk kategori **ALL-positif**.")

        st.caption("⚠️ Hasil hanya untuk keperluan riset/akademik — bukan alat diagnosis klinis.")

    else:
        st.error('Model gagal dimuat. Periksa file `.h5` di folder `models/`.')

else:
    st.markdown("""
<div style="background:#1e293b;border:2px dashed #334155;border-radius:1rem;
            padding:3rem;text-align:center;color:#64748b">
  <p style="font-size:2.5rem;margin:0">📁</p>
  <h3 style="margin:0.5rem 0;color:#94a3b8">Belum ada gambar yang di-upload</h3>
  <p style="margin:0">Klik tombol upload di atas atau seret gambar ke area tersebut.</p>
</div>
""", unsafe_allow_html=True)

    with st.expander('Panduan penggunaan'):
        st.markdown("""
1. Pilih model di sidebar (default: CBAM — model terbaik)
2. Klik **Browse files** atau seret gambar ke area upload
3. Tunggu preprocessing dan prediksi selesai (< 5 detik)
4. Lihat hasil: kelas prediksi, confidence score, dan interpretasi

**Tips untuk hasil terbaik:**
- Resolusi ≥ 256×256 px, gambar tidak blur
- Sel terlihat jelas, pencahayaan cukup
- Format: JPG, PNG, atau BMP
""")