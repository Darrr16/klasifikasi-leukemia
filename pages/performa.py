"""
pages/performa.py
==================
Performa model — membaca hasil dari results/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from helpers import (
    load_metrics, load_history, load_results,
    CLASS_NAMES, PLOT_FILES, RESULTS_DIR,
)

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0f172a',
    'axes.facecolor'   : '#1e293b',
    'axes.edgecolor'   : '#334155',
    'axes.labelcolor'  : '#94a3b8',
    'xtick.color'      : '#94a3b8',
    'ytick.color'      : '#94a3b8',
    'text.color'       : '#f1f5f9',
    'grid.color'       : '#334155',
    'legend.facecolor' : '#1e293b',
    'legend.edgecolor' : '#334155',
})

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('📊 Performa Model')
st.caption('Semua data diambil dari folder `results/` yang tersimpan saat training.')

# ── Pilih model ────────────────────────────────────────────────────────────────
MODEL_OPTIONS = ['EfficientNetB0 Baseline', 'EfficientNetB0 + CBAM']
selected = st.sidebar.radio('Model:', MODEL_OPTIONS, index=1)
st.sidebar.caption('Sumber data:\n`results/*_metrics.json`\n`results/*_history.npz`\n`results/*_results.npz`')

metrics = load_metrics(selected)
history = load_history(selected)
results = load_results(selected)

if metrics is None:
    st.error(f'File metrics tidak ditemukan untuk **{selected}**. '
             f'Pastikan `results/*_metrics.json` tersedia.')
    st.stop()

# ── Overview metrik ───────────────────────────────────────────────────────────
st.markdown("### Ringkasan Metrik")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric('Test Accuracy',  f"{metrics.get('test_accuracy',0)*100:.2f}%")
with c2: st.metric('Test Loss',      f"{metrics.get('test_loss',0):.4f}")
with c3: st.metric('MAE',            f"{metrics.get('mae',0):.4f}")
with c4: st.metric('Best Val Acc',   f"{metrics.get('best_val_acc',0)*100:.2f}%")
with c5: st.metric('Best Val Loss',  f"{metrics.get('best_val_loss',0):.4f}")

# ── Perbandingan kedua model ───────────────────────────────────────────────────
with st.expander('Bandingkan kedua model', expanded=False):
    m1 = load_metrics('EfficientNetB0 Baseline')
    m2 = load_metrics('EfficientNetB0 + CBAM')
    if m1 and m2:
        comp_df = pd.DataFrame({
            'Metrik'   : ['Accuracy (%)', 'Test Loss', 'MAE', 'Best Val Acc (%)'],
            'Baseline' : [round(m1.get('test_accuracy',0)*100,4), round(m1.get('test_loss',0),4),
                          round(m1.get('mae',0),4), round(m1.get('best_val_acc',0)*100,4)],
            'CBAM'     : [round(m2.get('test_accuracy',0)*100,4), round(m2.get('test_loss',0),4),
                          round(m2.get('mae',0),4), round(m2.get('best_val_acc',0)*100,4)],
        })
        comp_df['Delta (CBAM−Baseline)'] = comp_df['CBAM'] - comp_df['Baseline']
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        cmp_img = os.path.join(RESULTS_DIR, 'models_comparison.png')
        if os.path.exists(cmp_img):
            st.image(cmp_img, caption='Grafik perbandingan', use_container_width=True)
    else:
        st.info('Salah satu file metrics tidak ditemukan.')

st.markdown('---')

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Confusion Matrix',
    'Kurva Training',
    'Classification Report',
    'Sample Prediksi',
    'Raw JSON',
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    plots    = PLOT_FILES.get(selected, {})
    img_norm = plots.get('confusion_normalized', '')
    img_raw  = plots.get('confusion_raw', '')

    if os.path.exists(img_norm) and os.path.exists(img_raw):
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_norm, caption=f'{selected} — Normalized',
                     use_container_width=True)
        with c2:
            st.image(img_raw, caption=f'{selected} — Raw Counts',
                     use_container_width=True)
    elif results is not None and 'true_classes' in results and 'predicted_classes' in results:
        y_true = results['true_classes']
        y_pred = results['predicted_classes']
        cm     = confusion_matrix(y_true, y_pred)
        c1, c2 = st.columns(2)
        with c1:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Normalized')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Raw Counts')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.warning('File confusion matrix (PNG atau results.npz) tidak ditemukan.')

    # Insight
    if results is not None and 'true_classes' in results and 'predicted_classes' in results:
        y_true = results['true_classes']
        y_pred = results['predicted_classes']
        cm     = confusion_matrix(y_true, y_pred)
        st.markdown('---')
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown('**Prediksi Benar per Kelas:**')
            for i, cls in enumerate(CLASS_NAMES):
                total = cm[i].sum(); benar = cm[i, i]
                st.markdown(f'- **{cls}**: {benar}/{total} ({benar/total*100:.1f}%)')
        with ic2:
            st.markdown('**Kesalahan Terbanyak:**')
            cm_tmp = cm.copy().astype(float)
            np.fill_diagonal(cm_tmp, 0)
            for _ in range(3):
                idx = np.unravel_index(cm_tmp.argmax(), cm_tmp.shape)
                if cm_tmp[idx] == 0: break
                st.markdown(f'- **{CLASS_NAMES[idx[0]]}** → **{CLASS_NAMES[idx[1]]}**: {cm[idx]} kasus')
                cm_tmp[idx] = 0

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: KURVA TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    img_hist = PLOT_FILES.get(selected, {}).get('training_history', '')

    if os.path.exists(img_hist):
        st.image(img_hist, caption=f'{selected} — Training History',
                 use_container_width=True)
    elif history is not None:
        loss     = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        acc      = history.get('accuracy', [])
        val_acc  = history.get('val_accuracy', [])
        epochs   = range(1, len(loss) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(epochs, loss,     lw=2, color='#f87171', label='Train Loss')
        ax1.plot(epochs, val_loss, lw=2, color='#60a5fa', label='Val Loss')
        best_e = int(np.argmin(val_loss)) + 1
        ax1.scatter(best_e, val_loss[best_e-1], s=100, color='#fbbf24',
                    zorder=5, label=f'Best epoch {best_e}')
        ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=.2)

        ax2.plot(epochs, acc,     lw=2, color='#4ade80', label='Train Acc')
        ax2.plot(epochs, val_acc, lw=2, color='#fb923c', label='Val Acc')
        best_e2 = int(np.argmax(val_acc)) + 1
        ax2.scatter(best_e2, val_acc[best_e2-1], s=100, color='#fbbf24',
                    zorder=5, label=f'Best epoch {best_e2}')
        ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=.2)

        fig.suptitle(f'{selected} — Training History', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning('File training history tidak ditemukan.')

    if history:
        st.markdown('---')
        hc1, hc2, hc3 = st.columns(3)
        vl = history.get('val_loss', [0]); va = history.get('val_accuracy', [0])
        with hc1: st.metric('Best Val Loss', f"{min(vl):.4f}", f"Epoch {int(np.argmin(vl))+1}")
        with hc2: st.metric('Best Val Acc',  f"{max(va)*100:.2f}%", f"Epoch {int(np.argmax(va))+1}")
        with hc3: st.metric('Total Epochs',  len(history.get('loss', [])))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if results is not None and 'true_classes' in results and 'predicted_classes' in results:
        y_true = results['true_classes']
        y_pred = results['predicted_classes']

        st.code(classification_report(y_true, y_pred, target_names=CLASS_NAMES), language=None)

        cm   = confusion_matrix(y_true, y_pred)
        FP   = cm.sum(axis=0) - np.diag(cm)
        FN   = cm.sum(axis=1) - np.diag(cm)
        TP   = np.diag(cm)
        TN   = cm.sum() - (FP + FN + TP)
        prec = np.where(TP+FP>0, TP/(TP+FP), 0)
        rec  = np.where(TP+FN>0, TP/(TP+FN), 0)
        f1   = np.where(prec+rec>0, 2*prec*rec/(prec+rec), 0)
        spec = np.where(TN+FP>0, TN/(TN+FP), 0)

        report_df = pd.DataFrame({
            'Kelas'      : CLASS_NAMES,
            'Precision'  : np.round(prec, 4),
            'Recall'     : np.round(rec,  4),
            'F1-Score'   : np.round(f1,   4),
            'Specificity': np.round(spec, 4),
            'Support'    : cm.sum(axis=1),
        })
        st.dataframe(report_df, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(CLASS_NAMES)); w = 0.2
        ax.bar(x-w*1.5, prec, w, label='Precision',   color='#f87171')
        ax.bar(x-w*.5,  rec,  w, label='Recall',      color='#60a5fa')
        ax.bar(x+w*.5,  f1,   w, label='F1-Score',    color='#4ade80')
        ax.bar(x+w*1.5, spec, w, label='Specificity', color='#fbbf24')
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.08); ax.legend(); ax.grid(axis='y', alpha=.2)
        ax.set_title(f'{selected} — Per-Class Metrics')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning('File `*_results.npz` tidak ditemukan atau tidak mengandung `true_classes`/`predicted_classes`.')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SAMPLE PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    img_samp = PLOT_FILES.get(selected, {}).get('sample_predictions', '')
    if os.path.exists(img_samp):
        st.image(img_samp, caption=f'{selected} — Sample Predictions',
                 use_container_width=True)
        st.caption('🟢 Label hijau = prediksi benar  |  🔴 Label merah = prediksi salah')
    else:
        st.warning('Gambar sample predictions tidak ditemukan di folder `results/`.')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RAW JSON
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f'**Isi file metrics untuk {selected}:**')
    st.json(metrics)
    st.download_button(
        '⬇️ Download metrics.json',
        data=json.dumps(metrics, indent=2),
        file_name=f'{selected.replace(" ", "_")}_metrics.json',
        mime='application/json',
    )