# 🔬 Klasifikasi Leukemia — Blood Cell Classification

Sistem deteksi otomatis **Acute Lymphoblastic Leukemia (ALL)** menggunakan deep learning dengan arsitektur **EfficientNetB0 + CBAM**.

## 🌐 Live Demo

**[Kunjungi aplikasi →](https://klasifikasi-leukemia-streamlit.streamlit.app)**

## 📋 Tentang Proyek

Aplikasi ini mengklasifikasikan gambar sel darah menjadi 4 kategori:

| Kelas       | Label                  | Status |
| ----------- | ---------------------- | ------ |
| `EarlyPreB` | Early Precursor B-cell | ALL+   |
| `PreB`      | Precursor B-cell       | ALL+   |
| `ProB`      | Pro-B-cell             | ALL+   |
| `benign`    | Sel Normal / Sehat     | Normal |

### Arsitektur Model

- **Baseline**: EfficientNetB0 + classification head standar
- **Best**: EfficientNetB0 + Convolutional Block Attention Module (CBAM)

### Dataset

Dataset berasal dari [Blood Cell Cancer [ALL] — Kaggle](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)

- **Total**: 1,500 gambar
- **Split**: 90% training, 10% testing
- **Ukuran**: 224 × 224 px

## 🚀 Fitur Aplikasi

### 1. 🔍 Prediksi

Upload gambar sel darah dan dapatkan:

- Hasil klasifikasi otomatis
- Confidence score untuk setiap kelas
- Penjelasan medis dari hasil prediksi

### 2. 📊 Performa Model

Lihat metrik performa lengkap:

- Accuracy, precision, recall, F1-score
- Confusion matrix (raw & normalized)
- Kurva training history
- Contoh prediksi

### 3. 📈 Data Explorer

Analisis dataset:

- Distribusi kelas
- Galeri gambar per kategori
- Contoh augmentasi data

### 4. 🏗️ Arsitektur

Detail teknis:

- Diagram model EfficientNetB0
- Penjelasan CBAM
- Konfigurasi training

## 📦 Tech Stack

```
Python 3.10
├── TensorFlow / Keras (Deep Learning)
├── Streamlit (Web UI)
├── OpenCV (Image Processing)
├── NumPy, Pandas (Data Processing)
├── Matplotlib, Seaborn (Visualization)
└── scikit-learn (Metrics)
```

## 💻 Menjalankan Lokal

### Prasyarat

- Python 3.10+
- pip atau conda

### Setup

1. **Clone repository**

   ```bash
   git clone https://github.com/Darr16/klasifikasi-leukemia.git
   cd klasifikasi-leukemia
   ```

2. **Buat virtual environment**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**

   ```bash
   streamlit run app.py
   ```

5. **Akses aplikasi**
   ```
   http://localhost:8501
   ```

## 📁 Struktur Proyek

```
klasifikasi-leukemia/
├── app.py                          # Entry point Streamlit
├── requirements.txt                # Dependencies
├── README.md                       # (ini file)
│
├── pages/                          # Halaman aplikasi
│   ├── prediksi.py                # Prediksi gambar
│   ├── performa.py                # Metrik & hasil
│   ├── data_explorer.py           # Analisis dataset
│   ├── arsitektur.py              # Detail model
│   └── dokumentasi.py             # Dokumentasi
│
├── src/                           # Logic & utilities
│   ├── data_preprocessing.py      # Preprocessing & augmentasi
│   ├── models.py                  # Definisi model
│   └── training_utils.py          # Training & evaluasi
│
├── utils/                         # Helper functions
│   ├── helpers.py                 # Shared utilities
│   └── __init__.py
│
├── models/                        # Pre-trained models
│   ├── efficientnetb0_baseline_best.h5
│   └── efficientnetb0_cbam_best.h5
│
└── results/                       # Metrics & visualizations
    ├── efficientnetb0_cbam_metrics.json
    ├── efficientnetb0_cbam_history.npz
    └── ...
```

## 📊 Hasil Performa

### EfficientNetB0 + CBAM (Best Model)

| Metrik               | Nilai   |
| -------------------- | ------- |
| Test Accuracy        | ~95%    |
| Precision (benign)   | ~96%    |
| Recall (ALL classes) | ~94-97% |
| F1-Score (avg)       | ~95%    |

## ⚠️ Disclaimer

> Aplikasi ini dirancang untuk **keperluan penelitian dan akademik** saja. **Bukan alat diagnosis klinis** dan tidak boleh digunakan untuk diagnosis medis tanpa konsultasi dengan profesional kesehatan yang berkualifikasi.

## 📚 Referensi

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module
3. Blood Cell Cancer [ALL] Dataset — Kaggle

## 👤 Penulis

Daftar Rasyid (Darr16)  
Proyek Akhir PI — SMT6

## 📄 Lisensi

MIT License — Bebas digunakan untuk keperluan penelitian dan akademik

## 📞 Kontak & Support

- Email : gobalgabel45@gmail.com

**Terakhir diupdate**: Juli 2026
