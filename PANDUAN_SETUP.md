# Panduan Setup: Blood Cell Classification — EfficientNetB0 + CBAM

Panduan ini menjelaskan langkah-langkah lengkap untuk menyiapkan, menjalankan, dan mengembangkan proyek klasifikasi sel darah menggunakan **EfficientNetB0 + CBAM** di lingkungan lokal (VSCode).

> **Catatan Penting — Python 3.12:** Jika Python global Anda adalah versi 3.12, TensorFlow 2.13 belum memiliki dukungan penuh untuk versi tersebut. Panduan ini mencakup cara menginstal Python 3.10 secara berdampingan dan membuat virtual environment yang mengarah ke versi tersebut — tanpa mengganggu instalasi Python 3.12 yang sudah ada.

---

## Daftar Isi

1. [Prasyarat Sistem](#1-prasyarat-sistem)
2. [Struktur Proyek](#2-struktur-proyek)
3. [Persiapan Lingkungan (Environment)](#3-persiapan-lingkungan-environment)
4. [Instalasi Dependensi](#4-instalasi-dependensi)
5. [Konfigurasi Dataset](#5-konfigurasi-dataset)
6. [Setup VSCode](#6-setup-vscode)
7. [Menjalankan Notebook](#7-menjalankan-notebook)
8. [Alur Eksekusi Notebook (Cell-by-Cell)](#8-alur-eksekusi-notebook-cell-by-cell)
9. [Pemecahan Masalah (Troubleshooting)](#9-pemecahan-masalah-troubleshooting)
10. [Catatan Teknis Arsitektur](#10-catatan-teknis-arsitektur)

---

## 1. Prasyarat Sistem

Pastikan semua komponen berikut sudah terpasang di komputer Anda sebelum memulai.

**Python 3.10** — Ini adalah versi yang digunakan untuk virtual environment proyek ini. TensorFlow 2.13 belum mendukung Python 3.12 secara penuh, sehingga Python 3.10 digunakan di dalam venv meskipun Python 3.12 tetap menjadi versi global sistem Anda. Unduh installer Python 3.10 dari [python.org/downloads](https://www.python.org/downloads/release/python-31011/). Saat proses instalasi, **jangan centang opsi "Add Python to PATH"** agar tidak menimpa atau mengganggu Python 3.12 yang sudah aktif.

**Git** — Diperlukan untuk kloning repositori. Unduh dari [git-scm.com](https://git-scm.com).

**VSCode** — Unduh dari [code.visualstudio.com](https://code.visualstudio.com). Setelah terinstal, pasang ekstensi berikut melalui panel Extensions (`Ctrl+Shift+X`):

- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **Pylance** (Microsoft)

**GPU (Opsional namun sangat disarankan)** — Jika menggunakan NVIDIA GPU, pasang CUDA Toolkit 11.8 dan cuDNN 8.6 untuk akselerasi TensorFlow-GPU. Periksa kompatibilitas di [tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

---

## 2. Struktur Proyek

Setelah setup selesai, proyek akan memiliki struktur berikut:

```
dashboard-blood-cell-efficientnet/
├── FGD-3.ipynb                          ← Notebook utama penelitian
├── requirements.txt                      ← Daftar dependensi Python
├── README.md                            ← Dokumentasi proyek
│
├── data/
│   ├── Blood cell Cancer [ALL]/         ← Dataset mentah (dari Kaggle)
│   │   ├── EarlyPreB/
│   │   ├── PreB/
│   │   ├── ProB/
│   │   └── benign/
│   └── tmp/                             ← Data yang sudah diproses (auto-generated)
│       ├── prepared_data/               ← Training: original + segmented
│       └── prepared_test/               ← Test: resize only
│
├── src/
│   ├── data_preprocessing.py            ← Pipeline preprocessing & augmentasi
│   ├── models.py                        ← EfficientNetB0 Baseline & CBAM
│   └── training_utils.py                ← Training, evaluasi & visualisasi
│
├── models/                              ← Model terbaik (dari ModelCheckpoint)
│   ├── efficientnetb0_baseline_best.h5
│   └── efficientnetb0_cbam_best.h5
│
├── results/                             ← Output visualisasi & metrik
│   ├── EfficientNetB0 Baseline_training_history.png
│   ├── EfficientNetB0 + CBAM_training_history.png
│   ├── EfficientNetB0 Baseline_confusion_matrix_raw.png
│   ├── EfficientNetB0 Baseline_confusion_matrix_normalized.png
│   ├── EfficientNetB0 + CBAM_confusion_matrix_raw.png
│   ├── EfficientNetB0 + CBAM_confusion_matrix_normalized.png
│   ├── EfficientNetB0 Baseline_sample_predictions.png
│   ├── EfficientNetB0 + CBAM_sample_predictions.png
│   ├── models_comparison.png
│   ├── efficientnetb0_baseline.h5
│   ├── efficientnetb0_cbam.h5
│   ├── efficientnetb0_baseline_metrics.json
│   └── efficientnetb0_cbam_metrics.json
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
└── utils/
    ├── __init__.py
    └── (utility files untuk app deployment)
```

---

## 3. Persiapan Lingkungan (Environment)

Sangat disarankan menggunakan **virtual environment** agar dependensi proyek tidak bertabrakan dengan instalasi Python global. Karena Python global Anda adalah 3.12, venv ini akan dibuat secara eksplisit menggunakan Python 3.10 yang sudah diinstal di Langkah 1.

### Langkah 3.1 — Buka Terminal di VSCode

Buka VSCode, lalu buka terminal terintegrasi dengan menekan `` Ctrl+` `` atau melalui menu **Terminal → New Terminal**.

### Langkah 3.2 — Buat Folder Proyek

```bash
mkdir dashboard-blood-cell-efficientnet
cd dashboard-blood-cell-efficientnet
```

### Langkah 3.3 — Verifikasi Python 3.10 Terdeteksi

Sebelum membuat venv, pastikan Python 3.10 yang baru diinstal sudah dapat dikenali oleh sistem. Buka terminal baru (bukan terminal lama yang masih aktif), lalu jalankan:

```bash
# Windows — Python Launcher menampilkan semua versi yang terinstal
py -0
```

Output yang diharapkan:

```
 -V:3.12 *        Python 3.12 (active)
 -V:3.10          Python 3.10
```

Jika perintah `py` tidak dikenal, gunakan path langsung untuk memverifikasi:

```bash
# Windows — path default installer Python 3.10
"C:\Users\<NamaAnda>\AppData\Local\Programs\Python\Python310\python.exe" --version
```

```bash
# macOS / Linux
python3.10 --version
```

### Langkah 3.4 — Buat Virtual Environment dengan Python 3.10

Perintah berikut membuat venv yang secara eksplisit menggunakan interpreter Python 3.10, bukan 3.12 yang aktif secara global. Python 3.12 global Anda tidak akan tersentuh sama sekali.

**Windows (menggunakan Python Launcher — cara yang disarankan):**

```bash
py -3.10 -m venv venv
```

**Windows (menggunakan path eksplisit, jika `py` tidak tersedia):**

```bash
"C:\Users\<NamaAnda>\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv
```

**macOS / Linux:**

```bash
python3.10 -m venv venv
```

Setelah perintah selesai, akan muncul folder `venv/` di dalam direktori proyek.

### Langkah 3.5 — Aktifkan Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

> Jika muncul error terkait ExecutionPolicy, jalankan perintah berikut terlebih dahulu, lalu ulangi aktivasi:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

Setelah aktif, prompt terminal akan menampilkan `(venv)` di awal baris.

### Langkah 3.6 — Verifikasi Versi Python di Dalam Venv

Langkah ini penting untuk memastikan venv benar-benar menggunakan Python 3.10, bukan 3.12.

```bash
python --version
# Output yang diharapkan: Python 3.10.x
```

Jika output masih menampilkan 3.12, berarti venv dibuat dengan interpreter yang salah. Hapus folder `venv/` dan ulangi Langkah 3.4 menggunakan perintah dengan path eksplisit.

### Langkah 3.7 — Pilih Interpreter di VSCode

Tekan `Ctrl+Shift+P` → ketik **"Python: Select Interpreter"** → pilih interpreter yang menunjukkan path `.\venv\Scripts\python.exe` (Windows) atau `./venv/bin/python` (Mac/Linux) **dengan label versi 3.10**. Ini memastikan kernel Jupyter di notebook juga menggunakan Python 3.10 yang sama.

---

## 4. Instalasi Dependensi

### Langkah 4.1 — Upgrade pip

```bash
pip install --upgrade pip
```

### Langkah 4.2 — Install dari requirements.txt

Pastikan file `requirements.txt` sudah ada di folder proyek, lalu jalankan:

```bash
pip install -r requirements.txt
```

Proses instalasi memerlukan waktu beberapa menit tergantung kecepatan internet.

### Langkah 4.3 — Verifikasi Instalasi

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Output yang diharapkan (contoh):

```
TF: 2.13.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Jika tidak ada GPU, output GPU akan berupa `[]` — training tetap bisa berjalan menggunakan CPU, namun lebih lambat.

---

## 5. Konfigurasi Dataset

### Langkah 5.1 — Unduh Dataset

Dataset yang digunakan adalah **"Blood Cell Cancer [ALL]"** dari Kaggle.

1. Buka: [https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
2. Klik **Download** (perlu akun Kaggle — gratis)
3. Ekstrak file ZIP yang diunduh

### Langkah 5.2 — Tempatkan Dataset

Letakkan folder dataset di dalam proyek dengan struktur berikut:

```
dashboard-blood-cell-efficientnet/
└── data/
    └── Blood cell Cancer [ALL]/
        ├── EarlyPreB/      ← ±3.000 gambar
        ├── PreB/           ← ±3.000 gambar
        ├── ProB/           ← ±3.000 gambar
        └── benign/         ← ±3.000 gambar
```

> **Catatan:** Pastikan nama folder menggunakan spasi dan kapitalisasi yang tepat: `Blood cell Cancer [ALL]`. Path ini sesuai dengan nilai `CONFIG['data_dir']` di notebook.

### Langkah 5.3 — Buat Folder data/tmp

```bash
mkdir -p data/tmp
```

Folder ini akan diisi otomatis oleh notebook saat preprocessing berjalan.

---

## 6. Setup VSCode

### Langkah 6.1 — Buka Folder Proyek

Di VSCode: **File → Open Folder** → pilih folder `dashboard-blood-cell-efficientnet`.

### Langkah 6.2 — Salin File Source Code

Pastikan semua file berikut sudah ada di folder `src/`:

- `src/data_preprocessing.py`
- `src/models.py`
- `src/training_utils.py`

Buat folder jika belum ada:

```bash
mkdir -p src models results config utils
```

### Langkah 6.3 — Konfigurasi Jupyter Kernel

1. Buka file `FGD-3.ipynb` di VSCode
2. Klik **"Select Kernel"** di pojok kanan atas editor notebook
3. Pilih **"Python Environments"** → pilih environment `venv` yang sudah dibuat

### Langkah 6.4 — (Opsional) Konfigurasi GPU Memory Growth

Tambahkan kode berikut di awal notebook jika menggunakan GPU untuk mencegah TensorFlow mengalokasikan seluruh memori GPU sekaligus:

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
```

---

## 7. Menjalankan Notebook

### Opsi A — Jalankan Semua Cell Sekaligus

Klik tombol **"Run All"** (▶▶) di toolbar notebook VSCode atau gunakan shortcut `Ctrl+F9`.

> **Perhatian:** Proses K-means segmentation pada Section 4.3 memerlukan waktu **15–45 menit** tergantung ukuran dataset dan kecepatan CPU. Proses ini hanya perlu dilakukan **sekali** — jika folder `data/tmp/prepared_data` sudah terisi, notebook akan otomatis melewati proses ini.

### Opsi B — Jalankan Per Section

Untuk pemahaman yang lebih baik, jalankan cell per cell menggunakan `Shift+Enter`. Urutan eksekusi yang benar:

```
Section 1 → Section 2 → Section 3 → Section 4.1 → 4.2 → 4.3 → 4.4 → 4.5
→ Section 5.1 → 5.2 → Section 6.1 → 6.2 → 6.3
→ Section 7 → Section 8 → Section 9 → Section 10
```

### Estimasi Waktu Training

| Kondisi                     | Estimasi Waktu per Model |
| --------------------------- | ------------------------ |
| CPU saja (Intel i7/Ryzen 7) | 3–6 jam                  |
| NVIDIA GTX 1060 / RTX 2060  | 30–60 menit              |
| NVIDIA RTX 3080 / 4080      | 10–20 menit              |
| Google Colab (T4 GPU)       | 20–40 menit              |

---

## 8. Alur Eksekusi Notebook (Cell-by-Cell)

Bagian ini menjelaskan apa yang terjadi di setiap section notebook.

**Section 1 — Import & Setup:** Mengimpor semua library, menetapkan random seed untuk reprodusibilitas, dan mengimpor modul kustom dari folder `src/`.

**Section 2 — Configuration:** Mendefinisikan `CONFIG` dictionary yang berisi semua hyperparameter. Modifikasi nilai di sini jika ingin bereksperimen (misalnya mengubah `batch_size`, `epochs`, atau `dropout_rate`).

**Section 3 — Data Exploration:** Memverifikasi keberadaan dataset, menghitung jumlah sampel per kelas, dan menampilkan bar chart distribusi kelas.

**Section 4.1 — Folder & Split:** Membuat struktur folder `data/tmp/` dan melakukan stratified split 90:10 (train:test).

**Section 4.2 — Test Processing:** Menyimpan gambar test yang sudah di-resize ke `data/tmp/prepared_test/`. Proses ini cepat (±1–2 menit).

**Section 4.3 — Train Processing:** Proses terlama — melakukan K-means segmentation pada setiap gambar training dan menyimpan dua versi (original + segmented) ke `data/tmp/prepared_data/`. **Jika sudah pernah dijalankan, cell ini akan dilewati otomatis.**

**Section 4.4 — Generators:** Membuat DataFrames dan Keras ImageDataGenerators. Augmentasi (rotasi, flip, zoom, shift) diterapkan hanya pada data training.

**Section 5 — Model Creation:** Membangun dua model:

- `baseline_model`: EfficientNetB0 dengan classification head standar
- `cbam_model`: EfficientNetB0 ditambah CBAM block sebelum global pooling

**Section 6 — Training:** Melatih kedua model dengan callbacks (EarlyStopping + ReduceLROnPlateau + ModelCheckpoint). Model terbaik (berdasarkan `val_accuracy`) disimpan otomatis ke folder `models/`.

**Section 7 — Training Analysis:** Menampilkan dan menyimpan kurva loss dan accuracy.

**Section 8 — Evaluation:** Menjalankan inferensi pada test set, menampilkan confusion matrix (raw dan normalized), classification report, dan grid prediksi sampel.

**Section 9 — Comparison:** Perbandingan side-by-side kedua model dengan bar chart akurasi dan F1 per kelas. Model dan metrik JSON disimpan ke folder `results/`.

**Section 10 — Summary:** Ringkasan final seluruh hasil penelitian.

---

## 9. Pemecahan Masalah (Troubleshooting)

**Venv terbuat tetapi masih menggunakan Python 3.12**

Ini terjadi jika venv dibuat tanpa menentukan versi secara eksplisit (misalnya menggunakan `python -m venv venv` biasa). Solusinya adalah menghapus folder `venv/` yang salah dan membuatnya ulang menggunakan perintah eksplisit dari Langkah 3.4. Pastikan Python 3.10 sudah terdeteksi terlebih dahulu dengan menjalankan `py -0` (Windows) atau `python3.10 --version` (Mac/Linux).

```bash
# Hapus venv lama
rmdir /s /q venv        # Windows
rm -rf venv             # macOS / Linux

# Buat ulang dengan Python 3.10
py -3.10 -m venv venv   # Windows
python3.10 -m venv venv # macOS / Linux
```

**Error: `ModuleNotFoundError: No module named 'src.data_preprocessing'`**

Pastikan Anda menjalankan notebook dari root directory proyek. Di VSCode, periksa direktori kerja aktif dengan menjalankan cell berikut:

```python
import os; print(os.getcwd())
```

Output harus berakhir dengan nama folder proyek Anda.

**Error: `include_preprocessing` is not a valid argument**

Ini terjadi jika versi TensorFlow < 2.4. Solusinya, buka `src/models.py` dan hapus parameter `include_preprocessing=True` dari kedua fungsi `EfficientNetB0(...)`. Tambahkan layer preprocessing manual sebagai gantinya:

```python
x = tf.keras.layers.Rescaling(1./255)(inputs)
```

Sesuaikan juga rescale di generator menjadi `rescale=None` (hapus rescaling di generator).

**Error: `OOM (Out of Memory)` saat training**

Kurangi `batch_size` di `CONFIG` dari 32 menjadi 16 atau 8. Selain itu, aktifkan GPU memory growth seperti dijelaskan di Langkah 6.4.

**Training sangat lambat tanpa GPU**

Pertimbangkan menggunakan Google Colab (gratis, dengan GPU T4). Upload notebook dan file `src/` ke Google Drive, lalu mount drive di Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Error: Dataset tidak ditemukan**

Pastikan path dataset tepat: `data/Blood cell Cancer [ALL]/`. Perhatikan spasi dan tanda kurung siku. Verifikasi dengan:

```python
import os
print(os.path.exists('data/Blood cell Cancer [ALL]'))
```

---

## 10. Catatan Teknis Arsitektur

### Mengapa EfficientNetB0?

EfficientNetB0 menggunakan prinsip _compound scaling_ — menyeimbangkan kedalaman (depth), lebar (width), dan resolusi (resolution) secara bersamaan dengan koefisien yang ditentukan melalui Neural Architecture Search. Dibandingkan MobileNetV2, EfficientNetB0 umumnya mencapai akurasi lebih tinggi pada jumlah parameter yang setara, menjadikannya pilihan tepat untuk penelitian yang mengutamakan efisiensi.

### Mengapa CBAM?

CBAM (Convolutional Block Attention Module) bekerja dalam dua tahap:

**Channel Attention** — Menjawab pertanyaan "fitur _apa_ yang penting?". Dua deskriptor global (average-pool dan max-pool) dikompresi melalui MLP bersama, lalu dijumlahkan dan dilewatkan sigmoid. Hasilnya mengkalibrasi ulang bobot setiap channel.

**Spatial Attention** — Menjawab pertanyaan "fitur di _mana_ yang penting?". Average dan max pooling dilakukan sepanjang dimensi channel, dikoncatenasi, lalu dikonvolusi dengan filter 7×7 untuk menghasilkan peta perhatian 2D.

Dengan menempatkan CBAM setelah backbone EfficientNetB0 (sebelum global pooling), model dapat secara selektif menekankan region sel yang paling relevan untuk klasifikasi, yang sangat berguna dalam analisis citra medis.

### Fine-tuning (Opsional)

Setelah training awal dengan backbone frozen, Anda dapat melakukan fine-tuning untuk meningkatkan akurasi:

```python
# Unfreeze beberapa layer terakhir backbone
base_model = baseline_model.layers[1]  # layer index EfficientNetB0
base_model.trainable = True

# Freeze semua layer kecuali 30 terakhir
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile dengan LR yang jauh lebih kecil
baseline_model = compile_model(
    baseline_model,
    initial_learning_rate=1e-5,  # 100x lebih kecil dari training awal
    decay_steps=20,
    decay_rate=0.9
)

# Lanjutkan training
baseline_model.fit(train_gen, epochs=20, ...)
```

---

_Panduan ini dibuat untuk proyek penelitian Blood Cell Classification — EfficientNetB0 + CBAM._
