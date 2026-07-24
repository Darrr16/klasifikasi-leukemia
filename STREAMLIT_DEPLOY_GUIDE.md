# Panduan Lengkap Deploy Aplikasi Streamlit

Panduan ini khusus untuk proyek klasifikasi citra leukimia yang menggunakan Streamlit dan file utama aplikasi adalah app.py.

---

## 1. Persiapan sebelum deploy

Pastikan hal-hal berikut sudah tersedia:

- File utama aplikasi: app.py
- Folder halaman: pages/
- Folder kode pendukung: src/, utils/
- Folder model: models/
- File dependency: requirements.txt
- Dataset tidak wajib di-upload jika model sudah tersedia di repo

Project Anda sudah memiliki struktur yang cukup baik untuk deploy ke Streamlit Cloud.

---

## 2. Pastikan file yang dibutuhkan ada

Cek apakah file berikut ada di root proyek:

- app.py
- requirements.txt
- pages/
- src/
- utils/
- models/
- results/

Jika ada file besar yang tidak ingin ikut di GitHub, pastikan model yang dibutuhkan tetap tersedia di repo atau di tempat yang bisa diunduh saat runtime.

---

## 3. Periksa requirements.txt

Untuk deployment di Streamlit Community Cloud, pastikan file requirements.txt berisi paket yang dibutuhkan.

Contoh yang cocok untuk project Anda:

```txt
streamlit>=1.36.0
tensorflow-cpu==2.13.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
opencv-python==4.8.0.76
Pillow==10.0.0
imutils==0.5.4
matplotlib==3.7.2
seaborn==0.12.2
```

> Catatan: untuk Streamlit Cloud, lebih aman memakai tensorflow-cpu dibanding tensorflow biasa karena lebih kompatibel pada environment cloud.

---

## 4. Jalankan aplikasi lokal dulu

Sebelum deploy, pastikan app berjalan di komputer Anda.

Buka terminal di folder project, lalu jalankan:

```bash
streamlit run app.py
```

Jika berhasil, biasanya browser akan terbuka ke alamat:

```txt
http://localhost:8501
```

Jika tidak bisa dibuka, cek apakah ada error seperti:

- modul tidak ditemukan
- model tidak ada
- path file salah
- dependency belum terinstall

---

## 5. Siapkan repository GitHub

Streamlit Cloud biasanya deploy dari GitHub.

### Langkah-langkah:

1. Buat akun GitHub jika belum punya
2. Buat repository baru
3. Upload project ke repository tersebut

Contoh perintah:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/NAMA_USER/NAMA_REPO.git
git push -u origin main
```

---

## 6. Deploy ke Streamlit Community Cloud

### Langkah-langkah:

1. Buka situs: https://streamlit.io/cloud
2. Login dengan akun GitHub
3. Klik tombol New app
4. Pilih repository yang sudah Anda push
5. Pilih branch (biasanya main)
6. Pilih file utama aplikasi: app.py
7. Klik Deploy

### Pengaturan penting:

- Main file path: app.py
- Python version: pilih yang kompatibel (disarankan 3.10 atau 3.11)
- Requirements akan dibaca otomatis dari requirements.txt

---

## 7. Tunggu proses build

Streamlit Cloud akan:

- menginstall dependency dari requirements.txt
- membuat environment Python
- menjalankan aplikasi Anda

Proses ini bisa memakan waktu beberapa menit.

Jika deploy gagal, biasanya penyebabnya:

- requirements salah
- file app.py error
- model file tidak ada
- ukuran repository terlalu besar
- paket berat seperti TensorFlow memakan waktu lama

---

## 8. Masalah umum dan solusinya

### A. ModuleNotFoundError

Contoh:

```txt
ModuleNotFoundError: No module named 'streamlit'
```

Solusi:

- pastikan streamlit ada di requirements.txt
- pastikan dependency berhasil diinstall

### B. Error saat memuat model

Contoh:

```txt
Model tidak ditemukan
```

Solusi:

- cek apakah file model ada di folder models/
- pastikan path model benar
- pastikan file ikut terupload ke GitHub

### C. Build lama atau gagal karena TensorFlow

Solusi:

- gunakan tensorflow-cpu di requirements.txt
- hindari paket yang tidak perlu
- pastikan Python version yang dipilih kompatibel

### D. Aplikasi berhasil deploy tapi halaman kosong

Solusi:

- cek log error di Streamlit Cloud
- pastikan app.py tidak memanggil file yang tidak ada
- cek apakah halaman pages/ ada dan diimport dengan benar

---

## 9. Tips agar deploy lebih aman

- Simpan semua file penting di repository GitHub
- Pastikan requirements.txt sudah lengkap
- Gunakan file utama app.py
- Hindari path absolut yang mengarah ke komputer lokal
- Test dulu secara lokal sebelum deploy
- Jika ukuran repo besar, pertimbangkan Git LFS atau hosting file model eksternal

---

## 10. Checklist sebelum deploy

Sebelum klik Deploy, pastikan:

- [ ] app.py bisa jalan lokal
- [ ] requirements.txt lengkap
- [ ] models/ ada dan bisa dibaca
- [ ] semua folder penting sudah di GitHub
- [ ] repo sudah di-push ke GitHub
- [ ] branch utama adalah main

---

## 11. Alternatif jika ingin deploy cepat

Kalau Anda ingin deploy paling cepat, gunakan cara ini:

1. Pastikan repo sudah di GitHub
2. Buka Streamlit Cloud
3. Pilih repo dan file app.py
4. Tunggu build selesai

Biasanya ini sudah cukup untuk project sederhana seperti milik Anda.

---

## 12. Catatan khusus untuk project Anda

Project Anda sudah sangat cocok untuk Streamlit karena:

- ada app.py sebagai entry point
- ada halaman terpisah di folder pages/
- ada model .h5 yang dipakai aplikasi
- ada visualisasi metrics dan hasil prediksi

Yang perlu diperhatikan khusus:

- model file harus tersedia saat deploy
- TensorFlow harus kompatibel dengan environment cloud
- pastikan requirements.txt tidak ketinggalan paket seperti Pillow, opencv, scikit-learn, dan streamlit

---

## 13. Link yang biasanya dipakai

- Streamlit Cloud: https://streamlit.io/cloud
- Streamlit Documentation: https://docs.streamlit.io/
- GitHub: https://github.com/

---

Jika Anda ingin, langkah berikutnya bisa saya bantu lanjutkan menjadi versi yang lebih spesifik untuk project Anda, misalnya:

- checklist file yang harus diupload
- versi requirements yang paling cocok untuk Streamlit Cloud
- format README untuk repo GitHub
- cara agar deploy lebih cepat dan stabil
