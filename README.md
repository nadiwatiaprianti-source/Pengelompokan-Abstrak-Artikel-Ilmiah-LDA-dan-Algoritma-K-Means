# Sistem "Cek Pengelompokan Abstrak" (desain pembimbing — Paper 3)

Pengguna menempel SATU abstrak -> sistem menentukan klaster terdekat dan
menampilkan artikel-artikel sejenis (Judul, Abstrak, Keyword).
Mesin sama dengan tesis: TF-IDF + LDA (Gensim) + Early Fusion + SVD/L2 + K-Means.

## Berkas
- app.py            : antarmuka (Langkah 1 input, Langkah 2 hasil)
- engine.py         : melatih korpus + menempatkan abstrak baru + cari termirip
- pipeline_core.py  : konfigurasi & preprocessing (dipakai engine)
- contoh_korpus.csv : korpus contoh (kolom Judul, ABSTRAK, Keyword)

## Menjalankan
    pip install -r requirements.txt
    streamlit run app.py

## Cara pakai
1. Unggah CSV korpus (idealnya 1.800 abstrak Anda; kolom Judul/Keyword opsional),
   atau klik "Gunakan data contoh".
2. Pastikan pemetaan kolom benar (Abstrak wajib).
3. Tempel satu abstrak di kotak kiri -> klik "Cek Pengelompokan Abstrak".
4. Lihat klaster + tema dan daftar artikel sejenis di kanan.

## Catatan
- Korpus dilatih sekali, lalu abstrak baru DITEMPATKAN ke klaster terdekat
  (tidak melatih ulang) -> cepat dan konsisten.
- Untuk hasil sesuai tesis, gunakan dataset 1.800 abstrak Bahasa Inggris.
