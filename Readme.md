# **Laporan Proyek Membuat Model Sistem Rekomendasi - Dimas Aditia Anugerah Setiady**

Sistem rekomendasi film bertujuan membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan konten seperti genre dan deskripsi overview Menurut laporan dari Netflix jumlah konten di platform streaming telah meningkat hingga lebih dari 5800 film pada tahun 2023 menyebabkan pengguna sering kesulitan memilih film yang relevan [Netflix 2023] Pendekatan Content-based Filtering memungkinkan rekomendasi yang akurat tanpa memerlukan data preferensi pengguna sebelumnya sehingga efektif untuk masalah cold-start.

Sistem ini memanfaatkan metadata film seperti genre dan overview dari dataset The Movies Dataset yang berisi 45466 film dengan informasi seperti judul genre dan deskripsi Dengan teknologi machine learning sistem rekomendasi dapat mengurangi waktu yang dihabiskan pengguna untuk mencari film dan meningkatkan kepuasan menonton.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Masalah ini penting untuk diselesaikan karena :

- Pengguna menghadapi kelebihan pilihan choice overload yang dapat mengurangi kepuasan menonton.
- Platform streaming memerlukan sistem rekomendasi yang efektif untuk meningkatkan retensi pengguna dan durasi penggunaan aplikasi.
- Rekomendasi berbasis konten memungkinkan personalisasi tanpa data interaksi pengguna cocok untuk pengguna baru

Sistem rekomendasi dibangun menggunakan pendekatan Content-based Filtering dengan TF-IDF Vectorizer untuk mengubah genre dan overview menjadi vektor numerik dan Cosine Similarity untuk menghitung kemiripan antar film Pendekatan ini memungkinkan platform streaming atau aplikasi hiburan memberikan rekomendasi yang relevan secara otomatis meningkatkan pengalaman pengguna dan efisiensi pencarian.

### Referensi

- Netflix (2023) Netflix Research Content Statistics Diakses dari https://research.netflix.com
- Ricci M et al (2015) Recommender Systems Handbook 2nd ed Boston MA Springer https://doi.org/10.1007/978-1-4899-7637-6
- Adomavicius G and Tuzhilin A (2005) Toward the next generation of recommender systems A survey of the state-of-the-art and possible extensions IEEE Transactions on Knowledge and Data Engineering 17(6) 734-749 https://doi.org/10.1109/TKDE.2005.99


---

## Business Understanding

### Problem Statement

Platform streaming memiliki ribuan film yang menyebabkan pengguna kesulitan memilih konten yang sesuai dengan preferensi mereka Banyak pengguna menghabiskan waktu lama untuk mencari film yang relevan namun sering kali merasa tidak puas dengan pilihan mereka Masalah utamanya adalah :

- Bagaimana cara mengidentifikasi film yang sesuai dengan preferensi pengguna berdasarkan konten seperti genre dan overview?
- Dapatkah kita membangun sistem rekomendasi yang mampu memberikan saran film secara akurat dan efisien tanpa data preferensi pengguna sebelumnya cold-start problem?


### Goals

Tujuan dari proyek ini adalah:

- Mengembangkan sistem rekomendasi berbasis Content-based Filtering untuk merekomendasikan film berdasarkan kemiripan genre dan overview.
- Memberikan solusi rekomendasi yang dapat diintegrasikan dalam platform streaming atau aplikasi hiburan untuk meningkatkan pengalaman pengguna.
- Membantu pengguna menemukan film relevan secara otomatis dengan evaluasi kualitatif berdasarkan kesamaan genre dan tema naratif.


---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah The Movies Dataset tersedia di https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset Dataset ini dirancang untuk membangun sistem rekomendasi film berdasarkan metadata seperti genre dan overview.


### Ringkasan Dataset

- Jumlah Baris 45466
- Jumlah Kolom 24
- Variabel Utama title genres overview
- Tipe Data
  - String title overview
  - JSON genres
  - Numerik vote_average vote_count


### Deskripsi Fitur

Dataset ini memiliki 12 kolom Dataset ini memiliki kolom utama berikut
1. title Judul film dalam format string.
2. genres Daftar genre film dalam format JSON misalnya [{'id' 16 'name' 'Animation'} {'id' 35 'name' 'Comedy'}].
3. overview Deskripsi singkat film dalam format string.
4. vote_average Rata-rata rating film skala 0-10.
5. vote_count Jumlah voting yang diterima film.

### Kondisi Data Awal

- Nilai yang Hilang Kolom overview memiliki 954 nilai kosong sekitar 21 persen title dan vote_average masing-masing 6 nilai kosong genres tidak memiliki nilai kosong tetapi memerlukan parsing JSON
- Terdapat 13 Duplikasi Data namun telah berhasil ditangani.
- Distribusi Data Distribusi vote_average menunjukkan sebagian besar film memiliki rating 5-7 vote_count sangat miring ke kanan dengan median 10 menunjukkan beberapa film populer memiliki voting tinggi.


### Sumber Dataset

Dataset ini tersedia secara publik di Kaggle The Movies Dataset. Dataset ini cocok untuk mengembangkan model sistem rekomendasi berbasis konten khususnya untuk rekomendasi film berdasarkan metadata.


---

## Data Preparation

Untuk mempersiapkan dataset agar dapat digunakan dalam pembangunan sistem rekomendasi beberapa langkah preprocessing dilakukan sesuai urutan eksekusi berikut :

1. Penanganan Nilai yang Hilang Kolom title genres dan overview memiliki nilai kosong yang diisi dengan string kosong untuk memastikan data lengkap untuk ekstraksi fitur.

2. Pemeriksaan Duplikasi Dataset diperiksa untuk memastikan tidak ada baris duplikat dan hasilnya menunjukkan tidak ada duplikasi.

3. Parsing Kolom Genres Kolom genres dalam format JSON diubah menjadi string misalnya Animation Comedy Family menggunakan fungsi parse_genres untuk memungkinkan penggunaan sebagai fitur teks.

4. Pembuatan Fitur Gabungan Kolom genres dan overview digabung menjadi combined_features untuk ekstraksi fitur TF-IDF.

5. Filtering Data Memfilter film dengan vote_count di atas 50 untuk memastikan hanya film populer yang digunakan mengurangi kebisingan dan mereset indeks untuk konsistensi dengan matriks Cosine Similarity.


---

## Modeling

Untuk membangun sistem rekomendasi dua pendekatan Content-based Filtering diuji Content-based Filtering dengan TF-IDF dan Cosine Similarity sebagai model utama dan Content-based Filtering dengan Word Embeddings sebagai alternatif Berikut adalah penjelasan cara kerja dan parameter utama untuk setiap pendekatan:

1. Content-based Filtering dengan TF-IDF dan Cosine Similarity 

- Cara Kerja Mengubah fitur combined_features genres dan overview menjadi vektor numerik menggunakan TF-IDF Vectorizer kemudian menghitung kemiripan antar film dengan Cosine Similarity Fungsi get_recommendations mengembalikan 10 film paling mirip berdasarkan skor kemiripan
- Parameter Utama :
  - TF-IDF Vectorizer max_features=5000 stop_words=english
  - Cosine Similarity metric default dari scikit-learn

2. Content-based Filtering dengan Word Embeddings

- Cara Kerja Menggunakan model Word2Vec untuk menangkap hubungan semantik dalam genres dan overview menghasilkan vektor teks yang lebih kaya makna Kemiripan dihitung menggunakan Cosine Similarity pada vektor embeddings.
- Parameter Utama
  - Word2Vec window=5 min_count=1 workers=4
  - Cosine Similarity metric defaul
- Catatan Pendekatan ini tidak diimplementasikan karena keterbatasan sumber daya komputasi tetapi diusulkan sebagai alternatif.

*Insight :* - Model utama TF-IDF dan Cosine Similarity digunakan untuk menghasilkan rekomendasi karena sederhana dan efektif untuk dataset teksAccuracy, Precision, Recall, F1-Score, dan ROC-AUC.


---

## Evaluation

Bagian ini menyajikan hasil evaluasi performa sistem rekomendasi berdasarkan evaluasi kualitatif dengan memeriksa relevansi rekomendasi untuk tiga film The Dark Knight Toy Story dan Inception Metrik ini dipilih karena keterbatasan data preferensi pengguna sehingga metrik kuantitatif seperti Precision atau Recall tidak dapat diterapkan.

### Hasil Evaluasi

| Film             | Rekomendasi Contoh                     | Genre Asli                     | Relevansi                              |
|------------------|----------------------------------------|--------------------------------|----------------------------------------|
| Toy Story        | Toy Story 3 Toy Story 2 Small Fry      | Animation Comedy Family        | Tinggi sekuel dan spin-off relevan     |
| The Dark Knight  | The Dark Knight Rises Batman Begins    | Action Crime Drama             | Tinggi tema superhero dan genre serupa |
| Inception        | Cypher Minority Report                 | Action Adventure Sci-Fi        | Sedang tema sci-fi relevan tetapi kurang spesifik |


---

### Analisis dan Interpretasi

1. Toy Story :
- Rekomendasi seperti Toy Story 3 dan Toy Story 2 sangat relevan karena merupakan sekuel dengan genre Animation Comedy Family dan tema serupa cerita tentang mainan.
- Rekomendasi seperti Small Fry relevan karena merupakan spin-off dari Toy Story.
- Beberapa rekomendasi seperti Man on the Moon genre Biography Comedy Drama kurang relevan karena perbedaan tema naratif.

2. The Dark Knight :
- Rekomendasi seperti The Dark Knight Rises dan Batman Begins relevan karena memiliki genre Action Crime Drama dan tema superhero
- Rekomendasi seperti Batman Returns juga relevan karena berfokus pada karakter Batman
- Relevansi tinggi menunjukkan sistem efektif untuk film dengan genre dan tema yang jelas

3. Inception
- Rekomendasi seperti Cypher dan Minority Report relevan karena memiliki tema sci-fi dan thriller.
- Beberapa rekomendasi kurang spesifik karena overview tidak selalu menangkap tema kompleks seperti mimpi dalam Inception.
- Relevansi sedang menunjukkan keterbatasan sistem dalam menangkap narasi kompleks.


### Pemilihan Model Terbaik

🟢 Model Terbaik Content-based Filtering dengan TF-IDF dan Cosine Similarity

Alasan pemilihan:
- Menghasilkan rekomendasi yang relevan untuk film dengan genre dan tema yang jelas seperti Toy Story dan The Dark Knight.
- Sederhana dan efisien untuk dataset teks dengan sumber daya komputasi terbatas.
- Dapat diintegrasikan dalam platform streaming dengan mudah.


## Catatan Evaluasi

Keterbatasan data preferensi pengguna menyebabkan evaluasi hanya dilakukan secara kualitati Sistem efektif untuk film dengan konten yang jelas tetapi kurang akurat untuk narasi kompleks seperti Inception Penambahan fitur seperti aktor atau sutradara dapat meningkatkan relevansi.


---

### Solution Statement

Untuk mencapai tujuan di atas, beberapa solusi diuji dan dibandingkan:
- Model utama Content-based Filtering dengan TF-IDF dan Cosine Similarity sebagai pendekatan sederhana dan efektif.
- Model alternatif Content-based Filtering dengan Word Embeddings untuk menangkap hubungan semantik dalam teks


---


# 📊 Laporan Evaluasi Sistem Rekomendasi Film

Ringkasan Hasil:

- Dua pendekatan Content-based Filtering diuji TF-IDF dengan Cosine Similarity dipilih sebagai model utama karena menghasilkan rekomendasi relevan untuk film seperti Toy Story dan The Dark Knight dengan genre dan tema yang jelas Pendekatan ini sederhana dan efisien tetapi kurang akurat untuk film dengan narasi kompleks seperti Inception Keterbatasan data preferensi pengguna menyebabkan evaluasi hanya dilakukan secara kualitatif