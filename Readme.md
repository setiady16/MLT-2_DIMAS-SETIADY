# Laporan Proyek Sistem Rekomendasi Film

## Identitas
- Nama: Dimas Aditia Anugerah Setiady
- Email: mc240d5y0910@student.devacademy.id
- ID Dicoding: MC240D5Y0910

## Latar Belakang
Sistem rekomendasi film dibuat untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan konten film, seperti genre dan deskripsi. Pendekatan *Content-based Filtering* dipilih karena efektif untuk masalah *cold-start* dan memanfaatkan metadata film yang kaya dari dataset *The Movies Dataset*.

## Tujuan
- Mengembangkan sistem rekomendasi film berbasis *Content-based Filtering*.
- Menyediakan rekomendasi yang relevan berdasarkan kesamaan genre dan overview.
- Mengevaluasi hasil rekomendasi secara kualitatif.

## Dataset
Dataset *The Movies Dataset* dari Kaggle ([link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)) digunakan, dengan file utama `movies_metadata.csv` yang berisi informasi seperti judul, genre, overview, rating, dan jumlah voting.

## Metode
- **Pendekatan**: *Content-based Filtering* menggunakan TF-IDF Vectorizer untuk ekstraksi fitur teks dan Cosine Similarity untuk menghitung kemiripan antar film.
- **Data Understanding**: 
  - Memuat dataset dan menampilkan 5 baris pertama untuk memahami struktur data.
  - Melakukan analisis univariate dengan mengecek informasi dataset, statistik deskriptif kolom numerik (vote_average, vote_count), dan jumlah missing value pada kolom utama (title, genres, overview, vote_average).
  - Visualisasi distribusi vote_average dan vote_count menggunakan histogram dengan Kernel Density Estimation (KDE).
  - **Preprocessing**: 
  - Penanganan Missing Value: Mengganti nilai kosong pada kolom title, genres, dan overview dengan string kosong ('').
  - Penghapusan Duplikat: Mengidentifikasi dan menghapus data duplikat, lalu mereset indeks.
  - Parsing Kolom Genres: Mengubah format JSON pada kolom genres menjadi string yang berisi nama genre, dipisahkan oleh spasi.
  - Pembuatan Fitur Gabungan: Menggabungkan kolom genres dan overview menjadi combined_features.
  - Filtering: Memfilter film dengan vote_count > 50 untuk memastikan kualitas data.
- **Modeling**: 
  - Ekstraksi Fitur: Menggunakan TF-IDF Vectorizer (max_features=5000, stop_words='english') untuk mengubah combined_features menjadi representasi numerik.
  - Perhitungan Kemiripan: Menghitung Cosine Similarity antar film berdasarkan matriks TF-IDF.
  - Fungsi Rekomendasi: Membuat fungsi get_recommendations yang menerima judul film dan mengembalikan 10 film paling mirip berdasarkan skor kemiripan.
- **Evaluasi**: Evaluasi Kualitatif: Memeriksa relevansi rekomendasi untuk tiga film: The Dark Knight, Toy Story, dan Inception. Hasil rekomendasi dibandingkan dengan ekspektasi berdasarkan genre dan tema film.

## Hasil
- Sistem berhasil merekomendasikan film berdasarkan kesamaan konten. Contoh:
  - Untuk The Dark Knight, rekomendasi termasuk The Dark Knight Rises, Batman Returns, dan Batman: Under the Red Hood, yang relevan karena memiliki genre aksi, kriminal, dan tema superhero.
  - Untuk Toy Story, rekomendasi termasuk Toy Story 3 dan Toy Story 2, yang sangat relevan karena merupakan sekuel dengan genre animasi dan tema serupa.
  - Untuk Inception, rekomendasi termasuk Cypher dan Minority Report, yang memiliki tema sci-fi dan thriller, meskipun beberapa rekomendasi kurang spesifik.
  
- Keterbatasan: 
  - Hanya menggunakan `genres` dan `overview`, sehingga fitur seperti aktor atau sutradara belum dimanfaatkan.
  - Evaluasi bersifat kualitatif, tanpa metrik kuantitatif seperti Precision@K atau Recall@K karena keterbatasan data preferensi pengguna.

## Kesimpulan
Sistem *Content-based Filtering* efektif untuk merekomendasikan film berdasarkan metadata, tetapi dapat ditingkatkan dengan fitur tambahan dan evaluasi kuantitatif untuk meningkatkan relevansi dan personalisasi rekomendasi.

## File Submission
- `Sistem_Rekomendasi_Dimas_Setiady.ipynb`: Notebook berisi kode dan dokumentasi.
- `sistem_rekomendasi_film.py`: File Python berisi implementasi sistem rekomendasi.
- `README.md`: Laporan proyek ini.