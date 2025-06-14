# **Laporan Proyek Akhir: Sistem Rekomendasi Film Berbasis Content-Based Filtering - Dimas Aditia Anugerah Setiady**

## Project Overview

### Latar Belakang

Di era digital, platform streaming seperti Netflix, Disney+, dan Amazon Prime Video menyediakan ribuan film, yang sering kali membuat pengguna kewalahan dalam memilih konten yang sesuai dengan preferensi mereka. Fenomena ini dikenal sebagai information overload, di mana banyaknya pilihan justru menghambat pengambilan keputusan (Adomavicius & Tuzhilin, 2005). Sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan film yang relevan berdasarkan preferensi mereka, meningkatkan kepuasan pengguna, dan memperpanjang waktu interaksi dengan platform (Ricci et al., 2011). Pendekatan Content-Based Filtering memanfaatkan fitur konten, seperti genre, sinopsis (overview), dan tagline, untuk menghitung kemiripan antar film, menjadikannya metode yang efektif untuk rekomendasi personal (Lops et al., 2011). Proyek ini menggunakan The Movies Dataset dari Kaggle untuk membangun sistem rekomendasi berbasis Content-Based Filtering yang merekomendasikan film berdasarkan kemiripan teks.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Tanpa sistem rekomendasi yang efektif, pengguna mungkin menghabiskan waktu lama untuk mencari film atau bahkan meninggalkan platform karena frustrasi. Studi menunjukkan bahwa rekomendasi yang relevan dapat meningkatkan user engagement hingga 30% dan mengurangi churn rate (Gomez-Uribe & Hunt, 2015). Selain itu, sistem rekomendasi membantu platform streaming mengoptimalkan inventaris konten mereka dengan mempromosikan film yang mungkin terlewatkan oleh pengguna. Dalam konteks ini, Content-Based Filtering dipilih karena kemampuannya untuk merekomendasikan item baru tanpa memerlukan data interaksi pengguna yang besar, yang sering kali menjadi kendala pada pendekatan Collaborative Filtering (Lops et al., 2011).

### Bagaimana Masalah Ini Diselesaikan

Proyek ini mengembangkan sistem rekomendasi menggunakan TF-IDF Vectorizer untuk mengubah fitur teks (genres, overview, tagline) menjadi representasi numerik, diikuti oleh Cosine Similarity untuk mengukur kemiripan antar film. Sistem dievaluasi dengan metrik Precision, Recall, dan F1-Score, menargetkan Precision rata-rata di atas 85%.

### Referensi

- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734–749. https://doi.org/10.1109/TKDE.2005.99

- Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems, 6(4), 1–19. https://doi.org/10.1145/2843948

- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In F. Ricci, L. Rokach, B. Shapira, & P. B. Kantor (Eds.), Recommender systems handbook (pp. 73–105). Springer. https://doi.org/10.1007/978-0-387-85820-3_3

- Ricci, F., Rokach, L., Shapira, B., & Kantor, P. B. (Eds.). (2011). Recommender systems handbook. Springer. https://doi.org/10.1007/978-0-387-85820-3


---

## Business Understanding

### Problem Statement

Berikut adalah pernyataan masalah yang diidentifikasi dalam konteks proyek ini:

- Pernyataan Masalah 1: Pengguna platform streaming menghabiskan waktu lama untuk memilih film karena jumlah pilihan yang sangat banyak, menyebabkan information overload dan menurunkan kepuasan pengguna.
- Pernyataan Masalah 2: Banyak film berkualitas di platform streaming tidak ditemukan oleh pengguna karena kurangnya sistem rekomendasi yang dapat menyesuaikan preferensi berdasarkan konten film seperti genre dan sinopsis.
- Pernyataan Masalah 3: Platform streaming memerlukan sistem rekomendasi yang mampu menghasilkan rekomendasi relevan tanpa ketergantungan pada data interaksi pengguna yang besar, yang sering kali tidak tersedia untuk pengguna baru atau film baru.

### Goals

Tujuan dari proyek ini adalah:

- Mengembangkan sistem rekomendasi yang menghasilkan top-10 rekomendasi film relevan berdasarkan kemiripan genres, overview, dan tagline.
- Mencapai Precision rata-rata di atas 85% untuk memastikan rekomendasi memiliki genre yang sesuai dengan film input.
- Mengoptimalkan model dengan fitur tambahan (tagline) dan preprocessing data yang tepat.

### Solution Statement

- Menggunakan Content-Based Filtering dengan TF-IDF Vectorizer untuk mengubah teks (genres, overview, tagline) menjadi representasi numerik.
- Menghitung kemiripan antar film dengan Cosine Similarity untuk menghasilkan top-10 rekomendasi.
- Mengevaluasi model dengan Precision, Recall, dan F1-Score, serta analisis kualitatif.
- Menerapkan preprocessing seperti penanganan missing values, penghapusan duplikat, dan filtering vote_count untuk kualitas data.


---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah The Movies Dataset tersedia di https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset  Dataset ini berisi metadata 45.466 film dengan 24 kolom. Berikut deskripsi fitur:

1. adult: Film untuk audiens dewasa (True/False).
2. belongs_to_collection: Koleksi/seri film.
3. budget: Anggaran produksi (dolar).
4. genres: Genre film (JSON).
5. homepage: URL situs resmi.
6. id: ID unik film.
7. imdb_id: ID IMDb.
8. original_language: Bahasa asli.
9. original_title: Judul asli.
10. overview: Sinopsis film.
11. popularity: Skor popularitas TMDB.
12. poster_path: Path poster.
13. production_companies: Perusahaan produksi (JSON).
14. production_countries: Negara produksi (JSON).
15. release_date: Tanggal rilis.
16. revenue: Pendapatan (dolar).
17. runtime: Durasi (menit).
18. spoken_languages: Bahasa film (JSON).
19. status: Status (misalnya, Released).
20. tagline: Slogan promosi.
21. title: Judul Inggris.
22. video: Ada video terkait (True/False).
23. vote_average: Rata-rata rating (0-10).
24. vote_count: Jumlah rating.


### Eksplorasi Data

- Jumlah Data: 45.466 baris, 24 kolom.
- Missing Values: overview (954 kosong), tagline (25.054 kosong), title (6 kosong).
- Duplikat: 13 data duplikat.
- Statistik Deskriptif:
  - vote_average: Rata-rata 5.62, min 0, max 10.
  - vote_count: Rata-rata 109.9, max 14.075, distribusi skewed.
  - Numerik vote_average vote_count
- Visualisasi: vote_average terkonsentrasi pada 5-7, vote_count didominasi nilai rendah.

*Insight :*
- Fitur genres, overview, dan tagline kunci untuk rekomendasi.
- Missing values pada tagline perlu penanganan.
- Filtering vote_count > 50 untuk fokus pada film populer.


---

## Data Preparation

### Teknik Preprocessing

1. Penanganan Missing Values:
  - overview, genres, title, tagline diisi dengan ''.

2. Penghapusan Duplikat:
  - Menghapus 13 duplikat dengan drop_duplicates(), indeks diatur ulang.

3. Parsing Kolom Genres:
  - JSON genres diubah menjadi teks (misalnya, "Animation Comedy Family").

4. Pembuatan Fitur Gabungan:
  - combined_features = genres + overview + tagline.

5. Filtering Data:
  - Film dengan vote_count ≤ 50 difilter, tersisa 10.041 baris.

6. Ekstraksi Fitur dengan TF-IDF:
  - TfidfVectorizer (stop_words='english', max_features=5000) menghasilkan matriks TF-IDF.

7. Penambahan Fitur Tagline (Iteratif):
  - Setelah evaluasi awal tanpa tagline (menghasilkan Precision Toy Story 0.60), tagline ditambahkan ke combined_features untuk meningkatkan relevansi.
  - tagline diisi dengan string kosong untuk missing values (df['tagline'] = df['tagline'].fillna('')).

*Insight :*
- Preprocessing menghasilkan dataset bersih.
- tagline meningkatkan kualitas fitur teks.
- TF-IDF siap untuk Cosine Similarity.


---

## Modeling

### Pendekatan Modeling

- Ekstraksi Fitur: Matriks TF-IDF dari combined_features.
- Perhitungan Kemiripan: Cosine Similarity antar film.
- Fungsi Rekomendasi: Top-10 film berdasarkan skor kemiripan.

### Hasil Rekomendasi (Top-N)

- The Dark Knight:
  1. The Dark Knight Rises
  2. Batman Returns
  3. The Lego Batman Movie
  4. Batman Forever
  5. Batman: The Dark Knight Returns, Part 2
  6. Batman vs Dracula
  7. Batman: Under the Red Hood
  8. Batman: Year One
  9. Batman: Mask of the Phantasm
  10. Batman: The Dark Knight Returns, Part 1

- Toy Story:
  1. Toy Story 3
  2. Toy Story 2
  3. Small Fry
  4. The 40 Year Old Virgin
  5. Man on the Moon
  6. Factory Girl
  7. Rebel Without a Cause
  8. Class of 1984
  9. Manhattan
  10. For Love or Money

- Inception:
  1. Transformers: Revenge of the Fallen
  2. Cypher
  3. Repeaters
  4. Renaissance
  5. Seconds
  6. Pitch Perfect 2
  7. Thick as Thieves
  8. Mission: Impossible - Rogue Nation
  9. Stolen
  10. Adventures of Arsene Lupin


*Insight :*
- The Dark Knight: Rekomendasi relevan (tema Batman, Action/Crime).
- Toy Story: Tiga rekomendasi relevan (Toy Story 3, Toy Story 2, Small Fry), beberapa kurang sesuai (Comedy umum).
- Inception: Mayoritas relevan (Sci-Fi/Thriller), kecuali Pitch Perfect 2.
- tagline meningkatkan relevansi Toy Story.


---


## Evaluation

### Metrik Evaluasi

- Precision: Proporsi rekomendasi relevan (genre sama) dari top-10.
- Recall: Proporsi rekomendasi relevan dari total film relevan.
- F1-Score: Harmonic mean Precision dan Recall.

### Hasil Evaluasi

- The Dark Knight: Precision 1.00, Recall 0.0015, F1-Score 0.00.
- Toy Story: Precision 0.70, Recall 0.0018, F1-Score 0.00.
- Inception: Precision 0.90, Recall 0.0020, F1-Score 0.00.
- Rata-rata: Precision 0.87, Recall 0.0018, F1-Score 0.00.

*Insight :*
- Precision 0.87 memenuhi target >85%.
- Peningkatan Precision Toy Story (0.60 ke 0.70) karena tagline.
- Recall rendah normal untuk top-10.
- Kualitatif: The Dark Knight sangat relevan, Toy Story dan Inception ada rekomendasi kurang sesuai.


### Kesimpulan

- Model mencapai Precision 0.87, melewati target 85%.
- tagline meningkatkan relevansi.
- Preprocessing memastikan dataset berkualitas.
- Perbaikan seperti keywords dapat meningkatkan performa.



