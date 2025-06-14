# Laporan Proyek Sistem Rekomendasi Film

## Identitas
- Nama: [Dimas Aditia Anugerah Setiady]
- Email: [mc240d5y0910@student.devacademy.id]
- ID Dicoding: [MC240D5Y0910]

## Latar Belakang
Sistem rekomendasi film dibuat untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan konten film, seperti genre dan deskripsi. Pendekatan *Content-based Filtering* dipilih karena efektif untuk masalah *cold-start* dan memanfaatkan metadata film yang kaya.

## Tujuan
- Mengembangkan sistem rekomendasi film berbasis *Content-based Filtering*.
- Menyediakan rekomendasi yang relevan berdasarkan kesamaan genre dan overview.
- Mengevaluasi hasil rekomendasi secara kualitatif.

## Dataset
Dataset *The Movies Dataset* dari Kaggle ([link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)) digunakan, dengan file utama `movies_metadata.csv` yang berisi informasi seperti judul, genre, overview, dan rating.

## Metode
- **Pendekatan**: *Content-based Filtering* menggunakan TF-IDF Vectorizer dan Cosine Similarity.
- **Preprocessing**: Menangani missing value, parsing JSON pada kolom genres, dan membuat fitur gabungan (genres + overview).
- **Modeling**: Ekstraksi fitur teks dengan TF-IDF, menghitung kesamaan dengan Cosine Similarity, dan membuat fungsi rekomendasi.
- **Evaluasi**: Kualitatif, dengan memeriksa relevansi rekomendasi untuk film seperti *The Dark Knight*, *Toy Story*, dan *Inception*.

## Hasil
- Sistem berhasil merekomendasikan film berdasarkan kesamaan konten.
- Contoh: Untuk *The Dark Knight*, rekomendasi cenderung berupa film aksi atau thriller dengan tema serupa.
- Keterbatasan: Evaluasi hanya kualitatif; metrik kuantitatif seperti *Precision@K* dapat ditambahkan dengan data rating.

## Kesimpulan
- Sistem *Content-based Filtering* efektif untuk merekomendasikan film berdasarkan metadata.
