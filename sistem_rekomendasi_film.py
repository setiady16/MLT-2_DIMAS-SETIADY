# sistem_rekomendasi_film.py
# Proyek Akhir: Sistem Rekomendasi Film Berbasis Content-based Filtering
# Nama: Dimas Aditia Anugerah Setiady
# Email: mc240d5y0910@student.devacademy.id
# ID Dicoding: MC240D5Y0910

# Import Library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.simplefilter('ignore')

# Data Understanding
# Load dataset
try:
    df = pd.read_csv(r'C:\Users\ADVAN\Documents\The Movies Dataset\movies_metadata.csv')
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("Error: File 'movies_metadata.csv' tidak ditemukan di path yang diberikan.")
    exit()
except Exception as e:
    print(f"Error saat memuat dataset: {e}")
    exit()

# Cek 5 baris pertama dataset
print("\n5 Baris Pertama Dataset:")
print(df.head())

# Melihat info dataset
print("\nInformasi Dataset:")
print(df.info())

# Statistik deskriptif untuk kolom numerik
print("\nStatistik Deskriptif Kolom Numerik:")
print(df[['vote_average', 'vote_count']].describe())

# Cek missing value pada kolom utama
print("\nJumlah Missing Value:")
print(df[['title', 'genres', 'overview', 'vote_average']].isnull().sum())

# Visualisasi Univariate
# Distribusi vote_average
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'].dropna(), bins=20, kde=True)
plt.title('Distribusi Rata-rata Rating Film')
plt.xlabel('Vote Average')
plt.ylabel('Frekuensi')
plt.show()

# Distribusi vote_count
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_count'].dropna(), bins=20, kde=True)
plt.title('Distribusi Jumlah Voting Film')
plt.xlabel('Vote Count')
plt.ylabel('Frekuensi')
plt.show()

# Data Preparation
# 1. Cek missing value
print("\nJumlah Missing Value (Semua Kolom):")
print(df.isnull().sum())

# 2. Tangani missing value
df['overview'] = df['overview'].fillna('')  # Ganti NaN dengan string kosong
df['genres'] = df['genres'].fillna('')      # Ganti NaN dengan string kosong
df['title'] = df['title'].fillna('')        # Ganti NaN dengan string kosong

# 3. Cek data duplikat
duplicates = df.duplicated().sum()
print(f"\nJumlah duplikat: {duplicates}")

# 4. Hapus data duplikat
df = df.drop_duplicates().reset_index(drop=True)

# Cek ulang jumlah duplikat
print(f"Jumlah duplikat setelah dibersihkan: {df.duplicated().sum()}")

# 5. Parsing kolom genres
def parse_genres(genres_str):
    try:
        genres = json.loads(genres_str.replace("'", "\""))
        return ' '.join([genre['name'] for genre in genres])
    except Exception as e:
        print(f"Error parsing genres: {genres_str}, Error: {e}")
        return ''

df['genres'] = df['genres'].apply(parse_genres)

# 6. Buat fitur gabungan (genres + overview)
df['combined_features'] = df['genres'] + ' ' + df['overview']

# 7. Filter film dengan vote_count > 50 untuk memastikan kualitas
df = df[df['vote_count'] > 50].reset_index(drop=True)  # Reset indeks setelah filtering

# 8. Cek hasil preprocessing
print("\nHasil Preprocessing (5 Baris Pertama):")
print(df[['title', 'combined_features']].head())

# Modeling
# Ekstraksi fitur dengan TF-IDF
try:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    print("Matriks TF-IDF berhasil dibuat!")
except Exception as e:
    print(f"Error saat membuat matriks TF-IDF: {e}")
    exit()

# Hitung Cosine Similarity
try:
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Matriks Cosine Similarity berhasil dibuat!")
except Exception as e:
    print(f"Error saat menghitung Cosine Similarity: {e}")
    exit()

# Fungsi aman untuk mendapatkan rekomendasi
def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_k=10):
    matches = df[df['title'].str.lower() == title.lower()]
    
    if matches.empty:
        return f"Judul '{title}' tidak ditemukan di dataset."

    idx = matches.index[0]
    
    try:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices].reset_index(drop=True)
    except Exception as e:
        return f"Error saat merekomendasikan film '{title}': {e}"

# Contoh penggunaan
print("\nContoh Rekomendasi untuk 'The Dark Knight':")
print(get_recommendations('The Dark Knight'))

# Evaluasi
print("\nEvaluasi Kualitatif:")
test_movies = ['The Dark Knight', 'Toy Story', 'Inception']
for movie in test_movies:
    print(f"\nRekomendasi untuk {movie}:")
    print(get_recommendations(movie))