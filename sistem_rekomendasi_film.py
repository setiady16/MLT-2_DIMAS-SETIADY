# sistem_rekomendasi_film.py
# Proyek Akhir: Sistem Rekomendasi Film Berbasis Content-Based Filtering
# Nama: Dimas Aditia Anugerah Setiady
# Email: mc240d5y0910@student.devacademy.id
# ID Dicoding: MC240D5Y0910
#
# Project Overview:
# - Latar Belakang: Platform streaming menghadapi information overload, menyulitkan pengguna memilih film
#   (Adomavicius & Tuzhilin, 2005). Sistem rekomendasi berbasis Content-Based Filtering membantu
#   merekomendasikan film relevan berdasarkan genres, overview, dan tagline (Lops et al., 2011).
# - Tujuan: Membangun model dengan Precision >85%, menggunakan TF-IDF dan Cosine Similarity.
# - Manfaat: Meningkatkan pengalaman pengguna dan engagement platform streaming.
#
# Business Understanding:
# - Problem Statements:
#   1. Information overload menyebabkan pengguna menghabiskan waktu lama memilih film.
#   2. Film berkualitas terlewatkan karena kurangnya rekomendasi berbasis konten.
#   3. Perlu sistem rekomendasi tanpa ketergantungan data interaksi pengguna besar.
# - Goals: Menghasilkan top-10 rekomendasi relevan dengan Precision >85%.
# - Solution: Content-Based Filtering dengan TF-IDF dan Cosine Similarity, evaluasi dengan
#   Precision, Recall, F1-Score.

# Import Library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import argparse
import os
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.simplefilter('ignore')

# Parsing argumen untuk path dataset
parser = argparse.ArgumentParser(description='Sistem Rekomendasi Film Berbasis Content-Based Filtering')
parser.add_argument('--data_path', type=str, default='movies_metadata.csv',
                    help='Path ke file movies_metadata.csv')
args = parser.parse_args()

# Data Understanding
def load_data(data_path):
    """Memuat dataset dan menangani error."""
    try:
        df = pd.read_csv(data_path)
        print("Dataset berhasil dimuat!")
        return df
    except FileNotFoundError:
        print(f"Error: File '{data_path}' tidak ditemukan.")
        exit()
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        exit()

df = load_data(args.data_path)

# Cek info dataset
print("\nInformasi Dataset:")
print(df.info())

# Statistik deskriptif
print("\nStatistik Deskriptif Kolom Numerik:")
print(df[['vote_average', 'vote_count']].describe())

# Cek missing value
print("\nJumlah Missing Value (Kolom Utama):")
print(df[['title', 'genres', 'overview', 'tagline', 'vote_average']].isnull().sum())

# Visualisasi distribusi
def plot_distributions(df):
    """Membuat dan menyimpan plot distribusi vote_average dan vote_count."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vote_average'].dropna(), bins=20, kde=True)
    plt.title('Distribusi Rata-rata Rating Film')
    plt.xlabel('Vote Average')
    plt.ylabel('Frekuensi')
    plt.savefig('vote_average_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['vote_count'].dropna(), bins=20, kde=True)
    plt.title('Distribusi Jumlah Voting Film')
    plt.xlabel('Vote Count')
    plt.ylabel('Frekuensi')
    plt.savefig('vote_count_distribution.png')
    plt.close()

plot_distributions(df)

# Data Preparation
print("\nJumlah Missing Value (Semua Kolom):")
print(df.isnull().sum())

# Tangani missing value
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['title'] = df['title'].fillna('')
df['tagline'] = df['tagline'].fillna('')  # Tambahkan tagline

# Hapus duplikat
duplicates = df.duplicated().sum()
print(f"\nJumlah duplikat: {duplicates}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Jumlah duplikat setelah dibersihkan: {df.duplicated().sum()}")

# Parsing genres
def parse_genres(genres_str):
    """Mengubah format JSON genres menjadi teks."""
    try:
        genres = json.loads(genres_str.replace("'", "\""))
        return ' '.join([genre['name'] for genre in genres])
    except:
        return ''

df['genres'] = df['genres'].apply(parse_genres)

# Buat fitur gabungan (genres + overview + tagline)
df['combined_features'] = df['genres'] + ' ' + df['overview'] + ' ' + df['tagline']

# Filter vote_count > 50
df = df[df['vote_count'] > 50].reset_index(drop=True)
print(f"\nJumlah film setelah filtering vote_count > 50: {len(df)}")

# Cek hasil preprocessing
print("\nHasil Preprocessing (5 Baris Pertama):")
print(df[['title', 'combined_features']].head())

# Modeling
try:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    print("Matriks TF-IDF berhasil dibuat!")
except Exception as e:
    print(f"Error saat membuat matriks TF-IDF: {e}")
    exit()

try:
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Matriks Cosine Similarity berhasil dibuat!")
except Exception as e:
    print(f"Error saat menghitung Cosine Similarity: {e}")
    exit()

def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_k=10):
    """Mengembalikan top-k rekomendasi untuk judul film tertentu."""
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        return f"Judul '{title}' tidak ditemukan di dataset."
    idx = matches.index[0]
    try:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]
        movie_indices = [i[0] for i in sim_scores]
        return df[['title', 'genres']].iloc[movie_indices].reset_index(drop=True)
    except Exception as e:
        return f"Error saat merekomendasikan film '{title}': {e}"

# Evaluasi
def has_same_genre(genres1, genres2):
    """Memeriksa apakah dua film memiliki setidaknya satu genre yang sama."""
    set1 = set(genres1.split())
    set2 = set(genres2.split())
    return len(set1.intersection(set2)) > 0

def calculate_metrics(title, recommendations, df):
    """Menghitung Precision, Recall, dan F1-Score untuk rekomendasi."""
    if isinstance(recommendations, str):  # Jika error/tidak ditemukan
        return 0.0, 0.0, 0.0
    
    input_genres = df[df['title'].str.lower() == title.lower()]['genres'].iloc[0]
    relevant_items = df[df['genres'].apply(lambda x: has_same_genre(x, input_genres))]
    
    y_true = [has_same_genre(rec['genres'], input_genres) for _, rec in recommendations.iterrows()]
    precision = precision_score(y_true, [1]*len(y_true), zero_division=0)
    recall = len(recommendations[y_true]) / len(relevant_items) if len(relevant_items) > 0 else 0.0
    f1 = f1_score(y_true, [1]*len(y_true), zero_division=0)
    
    return precision, recall, f1

# Uji rekomendasi dan evaluasi
test_movies = ['The Dark Knight', 'Toy Story', 'Inception']
results = []

print("\nEvaluasi Sistem Rekomendasi:")
for movie in test_movies:
    print(f"\nRekomendasi untuk {movie}:")
    recommendations = get_recommendations(movie)
    print(recommendations)
    
    if not isinstance(recommendations, str):
        precision, recall, f1 = calculate_metrics(movie, recommendations, df)
        results.append({'Film': movie, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
        print(f"Metrik Evaluasi untuk {movie}:")
        print(f"Precision: {precision:.2f}, Recall: {recall:.4f}, F1-Score: {f1:.2f}")

# Ringkasan hasil evaluasi
if results:
    results_df = pd.DataFrame(results)
    print("\nRingkasan Metrik Evaluasi:")
    print(results_df)
    print("\nRata-rata Metrik:")
    print(results_df.mean(numeric_only=True))

# Simpan hasil evaluasi sebagai CSV
results_df.to_csv('evaluation_results.csv', index=False)
print("\nHasil evaluasi disimpan ke 'evaluation_results.csv'.")