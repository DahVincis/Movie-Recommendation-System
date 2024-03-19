import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import ast  # For parsing strings of lists and dictionaries

# Load the datasets
movies_metadata_path = '/Users/dre/Desktop/Movie_Recommendation/archive/movies_metadata.csv'
keywords_path = '/Users/dre/Desktop/Movie_Recommendation/archive/keywords.csv'
credits_path = '/Users/dre/Desktop/Movie_Recommendation/archive/credits.csv'

movies = pd.read_csv(movies_metadata_path, low_memory=False)
keywords = pd.read_csv(keywords_path)
credits = pd.read_csv(credits_path)

# Convert 'id' columns to integers to avoid merge conflicts
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
credits['id'] = pd.to_numeric(credits['id'], errors='coerce')

# Drop rows with NaN 'id' values after conversion, if any
movies.dropna(subset=['id'], inplace=True)
keywords.dropna(subset=['id'], inplace=True)
credits.dropna(subset=['id'], inplace=True)

# Ensure 'id' is int64 across all DataFrames to ensure consistent merge behavior
movies['id'] = movies['id'].astype('int64')
keywords['id'] = keywords['id'].astype('int64')
credits['id'] = credits['id'].astype('int64')

# Preprocess keywords and credits
keywords['keywords'] = keywords['keywords'].apply(ast.literal_eval).apply(lambda k: ' '.join([i['name'] for i in k]) if isinstance(k, list) else '')
credits['cast'] = credits['cast'].apply(ast.literal_eval)
credits['crew'] = credits['crew'].apply(ast.literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return None

def get_top_cast(x):
    return ' '.join([i['name'] for i in x[:3]]) if len(x) > 0 else ''

credits['director'] = credits['crew'].apply(get_director)
credits['top_cast'] = credits['cast'].apply(get_top_cast)

# Merge datasets
movies = movies.merge(keywords, on='id').merge(credits[['id', 'director', 'top_cast']], on='id')

# Convert 'genres' from stringified JSON to space-separated string of genre names
movies['genres'] = movies['genres'].apply(ast.literal_eval).apply(lambda x: ' '.join([i['name'] for i in x]) if isinstance(x, list) else '')

# Extract release year from 'release_date'
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year

# Combine features
movies['combined_features'] = movies[['title', 'genres', 'director', 'top_cast', 'keywords']].apply(
    lambda x: ' '.join(x.dropna().values).lower(), axis=1
)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

def recommend_movies_based_on_input(user_input, top_n=7):
    user_input_tfidf = tfidf_vectorizer.transform([user_input.lower()])
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1: top_n+1]]
    # Include 'release_year' and 'vote_average' in the output
    return movies.iloc[movie_indices][['title', 'genres', 'director', 'release_year', 'vote_average']]

# Example user input
user_input = input("Enter your search (e.g., genre, director, actor names, etc.): ")
recommendations = recommend_movies_based_on_input(user_input)
print(tabulate(recommendations, headers='keys', tablefmt='psql', showindex=False))
