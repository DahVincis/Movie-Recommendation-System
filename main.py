from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tabulate import tabulate

# Load the dataset
filePath = 'movies.csv'
data = pd.read_csv(filePath)

# Combine the features of interest into a single string for the recommendation engine
data['combinedFeatures'] = data[['name', 'genre', 'directors_name', 'stars_name', 'description']].apply(lambda x: ' '.join(x.dropna().values), axis=1).str.lower()

# Initialize a TF-IDF Vectorizer
tfidfVectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features
tfidfMatrix = tfidfVectorizer.fit_transform(data['combinedFeatures'])

def recommendMoviesBasedOnInput(userInput, topN=7):
    # Vectorize the user input using the same TF-IDF vectorizer
    userInputTfidf = tfidfVectorizer.transform([userInput.lower()])
    
    # Calculate cosine similarity between the user input and all movies
    cosineSim = cosine_similarity(userInputTfidf, tfidfMatrix)
    
    # Get similarity scores for all movies and sort them
    simScores = list(enumerate(cosineSim[0]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the topN similar movies
    movieIndices = [i[0] for i in simScores[1: topN+1]]  # Skip the first one as it will be the input itself if it matches exactly
    
    # Select the topN similar movies and process the names for concise output
    recommendedMovies = data.iloc[movieIndices].copy()
    recommendedMovies['genre'] = recommendedMovies['genre'].apply(lambda x: ', '.join(x.split(',')[:2]))  # Show first 2 genres
    recommendedMovies['stars_name'] = recommendedMovies['stars_name'].apply(lambda x: ', '.join(x.split(',')[:3]))  # Show first 3 stars
    recommendedMovies['directors_name'] = recommendedMovies['directors_name'].apply(lambda x: x.split(',')[0])  # Show primary director
    
    # Return the topN most similar movies with adjusted output
    return recommendedMovies[['name', 'year', 'genre', 'directors_name', 'rating', 'stars_name']]

# Loop until the user inputs "quit"
while True:
    userInput = input("Enter your search (e.g., genre, director, actor names, etc.) or 'quit' to exit: ")
    if userInput.lower() == 'quit':
        break

    # Get recommendations based on the user input
    recommendations = recommendMoviesBasedOnInput(userInput, topN=7)

    # Print the recommendations
    print(tabulate(recommendations, headers='keys', tablefmt='psql', showindex=False))