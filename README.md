
# Movie Recommendation System

## Overview
This script provides movie recommendations based on user input using a TF-IDF vectorization and cosine similarity approach. It utilizes data from a CSV file containing movies and their features like genre, directors, stars, and descriptions.

## Requirements
- Python 3.x
- Pandas: `pip install pandas`
- Scikit-learn: `pip install scikit-learn`
- Tabulate: `pip install tabulate`

## Dataset
The script reads a file named `movies.csv` that should be structured with columns: `name`, `genre`, `directors_name`, `stars_name`, `year`, `rating`, and `description`.

## Usage
1. Ensure that the `movies.csv` file is in the same directory as the script.
2. Run the script using Python. For example:
   ```bash
   python movie_recommender.py
   ```
3. When prompted, enter your search query related to the movies (e.g., genre, director, actor names). You can input 'quit' to exit the script.

## Functionality
- **Data Preparation**: Combines relevant features into a single feature for vectorization.
- **TF-IDF Vectorization**: Converts the combined text data into a matrix of TF-IDF features.
- **Cosine Similarity**: Computes the similarity between the user input and all movies in the dataset.
- **Recommendations**: Outputs the top 7 recommended movies based on the query.

Enjoy finding new movies to watch!
