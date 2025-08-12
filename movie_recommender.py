#!/usr/bin/env python3
"""
Enhanced Unified Movie Recommender

Features:
- Improved error handling and user feedback
- Added runtime performance logging
- Better input validation
- More informative output formatting
- Default fallback recommendations
"""

import os
import argparse
import difflib
import time
from typing import List, Tuple, Dict, Union
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration / Constants
# ---------------------------
DATA_DIR = "data"
MOVIES_CSV = os.path.join(DATA_DIR, "movies_copy.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings_copy.csv")

NUM_CLUSTERS = 20
SVD_RANDOM_STATE = 42
TFIDF_STOP_WORDS = 'english'
MIN_RATING_COUNT = 10  # Minimum ratings for a movie to be considered in popularity

# ---------------------------
# Data Loading Utilities
# ---------------------------
def load_data(movies_path: str = MOVIES_CSV, ratings_path: str = RATINGS_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and ratings CSVs into pandas DataFrames with validation."""
    try:
        logger.info("Loading data files...")

        # 🔹 Check file existence
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        # 🔹 Quick permission/lock test
        try:
            with open(movies_path, "r", encoding="utf-8") as f:
                logger.info(f"Movies.csv first line: {f.readline().strip()}")
            with open(ratings_path, "r", encoding="utf-8") as f:
                logger.info(f"Ratings.csv first line: {f.readline().strip()}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied when reading one of the CSV files.\n"
                f"Possible causes:\n"
                f"- The file is open in Excel or another program (close it).\n"
                f"- Windows blocked the file after extraction (Right-click → Properties → Unblock).\n"
                f"- Antivirus or OneDrive is locking the file temporarily.\n"
                f"File paths:\n"
                f"  Movies: {movies_path}\n"
                f"  Ratings: {ratings_path}"
            )

        # 🔹 Load with pandas
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        # 🔹 Validate required columns
        required_movie_cols = {'movieId', 'title'}
        required_rating_cols = {'userId', 'movieId', 'rating'}

        if not required_movie_cols.issubset(movies.columns):
            raise ValueError(f"movies.csv missing required columns: {required_movie_cols}")
        if not required_rating_cols.issubset(ratings.columns):
            raise ValueError(f"ratings.csv missing required columns: {required_rating_cols}")

        logger.info(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
        return movies, ratings

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


# ---------------------------
# Content-based Recommender
# ---------------------------
def build_tfidf_matrix(movies: pd.DataFrame) -> Tuple:
    """Build TF-IDF matrix from movie metadata."""
    logger.info("Building TF-IDF matrix...")
    start_time = time.time()
    
    movies = movies.copy()
    # Combine title and genres for better recommendations
    movies['meta'] = movies['title'] + ' ' + movies['genres'].fillna('')
    vectorizer = TfidfVectorizer(stop_words=TFIDF_STOP_WORDS)
    tfidf_matrix = vectorizer.fit_transform(movies['meta'])
    
    logger.info(f"TF-IDF matrix built in {time.time() - start_time:.2f}s")
    return tfidf_matrix, vectorizer, movies

def get_closest_title(user_input: str, titles: List[str], cutoff: float = 0.6) -> Union[str, None]:
    """Fuzzy match movie title."""
    matches = difflib.get_close_matches(user_input, titles, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def content_recommendations(movies: pd.DataFrame, tfidf_matrix, user_input: str, top_n: int = 10) -> Dict:
    """Generate content-based recommendations."""
    titles = movies['title'].tolist()
    matched = get_closest_title(user_input, titles)
    
    if not matched:
        return {"error": f"No close match found for '{user_input}'. Try being more specific."}
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    idx = movies[movies['title'] == matched].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    return {
        "matched_movie": matched,
        "recommendations": [
            (movies.iloc[i]['title'], float(score)) 
            for i, score in sim_scores[1:top_n+1]
        ]
    }

# ---------------------------
# Popularity-based Recommender
# ---------------------------
def build_popularity_df(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Calculate popularity scores."""
    logger.info("Calculating popularity scores...")
    
    rating_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    df = pd.merge(movies, rating_stats, on='movieId', how='left')
    df.fillna({'avg_rating': 0, 'rating_count': 0}, inplace=True)
    
    # Filter movies with minimum ratings
    df = df[df['rating_count'] >= MIN_RATING_COUNT]
    
    # Normalize and calculate score
    df['rating_count_norm'] = df['rating_count'] / df['rating_count'].max()
    df['popularity_score'] = (df['avg_rating'] * 0.7) + (df['rating_count_norm'] * 0.3 * 5)
    
    return df

def popularity_recommendations(pop_df: pd.DataFrame, top_n: int = 10) -> List[Tuple]:
    """Generate popularity-based recommendations."""
    return pop_df.sort_values('popularity_score', ascending=False).head(top_n)[
        ['title', 'popularity_score', 'rating_count']
    ].to_records(index=False).tolist()

# ---------------------------
# Main Execution
# ---------------------------
def main():
    """Main interactive recommendation loop."""
    print("\n=== Enhanced Movie Recommender ===")
    print("1. Content-based recommendations")
    print("2. Popularity-based recommendations")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nChoose an option (1-3): ").strip()
            
            if choice == '3':
                print("Goodbye!")
                break
                
            if choice not in ('1', '2'):
                print("Invalid choice. Please enter 1, 2, or 3.")
                continue
                
            top_n = input(f"Number of recommendations (default 10): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 10
            
            # Load data
            movies, ratings = load_data()
            
            if choice == '1':
                query = input("Enter movie title: ").strip()
                _, _, movies_with_meta = build_tfidf_matrix(movies)
                tfidf_matrix, _, _ = build_tfidf_matrix(movies)
                result = content_recommendations(movies_with_meta, tfidf_matrix, query, top_n)
                
                if 'error' in result:
                    print(f"\nError: {result['error']}")
                else:
                    print(f"\nMovies similar to '{result['matched_movie']}':")
                    for title, score in result['recommendations']:
                        print(f"- {title} (similarity: {score:.2f})")
            
            elif choice == '2':
                pop_df = build_popularity_df(movies, ratings)
                recs = popularity_recommendations(pop_df, top_n)
                print("\nMost Popular Movies:")
                for title, score, count in recs:
                    print(f"- {title} (score: {score:.2f}, ratings: {count})")
                    
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            print("An error occurred. Please try again.")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("A fatal error occurred. Please check the logs.")
