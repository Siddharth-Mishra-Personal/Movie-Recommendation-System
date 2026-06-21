# 🎬 Movie Recommendation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ml-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project that implements content-based filtering algorithms to suggest movies to users based on shared traits such as genres, keywords, cast, and directors. 

---

## 📌 Project Overview

With thousands of movies available on streaming platforms, discovering content can be overwhelming. This project implements a **Content-Based Recommendation Engine** that calculates mathematical similarities between films. By analyzing structured text meta-data (like tags, plot keywords, and actors), the engine identifies patterns in what a user enjoys and surfaces highly relevant alternative suggestions.

### 🚀 Key Features
*   **Text Processing & Tokenization**: Merges distinct text fields (genres, keywords, cast, crew) into single metadata tags.
*   **Vectorization**: Implements **CountVectorizer** or **TF-IDF** to convert raw string content into structured numeric feature matrices.
*   **Similarity Computation**: Utilizes **Cosine Similarity** to quantify spatial distances between thousands of movie profiles simultaneously.
*   **Dynamic Search & Predict**: Input any target film title to instantaneously receive a list of top-ranked lookalike recommendations.

---

## 🛠️ Tech Stack & Core Libraries

*   **Core Logic**: `Python 3.8+`
*   **Data Wrangling**: `pandas`, `numpy`
*   **Machine Learning / Math**: `scikit-learn` (`CountVectorizer`, `cosine_similarity`)
*   **Interface**: `Jupyter Notebook`

---

## 📂 Project Architecture & Methodology

[Raw Movie Data] ➔ [Feature Engineering: Tag Merging] ➔ [Text Vectorization] ➔ [Cosine Similarity Matrix] ➔ [Top-N Recommendations Engine]


1. **Data Ingestion**: Loading movie attributes including titles, overviews, genres, cast lineups, and technical credits.
2. **Text Cleaning & Preprocessing**: 
   * Stripping spaces from entity strings (e.g., transforming `"Johnny Depp"` into `"JohnnyDepp"` so the vectorizer reads it as a unique singular token).
   * Combining descriptive attributes into an aggregated text block per film.
3. **Feature Matrix Construction**: Transforming text blocks into a frequency count array using a bag-of-words model.
4. **Distance Evaluation**: Constructing an $N \times N$ cosine similarity matrix mapping how closely aligned every movie is to every other entry in the database.
5. **Ranking Execution**: Fetching the target index for a user-input film, sorting similarity values in descending order, and extracting the top matches.

---

## 🚀 Getting Started

### Prerequisites
Make sure your system has Python 3.8 or higher installed.

### Installation & Execution Guide

1. **Clone the repository:**
```bash
   git clone [https://github.com/Siddharth-Mishra-Personal/Movie-Recommendation-System.git](https://github.com/Siddharth-Mishra-Personal/Movie-Recommendation-System.git)
   cd Movie-Recommendation-System
