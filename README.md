# 📚 BookWise – AI-Powered Book Recommendation System

A content-based book recommender built with **Python, Scikit-learn and Streamlit**. Pick a book and BookWise
finds similar ones using **TF-IDF vectors + cosine similarity**. The recommendations come from a trained, saved
model, not from hard-coded results.

## Overview
`python src/train_model.py` cleans the CSV, builds a TF-IDF model and saves it to `models/`.
`streamlit run app.py` loads those files and serves the recommendations. If the dataset has ratings, an optional
hybrid score (default 70 % similarity + 30 % popularity) re-ranks the best content matches.

## Features
- Partial, case-insensitive search by title **or** author (word order does not matter)
- Top-5 / 10 / 15 recommendations with real similarity percentages and cover images
- Optional hybrid ranking with configurable weights (only if the data has ratings)
- Popular books: top rated, most rated, top authors, with a minimum-ratings filter
- Dataset explorer: rating and year histograms, top authors and genres, model info, offline evaluation
- Works with different column names; every column except the title is optional
- Friendly messages for a missing dataset, missing model, empty search, missing image / description / rating

## Machine learning
| Idea | Role in BookWise |
|---|---|
| **Content-based filtering** | Recommends books whose *content* resembles the selected book. No user history needed. |
| **TF-IDF** | Turns each book's text into numbers. Rare-but-frequent-in-this-book words get high weight; common words get low weight. |
| **Cosine similarity** | Compares two vectors by angle (1 = same direction, 0 = nothing in common), so text length does not bias the result. |
| **Hybrid re-ranking** | `final = w_c × content + w_p × popularity` on the most similar candidates. Popularity is a Bayesian weighted rating, so books with very few votes cannot dominate. |

Text features: title, author (×2), genre (×2), publisher, description, publication decade. Multi-word genres and
publishers become single tokens (`science_fiction`, `tor_books`).

## Architecture
```
User → Streamlit UI (app.py) → Recommendation Engine (src/recommender.py)
     → TF-IDF / Similarity Model (models/*.pkl) → Book Dataset (data/*.csv) → Recommended Books
```
Training flow: `CSV → clean → missing values → duplicates → features → text preprocessing → TF-IDF → cosine / on-demand → models/`

## Project structure
```
book-recommendation-system/
├── app.py                      Streamlit front-end
├── requirements.txt  .gitignore  README.md
├── data/        books.csv (yours)   sample_books.csv (demo file)
├── models/      tfidf_vectorizer.pkl  tfidf_matrix.pkl  similarity_matrix.pkl  books.pkl  model_info.json
├── src/
│   ├── data_preprocessing.py   column detection, cleaning, feature engineering
│   ├── train_model.py          training script
│   ├── recommender.py          search + recommendation + popularity functions
│   ├── evaluation.py           offline metrics
│   └── utils.py                paths, constants, exceptions, formatting helpers
├── notebooks/book_recommendation_analysis.ipynb
├── tests/test_pipeline.py
├── docs/        VIVA_PREPARATION.md   PROJECT_REPORT.md
└── assets/logo.png
```
`similarity_matrix.pkl` exists only for small catalogues (≤ 3000 books). Larger catalogues compute one row of
similarities on demand from the sparse `tfidf_matrix.pkl`, which keeps memory low.

## Dataset schema
Only a title column is required. Names are matched ignoring case, spaces, `_` and `-`.

| Meaning | Accepted names (examples) | Required |
|---|---|---|
| Title | `Book-Title`, `title`, `original_title` | **Yes** |
| Author | `Book-Author`, `author`, `authors` | No |
| Year | `Year-Of-Publication`, `year`, `original_publication_year` | No |
| Publisher | `Publisher` | No |
| Genre | `Genre`, `genres`, `categories` (`A, B`, `A\|B` or `['A','B']`) | No |
| Description | `Description`, `summary`, `synopsis` | No |
| Image | `Image-URL-L/M/S`, `image_url`, `thumbnail` | No |
| Rating | `Book-Rating`, `average_rating` | No |
| Rating count | `ratings_count`, `rating_count` | No |
| ISBN | `ISBN`, `isbn13` | Only to merge a separate ratings file |

**Book-Crossing style data** (Books.csv + Ratings.csv): put Books.csv at `data/books.csv`, Ratings.csv at
`data/ratings.csv`, and train with `python src/train_model.py --ratings data/ratings.csv`. Ratings of 0 are treated
as "no rating". `data/sample_books.csv` has 52 real book titles for testing; its **ratings and rating counts are
placeholder values**, not real statistics.

## Installation
```bash
git clone <repository-url>
cd book-recommendation-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tested with Python 3.12, Streamlit 1.65, pandas 3.0, scikit-learn 1.8. Python 3.9+ should work.

## Training
```bash
python src/train_model.py                                   # uses data/books.csv (or the sample file if missing)
python src/train_model.py --ratings data/ratings.csv        # with a separate ratings file
python src/train_model.py --max-features 20000 --min-df 2   # for very large datasets
```

## Running
```bash
streamlit run app.py
```

## Testing and debugging
```bash
python -m unittest discover tests -v      # 12 end-to-end tests, uses a temporary model folder
pip install jupyter && jupyter notebook notebooks/book_recommendation_analysis.ipynb
```
| Symptom | Fix |
|---|---|
| "The trained model was not found" | Run `python src/train_model.py` |
| "No title column found" | Rename your title column, or add an alias in `COLUMN_ALIASES` in `src/data_preprocessing.py` |
| `ModuleNotFoundError: src` | Run commands from the project root folder |
| Model out of date after changing data | Re-run training; the app reloads the new model automatically |
| Covers show a coloured letter | The dataset has no image URL (or it is empty); this is expected |
| Slow training on a huge CSV | Lower `--max-features`, raise `--min-df` |

## Evaluation
With no user-interaction data, accuracy and true Precision@K / Recall@K cannot be computed. The app reports
similarity statistics, catalogue coverage, intra-list diversity, and *proxy* metrics (same author / shared genre).
`precision_at_k` and `recall_at_k` are included for when ground-truth data exists.

## Screenshots
| Home | Recommender |
|---|---|
| _add `screenshots/home.png`_ | _add `screenshots/recommender.png`_ |
| **Popular books** | **Dataset explorer** |
| _add `screenshots/popular.png`_ | _add `screenshots/explorer.png`_ |

## Future enhancements
Collaborative filtering · user profiles · personalised recommendations · login system · feedback (like/dislike) ·
stronger hybrid models · LLM-based explanations · deployment (Streamlit Community Cloud / Docker).
