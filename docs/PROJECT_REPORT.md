# Project Report Content – BookWise: AI-Powered Book Recommendation System

> Replace the bracketed items with your own values (shown in the app under *Dataset Explorer → Model & evaluation*).

## Abstract
Readers face an overwhelming number of books. This project presents BookWise, a content-based book recommendation system built with Python, Scikit-learn and Streamlit. Book metadata (title, author, genre, publisher, description, publication decade) is cleaned and merged into a text feature, converted to numerical vectors with TF-IDF, and compared using cosine similarity. For a selected book, the system returns the most similar titles with similarity scores. When rating data is available, an optional hybrid score blends content similarity with a Bayesian popularity score. The trained model is saved and reused by a Streamlit web application that also provides search, popular-book rankings and dataset analytics.

## Introduction
Recommendation systems help users discover relevant items in large catalogues and are used by online stores, streaming platforms and libraries. Books are well suited to text-based machine learning because they have rich descriptive metadata.

## Problem Statement
Given a book a user likes, automatically suggest other books they may enjoy, using only the information available in a book dataset.

## Objectives
1. Build a reusable ML pipeline from CSV data to recommendations.
2. Use TF-IDF and cosine similarity for content-based recommendation.
3. Add an optional rating/popularity hybrid score.
4. Provide a clean web interface with search, covers, similarity scores and analytics.
5. Evaluate the system honestly given the available data.

## Existing System
Manual browsing, keyword search and bestseller lists. Keyword search needs the exact title or author; bestseller lists are the same for everyone and ignore what the reader just enjoyed.

## Proposed System
BookWise recommends books by content similarity to a selected title, supports partial search by title or author, and optionally re-ranks results using ratings.

## Literature / Background
- **Recommender systems** are usually grouped into content-based, collaborative and hybrid approaches (Ricci, Rokach & Shapira, *Recommender Systems Handbook*).
- **TF-IDF** (Spärck Jones, 1972; Salton & Buckley, 1988) weights terms by their discriminative power.
- **Cosine similarity** is the standard measure for comparing sparse text vectors (Manning, Raghavan & Schütze, *Introduction to Information Retrieval*).
- **Weighted rating** (Bayesian average) prevents items with few votes from dominating rankings.

## System Requirements
**Software:** Python 3.9+, Streamlit, pandas, NumPy, SciPy, scikit-learn, Plotly, Matplotlib, Pillow, joblib. **Hardware:** a normal laptop (4 GB RAM is enough for catalogues of tens of thousands of books). **Data:** a CSV with at least a title column.

## Methodology
Dataset → data cleaning → missing-value handling → duplicate removal → feature engineering → text preprocessing → TF-IDF → cosine similarity / nearest-row search → recommendation engine → Streamlit application.

## Data Preprocessing
- Automatic detection and renaming of columns; optional columns may be absent.
- Lowercasing, accent removal, punctuation removal, whitespace normalisation (digits kept).
- Invalid years set to missing; ratings of 0 treated as "no rating".
- Records without a title removed; books with the same cleaned title and author merged and their ratings combined.
- Missing author, publisher, genre and description handled without dropping the book.

## Feature Engineering
A single "soup" per book: title + author×2 + genre×2 + publisher + description + publication decade. Multi-word genres and publishers become single tokens. Popularity is a Bayesian weighted rating `v/(v+m)·R + m/(v+m)·C`, min-max scaled to [0, 1].

## Machine Learning Algorithm
`TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True, max_features=30000)` produces an L2-normalised sparse matrix. Similarity between books *a* and *b* is `cos(a, b) = (a·b)/(‖a‖‖b‖)`. TF-IDF suits the task because it highlights distinctive words; cosine similarity suits it because it ignores text length and handles sparse vectors efficiently. Hybrid score: `final = w_c·content + w_p·popularity` (default 0.70 / 0.30), applied to the 200 most similar candidates.

## System Architecture
User → Streamlit UI → Recommendation Engine → TF-IDF/Similarity Model → Book Dataset → Recommended Books. Training and serving are separate: `train_model.py` writes `models/*.pkl`; `app.py` only loads them (cached with `st.cache_resource`).

## Implementation
Modules: `data_preprocessing.py` (cleaning and features), `train_model.py` (training and saving), `recommender.py` (search, recommendation, popularity, statistics), `evaluation.py` (offline metrics), `utils.py` (constants, exceptions, helpers), `app.py` (interface). For catalogues above 3000 books, the full similarity matrix is not stored; one similarity row is computed on demand from the sparse matrix.

## Streamlit Interface
Sidebar navigation with five pages: Home (project summary and metrics), Book Recommender (search, selection, 5/10/15 results as cards with cover, author, genre, rating, year, similarity bar, description), Popular Books (top rated, most rated, top authors with a minimum-ratings filter), Dataset Explorer (Plotly charts, model info, evaluation) and About.

## Results
*Insert your own screenshots and numbers.* Dataset used: [file name]; books after cleaning: [n]; TF-IDF terms: [n]; similarity mode: [dense / on-demand]. Include screenshots of three example searches and the recommended lists. Do not quote results that were not produced by your run.

## Evaluation
Accuracy is not appropriate: recommending is a ranking task and a catalogue has no per-user ground truth. Measures used: (1) similarity-score analysis, (2) catalogue coverage, (3) intra-list diversity, (4) proxy agreement (same author / shared genre; **not** true precision). Precision@K = relevant items in top K ÷ K and Recall@K = relevant items in top K ÷ all relevant items are implemented for use when real user-interaction data is available. Record your measured values: avg top-1 similarity [ ], coverage [ ], diversity [ ].

## Advantages
No user history required; transparent and explainable; fast; works with partial metadata; runs on a laptop; modular and reusable.

## Limitations
Quality depends on the text available; no personalisation; cold start for books without descriptions; popularity needs rating data; offline metrics are proxies rather than user satisfaction.

## Future Enhancement
Collaborative filtering, user profiles and login, like/dislike feedback, sentence-embedding or LLM-based similarity and explanations, A/B testing with real users, cloud deployment.

## Conclusion
BookWise shows a complete ML workflow: data cleaning, feature engineering, TF-IDF modelling, similarity-based recommendation and an interactive interface. It is modular, honest about its evaluation limits, and ready to extend.

## References
1. F. Ricci, L. Rokach, B. Shapira (eds.), *Recommender Systems Handbook*, Springer.
2. C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.
3. K. Spärck Jones, "A statistical interpretation of term specificity and its application in retrieval", *Journal of Documentation*, 1972.
4. G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval", *Information Processing & Management*, 1988.
5. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12, 2011.
6. Streamlit documentation, https://docs.streamlit.io
