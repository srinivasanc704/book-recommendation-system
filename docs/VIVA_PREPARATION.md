# VIVA PREPARATION – BookWise

**1. What is a recommendation system?**
Software that predicts which items (books, movies, products) a user will like and suggests them.

**2. Why a book recommendation system?**
Readers face thousands of choices. Book data is rich in text (title, author, genre, description), which suits text-based ML, and the results are easy to demonstrate.

**3. What is content-based filtering?**
It recommends items whose own features are similar to an item the user liked. It needs no data about other users.

**4. What is TF-IDF?**
Term Frequency × Inverse Document Frequency. A word scores high if it appears often in one book but in few other books. Common words ("the", "book") score low.

**5. What is cosine similarity?**
The cosine of the angle between two vectors. 1 means the same direction (very similar), 0 means nothing in common.

**6. Why use cosine similarity?**
It ignores text length (a short and a long description can still match), and it works well on sparse TF-IDF vectors.

**7. What is feature engineering?**
Creating useful model inputs from raw data. Here we combine title, author, genre, publisher, description and decade into one text "soup", repeating author and genre so they count more.

**8. How does the pipeline work?**
CSV → clean → remove duplicates → build soup → TF-IDF → save model (`train_model.py`). Then the app loads the model, the user picks a book, we compute cosine similarity with all books, drop the book itself, sort, and show the Top-N.

**9. Why Streamlit?**
It builds a web interface in pure Python, has built-in widgets and caching, and runs locally with one command.

**10. Why is accuracy not the main metric?**
Recommending is ranking, not classifying. A catalogue has no "correct label" for a user. We would need real user clicks or ratings for Precision@K / Recall@K. Without them we measure similarity strength, diversity, and coverage.

**11. What are the limitations?**
Depends on text quality; no personalisation (cannot learn one user's taste); cold-start for books with little text; popularity re-ranking needs ratings; offline metrics are only proxies.

**12. How can it be improved?**
Collaborative filtering, user profiles and feedback, sentence-embedding or LLM models, a proper user study, deployment.

## Extra questions examiners like
- **Why are the similarity values small (e.g. 8–45 %)?** TF-IDF vectors are sparse, so two related books share only a few words. Ranking matters more than the absolute value.
- **How do you avoid a 5-star book with one vote ranking first?** Bayesian weighted rating plus a minimum-ratings threshold.
- **What is the hybrid formula?** `final = 0.7 × content + 0.3 × popularity`; both parts are scaled to 0–1 inside the candidate pool; weights are configurable in the UI.
- **Why not store the full similarity matrix?** n books → n² numbers. For 100 000 books that is 40 GB. We store the sparse TF-IDF matrix and compute one row when needed.
- **Where do duplicate books go?** Same cleaned title + author = one book; ratings are combined.
