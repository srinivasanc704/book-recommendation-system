"""Recommendation engine: loads the trained model and answers queries.

The heart of the project is ``BookRecommender.recommend_books``:
    1. find the selected book,
    2. compute cosine similarity between its TF-IDF vector and every other book,
    3. exclude the book itself, sort, return the Top-N,
    4. (optional) re-rank the best candidates with a rating/popularity score.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data_preprocessing import genre_counts, weighted_rating
from src.utils import (BOOKS_FILE, DEFAULT_CONTENT_WEIGHT, DEFAULT_POPULARITY_WEIGHT,
                       HYBRID_CANDIDATE_POOL, MODEL_INFO_FILE, MODELS_DIR, SIMILARITY_FILE,
                       TFIDF_MATRIX_FILE, VECTORIZER_FILE, BookNotFoundError,
                       ModelNotFoundError)

RESULT_COLUMNS = ["rank", "book_id", "title", "author", "genre", "year", "rating", "rating_count",
                  "publisher", "image_url", "description", "display_name",
                  "content_similarity", "popularity_score", "final_score"]


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, sorted from largest to smallest."""
    k = min(k, len(values))
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-values, k - 1)[:k]
    return part[np.argsort(-values[part], kind="stable")]


class BookRecommender:
    """Content-based (optionally hybrid) book recommender."""

    def __init__(self, books: pd.DataFrame, vectorizer: Any, tfidf_matrix: csr_matrix,
                 similarity_matrix: Optional[np.ndarray] = None,
                 info: Optional[dict[str, Any]] = None) -> None:
        self.books = books.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = similarity_matrix
        self.info = info or {}
        self._label_to_index = {name: i for i, name in enumerate(self.books["display_name"])}
        self._title_lower = self.books["title"].str.lower()

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "BookRecommender":
        """Load the files created by ``train_model.py``."""
        models_dir = Path(models_dir)
        required = [BOOKS_FILE, VECTORIZER_FILE, TFIDF_MATRIX_FILE]
        missing = [name for name in required if not (models_dir / name).exists()]
        if missing:
            raise ModelNotFoundError(
                f"Model files not found in '{models_dir}': {', '.join(missing)}. "
                "Run `python src/train_model.py` first.")
        try:
            books = joblib.load(models_dir / BOOKS_FILE)
            vectorizer = joblib.load(models_dir / VECTORIZER_FILE)
            tfidf = joblib.load(models_dir / TFIDF_MATRIX_FILE)
            sim_path = models_dir / SIMILARITY_FILE
            similarity = joblib.load(sim_path) if sim_path.exists() else None
            info_path = models_dir / MODEL_INFO_FILE
            info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        except Exception as exc:  # corrupted pickle, version mismatch, ...
            raise ModelNotFoundError(
                f"Model files could not be loaded ({type(exc).__name__}). "
                "Re-run `python src/train_model.py`.") from exc
        if len(books) != tfidf_matrix_rows(tfidf):
            raise ModelNotFoundError("Model files are out of sync. Re-run `python src/train_model.py`.")
        return cls(books, vectorizer, tfidf, similarity, info)

    # ------------------------------------------------------------- properties
    @property
    def n_books(self) -> int:
        return len(self.books)

    @property
    def has_ratings(self) -> bool:
        return bool((self.books["rating_count"] > 0).any())

    # ----------------------------------------------------------------- lookup
    def get_book(self, book: Any) -> pd.Series:
        """Return the book row for a display label, a title or a book_id."""
        idx = self.find_book(book)
        if idx is None:
            raise BookNotFoundError(f"Book not found: {book!r}")
        return self.books.iloc[idx]

    def find_book(self, book: Any) -> Optional[int]:
        """Resolve a label ('Title — Author'), exact title or book_id to a row index."""
        if isinstance(book, (int, np.integer)):
            return int(book) if 0 <= int(book) < self.n_books else None
        if not isinstance(book, str) or not book.strip():
            return None
        text = book.strip()
        if text in self._label_to_index:
            return self._label_to_index[text]
        matches = np.flatnonzero((self._title_lower == text.lower()).to_numpy())
        if len(matches) == 0:
            return None
        if len(matches) > 1:  # same title by several authors: take the most-rated one
            matches = matches[np.argsort(-self.books["rating_count"].to_numpy()[matches], kind="stable")]
        return int(matches[0])

    def search_books(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Case-insensitive partial search over title and author.

        Every word of the query must appear in the title or author, in any
        order, so 'rowling harry' and 'harry pot' both work.
        """
        from src.data_preprocessing import clean_text_for_model
        tokens = clean_text_for_model(query).split()
        if not tokens:
            return self.books.iloc[0:0]
        text = self.books["search_text"]
        mask = pd.Series(True, index=self.books.index)
        for token in tokens:
            mask &= text.str.contains(token, regex=False)
        hits = self.books[mask].copy()
        if hits.empty:
            return hits
        phrase = " ".join(tokens)
        title_clean = hits["search_text"]
        hits["_rank"] = np.where(title_clean.str.startswith(phrase), 0,
                                 np.where(self._title_lower[hits.index].str.contains(tokens[0], regex=False), 1, 2))
        hits = hits.sort_values(["_rank", "rating_count"], ascending=[True, False], kind="stable")
        return hits.drop(columns="_rank").head(limit)

    # ------------------------------------------------------------- similarity
    def similarity_to(self, index: int) -> np.ndarray:
        """Cosine similarity between one book and all books (values 0..1)."""
        if self.similarity_matrix is not None:
            return np.asarray(self.similarity_matrix[index], dtype=np.float64)
        return cosine_similarity(self.tfidf_matrix[index], self.tfidf_matrix).ravel().astype(np.float64)

    # --------------------------------------------------------- recommendation
    def recommend_books(self, book_title: Any, n_recommendations: int = 10,
                        content_weight: float = DEFAULT_CONTENT_WEIGHT,
                        popularity_weight: float = DEFAULT_POPULARITY_WEIGHT,
                        use_hybrid: Optional[bool] = None,
                        candidate_pool: int = HYBRID_CANDIDATE_POOL) -> pd.DataFrame:
        """Return the Top-N most similar books as a DataFrame.

        Columns include ``content_similarity`` (real cosine similarity, 0..1),
        ``popularity_score`` and ``final_score``. With ``use_hybrid=False`` (or when
        the dataset has no ratings) final_score equals content_similarity.
        Books with zero similarity are never returned, so fewer than N rows
        can come back for small catalogues.
        """
        idx = self.find_book(book_title)
        if idx is None:
            raise BookNotFoundError(f"Book not found: {book_title!r}")
        n = max(1, int(n_recommendations))
        if use_hybrid is None:
            use_hybrid = True
        use_hybrid = bool(use_hybrid and self.has_ratings and popularity_weight > 0)

        sims = self.similarity_to(idx)
        sims[idx] = -1.0  # a book must never recommend itself
        popularity = self.books["popularity_score"].to_numpy(dtype=np.float64)

        if use_hybrid:
            total = float(content_weight) + float(popularity_weight)
            w_content, w_pop = ((DEFAULT_CONTENT_WEIGHT, DEFAULT_POPULARITY_WEIGHT) if total <= 0
                                else (content_weight / total, popularity_weight / total))
            # Re-rank only books that are already content-similar, otherwise a very
            # popular but unrelated book could win.
            pool = _top_k_indices(sims, max(candidate_pool, n * 5))
            pool = pool[sims[pool] > 0]
            # Raw cosine values are usually small (e.g. 0.05-0.4) while popularity spans
            # 0-1, so both are scaled to 0-1 *within the candidate pool* before mixing.
            # (The raw cosine value is still what we report as "similarity".)
            top_sim = float(sims[pool].max()) if len(pool) else 0.0
            content_part = sims[pool] / top_sim if top_sim > 0 else sims[pool]
            pop_pool = popularity[pool]
            pop_span = float(pop_pool.max() - pop_pool.min()) if len(pool) else 0.0
            pop_part = (pop_pool - pop_pool.min()) / pop_span if pop_span > 0 else np.zeros_like(pop_pool)
            final = w_content * content_part + w_pop * pop_part
            order = _top_k_indices(final, n)
            chosen, final_scores = pool[order], final[order]
        else:
            top = _top_k_indices(sims, n)
            chosen = top[sims[top] > 0]
            final_scores = sims[chosen]

        result = self.books.iloc[chosen].copy()
        result["content_similarity"] = sims[chosen]
        result["popularity_score"] = popularity[chosen]
        result["final_score"] = final_scores
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result[RESULT_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------- popularity
    def top_rated_books(self, n: int = 10, min_ratings: int = 10) -> pd.DataFrame:
        """Best average rating among books with at least ``min_ratings`` ratings.

        Sorted by the Bayesian weighted rating so tiny vote counts do not dominate.
        """
        if not self.has_ratings:
            return self.books.iloc[0:0]
        eligible = self.books[self.books["rating_count"] >= max(1, min_ratings)].copy()
        if eligible.empty:
            return eligible
        eligible["weighted_rating"] = weighted_rating(eligible, m=float(max(1, min_ratings)))
        return eligible.sort_values(["weighted_rating", "rating_count"], ascending=False).head(n)

    def most_popular_books(self, n: int = 10) -> pd.DataFrame:
        """Books with the largest number of ratings."""
        if not self.has_ratings:
            return self.books.iloc[0:0]
        return self.books.sort_values("rating_count", ascending=False, kind="stable").head(n)

    def top_authors(self, n: int = 10, min_ratings: int = 10) -> pd.DataFrame:
        """Authors ranked by rating (weighted by vote count) with enough ratings."""
        books = self.books[self.books["author"] != "Unknown Author"].copy()
        if self.has_ratings:
            books["_sum"] = books["rating"].fillna(0) * books["rating_count"]
            grouped = books.groupby("author").agg(books=("title", "count"),
                                                  total_ratings=("rating_count", "sum"),
                                                  _sum=("_sum", "sum")).reset_index()
            grouped = grouped[grouped["total_ratings"] >= max(1, min_ratings)]
            grouped["avg_rating"] = grouped["_sum"] / grouped["total_ratings"]
            return grouped.drop(columns="_sum").sort_values(
                ["avg_rating", "total_ratings"], ascending=False).head(n).reset_index(drop=True)
        grouped = books.groupby("author").size().reset_index(name="books")
        return grouped.sort_values("books", ascending=False).head(n).reset_index(drop=True)

    # --------------------------------------------------------------- analytics
    def dataset_stats(self) -> dict[str, Any]:
        """Real statistics computed from the loaded catalogue."""
        books = self.books
        rated = books[books["rating_count"] > 0]
        years = books["year"].dropna()
        return {
            "n_books": int(len(books)),
            "n_authors": int(books.loc[books["author"] != "Unknown Author", "author"].nunique()),
            "n_publishers": int(books.loc[books["publisher"] != "", "publisher"].nunique()),
            "n_genres": int(len(genre_counts(books))),
            "avg_rating": float(np.average(rated["rating"], weights=rated["rating_count"])) if len(rated) else None,
            "n_rated_books": int(len(rated)),
            "year_min": int(years.min()) if len(years) else None,
            "year_max": int(years.max()) if len(years) else None,
            "has_images": bool(books["image_url"].notna().any()),
        }


def tfidf_matrix_rows(matrix: Any) -> int:
    """Number of rows of a (sparse) matrix."""
    return int(matrix.shape[0])


# ----------------------------------------------------------------------------
# Module-level convenience function:  recommend_books("Dune", 10)
# ----------------------------------------------------------------------------
_DEFAULT_RECOMMENDER: Optional[BookRecommender] = None


def recommend_books(book_title: str, n_recommendations: int = 10, **kwargs: Any) -> pd.DataFrame:
    """Load the default model once and return recommendations for ``book_title``."""
    global _DEFAULT_RECOMMENDER
    if _DEFAULT_RECOMMENDER is None:
        _DEFAULT_RECOMMENDER = BookRecommender.load()
    return _DEFAULT_RECOMMENDER.recommend_books(book_title, n_recommendations, **kwargs)
