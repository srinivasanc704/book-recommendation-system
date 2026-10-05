"""Evaluation helpers for the recommender.

Important honesty note
----------------------
A content-based system built from a *catalogue* has no ground truth about what
a user actually liked, so classic Accuracy / real Precision@K / Recall@K cannot
be computed from it. What we CAN measure honestly is:

* similarity-score statistics (how strong are the matches?)
* catalogue coverage (how much of the catalogue ever gets recommended?)
* intra-list diversity (are the Top-K books all near-duplicates?)
* metadata agreement - a PROXY for relevance: how often a recommendation shares
  an author or a genre with the query book. It is NOT real Precision@K.

``precision_at_k`` / ``recall_at_k`` are provided for the day a real
user-interaction (ground-truth) dataset is available.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from src.data_preprocessing import parse_genres
from src.recommender import BookRecommender


def precision_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in list(recommended)[:k] if item in relevant_set) / k


def recall_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return sum(1 for item in list(recommended)[:k] if item in relevant_set) / len(relevant_set)


def evaluate_recommender(recommender: BookRecommender, k: int = 10, sample_size: int = 100,
                         seed: int = 42) -> dict[str, Any]:
    """Run the offline checks on a random sample of query books."""
    books = recommender.books
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(books), size=min(sample_size, len(books)), replace=False)

    top1, mean_topk, diversity = [], [], []
    author_hits, genre_hits, genre_eligible = [], [], 0
    recommended_ids: set[int] = set()
    lists_returned = 0

    for idx in sample:
        recs = recommender.recommend_books(int(idx), k, use_hybrid=False)
        if recs.empty:
            continue
        lists_returned += 1
        sims = recs["content_similarity"].to_numpy()
        top1.append(float(sims[0]))
        mean_topk.append(float(sims.mean()))
        recommended_ids.update(recs["book_id"].astype(int).tolist())

        # intra-list diversity = 1 - average pairwise cosine similarity inside the list
        if len(recs) > 1:
            vectors = recommender.tfidf_matrix[recs["book_id"].astype(int).to_numpy()]
            pair = (vectors @ vectors.T).toarray()
            off_diagonal = pair[~np.eye(len(recs), dtype=bool)]
            diversity.append(float(1.0 - off_diagonal.mean()))

        query = books.iloc[int(idx)]
        author_hits.append(float((recs["author"] == query["author"]).mean()))
        query_genres = set(parse_genres(query["genre"]))
        if query_genres:
            genre_eligible += 1
            shared = recs["genre"].map(lambda g: bool(query_genres & set(parse_genres(g))))
            genre_hits.append(float(shared.mean()))

    def mean_or_none(values: list[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    return {
        "k": k,
        "queries_evaluated": lists_returned,
        "avg_top1_similarity": mean_or_none(top1),
        "avg_topk_similarity": mean_or_none(mean_topk),
        "intra_list_diversity": mean_or_none(diversity),
        "catalogue_coverage": len(recommended_ids) / len(books) if len(books) else None,
        "same_author_rate_proxy": mean_or_none(author_hits),
        "shared_genre_rate_proxy": mean_or_none(genre_hits),
        "queries_with_genre": genre_eligible,
    }
