"""Train the content-based model and save it to ``models/``.

Usage
-----
    python src/train_model.py
    python src/train_model.py --data data/books.csv --ratings data/ratings.csv
    python src/train_model.py --max-features 20000 --min-df 2

Pipeline: CSV -> clean -> engineer features -> TF-IDF -> (similarity) -> save.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Allow ``python src/train_model.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from src.data_preprocessing import preprocess_books, read_csv_flexible  # noqa: E402
from src.utils import (BOOKS_FILE, DEFAULT_DATA_PATH, DENSE_MATRIX_LIMIT, MODEL_INFO_FILE,  # noqa: E402
                       MODELS_DIR, SAMPLE_DATA_PATH, SIMILARITY_FILE, TFIDF_MATRIX_FILE,
                       VECTORIZER_FILE, DatasetError)


def train(data_path: Path, models_dir: Path = MODELS_DIR, ratings_path: Optional[Path] = None,
          max_features: int = 30000, min_df: int = 1, ngram_range: tuple[int, int] = (1, 2),
          verbose: bool = True) -> dict[str, Any]:
    """Run the full training pipeline and return the saved model_info dict."""
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    log(f"[1/5] Loading dataset: {data_path}")
    raw = read_csv_flexible(data_path)
    ratings_raw = read_csv_flexible(ratings_path) if ratings_path else None

    log("[2/5] Cleaning data and engineering features ...")
    books, report = preprocess_books(raw, ratings_raw)
    log(f"      rows in: {report['rows_in']} | invalid removed: {report['invalid_removed']} | "
        f"duplicates merged: {report['duplicates_removed']} | books kept: {report['rows_out']}")
    log(f"      columns missing (handled gracefully): {report['columns_missing']}")

    log("[3/5] Fitting TF-IDF vectoriser ...")
    # TF-IDF: words that are frequent in one book but rare overall get high weight.
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df,
                                 ngram_range=ngram_range, sublinear_tf=True, dtype=np.float32)
    try:
        tfidf_matrix = vectorizer.fit_transform(books["soup"])
    except ValueError as exc:  # e.g. "empty vocabulary"
        raise DatasetError(f"TF-IDF failed - not enough text to learn from ({exc}).") from exc
    log(f"      matrix shape: {tfidf_matrix.shape} (books x terms), "
        f"non-zero entries: {tfidf_matrix.nnz}")

    log("[4/5] Similarity model ...")
    similarity_path = models_dir / SIMILARITY_FILE
    if len(books) <= DENSE_MATRIX_LIMIT:
        similarity = cosine_similarity(tfidf_matrix).astype(np.float32)
        joblib.dump(similarity, similarity_path)
        mode = "dense_matrix"
        log(f"      stored full {similarity.shape} cosine-similarity matrix")
    else:
        if similarity_path.exists():  # remove a stale file from an earlier, smaller run
            similarity_path.unlink()
        mode = "on_demand"
        log("      catalogue is large: similarities are computed on demand from the sparse "
            "TF-IDF matrix (saves memory)")

    log(f"[5/5] Saving model files to {models_dir}")
    joblib.dump(vectorizer, models_dir / VECTORIZER_FILE)
    joblib.dump(tfidf_matrix, models_dir / TFIDF_MATRIX_FILE)
    joblib.dump(books, models_dir / BOOKS_FILE)

    rated = books[books["rating_count"] > 0]
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": Path(data_path).name,
        "ratings_file": Path(ratings_path).name if ratings_path else None,
        "n_books": int(len(books)),
        "n_terms": int(tfidf_matrix.shape[1]),
        "similarity_mode": mode,
        "tfidf_params": {"max_features": max_features, "min_df": min_df,
                         "ngram_range": list(ngram_range), "stop_words": "english",
                         "sublinear_tf": True},
        "rating_min": float(rated["rating"].min()) if len(rated) else None,
        "rating_max": float(rated["rating"].max()) if len(rated) else None,
        "preprocessing_report": report,
    }
    (models_dir / MODEL_INFO_FILE).write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log("Done. Start the app with:  streamlit run app.py")
    return info


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BookWise content-based model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Books CSV file")
    parser.add_argument("--ratings", type=Path, default=None,
                        help="Optional separate ratings CSV (ISBN + rating per row)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    data_path = args.data
    if not data_path.exists():
        if data_path == DEFAULT_DATA_PATH and SAMPLE_DATA_PATH.exists():
            print(f"WARNING: {data_path} not found. Training on the small SAMPLE file instead:\n"
                  f"         {SAMPLE_DATA_PATH}\n"
                  "         Put your real dataset at data/books.csv and re-run for real results.\n")
            data_path = SAMPLE_DATA_PATH
        else:
            print(f"ERROR: dataset not found at {data_path}.")
            return 1
    try:
        train(data_path, args.models_dir, args.ratings, args.max_features, args.min_df)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
