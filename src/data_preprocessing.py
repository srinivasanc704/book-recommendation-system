"""Data loading, cleaning and feature engineering.

The goal of this module is to turn *any* reasonable book CSV into one clean
table with a fixed set of column names, so the rest of the project never has
to care what the original columns were called.

Canonical columns produced
--------------------------
book_id, isbn, title, author, year, publisher, genre, description, image_url,
rating, rating_count, rating_sum, popularity_score, display_name,
search_text, soup (the combined text that TF-IDF will vectorise)
"""
from __future__ import annotations

import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import FEATURE_WEIGHTS, DatasetError, is_missing

# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------
# Canonical name -> possible original names (compared after normalisation,
# i.e. lowercase with spaces / underscores / hyphens removed). The order is
# the priority order when several columns map to the same canonical name.
COLUMN_ALIASES: dict[str, list[str]] = {
    "isbn": ["ISBN", "isbn13", "isbn10"],
    "title": ["Book-Title", "title", "book_title", "original_title", "book name", "name"],
    "author": ["Book-Author", "author", "authors", "writer"],
    "year": ["Year-Of-Publication", "year", "publication_year", "original_publication_year",
             "published_year", "pub_year"],
    "publisher": ["Publisher", "publishing_house"],
    "image_url": ["Image-URL-L", "Image-URL-M", "Image-URL-S", "image_url", "image", "thumbnail",
                  "cover", "cover_url", "small_image_url"],
    "rating": ["Book-Rating", "average_rating", "avg_rating", "rating"],
    "rating_count": ["ratings_count", "rating_count", "num_ratings", "number_of_ratings",
                     "work_ratings_count"],
    "description": ["Description", "summary", "synopsis", "plot", "about", "book_desc"],
    "genre": ["Genre", "genres", "category", "categories", "subjects", "subject", "shelves"],
}
TEXT_COLUMNS = ["isbn", "title", "author", "publisher", "genre", "description", "image_url"]
MIN_VALID_YEAR = 1000


def _normalise_name(name: str) -> str:
    """'Book-Title' / 'book_title' / 'Book Title' all become 'booktitle'."""
    return re.sub(r"[\s_\-]+", "", str(name).lower())


def detect_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each canonical column to the original columns that match it."""
    lookup = {_normalise_name(col): col for col in df.columns}
    found: dict[str, list[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        matches: list[str] = []
        for alias in aliases:
            original = lookup.get(_normalise_name(alias))
            if original is not None and original not in matches:  # keep priority order, no repeats
                matches.append(original)
        if matches:
            found[canonical] = matches
    return found


# --------------------------------------------------------------------------
# Reading CSV files
# --------------------------------------------------------------------------
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV with automatic delimiter and encoding detection."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                sample = handle.read(20000)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            return pd.read_csv(path, sep=delimiter, encoding=encoding,
                               on_bad_lines="skip", low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise DatasetError(f"Could not read {path.name}: {last_error}")


# --------------------------------------------------------------------------
# Text cleaning helpers
# --------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces / tabs / newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def clean_display_text(value: Any) -> Optional[str]:
    """Light cleaning for text we SHOW to users (keeps case and punctuation).

    Removes HTML tags and stray whitespace; returns None for empty values.
    """
    if is_missing(value):
        return None
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = normalize_whitespace(text)
    return text or None


def clean_text_for_model(value: Any) -> str:
    """Heavier cleaning for text the MODEL reads.

    lowercase -> strip accents -> remove punctuation -> normalise whitespace.
    Digits are kept (e.g. the title '1984') and no words are dropped here;
    stop-words are handled later by TfidfVectorizer.
    """
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def parse_genres(value: Any) -> list[str]:
    """Turn "['Fantasy', 'Young Adult']", "Fantasy|Adventure" or "Fantasy, Adventure"
    into a clean list like ['Fantasy', 'Adventure']."""
    if is_missing(value):
        return []
    text = re.sub(r"[\[\]'\"]", "", str(value))
    parts = re.split(r"[,|;/]", text)
    return [normalize_whitespace(p).title() for p in parts if normalize_whitespace(p)]


def genre_tokens(genres: list[str]) -> str:
    """'Science Fiction' -> 'science_fiction' so a multi-word genre stays ONE token."""
    return " ".join(clean_text_for_model(g).replace(" ", "_") for g in genres)


# --------------------------------------------------------------------------
# Standardising the raw table
# --------------------------------------------------------------------------
def standardize_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Rename / merge the raw columns into the canonical column names."""
    found = detect_columns(raw)
    if "title" not in found:
        raise DatasetError(
            "No title column found. Expected something like 'Book-Title' or 'title'. "
            f"Columns in the file: {list(raw.columns)}"
        )
    std = pd.DataFrame(index=raw.index)
    for canonical in COLUMN_ALIASES:
        sources = found.get(canonical)
        if not sources:
            std[canonical] = np.nan
            continue
        # If several source columns exist (e.g. Image-URL-L / M / S) take the
        # first non-empty value in priority order.
        combined = raw[sources[0]]
        for extra in sources[1:]:
            combined = combined.where(combined.notna() & (combined.astype(str).str.strip() != ""),
                                      raw[extra])
        std[canonical] = combined

    for col in TEXT_COLUMNS:
        std[col] = std[col].map(clean_display_text)
    std["isbn"] = std["isbn"].map(lambda v: v.upper().replace("-", "") if isinstance(v, str) else v)

    current_year = datetime.now().year
    year = pd.to_numeric(std["year"], errors="coerce")
    std["year"] = year.where((year >= MIN_VALID_YEAR) & (year <= current_year + 1))
    std["rating"] = pd.to_numeric(std["rating"], errors="coerce")
    std["rating_count"] = pd.to_numeric(std["rating_count"], errors="coerce")
    return std, found


def standardize_ratings(ratings_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an external ratings file (ISBN + rating per row) to one row per ISBN."""
    found = detect_columns(ratings_raw)
    if "isbn" not in found or "rating" not in found:
        raise DatasetError(
            "The ratings file needs an ISBN column and a rating column "
            f"(found: {list(ratings_raw.columns)})."
        )
    tmp = pd.DataFrame({
        "isbn": ratings_raw[found["isbn"][0]].map(clean_display_text).map(
            lambda v: v.upper().replace("-", "") if isinstance(v, str) else v),
        "rating": pd.to_numeric(ratings_raw[found["rating"][0]], errors="coerce"),
    })
    tmp = tmp.dropna(subset=["isbn", "rating"])
    tmp = tmp[tmp["rating"] > 0]  # 0 means "implicit / no rating" in Book-Crossing
    grouped = tmp.groupby("isbn")["rating"].agg(ext_sum="sum", ext_count="count").reset_index()
    return grouped


# --------------------------------------------------------------------------
# Main preprocessing entry point
# --------------------------------------------------------------------------
def preprocess_books(raw: pd.DataFrame, ratings_raw: Optional[pd.DataFrame] = None
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw dataset and return ``(books, report)``.

    ``books`` has exactly one row per distinct (title, author) pair.
    ``report`` is a dict with counts that train_model.py saves for transparency.
    """
    rows_in = len(raw)
    std, found = standardize_columns(raw)

    # Remove invalid records: a book without a title cannot be recommended.
    valid = std["title"].notna()
    invalid_removed = int((~valid).sum())
    std = std[valid].copy()

    # ---- ratings: build (rating_sum, rating_count) per row -----------------
    if ratings_raw is not None:
        ext = standardize_ratings(ratings_raw)
        std = std.merge(ext, on="isbn", how="left")
        std["rating_sum"] = std["ext_sum"]
        std["rating_count"] = std["ext_count"]
        std = std.drop(columns=["ext_sum", "ext_count"])
        has_ratings = bool(std["rating_count"].fillna(0).sum() > 0)
    elif "rating" in found:
        std.loc[std["rating"] <= 0, "rating"] = np.nan  # 0 = "no rating"
        if "rating_count" in found:  # rows are already averages with a vote count
            std["rating_sum"] = std["rating"] * std["rating_count"]
        else:  # one row per individual rating
            std["rating_count"] = std["rating"].notna().astype(float)
            std["rating_sum"] = std["rating"]
        has_ratings = bool(std["rating_count"].fillna(0).sum() > 0)
    else:
        std["rating_sum"] = np.nan
        std["rating_count"] = np.nan
        has_ratings = False

    # ---- duplicate removal: same title + author = same book ----------------
    std["_key"] = std["title"].map(clean_text_for_model) + "||" + std["author"].map(clean_text_for_model)
    n_before = std["_key"].nunique() if len(std) else 0
    aggregations: dict[str, str] = {c: "first" for c in
                                    ["isbn", "title", "author", "year", "publisher",
                                     "genre", "description", "image_url"]}
    aggregations.update({"rating_sum": "sum", "rating_count": "sum"})
    books = std.groupby("_key", sort=False).agg(aggregations).reset_index(drop=True)
    duplicates_removed = len(std) - len(books)
    if n_before != len(books):  # defensive; should never happen
        raise DatasetError("Internal error while removing duplicates.")

    # ---- final tidy-up ----------------------------------------------------
    books["author"] = books["author"].fillna("Unknown Author")
    for col in ["publisher", "genre", "description"]:
        books[col] = books[col].fillna("")
    if has_ratings:
        counts = books["rating_count"].fillna(0)
        books["rating_count"] = counts.astype(int)
        books["rating"] = np.where(counts > 0, books["rating_sum"] / counts.replace(0, np.nan), np.nan)
    else:
        books["rating"] = np.nan
        books["rating_count"] = 0
    books = books.reset_index(drop=True)
    books.insert(0, "book_id", np.arange(len(books)))
    books["year"] = books["year"].astype("Int64")

    books = add_display_columns(books)
    books = add_popularity_features(books, has_ratings)
    books = add_soup(books)

    report = {
        "rows_in": int(rows_in),
        "invalid_removed": invalid_removed,
        "duplicates_removed": int(duplicates_removed),
        "rows_out": int(len(books)),
        "columns_detected": {k: v for k, v in found.items()},
        "columns_missing": [c for c in COLUMN_ALIASES if c not in found],
        "external_ratings_used": ratings_raw is not None,
        "has_ratings": has_ratings,
        "has_genre": bool((books["genre"] != "").any()),
        "has_description": bool((books["description"] != "").any()),
        "has_images": bool(books["image_url"].notna().any()),
    }
    if len(books) == 0:
        raise DatasetError("No valid books remain after cleaning the dataset.")
    return books, report


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def add_display_columns(books: pd.DataFrame) -> pd.DataFrame:
    """Add the label shown in dropdowns and a lowercase string used by search."""
    books = books.copy()
    labels = books["title"] + " \u2014 " + books["author"]
    dup = labels.duplicated(keep=False)
    labels = labels.where(~dup, labels + " (#" + books["book_id"].astype(str) + ")")
    books["display_name"] = labels
    books["search_text"] = (books["title"] + " " + books["author"]).map(clean_text_for_model)
    return books


def add_popularity_features(books: pd.DataFrame, has_ratings: bool) -> pd.DataFrame:
    """Add ``popularity_score`` in [0, 1] using a Bayesian weighted rating.

    weighted = v/(v+m) * R + m/(v+m) * C
      R = the book's average rating, v = its number of ratings,
      C = mean rating of all books, m = the typical (median) number of ratings.
    A book with very few ratings is pulled towards the global mean C, so one
    5-star vote cannot beat a book with thousands of 4.5-star votes.
    """
    books = books.copy()
    if not has_ratings:
        books["popularity_score"] = 0.0
        return books
    books["popularity_score"] = minmax(weighted_rating(books, m=None))
    return books


def weighted_rating(books: pd.DataFrame, m: Optional[float] = None) -> pd.Series:
    """Bayesian weighted rating (see ``add_popularity_features``)."""
    rated = books[(books["rating"].notna()) & (books["rating_count"] > 0)]
    if rated.empty:
        return pd.Series(0.0, index=books.index)
    global_mean = float(rated["rating"].mean())
    if m is None:
        m = max(1.0, float(rated["rating_count"].median()))
    votes = books["rating_count"].fillna(0).astype(float)
    avg = books["rating"].fillna(global_mean).astype(float)
    return (votes / (votes + m)) * avg + (m / (votes + m)) * global_mean


def minmax(series: pd.Series) -> pd.Series:
    """Scale a series to the range [0, 1]."""
    span = series.max() - series.min()
    if not np.isfinite(span) or span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def add_soup(books: pd.DataFrame) -> pd.DataFrame:
    """Combine the text columns into one 'soup' string per book.

    Each field is repeated according to FEATURE_WEIGHTS (e.g. author and genre
    twice) so that they influence TF-IDF more than a single mention would.
    """
    books = books.copy()
    parts = {
        "title": books["title"].map(clean_text_for_model),
        "author": books["author"].map(clean_text_for_model).where(books["author"] != "Unknown Author", ""),
        "genre": books["genre"].map(lambda g: genre_tokens(parse_genres(g))),
        # one token per publisher ('tor_books'), so the generic word "books" is not a match
        "publisher": books["publisher"].map(lambda v: clean_text_for_model(v).replace(" ", "_")),
        "description": books["description"].map(clean_text_for_model),
    }
    decade = books["year"].map(lambda y: f"decade_{int(y) // 10 * 10}" if pd.notna(y) else "")
    soup = decade
    for name, series in parts.items():
        repeat = FEATURE_WEIGHTS.get(name, 1)
        soup = soup + (" " + series) * repeat
    books["soup"] = soup.map(normalize_whitespace)
    return books


def genre_counts(books: pd.DataFrame) -> pd.Series:
    """Number of books per individual genre (multi-genre cells are split)."""
    exploded = books["genre"].map(parse_genres).explode().dropna()
    return exploded.value_counts()
