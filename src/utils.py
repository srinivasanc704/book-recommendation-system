"""Shared constants, exceptions and small helper functions.

Everything that more than one module needs lives here so that paths, file
names and default weights are defined in exactly one place.
"""
from __future__ import annotations

import hashlib
import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Paths (all relative to the project root, never absolute)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_DATA_PATH = DATA_DIR / "books.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_books.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"

# --------------------------------------------------------------------------
# Saved-model file names (written by train_model.py, read by recommender.py)
# --------------------------------------------------------------------------
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE = "tfidf_matrix.pkl"
SIMILARITY_FILE = "similarity_matrix.pkl"
BOOKS_FILE = "books.pkl"
MODEL_INFO_FILE = "model_info.json"

# --------------------------------------------------------------------------
# Model / recommendation defaults
# --------------------------------------------------------------------------
# Hybrid score = CONTENT_WEIGHT * similarity + POPULARITY_WEIGHT * popularity
DEFAULT_CONTENT_WEIGHT = 0.70
DEFAULT_POPULARITY_WEIGHT = 0.30
# Below this many books we also store the full dense similarity matrix.
# Above it we compute one row of similarities on demand (memory efficient).
DENSE_MATRIX_LIMIT = 3000
# How many content-similar candidates are re-ranked by the hybrid score.
HYBRID_CANDIDATE_POOL = 200
# How much each text field counts inside the combined "soup" (repetition = weight).
FEATURE_WEIGHTS = {"title": 1, "author": 2, "genre": 2, "publisher": 1, "description": 1}

# Fill in your own details; shown on the About page.
PROJECT_INFO = {
    "Project title": "BookWise - AI-Powered Book Recommendation System",
    "Student name": "<Your Name>",
    "Register / Roll no.": "<Your Roll No.>",
    "Programme": "B.Tech - Artificial Intelligence and Data Science (Semester V)",
    "Institution": "<Your College>",
    "Guide / Faculty": "<Faculty Name>",
}


# --------------------------------------------------------------------------
# Exceptions (the UI catches these and shows friendly messages)
# --------------------------------------------------------------------------
class BookWiseError(Exception):
    """Base class for all project-specific errors."""


class DatasetError(BookWiseError):
    """The dataset is missing, unreadable or has no usable title column."""


class ModelNotFoundError(BookWiseError):
    """Trained model files are missing; run ``python src/train_model.py``."""


class BookNotFoundError(BookWiseError):
    """The requested book is not in the catalogue."""


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------
def is_missing(value: Any) -> bool:
    """Return True for None, NaN, pandas NA and empty / 'nan' strings."""
    if value is None:
        return True
    try:
        if value != value:  # NaN is the only value not equal to itself
            return True
    except (TypeError, ValueError):
        pass
    if value is pd.NA:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "n/a", "na", "<na>"}


def format_year(value: Any) -> str:
    """'1997' for valid years, '' otherwise."""
    if is_missing(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def format_rating(value: Any, decimals: int = 2) -> str:
    """'4.32' for numeric ratings, 'N/A' when missing."""
    if is_missing(value):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def truncate_text(text: Any, max_chars: int = 220) -> str:
    """Shorten text to ``max_chars`` on a word boundary."""
    if is_missing(text):
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "..."


def esc(value: Any) -> str:
    """HTML-escape any value (titles can contain < > & characters)."""
    return html.escape("" if is_missing(value) else str(value))


def is_valid_url(value: Any) -> bool:
    """True only for http(s) URLs."""
    return isinstance(value, str) and bool(re.match(r"^https?://", value.strip(), re.I))


@lru_cache(maxsize=512)
def make_placeholder_cover(title: str, size: tuple[int, int] = (120, 180)) -> Image.Image:
    """Draw a simple coloured cover with the title's initial (used when no image URL)."""
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()
    colour = tuple(int(digest[i:i + 2], 16) // 2 + 40 for i in (0, 2, 4))
    image = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(image)
    try:
        font: Optional[ImageFont.ImageFont] = ImageFont.load_default(size=56)
    except TypeError:  # very old Pillow without the size argument
        font = ImageFont.load_default()
    letter = (title.strip()[:1] or "?").upper()
    draw.text((size[0] / 2, size[1] / 2), letter, fill="white", font=font, anchor="mm")
    return image
