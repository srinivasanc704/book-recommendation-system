"""End-to-end tests. Run with:  python -m unittest discover tests -v

They build a throw-away model in a temporary folder, so your real models/
directory is never touched.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import clean_text_for_model, detect_columns, preprocess_books  # noqa: E402
from src.evaluation import precision_at_k, recall_at_k  # noqa: E402
from src.recommender import BookRecommender  # noqa: E402
from src.train_model import train  # noqa: E402
from src.utils import SAMPLE_DATA_PATH, BookNotFoundError, DatasetError  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def test_column_detection_is_flexible(self):
        df = pd.DataFrame(columns=["book_title", "AUTHORS", "Image-URL-M", "average_rating"])
        found = detect_columns(df)
        self.assertEqual(found["title"], ["book_title"])
        self.assertIn("author", found)
        self.assertIn("rating", found)

    def test_missing_title_column_raises(self):
        with self.assertRaises(DatasetError):
            preprocess_books(pd.DataFrame({"foo": [1, 2]}))

    def test_cleaning_and_duplicates(self):
        raw = pd.DataFrame({"Book-Title": ["Dune", " dune", None, "Emma"],
                            "Book-Author": ["Frank Herbert", "Frank Herbert", "X", None],
                            "Year-Of-Publication": [1965, 0, 2000, 1815]})
        books, report = preprocess_books(raw)
        self.assertEqual(len(books), 2)
        self.assertEqual(report["invalid_removed"], 1)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertFalse(report["has_ratings"])
        self.assertEqual(books.loc[books["title"] == "Emma", "author"].iloc[0], "Unknown Author")

    def test_text_cleaning_keeps_digits_and_strips_accents(self):
        self.assertEqual(clean_text_for_model("  1984!! García  "), "1984 garcia")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        train(SAMPLE_DATA_PATH, Path(cls.tmp.name), verbose=False)
        cls.rec = BookRecommender.load(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_book_never_recommends_itself(self):
        recs = self.rec.recommend_books("The Hobbit", 10, use_hybrid=False)
        self.assertNotIn("The Hobbit", recs["title"].tolist())

    def test_scores_are_sorted_and_real(self):
        recs = self.rec.recommend_books("The Hobbit", 5, use_hybrid=False)
        sims = recs["content_similarity"].to_numpy()
        self.assertTrue(np.all(sims[:-1] >= sims[1:]))
        self.assertTrue(np.all((sims > 0) & (sims <= 1)))

    def test_similar_book_ranks_high(self):
        recs = self.rec.recommend_books("Harry Potter and the Philosopher's Stone", 2, use_hybrid=False)
        self.assertTrue(all("Harry Potter" in t for t in recs["title"]))

    def test_unknown_book_raises(self):
        with self.assertRaises(BookNotFoundError):
            self.rec.recommend_books("this book does not exist")

    def test_search_is_partial_and_case_insensitive(self):
        self.assertGreaterEqual(len(self.rec.search_books("HARRY pot")), 3)
        self.assertGreaterEqual(len(self.rec.search_books("tolkien")), 3)
        self.assertTrue(self.rec.search_books("   ").empty)

    def test_hybrid_changes_scores_but_not_candidates(self):
        pure = self.rec.recommend_books("Dune", 5, use_hybrid=False)
        hybrid = self.rec.recommend_books("Dune", 5, use_hybrid=True, popularity_weight=0.3)
        self.assertEqual(len(hybrid), 5)
        self.assertTrue((hybrid["content_similarity"] > 0).all())
        self.assertFalse(pure["final_score"].equals(hybrid["final_score"]))


class LargeCatalogueTests(unittest.TestCase):
    """Book-Crossing style data with a separate ratings file and the on-demand mode."""

    def test_on_demand_mode_and_external_ratings(self):
        rng = np.random.default_rng(0)
        words = [f"word{i}" for i in range(300)]
        n = 3200  # above DENSE_MATRIX_LIMIT
        books = pd.DataFrame({
            "ISBN": [f"{i:010d}" for i in range(n)],
            "Book-Title": [" ".join(rng.choice(words, 4)) + f" vol {i}" for i in range(n)],
            "Book-Author": [f"Author {i % 400}" for i in range(n)],
            "Year-Of-Publication": rng.integers(1950, 2020, n),
            "Publisher": [f"Pub {i % 50}" for i in range(n)],
        })
        ratings = pd.DataFrame({"User-ID": rng.integers(0, 500, 20000),
                                "ISBN": [f"{i:010d}" for i in rng.integers(0, n, 20000)],
                                "Book-Rating": rng.integers(0, 11, 20000)})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            books.to_csv(tmp_path / "b.csv", index=False, sep=";")
            ratings.to_csv(tmp_path / "r.csv", index=False)
            info = train(tmp_path / "b.csv", tmp_path / "m", tmp_path / "r.csv", verbose=False)
            self.assertEqual(info["similarity_mode"], "on_demand")
            self.assertFalse((tmp_path / "m" / "similarity_matrix.pkl").exists())
            rec = BookRecommender.load(tmp_path / "m")
            self.assertTrue(rec.has_ratings)
            recs = rec.recommend_books(int(rec.books.iloc[0]["book_id"]), 5)
            self.assertEqual(len(recs), 5)
            self.assertTrue((rec.top_rated_books(5, 5)["rating_count"] >= 5).all())


class MetricTests(unittest.TestCase):
    def test_precision_recall(self):
        self.assertAlmostEqual(precision_at_k([1, 2, 3, 4], [2, 4, 9], 4), 0.5)
        self.assertAlmostEqual(recall_at_k([1, 2, 3, 4], [2, 4, 9], 4), 2 / 3)


if __name__ == "__main__":
    unittest.main()
