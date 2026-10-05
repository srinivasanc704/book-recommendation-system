"""BookWise - AI-Powered Book Recommendation System (Streamlit front-end).

Run with:  streamlit run app.py

The app never trains anything. It loads the files created by
``python src/train_model.py`` and shows recommendations computed by them.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_preprocessing import genre_counts, parse_genres
from src.evaluation import evaluate_recommender
from src.recommender import BookRecommender
from src.utils import (BOOKS_FILE, DEFAULT_POPULARITY_WEIGHT, LOGO_PATH, MODELS_DIR, PROJECT_INFO,
                       BookNotFoundError, ModelNotFoundError, esc, format_rating, format_year,
                       is_missing, is_valid_url, make_placeholder_cover, truncate_text)

logger = logging.getLogger("bookwise")

st.set_page_config(page_title="BookWise", page_icon="📚", layout="wide")

PAGES = ["🏠 Home", "🔎 Book Recommender", "🔥 Popular Books", "📊 Dataset Explorer", "ℹ️ About"]
HOME, RECOMMENDER, POPULAR, EXPLORER, ABOUT = PAGES

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.bw-hero {background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
          padding: 2.2rem 2rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
.bw-hero h1 {margin: 0; font-size: 2.3rem; color: #fff;}
.bw-hero p {margin: .3rem 0 0 0; font-size: 1.1rem; opacity: .95;}
.bk-title {font-weight: 700; font-size: 1.05rem; line-height: 1.25; margin-bottom: .1rem;}
.bk-meta {opacity: .75; font-size: .9rem; margin-bottom: .35rem;}
.bk-chip {display: inline-block; background: rgba(124,58,237,.14); color: #7c3aed; border-radius: 999px;
          padding: 1px 10px; font-size: .75rem; margin: 0 4px 4px 0;}
.bk-desc {font-size: .85rem; opacity: .85; margin-top: .35rem;}
.bk-bar {background: rgba(128,128,128,.25); border-radius: 6px; height: 8px; width: 100%; margin: 2px 0 6px 0;}
.bk-bar > div {background: linear-gradient(90deg, #4f46e5, #db2777); height: 8px; border-radius: 6px;}
.bk-score {font-size: .8rem; opacity: .85;}
</style>
"""


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def show_plot(fig: Any) -> None:
    """Draw a Plotly figure full-width (works across Streamlit versions)."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.plotly_chart(fig, **kwargs)
            return
        except Exception:  # noqa: BLE001 - parameter name differs between versions
            continue


def show_table(df: pd.DataFrame) -> None:
    """Draw a dataframe full-width (works across Streamlit versions)."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.dataframe(df, hide_index=True, **kwargs)
            return
        except Exception:  # noqa: BLE001
            continue


def go_to(page: str) -> None:
    """Button callback that switches the sidebar page."""
    st.session_state["page"] = page


def model_signature() -> float:
    """Changes whenever the model is retrained, so the cached model reloads."""
    path = MODELS_DIR / BOOKS_FILE
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner="Loading the trained model ...")
def load_recommender(signature: float) -> BookRecommender:
    """Load the model once per server process (not on every rerun)."""
    return BookRecommender.load()


@st.cache_data(show_spinner=False)
def cached_stats(_rec: BookRecommender, signature: float) -> dict[str, Any]:
    return _rec.dataset_stats()


@st.cache_data(show_spinner=False)
def cached_genre_counts(_rec: BookRecommender, signature: float) -> pd.Series:
    return genre_counts(_rec.books)


def show_cover(row: pd.Series, width: int = 105) -> None:
    """Show the cover image, or a generated placeholder if there is no valid URL."""
    url = row.get("image_url")
    if is_valid_url(url):
        try:
            st.image(url.strip(), width=width)
            return
        except Exception:  # noqa: BLE001 - broken image must never break the page
            logger.warning("Could not render image %s", url)
    st.image(make_placeholder_cover(str(row.get("title", "?"))), width=width)


def book_card(row: pd.Series, rank: Optional[int] = None, similarity: Optional[float] = None,
              final_score: Optional[float] = None) -> None:
    """Render one book as a bordered card."""
    with st.container(border=True):
        image_col, text_col = st.columns([1, 3])
        with image_col:
            show_cover(row)
        with text_col:
            prefix = f"{rank}. " if rank else ""
            year = format_year(row.get("year"))
            meta = f"by {esc(row.get('author'))}" + (f" &middot; {year}" if year else "")
            html = [f"<div class='bk-title'>{prefix}{esc(row.get('title'))}</div>",
                    f"<div class='bk-meta'>{meta}</div>"]
            chips = "".join(f"<span class='bk-chip'>{esc(g)}</span>"
                            for g in parse_genres(row.get("genre"))[:3])
            if chips:
                html.append(f"<div>{chips}</div>")
            if not is_missing(row.get("rating")):
                votes = int(row.get("rating_count") or 0)
                votes_text = f" ({votes:,} ratings)" if votes > 0 else ""
                html.append(f"<div class='bk-score'>⭐ {format_rating(row.get('rating'))}{votes_text}</div>")
            if similarity is not None:
                pct = max(0.0, min(100.0, similarity * 100))
                html.append(f"<div class='bk-score'>Similarity: <b>{pct:.1f}%</b></div>"
                            f"<div class='bk-bar'><div style='width:{pct:.1f}%'></div></div>")
            if final_score is not None:
                html.append(f"<div class='bk-score'>Final score: {final_score:.3f}</div>")
            description = truncate_text(row.get("description"), 200)
            if description:
                html.append(f"<div class='bk-desc'>{esc(description)}</div>")
            st.markdown("".join(html), unsafe_allow_html=True)


def card_grid(frame: pd.DataFrame, with_scores: bool = False, numbered: bool = True) -> None:
    """Show books in a two-column grid of cards."""
    rows = list(frame.reset_index(drop=True).iterrows())
    for start in range(0, len(rows), 2):
        columns = st.columns(2)
        for col, (position, row) in zip(columns, rows[start:start + 2]):
            with col:
                book_card(row, rank=(position + 1) if numbered else None,
                          similarity=row["content_similarity"] if with_scores else None,
                          final_score=row["final_score"] if with_scores and "final_score" in row else None)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(rec: BookRecommender, signature: float) -> None:
    st.markdown("<div class='bw-hero'><h1>📚 BookWise</h1>"
                "<p>Discover your next favorite book using Machine Learning.</p></div>",
                unsafe_allow_html=True)
    st.write("BookWise is a **content-based recommender**. It turns each book's title, author, genre, "
             "publisher and description into TF-IDF vectors and finds the books whose vectors point in "
             "the most similar direction (cosine similarity). Pick a book you enjoyed and BookWise "
             "suggests what to read next.")
    stats = cached_stats(rec, signature)
    genres = cached_genre_counts(rec, signature)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Books", f"{stats['n_books']:,}")
    c2.metric("Authors", f"{stats['n_authors']:,}")
    c3.metric("Genres", f"{len(genres):,}" if len(genres) else "N/A")
    c4.metric("Average rating", format_rating(stats["avg_rating"]))
    st.markdown("### Start exploring")
    st.button("🔎 Start Exploring", type="primary", on_click=go_to, args=(RECOMMENDER,))
    st.caption("Tip: use the sidebar to open Popular Books, the Dataset Explorer or the About page.")


def page_recommender(rec: BookRecommender) -> None:
    st.title("🔎 Book Recommender")
    query = st.text_input("Search by title or author", placeholder="e.g. Harry Potter, Tolkien, Dune")
    matches = rec.search_books(query, limit=50) if query.strip() else rec.books.iloc[0:0]

    selected_label: Optional[str] = None
    if not query.strip():
        samples = rec.most_popular_books(3)["title"].tolist() if rec.has_ratings else rec.books["title"].head(3).tolist()
        st.info("Type part of a title or an author's name to begin."
                + (f" For example: {', '.join(samples)}." if samples else ""))
    elif matches.empty:
        st.warning("⚠️ Book not found. Try searching with another title.")
    else:
        st.caption(f"{len(matches)} matching book(s)" + (" (showing the first 50)" if len(matches) == 50 else ""))
        selected_label = st.selectbox("Select a book", matches["display_name"].tolist())

    n_recs = st.radio("Number of recommendations", [5, 10, 15], index=1, horizontal=True)
    use_hybrid, popularity_weight = False, 0.0
    with st.expander("⚙️ Ranking options"):
        if rec.has_ratings:
            use_hybrid = st.checkbox("Blend in rating / popularity (hybrid)", value=True)
            popularity_weight = st.slider("Popularity weight", 0.0, 1.0, DEFAULT_POPULARITY_WEIGHT, 0.05,
                                          disabled=not use_hybrid)
            st.caption(f"Final score = {1 - popularity_weight:.2f} × content similarity "
                       f"+ {popularity_weight:.2f} × popularity. Only the most content-similar "
                       "books are re-ranked, so unrelated bestsellers never appear.")
        else:
            st.caption("This dataset has no rating information, so pure content-based ranking is used.")

    if st.button("Recommend Books", type="primary", disabled=selected_label is None):
        try:
            result = rec.recommend_books(selected_label, n_recs, use_hybrid=use_hybrid,
                                         content_weight=1 - popularity_weight,
                                         popularity_weight=popularity_weight)
            st.session_state["rec_state"] = {"label": selected_label, "result": result,
                                             "hybrid": use_hybrid and rec.has_ratings}
        except BookNotFoundError:
            st.session_state.pop("rec_state", None)
            st.warning("⚠️ Book not found. Try searching with another title.")

    state = st.session_state.get("rec_state")
    if state:
        source = rec.get_book(state["label"])
        st.markdown("### Selected book")
        book_card(source)
        result: pd.DataFrame = state["result"]
        st.markdown("### Recommended books")
        if result.empty:
            st.info("No similar books were found for this title in the current dataset.")
        else:
            if len(result) < n_recs:
                st.caption(f"Only {len(result)} books share any content with this title.")
            card_grid(result, with_scores=True)
            with st.expander("Show scores as a table"):
                table = result[["rank", "title", "author", "content_similarity", "popularity_score", "final_score"]].copy()
                table["content_similarity"] = (table["content_similarity"] * 100).round(1).astype(str) + "%"
                show_table(table)


def page_popular(rec: BookRecommender) -> None:
    st.title("🔥 Popular Books")
    if not rec.has_ratings:
        st.info("This dataset has no rating information, so popularity rankings are not available. "
                "Showing the authors with the most books instead.")
        show_table(rec.top_authors(15))
        return
    max_votes = int(rec.books["rating_count"].max())
    upper = max(2, min(max_votes, 1000))
    min_ratings = st.slider("Minimum number of ratings", 1, upper, min(25, upper),
                            help="Books and authors with fewer ratings are ignored, "
                                 "so a single 5-star vote cannot dominate the ranking.")
    tab_rated, tab_popular, tab_authors = st.tabs(["⭐ Top rated", "🔥 Most popular", "✍️ Top authors"])
    with tab_rated:
        top = rec.top_rated_books(10, min_ratings)
        if top.empty:
            st.info("No books have that many ratings. Lower the minimum.")
        else:
            card_grid(top)
    with tab_popular:
        card_grid(rec.most_popular_books(10))
    with tab_authors:
        authors = rec.top_authors(10, min_ratings)
        if authors.empty:
            st.info("No authors have that many ratings. Lower the minimum.")
        else:
            authors = authors.rename(columns={"avg_rating": "average rating", "total_ratings": "total ratings"})
            authors["average rating"] = authors["average rating"].round(2)
            show_table(authors)


def page_explorer(rec: BookRecommender, signature: float) -> None:
    st.title("📊 Dataset Explorer")
    stats = cached_stats(rec, signature)
    books = rec.books
    cols = st.columns(5)
    cols[0].metric("Books", f"{stats['n_books']:,}")
    cols[1].metric("Authors", f"{stats['n_authors']:,}")
    cols[2].metric("Publishers", f"{stats['n_publishers']:,}" if stats["n_publishers"] else "N/A")
    cols[3].metric("Rated books", f"{stats['n_rated_books']:,}")
    years = f"{stats['year_min']}–{stats['year_max']}" if stats["year_min"] else "N/A"
    cols[4].metric("Years", years)

    tabs = st.tabs(["Ratings", "Publication years", "Top authors", "Top genres", "Model & evaluation", "Browse data"])
    with tabs[0]:
        rated = books[books["rating_count"] > 0]
        if rated.empty:
            st.info("No rating data in this dataset.")
        else:
            fig = px.histogram(rated, x="rating", nbins=20, template="plotly_white",
                               title="Distribution of average book ratings",
                               labels={"rating": "Average rating", "count": "Books"})
            fig.update_layout(yaxis_title="Number of books", bargap=0.05)
            show_plot(fig)
    with tabs[1]:
        year_data = books.dropna(subset=["year"]).copy()
        if year_data.empty:
            st.info("No publication-year data in this dataset.")
        else:
            year_data["year"] = year_data["year"].astype(int)
            fig = px.histogram(year_data, x="year", nbins=40, template="plotly_white",
                               title="Books by publication year")
            fig.update_layout(yaxis_title="Number of books", xaxis_title="Year", bargap=0.05)
            show_plot(fig)
    with tabs[2]:
        counts = (books[books["author"] != "Unknown Author"].groupby("author").size()
                  .sort_values(ascending=False).head(15).reset_index(name="books"))
        fig = px.bar(counts, x="books", y="author", orientation="h", template="plotly_white",
                     title="Authors with the most books")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
        show_plot(fig)
    with tabs[3]:
        genres = cached_genre_counts(rec, signature).head(15).reset_index()
        if genres.empty:
            st.info("No genre data in this dataset.")
        else:
            genres.columns = ["genre", "books"]
            fig = px.bar(genres, x="books", y="genre", orientation="h", template="plotly_white",
                         title="Most common genres")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
            show_plot(fig)
    with tabs[4]:
        info = rec.info
        if info:
            c1, c2, c3 = st.columns(3)
            c1.metric("TF-IDF terms", f"{info.get('n_terms', 0):,}")
            c2.metric("Similarity mode", str(info.get("similarity_mode", "?")).replace("_", " "))
            c3.metric("Trained on", str(info.get("data_file", "?")))
            st.caption(f"Model created: {info.get('created_at', 'unknown')}")
            with st.expander("Data-cleaning report from training"):
                st.json(info.get("preprocessing_report", {}))
        st.markdown("**Offline evaluation.** There are no real user clicks or ratings-per-user to "
                    "compare against, so accuracy / true Precision@K cannot be computed. These checks "
                    "use the catalogue itself; the 'agreement' numbers are *proxies*, not accuracy.")
        k = st.select_slider("K (list length)", options=[5, 10, 15], value=10)
        if st.button("Run evaluation on a random sample of 100 books"):
            with st.spinner("Evaluating ..."):
                st.session_state["eval_result"] = evaluate_recommender(rec, k=k, sample_size=100)
        result = st.session_state.get("eval_result")
        if result:
            def pct(v: Optional[float]) -> str:
                return "N/A" if v is None else f"{v * 100:.1f}%"
            rows = [("Queries evaluated", str(result["queries_evaluated"])),
                    ("Average similarity of the #1 recommendation", pct(result["avg_top1_similarity"])),
                    (f"Average similarity across the top {result['k']}", pct(result["avg_topk_similarity"])),
                    ("Intra-list diversity (higher = more varied lists)", pct(result["intra_list_diversity"])),
                    ("Catalogue coverage (share of books ever recommended)", pct(result["catalogue_coverage"])),
                    ("Proxy: recommendations by the same author", pct(result["same_author_rate_proxy"])),
                    ("Proxy: recommendations sharing a genre", pct(result["shared_genre_rate_proxy"]))]
            show_table(pd.DataFrame(rows, columns=["Metric", "Value"]))
    with tabs[5]:
        needle = st.text_input("Filter the table (title or author)", key="browse_filter")
        frame = rec.search_books(needle, limit=500) if needle.strip() else books.head(500)
        show_table(frame[["title", "author", "year", "publisher", "genre", "rating", "rating_count"]])


def page_about(rec: BookRecommender) -> None:
    st.title("ℹ️ About BookWise")
    st.markdown("### Project")
    st.write("BookWise is a college Machine Learning mini-project that recommends books similar to one "
             "the user selects. Recommendations come from a trained, reusable TF-IDF + cosine-similarity "
             "pipeline saved in the `models/` folder.")
    st.markdown("### Technology stack")
    st.markdown("Python · Pandas · NumPy · Scikit-learn · SciPy · Streamlit · Plotly · Pillow · Joblib")
    st.markdown("### Machine-learning algorithm")
    st.write("**TF-IDF** gives each word a weight: high if the word is frequent in one book but rare in "
             "the catalogue. **Cosine similarity** measures the angle between two TF-IDF vectors, so "
             "long and short descriptions can be compared fairly.")
    st.markdown("### Recommendation method")
    st.write("Content-based filtering: the selected book's vector is compared with every other book and "
             "the closest ones are returned. When rating data exists, an optional hybrid score "
             "(default 70% similarity + 30% popularity) re-ranks the best candidates.")
    st.markdown("### Dataset")
    info = rec.info
    stats = rec.dataset_stats()
    st.write(f"Currently loaded: **{stats['n_books']:,} books** from `{info.get('data_file', 'unknown')}`."
             " Any CSV with at least a title column works; other columns are optional.")
    st.markdown("### Future improvements")
    st.markdown("- Collaborative filtering and user profiles\n- Personalised, login-based recommendations\n"
                "- Feedback buttons (like / dislike) to improve ranking\n"
                "- Sentence-embedding or LLM-based explanations\n- Cloud deployment")
    st.markdown("### Student / project information")
    st.markdown("\n".join(f"- **{key}:** {esc(value)}" for key, value in PROJECT_INFO.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("page", HOME)

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=72)
        st.markdown("## 📚 BookWise")
        st.caption("AI-Powered Book Recommendation System")
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

    signature = model_signature()
    try:
        rec = load_recommender(signature)
    except ModelNotFoundError as exc:
        st.title("📚 BookWise")
        st.error("⚠️ The trained model was not found.")
        st.write(str(exc))
        st.code("python src/train_model.py", language="bash")
        st.stop()
        return
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while loading the model")
        st.error("⚠️ The model could not be loaded. Please re-run `python src/train_model.py`.")
        st.stop()
        return

    with st.sidebar:
        st.divider()
        st.caption(f"{rec.n_books:,} books loaded")

    pages: dict[str, Callable[[], None]] = {
        HOME: lambda: page_home(rec, signature),
        RECOMMENDER: lambda: page_recommender(rec),
        POPULAR: lambda: page_popular(rec),
        EXPLORER: lambda: page_explorer(rec, signature),
        ABOUT: lambda: page_about(rec),
    }
    try:
        pages[page]()
    except Exception:  # noqa: BLE001 - normal users should never see a traceback
        logger.exception("Unexpected error on page %s", page)
        st.error("⚠️ Something went wrong while showing this page. Please try another page or "
                 "re-run `python src/train_model.py`. Details are in the terminal log.")


main()
