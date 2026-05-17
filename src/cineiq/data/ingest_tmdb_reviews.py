import os
import time
import httpx
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import Session
from cineiq.data.models import Review

load_dotenv()

TMDB_BASE = "https://api.themoviedb.org/3"
_API_KEY = None


def _get_api_key():
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.getenv("TMDB_API_KEY")
        if not _API_KEY:
            raise RuntimeError("TMDB_API_KEY not set in environment / .env file")
    return _API_KEY


def _fetch_reviews_for_movie(client: httpx.Client, tmdb_id: int) -> list[str]:
    """Fetch all review pages for one movie. Returns list of review text strings."""
    reviews = []
    page = 1

    while True:
        resp = client.get(
            f"{TMDB_BASE}/movie/{tmdb_id}/reviews",
            params={"api_key": _get_api_key(), "page": page},
            timeout=10,
        )

        if resp.status_code == 429:
            time.sleep(2)
            continue
        if resp.status_code != 200:
            break

        data = resp.json()
        for r in data.get("results", []):
            content = r.get("content", "").strip()
            if content:
                reviews.append(content)

        if page >= data.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.1)

    return reviews


def _migrate_source_column(engine):
    """Add source column to existing DB if not present, backfill aclimdb rows."""
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE reviews ADD COLUMN source TEXT"))
            conn.execute(text("UPDATE reviews SET source = 'aclimdb' WHERE source IS NULL"))
            conn.commit()
            print("  Migrated: added 'source' column and tagged existing reviews as 'aclimdb'.")
        except Exception:
            pass  # Column already exists


def ingest_tmdb_reviews(engine):
    """
    Fetch reviews from the TMDB API for all movies that have a tmdb_id
    but no TMDB reviews yet, and insert them into the reviews table.
    """
    _migrate_source_column(engine)

    # Movies with tmdb_id that haven't been fetched from TMDB yet
    with engine.connect() as conn:
        movies = conn.execute(text("""
            SELECT m.tmdb_id, m.imdb_id, m.title
            FROM movies m
            WHERE m.tmdb_id IS NOT NULL
              AND m.imdb_id IS NOT NULL
              AND m.imdb_id NOT IN (
                  SELECT DISTINCT imdb_id FROM reviews WHERE source = 'tmdb'
              )
            ORDER BY m.movie_id
        """)).fetchall()

    total = len(movies)
    print(f"  Fetching TMDB reviews for {total} movies (skipping already-fetched)...")

    movies_with_reviews = 0
    total_inserted = 0

    with httpx.Client() as client:
        for i, (tmdb_id, imdb_id, title) in enumerate(movies, 1):
            texts = _fetch_reviews_for_movie(client, tmdb_id)

            if texts:
                rows = [
                    {
                        "imdb_id": imdb_id,
                        "review_text": t,
                        "label": None,
                        "vader_compound": None,
                        "source": "tmdb",
                    }
                    for t in texts
                ]
                with Session(engine) as session:
                    session.bulk_insert_mappings(Review, rows)
                    session.commit()
                movies_with_reviews += 1
                total_inserted += len(texts)

            if i % 500 == 0:
                print(f"  [{i}/{total}] {movies_with_reviews} movies with reviews, "
                      f"{total_inserted} reviews so far...")

            time.sleep(0.25)  # stay well within TMDB's 40 req/10s limit

    print(f"\n  Done. {total_inserted} TMDB reviews ingested for {movies_with_reviews} movies.")
    print(f"  {total - movies_with_reviews} movies had no TMDB reviews.")
    return total_inserted
