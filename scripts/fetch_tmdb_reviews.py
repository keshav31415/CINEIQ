"""
Fetch movie reviews from the TMDB API and store them in the reviews table.

Run this after 01_ingest_data.py and before 04_precompute_sentiment.py.
It is safe to re-run — already-fetched movies are skipped automatically.

Usage:
    python scripts/fetch_tmdb_reviews.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cineiq.db import get_engine
from cineiq.data.ingest_tmdb_reviews import ingest_tmdb_reviews

if __name__ == "__main__":
    print("=== Fetching TMDB Reviews ===")
    engine = get_engine()
    ingest_tmdb_reviews(engine)
