"""
Export reviews + movie metadata from the local DB to a CSV for Kaggle upload.

Output: data/exports/cineiq_reviews.csv
Run:    python scripts/export_reviews_for_kaggle.py
"""
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sqlalchemy import text
from cineiq.db import get_engine

OUT_PATH = Path('data/exports/cineiq_reviews.csv')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = get_engine()

print("Fetching reviews + movie metadata...")
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT
            m.movie_id,
            m.imdb_id,
            m.title,
            r.id        AS review_id,
            r.review_text,
            r.source
        FROM reviews r
        JOIN movies m ON r.imdb_id = m.imdb_id
        WHERE r.review_text IS NOT NULL
          AND TRIM(r.review_text) != ''
        ORDER BY m.movie_id, r.id
    """)).fetchall()

print(f"  {len(rows):,} reviews across movies.")

with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'imdb_id', 'title', 'review_id', 'review_text', 'source'])
    writer.writerows(rows)

size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"  Saved to {OUT_PATH}  ({size_mb:.1f} MB)")
print(f"  Ready to upload as a Kaggle dataset.")
