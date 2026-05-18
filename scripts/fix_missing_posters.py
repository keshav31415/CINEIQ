"""
Re-fetches poster_path for the ~170 movies that still have none.
Strategy:
  1. Try /movie/{tmdb_id}  (handles mis-fetched films + rate-limit retries)
  2. Fall back to /tv/{tmdb_id}  (Planet Earth, Band of Brothers, Cosmos, etc.)
Run: python scripts/fix_missing_posters.py
"""
import sys, sqlite3, time, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cineiq.config import get_config

cfg     = get_config()
API_KEY = os.environ.get('TMDB_API_KEY') or cfg['tmdb']['api_key']
DB_PATH = cfg['data']['database']

conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT movie_id, tmdb_id FROM movies "
    "WHERE tmdb_id IS NOT NULL AND (poster_path IS NULL OR poster_path = '')"
).fetchall()
conn.close()

print(f"Retrying {len(rows)} movies with missing posters...")

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'CineIQ/1.0'})


def get_with_retry(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = SESSION.get(url, params={'api_key': API_KEY}, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
        except Exception:
            time.sleep(1)
    return None


def fetch_poster(movie_id: int, tmdb_id: int) -> tuple[int, str | None]:
    # Try movie endpoint first
    data = get_with_retry(f"https://api.themoviedb.org/3/movie/{tmdb_id}")
    if data and data.get('poster_path'):
        return movie_id, data['poster_path']

    # Fall back to TV endpoint
    data = get_with_retry(f"https://api.themoviedb.org/3/tv/{tmdb_id}")
    if data and data.get('poster_path'):
        return movie_id, data['poster_path']

    return movie_id, None


results: dict[int, str] = {}
still_missing = 0

with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {pool.submit(fetch_poster, mid, tid): mid for mid, tid in rows}
    with tqdm(total=len(futures), unit='movie') as bar:
        for fut in as_completed(futures):
            mid, poster = fut.result()
            if poster:
                results[mid] = poster
            else:
                still_missing += 1
            bar.update(1)
            bar.set_postfix(found=len(results), missing=still_missing)

print(f"\nFound {len(results)} new posters  |  {still_missing} genuinely unavailable")

if results:
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "UPDATE movies SET poster_path = ? WHERE movie_id = ?",
        [(path, mid) for mid, path in results.items()]
    )
    conn.commit()
    conn.close()
    print(f"Updated {len(results)} rows in DB.")
