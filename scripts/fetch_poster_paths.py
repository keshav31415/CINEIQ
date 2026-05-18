"""
Fetches poster_path for all movies from TMDB API and stores in DB.
Uses 20 parallel threads — runs in ~8-10 min for 9700 movies.
Run: python scripts/fetch_poster_paths.py
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
    "SELECT movie_id, tmdb_id FROM movies WHERE tmdb_id IS NOT NULL"
).fetchall()
conn.close()

print(f"Fetching poster_path for {len(rows):,} movies...")

SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'CineIQ/1.0'})

def fetch_poster(movie_id: int, tmdb_id: int) -> tuple[int, str | None]:
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    try:
        r = SESSION.get(url, params={'api_key': API_KEY}, timeout=8)
        if r.status_code == 200:
            return movie_id, r.json().get('poster_path')
        if r.status_code == 429:
            time.sleep(2)
            r = SESSION.get(url, params={'api_key': API_KEY}, timeout=8)
            if r.status_code == 200:
                return movie_id, r.json().get('poster_path')
    except Exception:
        pass
    return movie_id, None


results: dict[int, str] = {}
errors = 0

with ThreadPoolExecutor(max_workers=20) as pool:
    futures = {pool.submit(fetch_poster, mid, tid): mid for mid, tid in rows}
    with tqdm(total=len(futures), unit='movie') as bar:
        for fut in as_completed(futures):
            mid, poster = fut.result()
            if poster:
                results[mid] = poster
            else:
                errors += 1
            bar.update(1)
            bar.set_postfix(found=len(results), errors=errors)

print(f"\nFetched {len(results):,} posters  |  {errors:,} missing/errors")
print("Updating database...")

conn = sqlite3.connect(DB_PATH)
conn.executemany(
    "UPDATE movies SET poster_path = ? WHERE movie_id = ?",
    [(path, mid) for mid, path in results.items()]
)
conn.commit()
conn.close()

print(f"Done. {len(results):,} poster_paths saved to DB.")
