import re
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select
from cineiq.data.models import Review, Movie
from cineiq.config import get_config


def _parse_aclimdb_split(split_dir: Path, url_files: dict, valid_imdb_ids: set):
    """
    Parse one split (train or test) of the aclImdb dataset.
    
    Args:
        split_dir: Path to aclImdb/train or aclImdb/test
        url_files: dict mapping sentiment label -> list of URL strings
        valid_imdb_ids: set of imdb_ids present in our movies table
    
    Returns:
        list of dicts ready for bulk insert, and count of skipped reviews
    """
    reviews = []
    skipped = 0

    for label_dir, label_name in [('pos', 'positive'), ('neg', 'negative')]:
        urls = url_files[label_dir]
        review_dir = split_dir / label_dir

        if not review_dir.exists():
            continue

        for review_file in sorted(review_dir.glob('*.txt')):
            # Filename format: [id]_[rating].txt (e.g., 200_8.txt)
            parts = review_file.stem.split('_')
            if len(parts) != 2:
                continue

            file_id = int(parts[0])
            user_rating = int(parts[1])

            # Map file_id to the URL line to get the imdb_id
            if file_id >= len(urls):
                skipped += 1
                continue

            url = urls[file_id]
            match = re.search(r'/title/(tt\d+)/', url)
            if not match:
                skipped += 1
                continue

            imdb_id = match.group(1)

            # Only keep reviews for movies in our database
            if imdb_id not in valid_imdb_ids:
                skipped += 1
                continue

            review_text = review_file.read_text(encoding='utf-8', errors='ignore').strip()
            if not review_text:
                skipped += 1
                continue

            reviews.append({
                'imdb_id': imdb_id,
                'review_text': review_text,
                'label': label_name,
                'vader_compound': None,
                'source': 'aclimdb'
            })

    return reviews, skipped


def ingest_reviews(engine):
    """
    Ingest genuinely mapped reviews from the Stanford aclImdb dataset.
    
    Instead of the Kaggle CSV (which has no movie IDs), we parse the original
    Stanford archive which includes URL mapping files that link each review
    to its real IMDB page URL, from which we extract the imdb_id.
    """
    cfg = get_config()
    aclimdb_dir = Path(cfg['data'].get('aclimdb_dir', 'data/raw/aclImdb'))

    if not aclimdb_dir.exists():
        print(f"  aclImdb directory not found at {aclimdb_dir}. Skipping reviews.")
        return 0

    # Get all valid imdb_ids from our database
    with Session(engine) as session:
        movies = session.execute(
            select(Movie.imdb_id).where(Movie.imdb_id.isnot(None))
        ).all()
        valid_imdb_ids = {m[0] for m in movies if m[0]}

    if not valid_imdb_ids:
        print("  No movies with imdb_ids found. Run ingest_movies first.")
        return 0

    print(f"  Found {len(valid_imdb_ids)} movies with imdb_ids in database.")

    all_reviews = []
    total_skipped = 0

    # Process both train and test splits
    for split in ['train', 'test']:
        split_dir = aclimdb_dir / split
        if not split_dir.exists():
            continue

        # Load URL mapping files for this split
        url_files = {}
        for label_dir in ['pos', 'neg']:
            url_file = split_dir / f'urls_{label_dir}.txt'
            if url_file.exists():
                url_files[label_dir] = url_file.read_text(encoding='utf-8').strip().split('\n')
            else:
                url_files[label_dir] = []
                print(f"  Warning: {url_file} not found.")

        reviews, skipped = _parse_aclimdb_split(split_dir, url_files, valid_imdb_ids)
        all_reviews.extend(reviews)
        total_skipped += skipped
        print(f"  {split}: {len(reviews)} mapped reviews, {skipped} skipped (no match)")

    # Bulk insert into database
    if all_reviews:
        with Session(engine) as session:
            chunk_size = 5000
            for i in range(0, len(all_reviews), chunk_size):
                chunk = all_reviews[i:i + chunk_size]
                session.bulk_insert_mappings(Review, chunk)
                session.commit()

    # Count unique movies that got reviews
    unique_movies = len({r['imdb_id'] for r in all_reviews})
    print(f"  Total: {len(all_reviews)} genuine reviews for {unique_movies} movies ingested.")
    print(f"  Skipped: {total_skipped} reviews (unmapped or missing in our catalog)")

    return len(all_reviews)
