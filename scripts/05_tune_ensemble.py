"""
Quick smoke test for the hybrid ensemble.
Picks a random user, runs the full two-stage pipeline, and prints results.
"""
import sys
import random
import pandas as pd
from pathlib import Path
from sqlalchemy import select, func

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from cineiq.db import get_session
from cineiq.data.models import Rating, Movie
from cineiq.config import get_config
from cineiq.training.ensemble import HybridEnsemble


def main():
    cfg = get_config()
    session = get_session()

    print("Loading ensemble...")
    ensemble = HybridEnsemble(
        models_dir=cfg['models']['output_dir'],
        alpha=cfg['ensemble']['alpha'],
        beta=cfg['ensemble']['beta'],
        gamma=cfg['ensemble']['gamma'],
        candidate_pool=cfg['ensemble']['candidate_pool'],
        top_k=cfg['ensemble']['final_top_k']
    )

    # Pick a random user who has rated at least 20 movies
    user_counts = pd.read_sql(
        select(Rating.user_id, func.count().label('cnt'))
        .group_by(Rating.user_id)
        .having(func.count() >= 20),
        session.bind
    )
    test_user = random.choice(user_counts['user_id'].tolist())

    # Get this user's rated movies
    user_ratings = pd.read_sql(
        select(Rating.movie_id, Rating.rating)
        .where(Rating.user_id == test_user),
        session.bind
    )
    rated_ids = user_ratings['movie_id'].tolist()

    print(f"\nTest User: {test_user} ({len(rated_ids)} ratings)")
    print("=" * 70)

    # Run the ensemble
    recs = ensemble.recommend(test_user, rated_ids)

    # Fetch titles for display
    movie_ids = [r['movie_id'] for r in recs]
    titles = pd.read_sql(
        select(Movie.movie_id, Movie.title, Movie.genres)
        .where(Movie.movie_id.in_(movie_ids)),
        session.bind
    )
    title_map = dict(zip(titles['movie_id'], titles['title']))
    genre_map = dict(zip(titles['movie_id'], titles['genres']))

    print(f"\nTop {len(recs)} Recommendations:")
    print("-" * 70)
    for i, rec in enumerate(recs, 1):
        mid = rec['movie_id']
        title = title_map.get(mid, 'Unknown')
        genres = genre_map.get(mid, '')
        print(f"\n  #{i}: {title}")
        print(f"      Genres:    {genres}")
        print(f"      Score:     {rec['score']:.4f}  "
              f"(SVD: {rec['svd_score']:.2f} | "
              f"Content: {rec['content_score']:.2f} | "
              f"Sentiment: {rec['sentiment_score']:.2f})")
        print(f"      Why:       {', '.join(rec['explanation'])}")

    # Also test cold-start
    print("\n" + "=" * 70)
    print("Cold-Start Test (seed: first 3 of user's liked movies)")
    liked = user_ratings[user_ratings['rating'] >= 4.0]['movie_id'].tolist()[:3]
    liked_titles = pd.read_sql(
        select(Movie.title).where(Movie.movie_id.in_(liked)),
        session.bind
    )
    print(f"  Seeds: {liked_titles['title'].tolist()}")

    cold_recs = ensemble.recommend_cold_start(liked, top_k=5)
    for i, rec in enumerate(cold_recs, 1):
        mid = rec['movie_id']
        t = title_map.get(mid) or "..."
        print(f"  #{i}: (id={mid}) score={rec['score']:.4f} | {', '.join(rec['explanation'])}")


if __name__ == "__main__":
    main()
