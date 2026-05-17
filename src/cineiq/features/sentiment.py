import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentScorer:
    """
    Uses VADER (rule-based, no training needed) to score review text.
    Produces a compound score in [-1.0, 1.0] per review,
    then aggregates all reviews for a movie into a single score.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score_review(self, text: str) -> float:
        """Return VADER compound score for a single review."""
        return self.analyzer.polarity_scores(text)['compound']

    def score_reviews_for_movie(self, reviews: list[str]) -> float:
        """Average compound score across all reviews for one movie."""
        if not reviews:
            return 0.0  # neutral fallback
        scores = [self.score_review(r) for r in reviews]
        return sum(scores) / len(scores)


def _compute_vader_sentiment(engine, scorer):
    """
    Primary signal: Run VADER on genuinely mapped Stanford aclImdb reviews.
    Only processes reviews that are linked to real movie IDs.
    
    Returns:
        dict {movie_id_str: float} for movies that have real reviews
    """
    from sqlalchemy import text

    query = text("""
        SELECT m.movie_id, r.review_text
        FROM reviews r
        JOIN movies m ON r.imdb_id = m.imdb_id
        WHERE r.review_text IS NOT NULL AND r.review_text != ''
        ORDER BY m.movie_id
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        print("  No genuine reviews found. VADER layer will be empty.")
        return {}

    movie_reviews = defaultdict(list)
    for movie_id, review_text in rows:
        movie_reviews[movie_id].append(review_text)

    print(f"  VADER: Scoring {sum(len(v) for v in movie_reviews.values())} "
          f"genuine reviews across {len(movie_reviews)} movies...")

    vader_cache = {}
    for movie_id, reviews in movie_reviews.items():
        avg_score = scorer.score_reviews_for_movie(reviews)
        vader_cache[str(movie_id)] = round(avg_score, 4)

    return vader_cache


def _compute_tmdb_sentiment(engine, exclude_ids: set):
    """
    Fallback signal: Derive sentiment from TMDB vote_average using
    Bayesian smoothing for movies that lack genuine text reviews.
    
    Uses the IMDB weighted rating formula:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = vote count for the movie
        m = minimum votes required (threshold = 50)
        R = average rating for the movie (0-10 scale)
        C = mean vote average across the entire catalog
    
    Then maps the weighted rating from [1, 10] to [-1, +1].
    
    Returns:
        dict {movie_id_str: float} for movies without text reviews
    """
    from sqlalchemy import text

    query = text("""
        SELECT movie_id, vote_average, vote_count
        FROM movies
        WHERE vote_average IS NOT NULL AND vote_count IS NOT NULL
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("  No TMDB vote data available for fallback.")
        return {}

    # Filter out movies that already have VADER scores
    df = df[~df['movie_id'].astype(str).isin(exclude_ids)]

    if df.empty:
        print("  All movies already covered by VADER. No fallback needed.")
        return {}

    # Bayesian average
    C = df['vote_average'].mean()
    m = 50  # minimum vote threshold

    v = df['vote_count'].values
    R = df['vote_average'].values

    weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C

    # Map from [1, 10] to [-1, +1]
    sentiment_scores = 2 * ((weighted_rating - 1.0) / (10.0 - 1.0)) - 1

    tmdb_cache = {}
    for i, row in df.iterrows():
        tmdb_cache[str(int(row['movie_id']))] = round(float(sentiment_scores[df.index.get_loc(i)]), 4)

    print(f"  TMDB Fallback: Derived sentiment for {len(tmdb_cache)} additional movies.")
    return tmdb_cache


def precompute_sentiment(engine, output_dir):
    """
    Hybrid sentiment computation:
      1. Primary: VADER on genuinely mapped Stanford aclImdb reviews
      2. Fallback: Bayesian TMDB vote distributions for uncovered movies
    
    Movies with real text reviews get authentic NLP-derived sentiment.
    Movies without reviews get a statistically smoothed audience score.
    
    Returns:
        sentiment_cache: dict {movie_id_str: float}
    """
    scorer = SentimentScorer()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: VADER on real reviews (primary signal)
    print("  Step 1/2: Computing VADER sentiment on genuine reviews...")
    vader_cache = _compute_vader_sentiment(engine, scorer)
    vader_ids = set(vader_cache.keys())

    # Step 2: TMDB fallback for uncovered movies
    print("  Step 2/2: Computing TMDB fallback for remaining movies...")
    tmdb_cache = _compute_tmdb_sentiment(engine, exclude_ids=vader_ids)

    # Merge: VADER takes priority, TMDB fills gaps
    sentiment_cache = {}
    sentiment_cache.update(tmdb_cache)   # Fallback goes first
    sentiment_cache.update(vader_cache)  # VADER overwrites any overlap

    # Save combined cache
    cache_path = out_path / 'imdb_sentiment_cache.json'
    with open(cache_path, 'w') as f:
        json.dump(sentiment_cache, f)

    # Save metadata about which source was used (for explainability)
    source_meta = {
        'vader_count': len(vader_cache),
        'tmdb_fallback_count': len(tmdb_cache),
        'total': len(sentiment_cache),
        'vader_movie_ids': list(vader_ids),
    }
    meta_path = out_path / 'sentiment_sources.json'
    with open(meta_path, 'w') as f:
        json.dump(source_meta, f)

    print(f"\n  Summary:")
    print(f"    VADER (genuine NLP):  {len(vader_cache)} movies")
    print(f"    TMDB (fallback):      {len(tmdb_cache)} movies")
    print(f"    Total coverage:       {len(sentiment_cache)} movies")
    print(f"  Cache saved to {cache_path}")

    return sentiment_cache
