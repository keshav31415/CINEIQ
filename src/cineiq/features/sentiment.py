import json
from pathlib import Path
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


def precompute_sentiment(engine, output_dir):
    """
    For every movie that has reviews in the DB, compute the aggregated
    VADER compound score and save as a JSON cache.

    Returns:
        sentiment_cache: dict {movie_id_str: float}
    """
    from sqlalchemy import text

    scorer = SentimentScorer()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Query reviews grouped by movie (via imdb_id -> movie_id join)
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
        print("  No reviews found linked to movies. Sentiment cache will be empty.")
        sentiment_cache = {}
    else:
        # Group reviews by movie_id
        from collections import defaultdict
        movie_reviews = defaultdict(list)
        for movie_id, review_text in rows:
            movie_reviews[movie_id].append(review_text)

        print(f"  Found reviews for {len(movie_reviews)} movies. Scoring with VADER...")

        sentiment_cache = {}
        scored_count = 0
        for movie_id, reviews in movie_reviews.items():
            avg_score = scorer.score_reviews_for_movie(reviews)
            sentiment_cache[str(movie_id)] = round(avg_score, 4)
            scored_count += len(reviews)

            if len(sentiment_cache) % 500 == 0:
                print(f"    Scored {len(sentiment_cache)} movies so far...")

        print(f"  Scored {scored_count} reviews across {len(sentiment_cache)} movies.")

    # Save cache
    cache_path = out_path / 'imdb_sentiment_cache.json'
    with open(cache_path, 'w') as f:
        json.dump(sentiment_cache, f)

    print(f"  Sentiment cache saved to {cache_path}")
    return sentiment_cache
