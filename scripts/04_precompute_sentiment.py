import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from cineiq.db import get_engine
from cineiq.config import get_config
from cineiq.features.sentiment import precompute_sentiment


def main():
    cfg = get_config()
    engine = get_engine()

    print("Precomputing hybrid sentiment scores...")
    print("  Primary: VADER on genuinely mapped Stanford aclImdb reviews")
    print("  Fallback: Bayesian TMDB vote distributions")
    print()
    sentiment_cache = precompute_sentiment(engine, cfg['models']['output_dir'])

    # Distribution stats
    if sentiment_cache:
        scores = list(sentiment_cache.values())
        avg = sum(scores) / len(scores)
        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)
        neu = len(scores) - pos - neg
        print(f"\n  Distribution:")
        print(f"    Positive (>0.05):  {pos}")
        print(f"    Negative (<-0.05): {neg}")
        print(f"    Neutral:           {neu}")
        print(f"    Average:           {avg:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
