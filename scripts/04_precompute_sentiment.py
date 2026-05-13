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

    print("Precomputing VADER sentiment scores...")
    sentiment_cache = precompute_sentiment(engine, cfg['models']['output_dir'])

    # Quick stats
    if sentiment_cache:
        scores = list(sentiment_cache.values())
        avg = sum(scores) / len(scores)
        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)
        neu = len(scores) - pos - neg
        print(f"\n  Summary:")
        print(f"    Movies with sentiment: {len(scores)}")
        print(f"    Average compound:      {avg:.4f}")
        print(f"    Positive (>0.05):      {pos}")
        print(f"    Negative (<-0.05):     {neg}")
        print(f"    Neutral:               {neu}")

    print("Done.")


if __name__ == "__main__":
    main()
