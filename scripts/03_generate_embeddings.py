import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import select

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from cineiq.db import get_engine, get_session
from cineiq.data.models import Movie
from cineiq.config import get_config
from cineiq.features.embeddings import generate_movie_embeddings, save_embeddings


def main():
    cfg = get_config()
    session = get_session()

    print("Loading movies from database...")
    query = select(Movie.movie_id, Movie.title, Movie.genres, Movie.overview)
    movies_df = pd.read_sql(query, session.bind)
    print(f"  Loaded {len(movies_df)} movies.")

    print("Generating embeddings with MiniLM...")
    embeddings, movie_id_to_idx = generate_movie_embeddings(
        movies_df,
        model_name=cfg['embeddings']['model_name'],
        batch_size=cfg['embeddings']['batch_size']
    )

    save_embeddings(embeddings, movie_id_to_idx, cfg['models']['output_dir'])
    print("Done.")


if __name__ == "__main__":
    main()
