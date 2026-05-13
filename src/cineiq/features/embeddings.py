import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


def generate_movie_embeddings(movies_df, model_name='all-MiniLM-L6-v2', batch_size=256):
    """
    Generate 384-dimensional embeddings for each movie by encoding
    a combined text of title + genres + overview.

    Args:
        movies_df: DataFrame with columns: movie_id, title, genres, overview
        model_name: SentenceTransformer model name
        batch_size: encoding batch size

    Returns:
        embeddings: np.ndarray of shape (n_movies, 384)
        movie_id_to_idx: dict mapping movie_id -> row index in the embeddings matrix
    """
    model = SentenceTransformer(model_name)

    # Build a rich text representation for each movie
    texts = []
    movie_id_to_idx = {}

    for idx, (_, row) in enumerate(movies_df.iterrows()):
        title = str(row['title']) if row['title'] else ''
        genres = str(row['genres']).replace('|', ', ') if row['genres'] else ''
        overview = str(row['overview']) if row['overview'] and str(row['overview']) != 'nan' else ''

        combined = f"{title}. {genres}. {overview}".strip()
        texts.append(combined)
        movie_id_to_idx[int(row['movie_id'])] = idx

    print(f"  Encoding {len(texts)} movie texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)

    return embeddings, movie_id_to_idx


def save_embeddings(embeddings, movie_id_to_idx, output_dir):
    """Save embeddings matrix and ID mapping to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / 'movie_embeddings.npy', embeddings)

    with open(out_path / 'movie_id_to_idx.json', 'w') as f:
        json.dump(movie_id_to_idx, f)

    print(f"  Saved embeddings: {embeddings.shape} to {out_path}")
    print(f"  Saved movie_id_to_idx: {len(movie_id_to_idx)} entries")
