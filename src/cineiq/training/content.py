import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentEngine:
    """
    Content-based recommendation using precomputed movie embeddings.
    All operations are in-memory NumPy — no vector DB needed for 45K items.
    """

    def __init__(self, embeddings, movie_id_to_idx):
        """
        Args:
            embeddings: np.ndarray of shape (n_movies, embed_dim)
            movie_id_to_idx: dict mapping movie_id (int) -> row index
        """
        self.embeddings = embeddings
        self.movie_id_to_idx = movie_id_to_idx
        # Reverse map: idx -> movie_id
        self.idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}

    def get_similar(self, movie_id, top_k=10):
        """
        Find top-K most similar movies to a given movie by cosine similarity.

        Returns:
            list of (movie_id, score) tuples, sorted by descending similarity
        """
        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return []

        query = self.embeddings[idx].reshape(1, -1)
        scores = cosine_similarity(query, self.embeddings)[0]

        # Exclude self
        scores[idx] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.idx_to_movie_id[i], float(scores[i])) for i in top_indices]

    def score_for_user(self, liked_movie_ids, n_total):
        """
        Build a user profile by averaging embeddings of liked movies,
        then compute cosine similarity against all movies.

        Args:
            liked_movie_ids: list of movie_id ints that the user rated highly
            n_total: total number of movies (for output array size)

        Returns:
            np.ndarray of shape (n_total,) with content similarity scores
        """
        liked_indices = [
            self.movie_id_to_idx[mid]
            for mid in liked_movie_ids
            if mid in self.movie_id_to_idx
        ]

        if not liked_indices:
            return np.zeros(n_total)

        # Average the embeddings of liked movies to form a user profile vector
        profile = self.embeddings[liked_indices].mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(profile, self.embeddings)[0]
        return scores
