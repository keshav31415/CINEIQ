import numpy as np
import json
from pathlib import Path


class HybridEnsemble:
    """
    Two-stage hybrid recommendation engine:
      Stage 1: Candidate generation — weighted blend of SVD + Content scores → top N
      Stage 2: Sentiment re-ranking — apply VADER bonus/penalty → final top K

    All operations are in-memory NumPy. No network calls, no DB queries at inference.
    """

    def __init__(self, models_dir,
                 alpha=0.5, beta=0.5, gamma=0.2,
                 candidate_pool=50, top_k=10):
        """
        Args:
            models_dir: path to directory containing all .npy and .json artifacts
            alpha: weight for SVD (collaborative) score in Stage 1
            beta: weight for Content score in Stage 1
            gamma: multiplier for sentiment bonus/penalty in Stage 2
            candidate_pool: how many candidates Stage 1 produces
            top_k: how many final recommendations to return
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.candidate_pool = candidate_pool
        self.top_k = top_k

        models = Path(models_dir)

        # --- Load SVD artifacts ---
        self.user_factors = np.load(models / 'svd_user_factors.npy')
        self.item_factors = np.load(models / 'svd_item_factors.npy')
        self.user_bias = np.load(models / 'svd_user_bias.npy')
        self.item_bias = np.load(models / 'svd_item_bias.npy')
        self.global_mean = np.load(models / 'svd_global_mean.npy')[0]

        with open(models / 'svd_mappings.json', 'r') as f:
            mappings = json.load(f)
        self.svd_user_map = {int(k): v for k, v in mappings['user_map'].items()}
        self.svd_item_map = {int(k): v for k, v in mappings['item_map'].items()}
        self.svd_idx_to_item = {v: k for k, v in self.svd_item_map.items()}

        # --- Load Content artifacts ---
        self.embeddings = np.load(models / 'movie_embeddings.npy')
        with open(models / 'movie_id_to_idx.json', 'r') as f:
            self.content_id_to_idx = {int(k): v for k, v in json.load(f).items()}
        self.content_idx_to_id = {v: k for k, v in self.content_id_to_idx.items()}

        # --- Load Sentiment cache ---
        with open(models / 'imdb_sentiment_cache.json', 'r') as f:
            self.sentiment_cache = json.load(f)

        # Precompute: set of all known movie IDs (intersection of both systems)
        self.all_movie_ids = sorted(
            set(self.svd_item_map.keys()) & set(self.content_id_to_idx.keys())
        )

        print(f"  Ensemble loaded: {len(self.all_movie_ids)} movies in common")
        print(f"  SVD: {self.user_factors.shape[0]} users, {self.item_factors.shape[0]} items")
        print(f"  Content: {self.embeddings.shape}")
        print(f"  Sentiment: {len(self.sentiment_cache)} movies scored")

    def _normalize(self, scores):
        """Min-max normalize to [0, 1]."""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _svd_scores_for_user(self, user_id):
        """Compute predicted SVD rating for all items for a given user."""
        user_idx = self.svd_user_map.get(user_id)
        if user_idx is None:
            return None  # unknown user → cold start

        # score = global_mean + user_bias + item_bias + dot(user_factor, item_factor)
        scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_idx]
        )
        return scores

    def _content_scores_for_user(self, liked_movie_ids):
        """Build user profile from liked movies, return cosine similarity to all."""
        from sklearn.metrics.pairwise import cosine_similarity

        liked_indices = [
            self.content_id_to_idx[mid]
            for mid in liked_movie_ids
            if mid in self.content_id_to_idx
        ]
        if not liked_indices:
            return np.zeros(self.embeddings.shape[0])

        profile = self.embeddings[liked_indices].mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(profile, self.embeddings)[0]
        return scores

    def recommend(self, user_id, rated_movie_ids):
        """
        Full two-stage recommendation pipeline.

        Args:
            user_id: int, the user's ID from the ratings table
            rated_movie_ids: list of movie_id ints the user has already rated

        Returns:
            list of dicts with keys: movie_id, score, svd_score, content_score,
            sentiment_score, explanation
        """
        rated_set = set(rated_movie_ids)

        # ===== STAGE 1: Candidate Generation =====
        # SVD scores
        raw_svd = self._svd_scores_for_user(user_id)
        has_svd = raw_svd is not None

        # Content scores (from user's liked movies, rating > 3.5)
        liked_ids = [mid for mid in rated_movie_ids if mid in self.content_id_to_idx]
        raw_content = self._content_scores_for_user(liked_ids)

        # Build a unified score array indexed by content index
        n_movies = self.embeddings.shape[0]
        stage1_scores = np.full(n_movies, -np.inf)

        for movie_id in self.all_movie_ids:
            content_idx = self.content_id_to_idx[movie_id]

            if movie_id in rated_set:
                continue  # exclude already-rated

            # Normalized SVD score for this movie
            svd_score = 0.0
            if has_svd and movie_id in self.svd_item_map:
                svd_idx = self.svd_item_map[movie_id]
                svd_score = raw_svd[svd_idx]

            content_score = raw_content[content_idx]

            # Weighted blend
            if has_svd:
                stage1_scores[content_idx] = (
                    self.alpha * svd_score + self.beta * content_score
                )
            else:
                # Cold start: content only
                stage1_scores[content_idx] = content_score

        # Pick top N candidates from Stage 1
        top_n_indices = np.argsort(stage1_scores)[::-1][:self.candidate_pool]

        # ===== STAGE 2: Sentiment Re-Ranking =====
        results = []
        for content_idx in top_n_indices:
            if stage1_scores[content_idx] == -np.inf:
                continue

            movie_id = self.content_idx_to_id.get(content_idx)
            if movie_id is None:
                continue

            s1_score = stage1_scores[content_idx]

            # Lookup precomputed VADER sentiment
            sent_score = self.sentiment_cache.get(str(movie_id), 0.0)

            # Final re-ranked score
            final_score = s1_score + self.gamma * sent_score

            # Individual component scores for explainability
            svd_raw = 0.0
            if has_svd and movie_id in self.svd_item_map:
                svd_raw = raw_svd[self.svd_item_map[movie_id]]
            content_raw = raw_content[content_idx]

            explanation = self._explain(svd_raw, content_raw, sent_score, has_svd)

            results.append({
                'movie_id': movie_id,
                'score': round(float(final_score), 4),
                'svd_score': round(float(svd_raw), 4),
                'content_score': round(float(content_raw), 4),
                'sentiment_score': round(float(sent_score), 4),
                'explanation': explanation
            })

        # Sort by final score and return top K
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:self.top_k]

    def _explain(self, svd_score, content_score, sentiment_score, has_svd):
        """Generate human-readable explanation strings."""
        reasons = []

        if has_svd and svd_score > 3.5:
            reasons.append("Users with similar taste rated this highly")
        if content_score > 0.5:
            reasons.append("Matches your genre and plot preferences")
        elif content_score > 0.3:
            reasons.append("Partially overlaps with your taste profile")
        if sentiment_score > 0.3:
            reasons.append("Strong positive audience sentiment")
        elif sentiment_score < -0.3:
            reasons.append("⚠ Mixed or negative audience reviews")

        if not reasons:
            reasons.append("Recommended based on overall compatibility")

        return reasons

    def recommend_cold_start(self, seed_movie_ids, top_k=None):
        """
        For new users with no rating history.
        Uses content-based only + sentiment re-ranking.

        Args:
            seed_movie_ids: list of movie_id ints the user says they like
            top_k: override for number of results
        """
        k = top_k or self.top_k
        raw_content = self._content_scores_for_user(seed_movie_ids)
        seed_set = set(seed_movie_ids)

        results = []
        for movie_id in self.all_movie_ids:
            if movie_id in seed_set:
                continue

            content_idx = self.content_id_to_idx[movie_id]
            content_score = raw_content[content_idx]
            sent_score = self.sentiment_cache.get(str(movie_id), 0.0)
            final_score = content_score + self.gamma * sent_score

            explanation = self._explain(0.0, content_score, sent_score, has_svd=False)

            results.append({
                'movie_id': movie_id,
                'score': round(float(final_score), 4),
                'svd_score': 0.0,
                'content_score': round(float(content_score), 4),
                'sentiment_score': round(float(sent_score), 4),
                'explanation': explanation
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
