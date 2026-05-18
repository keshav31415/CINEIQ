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

        # --- Load Sentiment cache (VADER/TMDB scalar fallback) ---
        with open(models / 'imdb_sentiment_cache.json', 'r') as f:
            self.sentiment_cache = json.load(f)

        # --- Load ABSA cache (optional — falls back to scalar if missing) ---
        absa_path = models / 'absa_sentiment_cache.json'
        if absa_path.exists():
            with open(absa_path, 'r') as f:
                self.absa_cache = json.load(f)
        else:
            self.absa_cache = {}

        # Precompute: set of all known movie IDs (intersection of both systems)
        self.all_movie_ids = sorted(
            set(self.svd_item_map.keys()) & set(self.content_id_to_idx.keys())
        )

        print(f"  Ensemble loaded: {len(self.all_movie_ids)} movies in common")
        print(f"  SVD: {self.user_factors.shape[0]} users, {self.item_factors.shape[0]} items")
        print(f"  Content: {self.embeddings.shape}")
        print(f"  Sentiment (scalar): {len(self.sentiment_cache)} movies scored")
        print(f"  Sentiment (ABSA):   {len(self.absa_cache)} movies scored")

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

    _ASPECTS = ['acting', 'plot', 'visuals', 'pacing', 'music']

    def _build_aspect_profile(self, rated_movie_ids: list) -> dict:
        """
        Average ABSA aspect scores across the user's rated movies.
        Returns {aspect: float} — None for aspects with no data.
        """
        sums = {asp: [] for asp in self._ASPECTS}
        for movie_id in rated_movie_ids:
            absa = self.absa_cache.get(str(movie_id))
            if not absa:
                continue
            for asp in self._ASPECTS:
                val = absa.get(asp)
                if val is not None:
                    sums[asp].append(val)
        return {asp: (sum(v) / len(v)) if v else None for asp, v in sums.items()}

    def _personalized_sentiment(self, movie_id: int, user_profile: dict) -> tuple[float, str | None]:
        """
        Compute a personalized sentiment score using ABSA aspect dot-product.
        Falls back to scalar VADER/TMDB score if ABSA unavailable.

        Returns (score, top_aspect_note) where top_aspect_note is a string
        describing the most influential aspect, or None if using fallback.
        """
        absa = self.absa_cache.get(str(movie_id))
        if not absa:
            return self.sentiment_cache.get(str(movie_id), 0.0), None

        scores, aspects_used = [], []
        for asp in self._ASPECTS:
            user_pref = user_profile.get(asp)
            movie_asp = absa.get(asp)
            if user_pref is not None and movie_asp is not None:
                scores.append(user_pref * movie_asp)
                aspects_used.append((asp, user_pref * movie_asp))

        if not scores:
            fallback = absa.get('overall')
            return (fallback if fallback is not None
                    else self.sentiment_cache.get(str(movie_id), 0.0)), None

        personalized = sum(scores) / len(scores)
        top_asp, top_val = max(aspects_used, key=lambda x: abs(x[1]))
        note = f"Strong on {top_asp}" if top_val > 0.1 else (
               f"Weak on {top_asp}" if top_val < -0.1 else None)
        return personalized, note

    def _vectorized_scores(self, user_id, rated_movie_ids):
        """
        Returns (svd_n, content_n, has_svd) — min-max normalized score arrays
        aligned to self.all_movie_ids order, with rated items zeroed out.
        Used by both recommend() and the Optuna tuning script.
        """
        rated_set = set(rated_movie_ids)

        raw_svd = self._svd_scores_for_user(user_id)
        has_svd = raw_svd is not None

        liked_ids = [mid for mid in rated_movie_ids if mid in self.content_id_to_idx]
        raw_content = self._content_scores_for_user(liked_ids)

        n = len(self.all_movie_ids)
        svd_scores = np.zeros(n)
        content_scores = np.zeros(n)

        for i, movie_id in enumerate(self.all_movie_ids):
            if has_svd and movie_id in self.svd_item_map:
                svd_scores[i] = raw_svd[self.svd_item_map[movie_id]]
            content_scores[i] = raw_content[self.content_id_to_idx[movie_id]]

        return self._normalize(svd_scores), self._normalize(content_scores), has_svd, rated_set

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
        svd_n, content_n, has_svd, rated_set = self._vectorized_scores(user_id, rated_movie_ids)

        # ===== STAGE 1: Candidate Generation =====
        if has_svd:
            stage1 = self.alpha * svd_n + self.beta * content_n
        else:
            stage1 = content_n.copy()

        # Exclude already-rated
        for i, mid in enumerate(self.all_movie_ids):
            if mid in rated_set:
                stage1[i] = -np.inf

        top_n_indices = np.argsort(stage1)[::-1][:self.candidate_pool]

        # ===== STAGE 2: Personalised Sentiment Re-Ranking =====
        user_profile = self._build_aspect_profile(rated_movie_ids)

        # Pre-fetch raw SVD + content for debug output
        liked_ids = [mid for mid in rated_movie_ids if mid in self.content_id_to_idx]
        raw_svd_full = self._svd_scores_for_user(user_id)
        raw_content_full = self._content_scores_for_user(liked_ids)

        results = []
        for i in top_n_indices:
            if stage1[i] == -np.inf:
                continue
            movie_id = self.all_movie_ids[i]

            sent_score, aspect_note = self._personalized_sentiment(movie_id, user_profile)
            final_score = stage1[i] + self.gamma * sent_score

            svd_raw = float(raw_svd_full[self.svd_item_map[movie_id]]) if (has_svd and movie_id in self.svd_item_map) else 0.0
            content_raw = float(raw_content_full[self.content_id_to_idx[movie_id]])

            results.append({
                'movie_id': movie_id,
                'score': round(float(final_score), 4),
                'svd_score': round(svd_raw, 4),
                'content_score': round(content_raw, 4),
                'sentiment_score': round(float(sent_score), 4),
                'explanation': self._explain(svd_raw, content_raw, sent_score, has_svd, aspect_note),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:self.top_k]

    def _explain(self, svd_score, content_score, sentiment_score, has_svd, aspect_note=None):
        """Generate human-readable explanation strings."""
        reasons = []

        if has_svd and svd_score > 3.5:
            reasons.append("Users with similar taste rated this highly")
        if content_score > 0.5:
            reasons.append("Matches your genre and plot preferences")
        elif content_score > 0.3:
            reasons.append("Partially overlaps with your taste profile")
        if aspect_note:
            reasons.append(aspect_note)
        elif sentiment_score > 0.3:
            reasons.append("Strong positive audience sentiment")
        elif sentiment_score < -0.3:
            reasons.append("Mixed or negative audience reviews")

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
        user_profile = self._build_aspect_profile(seed_movie_ids)

        results = []
        for movie_id in self.all_movie_ids:
            if movie_id in seed_set:
                continue

            content_idx = self.content_id_to_idx[movie_id]
            content_score = raw_content[content_idx]
            sent_score, aspect_note = self._personalized_sentiment(movie_id, user_profile)
            final_score = content_score + self.gamma * sent_score

            explanation = self._explain(0.0, content_score, sent_score, has_svd=False, aspect_note=aspect_note)

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
