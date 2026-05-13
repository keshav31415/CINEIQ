import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDTrainer:
    """
    Matrix Factorization via Truncated SVD on the user-item ratings matrix.
    
    Instead of relying on scikit-surprise (which is dead on Python 3.13),
    we use scipy.sparse.linalg.svds directly. This gives us:
      - User factors (U * sqrt(S))  shape: (n_users, n_factors)
      - Item factors (sqrt(S) * Vt) shape: (n_items, n_factors)
      - User/Item biases computed from mean-centering
    
    Prediction for user u, item i:
      score = global_mean + user_bias[u] + item_bias[i] + dot(U[u], V[i])
    """

    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.user_map = None  # raw_user_id -> matrix_row_index
        self.item_map = None  # raw_movie_id -> matrix_col_index

    def train(self, ratings_df):
        """
        Train SVD on a DataFrame with columns: user_id, movie_id, rating.
        Returns a dict of evaluation metrics.
        """
        # 1. Build contiguous ID mappings
        unique_users = sorted(ratings_df['user_id'].unique())
        unique_items = sorted(ratings_df['movie_id'].unique())
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {mid: idx for idx, mid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        print(f"  Matrix dimensions: {n_users} users x {n_items} items")

        # 2. Build sparse user-item matrix
        row_indices = ratings_df['user_id'].map(self.user_map).values
        col_indices = ratings_df['movie_id'].map(self.item_map).values
        values = ratings_df['rating'].values.astype(np.float32)

        R = csr_matrix((values, (row_indices, col_indices)),
                        shape=(n_users, n_items))

        # 3. Compute biases (mean-centering)
        self.global_mean = values.mean()

        # Per-user bias: average rating of that user minus global mean
        user_rating_sums = np.array(R.sum(axis=1)).flatten()
        user_rating_counts = np.array((R != 0).sum(axis=1)).flatten()
        user_rating_counts[user_rating_counts == 0] = 1  # avoid division by zero
        self.user_bias = (user_rating_sums / user_rating_counts) - self.global_mean

        # Per-item bias: average rating of that item minus global mean
        item_rating_sums = np.array(R.sum(axis=0)).flatten()
        item_rating_counts = np.array((R != 0).sum(axis=0)).flatten()
        item_rating_counts[item_rating_counts == 0] = 1
        self.item_bias = (item_rating_sums / item_rating_counts) - self.global_mean

        # 4. Mean-center the matrix before SVD
        #    For sparse efficiency, we only subtract from non-zero entries
        R_centered = R.copy().astype(np.float64)
        nonzero_rows, nonzero_cols = R_centered.nonzero()
        for i in range(len(nonzero_rows)):
            r, c = nonzero_rows[i], nonzero_cols[i]
            R_centered[r, c] -= (self.global_mean + self.user_bias[r] + self.item_bias[c])

        # 5. Truncated SVD
        #    n_factors must be < min(n_users, n_items)
        k = min(self.n_factors, min(n_users, n_items) - 1)
        print(f"  Running truncated SVD with k={k} factors...")
        U, sigma, Vt = svds(R_centered, k=k)

        # 6. Fold singular values into factors for easy dot-product inference
        sqrt_sigma = np.sqrt(sigma)
        self.user_factors = U * sqrt_sigma[np.newaxis, :]   # (n_users, k)
        self.item_factors = (Vt.T * sqrt_sigma[np.newaxis, :])  # (n_items, k)

        # 7. Evaluate: compute RMSE on the training data itself
        #    (Cross-validation would be done in the script via train/test split)
        predictions = self._predict_all(R)
        nonzero_mask = R.nonzero()
        actuals = np.array(R[nonzero_mask]).flatten()
        preds = predictions[nonzero_mask]
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        mae = np.mean(np.abs(actuals - preds))

        print(f"  Training RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return {'train_rmse': rmse, 'train_mae': mae}

    def _predict_all(self, R_sparse):
        """Predict ratings for all user-item pairs."""
        # score = global_mean + user_bias + item_bias + U @ V^T
        pred = (self.user_factors @ self.item_factors.T)
        pred += self.global_mean
        pred += self.user_bias[:, np.newaxis]
        pred += self.item_bias[np.newaxis, :]
        # Clip to valid rating range
        pred = np.clip(pred, 0.5, 5.0)
        return pred

    def export_artifacts(self, output_dir):
        """Save all trained artifacts to disk."""
        if self.user_factors is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        np.save(out_path / 'svd_user_factors.npy', self.user_factors)
        np.save(out_path / 'svd_item_factors.npy', self.item_factors)
        np.save(out_path / 'svd_user_bias.npy', self.user_bias)
        np.save(out_path / 'svd_item_bias.npy', self.item_bias)
        np.save(out_path / 'svd_global_mean.npy', np.array([self.global_mean]))

        # Convert numpy int keys to plain int for JSON serialization
        user_map_serializable = {str(k): int(v) for k, v in self.user_map.items()}
        item_map_serializable = {str(k): int(v) for k, v in self.item_map.items()}

        with open(out_path / 'svd_mappings.json', 'w') as f:
            json.dump({
                'user_map': user_map_serializable,
                'item_map': item_map_serializable
            }, f)

        print(f"  Artifacts exported to {out_path}/")
        print(f"    - user_factors: {self.user_factors.shape}")
        print(f"    - item_factors: {self.item_factors.shape}")
        print(f"    - {len(self.user_map)} users, {len(self.item_map)} items")
