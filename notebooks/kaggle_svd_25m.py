# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CineIQ — SVD Training on MovieLens 25M (Kaggle Notebook)              ║
# ║  Upgrades model from 100K ratings → 25M ratings                        ║
# ║  Output: svd_artifacts.zip  (drop contents into local models/)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELL 1: Install dependencies ────────────────────────────────────────────
# !pip install dagshub mlflow -q

# ── CELL 2: Download MovieLens 25M ──────────────────────────────────────────
import urllib.request, zipfile, os

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DEST = "/kaggle/working/ml-25m.zip"

if not os.path.exists("/kaggle/working/ml-25m/ratings.csv"):
    print("Downloading MovieLens 25M (~250MB)...")
    urllib.request.urlretrieve(URL, DEST)
    print("Extracting...")
    with zipfile.ZipFile(DEST, 'r') as z:
        z.extractall("/kaggle/working/")
    os.remove(DEST)
    print("Done.")
else:
    print("Already downloaded.")

# ── CELL 3: Load ratings ─────────────────────────────────────────────────────
import pandas as pd

print("Loading ratings...")
df = pd.read_csv("/kaggle/working/ml-25m/ratings.csv")
df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = df.drop(columns=['timestamp'])
print(f"  {len(df):,} ratings  |  {df['user_id'].nunique():,} users  |  {df['movie_id'].nunique():,} movies")

# ── CELL 4: SVDTrainer class ─────────────────────────────────────────────────
# (Copy-pasted from src/cineiq/training/collaborative.py)
# Key change: vectorized mean-centering for 25M scale

import numpy as np
import json
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDTrainer:
    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.user_map = None
        self.item_map = None

    def train(self, ratings_df):
        unique_users = sorted(ratings_df['user_id'].unique())
        unique_items = sorted(ratings_df['movie_id'].unique())
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {mid: idx for idx, mid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        print(f"  Matrix: {n_users:,} users x {n_items:,} items")

        row_indices = ratings_df['user_id'].map(self.user_map).values
        col_indices = ratings_df['movie_id'].map(self.item_map).values
        values = ratings_df['rating'].values.astype(np.float32)

        R = csr_matrix((values, (row_indices, col_indices)), shape=(n_users, n_items))

        self.global_mean = float(values.mean())

        user_rating_sums   = np.array(R.sum(axis=1)).flatten()
        user_rating_counts = np.array((R != 0).sum(axis=1)).flatten()
        user_rating_counts[user_rating_counts == 0] = 1
        self.user_bias = (user_rating_sums / user_rating_counts) - self.global_mean

        item_rating_sums   = np.array(R.sum(axis=0)).flatten()
        item_rating_counts = np.array((R != 0).sum(axis=0)).flatten()
        item_rating_counts[item_rating_counts == 0] = 1
        self.item_bias = (item_rating_sums / item_rating_counts) - self.global_mean

        # Vectorized mean-centering (replaces slow Python loop — critical for 25M scale)
        print("  Mean-centering sparse matrix...")
        R_centered = R.astype(np.float64).copy()
        nz_rows, nz_cols = R_centered.nonzero()
        R_centered.data -= (self.global_mean + self.user_bias[nz_rows] + self.item_bias[nz_cols])

        k = min(self.n_factors, min(n_users, n_items) - 1)
        print(f"  Running truncated SVD with k={k} factors...")
        U, sigma, Vt = svds(R_centered, k=k)

        sqrt_sigma = np.sqrt(sigma)
        self.user_factors = U  * sqrt_sigma[np.newaxis, :]
        self.item_factors = Vt.T * sqrt_sigma[np.newaxis, :]

        # Evaluate on a 100K sample (full prediction matrix is 162K x 62K = too large)
        print("  Evaluating on 100K sample...")
        sample_mask = np.random.choice(len(nz_rows), min(100_000, len(nz_rows)), replace=False)
        s_rows = nz_rows[sample_mask]
        s_cols = nz_cols[sample_mask]
        actuals = np.array(R[s_rows, s_cols]).flatten()
        preds = np.clip(
            self.global_mean
            + self.user_bias[s_rows]
            + self.item_bias[s_cols]
            + np.sum(self.user_factors[s_rows] * self.item_factors[s_cols], axis=1),
            0.5, 5.0
        )
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        mae  = float(np.mean(np.abs(actuals - preds)))
        print(f"  Sample RMSE: {rmse:.4f},  MAE: {mae:.4f}")
        return {'train_rmse': rmse, 'train_mae': mae, 'n_users': n_users, 'n_items': n_items}

    def export_artifacts(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'svd_user_factors.npy', self.user_factors)
        np.save(out / 'svd_item_factors.npy', self.item_factors)
        np.save(out / 'svd_user_bias.npy',    self.user_bias)
        np.save(out / 'svd_item_bias.npy',    self.item_bias)
        np.save(out / 'svd_global_mean.npy',  np.array([self.global_mean]))
        mappings = {
            'user_map': {str(k): int(v) for k, v in self.user_map.items()},
            'item_map': {str(k): int(v) for k, v in self.item_map.items()},
        }
        with open(out / 'svd_mappings.json', 'w') as f:
            json.dump(mappings, f)
        print(f"  Artifacts saved to {out}/")
        print(f"    user_factors: {self.user_factors.shape}")
        print(f"    item_factors: {self.item_factors.shape}")

# ── CELL 5: Train ────────────────────────────────────────────────────────────
trainer = SVDTrainer(n_factors=200)   # 200 factors vs 100 in local model
metrics = trainer.train(df)
trainer.export_artifacts("/kaggle/working/svd_artifacts/")

# ── CELL 6: Log to DagsHub MLflow ───────────────────────────────────────────
# Fill in your DagsHub credentials before running this cell
DAGSHUB_USER  = "keshav31415"
DAGSHUB_TOKEN = ""           # paste your DagsHub token here
DAGSHUB_REPO  = "CIENIQ"

import mlflow, dagshub

dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("svd-25m")

with mlflow.start_run(run_name="svd-ml25m-200factors"):
    mlflow.log_param("dataset",  "MovieLens-25M")
    mlflow.log_param("n_factors", 200)
    mlflow.log_param("n_users",  metrics['n_users'])
    mlflow.log_param("n_items",  metrics['n_items'])
    mlflow.log_metric("train_rmse", metrics['train_rmse'])
    mlflow.log_metric("train_mae",  metrics['train_mae'])
    print(f"Logged to DagsHub MLflow.")

# ── CELL 7: Zip artifacts for download ──────────────────────────────────────
import zipfile, glob

OUT_ZIP = "/kaggle/working/svd_artifacts.zip"
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fpath in glob.glob("/kaggle/working/svd_artifacts/*"):
        zf.write(fpath, arcname=os.path.basename(fpath))

size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
print(f"Zipped -> {OUT_ZIP}  ({size_mb:.0f} MB)")
print("Download svd_artifacts.zip, extract, and drop all files into your local models/ folder.")
print("The ensemble loads whatever .npy files are in models/ — no code changes needed.")
