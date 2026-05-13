import sys
import pandas as pd
import mlflow
import dagshub
import numpy as np
from pathlib import Path
from sqlalchemy import select

# Ensure src is in the path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from cineiq.db import get_session
from cineiq.data.models import Rating
from cineiq.config import get_config
from cineiq.training.collaborative import SVDTrainer

def main():
    cfg = get_config()
    session = get_session()

    # Initialize DagsHub + MLflow remote tracking
    dagshub.init(
        repo_owner=cfg['mlflow']['dagshub_repo_owner'],
        repo_name=cfg['mlflow']['dagshub_repo_name'],
        mlflow=True
    )

    print("Loading ratings from database...")
    query = select(Rating.user_id, Rating.movie_id, Rating.rating)
    ratings_df = pd.read_sql(query, session.bind)

    if ratings_df.empty:
        print("No ratings found. Run scripts/01_ingest_data.py first.")
        return

    print(f"Loaded {len(ratings_df)} ratings.")

    n_factors = cfg['svd']['n_factors']
    trainer = SVDTrainer(n_factors=n_factors)

    # MLflow tracking (now points to DagsHub)
    mlflow.set_experiment("CINEIQ_Collaborative_Filtering")
    with mlflow.start_run():
        mlflow.log_params({
            'n_factors': n_factors,
            'dataset_size': len(ratings_df),
            'n_unique_users': ratings_df['user_id'].nunique(),
            'n_unique_items': ratings_df['movie_id'].nunique(),
        })

        metrics = trainer.train(ratings_df)

        mlflow.log_metric("train_rmse", metrics['train_rmse'])
        mlflow.log_metric("train_mae", metrics['train_mae'])

        # Export artifacts locally
        trainer.export_artifacts(cfg['models']['output_dir'])

        # Log to DagsHub MLflow
        mlflow.log_artifacts(cfg['models']['output_dir'], artifact_path="svd_artifacts")
        print("Done. Logged to DagsHub MLflow.")

if __name__ == "__main__":
    main()
