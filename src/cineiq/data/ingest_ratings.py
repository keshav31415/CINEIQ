import pandas as pd
from sqlalchemy.orm import Session
from cineiq.data.models import Rating
from cineiq.config import get_config

def ingest_ratings(engine):
    cfg = get_config()
    ml_dir = cfg['data']['movielens_dir']
    
    try:
        ratings_df = pd.read_csv(f"{ml_dir}/ratings.csv")
    except FileNotFoundError as e:
        print(f"Skipping ratings ingestion due to missing file: {e}")
        return 0

    with Session(engine) as session:
        # Instead of row-by-row, bulk insert for speed
        ratings_dicts = ratings_df.to_dict('records')
        count = len(ratings_dicts)
        
        # We can chunk this to avoid memory issues on huge datasets
        chunk_size = 50000
        for i in range(0, count, chunk_size):
            chunk = ratings_dicts[i:i+chunk_size]
            # convert dict keys to match our ORM: movieId -> movie_id, userId -> user_id
            formatted_chunk = [
                {
                    'user_id': row['userId'],
                    'movie_id': row['movieId'],
                    'rating': row['rating'],
                    'timestamp': row['timestamp']
                } for row in chunk
            ]
            session.bulk_insert_mappings(Rating, formatted_chunk)
            session.commit()
            
    return count
