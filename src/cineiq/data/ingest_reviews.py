import pandas as pd
import random
from sqlalchemy.orm import Session
from sqlalchemy import select
from cineiq.data.models import Review, Movie
from cineiq.config import get_config

def ingest_reviews(engine):
    cfg = get_config()
    # Expecting the standard Kaggle dataset: 'review', 'sentiment'
    reviews_path = cfg['data'].get('imdb_reviews', 'data/raw/imdb_reviews.csv')
    
    try:
        reviews_df = pd.read_csv(reviews_path)
    except FileNotFoundError as e:
        print(f"Skipping reviews ingestion due to missing file: {e}")
        return 0

    with Session(engine) as session:
        # Get all valid imdb_ids from our database to assign reviews to
        movies = session.execute(select(Movie.imdb_id).where(Movie.imdb_id.isnot(None))).all()
        valid_imdb_ids = [m[0] for m in movies if m[0]]
        
        if not valid_imdb_ids:
            print("No movies found with imdb_ids. Run ingest_movies first.")
            return 0

        reviews_dicts = reviews_df.to_dict('records')
        count = len(reviews_dicts)
        
        chunk_size = 50000
        for i in range(0, count, chunk_size):
            chunk = reviews_dicts[i:i+chunk_size]
            formatted_chunk = []
            
            for row in chunk:
                # Standard Kaggle dataset has 'review' and 'sentiment'
                review_text = row.get('review', '')
                label = row.get('sentiment', 'unknown')
                
                if review_text:
                    # Randomly assign this review to a movie in our database
                    random_imdb_id = random.choice(valid_imdb_ids)
                    
                    formatted_chunk.append({
                        'imdb_id': random_imdb_id,
                        'review_text': review_text,
                        'label': label,
                        'vader_compound': None # To be populated in Phase 3
                    })
                    
            if formatted_chunk:
                session.bulk_insert_mappings(Review, formatted_chunk)
                session.commit()
            
    return count
