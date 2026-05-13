import pandas as pd
from sqlalchemy.orm import Session
from cineiq.data.models import Movie
from cineiq.config import get_config

def ingest_movies(engine):
    cfg = get_config()
    ml_dir = cfg['data']['movielens_dir']
    
    # In a real scenario, we'd add error handling for missing files.
    # For now, we assume the user has placed the files in data/raw/
    try:
        movies_df = pd.read_csv(f"{ml_dir}/movies.csv")
        links_df = pd.read_csv(f"{ml_dir}/links.csv")
        tmdb_df = pd.read_csv(cfg['data']['tmdb_movies'])
    except FileNotFoundError as e:
        print(f"Skipping movie ingestion due to missing file: {e}")
        return 0

    # 1. Merge MovieLens movies with links to get tmdbId and imdbId
    ml_merged = pd.merge(movies_df, links_df, on='movieId', how='left')
    
    # 2. Merge with TMDB metadata
    # TMDB dataset 'id' column is the tmdbId
    tmdb_df = tmdb_df.rename(columns={'id': 'tmdbId'})
    final_df = pd.merge(ml_merged, tmdb_df, on='tmdbId', how='left')
    
    # Format imdbId to "ttXXXXXXX"
    final_df['imdbId_formatted'] = final_df['imdbId'].apply(
        lambda x: f"tt{int(x):07d}" if pd.notnull(x) else None
    )

    # 3. Insert into DB
    with Session(engine) as session:
        count = 0
        for _, row in final_df.iterrows():
            movie = Movie(
                movie_id=row['movieId'],
                tmdb_id=row['tmdbId'] if pd.notnull(row['tmdbId']) else None,
                imdb_id=row['imdbId_formatted'],
                title=row['title_x'] if pd.notnull(row['title_x']) else row['title_y'],
                genres=row['genres_x'] if pd.notnull(row['genres_x']) else '',
                overview=row.get('overview', ''),
                release_year=None, # Would parse from title or release_date
                vote_average=row.get('vote_average', 0.0),
                vote_count=row.get('vote_count', 0)
            )
            session.add(movie)
            count += 1
            if count % 10000 == 0:
                session.commit()
        session.commit()
    return count
