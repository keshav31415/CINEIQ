import sys
from pathlib import Path

# Ensure the src directory is in the path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from cineiq.db import Base, get_engine
from cineiq.data.ingest_movies import ingest_movies
from cineiq.data.ingest_ratings import ingest_ratings
from cineiq.data.ingest_reviews import ingest_reviews

def main():
    print("Initializing Database Engine...")
    engine = get_engine()
    
    print("Recreating tables to ensure a clean run...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    print("Step 1/3: Ingesting movies...")
    n_movies = ingest_movies(engine)
    print(f"  -> {n_movies} movies loaded")
    
    print("Step 2/3: Ingesting ratings...")
    n_ratings = ingest_ratings(engine)
    print(f"  -> {n_ratings} ratings loaded")
    
    print("Step 3/3: Ingesting reviews...")
    n_reviews = ingest_reviews(engine)
    print(f"  -> {n_reviews} reviews loaded")

if __name__ == "__main__":
    main()
