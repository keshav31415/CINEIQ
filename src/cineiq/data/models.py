from sqlalchemy import Column, Integer, Float, Text, String, ForeignKey
from cineiq.db import Base

class Movie(Base):
    __tablename__ = 'movies'
    movie_id = Column(Integer, primary_key=True)     # MovieLens movieId
    tmdb_id = Column(Integer, index=True, nullable=True)
    imdb_id = Column(String(20), index=True, nullable=True)  # e.g., "tt0114709"
    title = Column(Text, nullable=False)
    genres = Column(Text)                             # pipe-separated
    overview = Column(Text)                           # from TMDB
    release_year = Column(Integer)
    vote_average = Column(Float)
    vote_count = Column(Integer)
    poster_path = Column(String(100))

class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    movie_id = Column(Integer, ForeignKey('movies.movie_id'), nullable=False)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer)

class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True, autoincrement=True)
    imdb_id = Column(String(20), ForeignKey('movies.imdb_id'), index=True)
    review_text = Column(Text, nullable=False)
    label = Column(String(10))       # 'positive' / 'negative'
    vader_compound = Column(Float, nullable=True)  # filled in Phase 3
    source = Column(String(20), default='aclimdb')  # 'aclimdb' or 'tmdb'
