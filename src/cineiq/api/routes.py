import sqlite3

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pathlib import Path
from sqlalchemy import select, text

from cineiq.data.models import Rating, Movie
from cineiq.api.schemas import (
    HealthResponse, RecommendationResponse, RecommendationItem,
    ColdStartRequest, MovieResponse, CatalogItem,
)

router = APIRouter()

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
MODEL_VERSION   = "v1.0-svd200-ml25m"
UI_DIR          = Path(__file__).parent.parent.parent.parent / "dashboard"


def _build_poster_url(poster_path: str | None) -> str | None:
    if poster_path and str(poster_path) != 'nan':
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    return None


def _enrich_recommendations(recs: list[dict], session) -> list[RecommendationItem]:
    """Fetch movie metadata from DB and build response items."""
    movie_ids = [r['movie_id'] for r in recs]
    movies = pd.read_sql(
        select(Movie.movie_id, Movie.title, Movie.genres, Movie.poster_path)
        .where(Movie.movie_id.in_(movie_ids)),
        session.bind
    )
    title_map = dict(zip(movies['movie_id'], movies['title']))
    genre_map = dict(zip(movies['movie_id'], movies['genres']))
    poster_map = dict(zip(movies['movie_id'], movies['poster_path']))

    items = []
    for rank, rec in enumerate(recs, 1):
        mid = rec['movie_id']
        genres_raw = genre_map.get(mid, '')
        genres = genres_raw.split('|') if genres_raw else []

        items.append(RecommendationItem(
            rank=rank,
            movie_id=mid,
            title=title_map.get(mid, 'Unknown'),
            genres=genres,
            score=rec['score'],
            poster_url=_build_poster_url(poster_map.get(mid)),
            explanation=rec['explanation'],
            debug={
                'svd_score': rec['svd_score'],
                'content_score': rec['content_score'],
                'sentiment_score': rec['sentiment_score'],
            }
        ))
    return items


# ─── Health ──────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    ensemble = request.app.state.ensemble
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        n_users=ensemble.user_factors.shape[0],
        n_items=ensemble.item_factors.shape[0],
        n_sentiment_scored=len(ensemble.sentiment_cache),
    )


# ─── Recommend for known user ───────────────────────────────────────
@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    request: Request,
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
):
    ensemble = request.app.state.ensemble
    session = request.app.state.db_session

    # Check if user exists in SVD model
    if user_id not in ensemble.svd_user_map:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in model. Use /recommend/cold-start instead.")

    # Get user's rated movies
    rated = pd.read_sql(
        select(Rating.movie_id).where(Rating.user_id == user_id),
        session.bind
    )
    rated_ids = rated['movie_id'].tolist()

    # Override top_k temporarily
    original_k = ensemble.top_k
    ensemble.top_k = top_k
    recs = ensemble.recommend(user_id, rated_ids)
    ensemble.top_k = original_k

    items = _enrich_recommendations(recs, session)

    return RecommendationResponse(
        user_id=user_id,
        model_version=MODEL_VERSION,
        recommendations=items,
    )


# ─── Cold-start recommendation ──────────────────────────────────────
@router.post("/recommend/cold-start", response_model=RecommendationResponse)
def cold_start(request: Request, body: ColdStartRequest):
    ensemble = request.app.state.ensemble
    session = request.app.state.db_session

    recs = ensemble.recommend_cold_start(body.seed_movie_ids, top_k=body.top_k)
    items = _enrich_recommendations(recs, session)

    return RecommendationResponse(
        user_id=None,
        model_version=MODEL_VERSION,
        recommendations=items,
    )


# ─── Similar movies ─────────────────────────────────────────────────
@router.get("/similar/{movie_id}", response_model=RecommendationResponse)
def similar(
    request: Request,
    movie_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
):
    ensemble = request.app.state.ensemble
    session = request.app.state.db_session

    if movie_id not in ensemble.content_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")

    # Use cold-start with a single seed movie
    recs = ensemble.recommend_cold_start([movie_id], top_k=top_k)
    items = _enrich_recommendations(recs, session)

    return RecommendationResponse(
        user_id=None,
        model_version=MODEL_VERSION,
        recommendations=items,
    )


# ─── Movie metadata ─────────────────────────────────────────────────
@router.get("/movie/{movie_id}", response_model=MovieResponse)
def get_movie(request: Request, movie_id: int):
    session = request.app.state.db_session
    ensemble = request.app.state.ensemble

    movie = pd.read_sql(
        select(Movie).where(Movie.movie_id == movie_id),
        session.bind
    )

    if movie.empty:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")

    row = movie.iloc[0]
    genres = row['genres'].split('|') if row['genres'] else []
    sentiment = ensemble.sentiment_cache.get(str(movie_id))

    return MovieResponse(
        movie_id=row['movie_id'],
        title=row['title'],
        genres=genres,
        overview=row['overview'] if str(row['overview']) != 'nan' else None,
        release_year=row['release_year'] if pd.notnull(row['release_year']) else None,
        vote_average=row['vote_average'] if pd.notnull(row['vote_average']) else None,
        vote_count=int(row['vote_count']) if pd.notnull(row['vote_count']) else None,
        poster_url=_build_poster_url(row['poster_path']),
        sentiment_score=sentiment,
    )


# ─── Catalog helpers ─────────────────────────────────────────────────

def _catalog_query(session, sql: str, params: dict, ensemble) -> list[CatalogItem]:
    rows = pd.read_sql(text(sql), session.bind, params=params)
    items = []
    for _, r in rows.iterrows():
        genres = r['genres'].split('|') if r.get('genres') else []
        items.append(CatalogItem(
            movie_id=int(r['movie_id']),
            title=r['title'],
            genres=genres,
            overview=r['overview'] if str(r.get('overview', '')) not in ('nan', 'None', '') else None,
            release_year=int(r['release_year']) if pd.notnull(r.get('release_year')) else None,
            vote_average=float(r['vote_average']) if pd.notnull(r.get('vote_average')) else None,
            poster_url=_build_poster_url(r.get('poster_path')),
            sentiment_score=ensemble.sentiment_cache.get(str(int(r['movie_id']))),
        ))
    return items


# ─── Search ──────────────────────────────────────────────────────────

@router.get("/search", response_model=list[CatalogItem])
def search(
    request: Request,
    q: str = Query(..., min_length=1),
    n: int = Query(default=24, ge=1, le=50),
):
    session  = request.app.state.db_session
    ensemble = request.app.state.ensemble
    return _catalog_query(session, """
        SELECT movie_id, title, genres, overview, release_year, vote_average, poster_path
        FROM movies WHERE title LIKE :q ORDER BY vote_average DESC LIMIT :n
    """, {"q": f"%{q}%", "n": n}, ensemble)


# ─── User rated movies ────────────────────────────────────────────────

@router.get("/user/{user_id}/rated")
def user_rated(request: Request, user_id: int):
    session = request.app.state.db_session
    rated = pd.read_sql(
        select(Rating.movie_id, Rating.rating).where(Rating.user_id == user_id),
        session.bind
    )
    if rated.empty:
        return {"user_id": user_id, "rated": []}
    top = rated.nlargest(1, 'rating').iloc[0]
    return {
        "user_id": user_id,
        "rated_ids": rated['movie_id'].tolist(),
        "top_movie_id": int(top['movie_id']),
    }


# ─── Catalog rows ────────────────────────────────────────────────────

@router.get("/catalog/genre", response_model=list[CatalogItem])
def catalog_genre(
    request: Request,
    genre: str = Query(...),
    n: int = Query(default=20, ge=1, le=40),
):
    session  = request.app.state.db_session
    ensemble = request.app.state.ensemble
    return _catalog_query(session, """
        SELECT movie_id, title, genres, overview, release_year, vote_average, poster_path
        FROM movies WHERE genres LIKE :g AND vote_count > 50
        ORDER BY vote_average DESC LIMIT :n
    """, {"g": f"%{genre}%", "n": n}, ensemble)


@router.get("/catalog/top-rated", response_model=list[CatalogItem])
def catalog_top_rated(
    request: Request,
    n: int = Query(default=20, ge=1, le=40),
):
    session  = request.app.state.db_session
    ensemble = request.app.state.ensemble
    return _catalog_query(session, """
        SELECT movie_id, title, genres, overview, release_year, vote_average, poster_path
        FROM movies WHERE vote_count > 200 ORDER BY vote_average DESC LIMIT :n
    """, {"n": n}, ensemble)


@router.get("/catalog/hidden-gems", response_model=list[CatalogItem])
def catalog_hidden_gems(
    request: Request,
    n: int = Query(default=20, ge=1, le=40),
):
    session  = request.app.state.db_session
    ensemble = request.app.state.ensemble
    return _catalog_query(session, """
        SELECT movie_id, title, genres, overview, release_year, vote_average, poster_path
        FROM movies WHERE vote_count BETWEEN 10 AND 80 AND vote_average >= 7.5
        ORDER BY vote_average DESC LIMIT :n
    """, {"n": n}, ensemble)


@router.get("/catalog/classics", response_model=list[CatalogItem])
def catalog_classics(
    request: Request,
    n: int = Query(default=20, ge=1, le=40),
):
    session  = request.app.state.db_session
    ensemble = request.app.state.ensemble
    return _catalog_query(session, """
        SELECT movie_id, title, genres, overview, release_year, vote_average, poster_path
        FROM movies WHERE release_year < 1990 AND vote_count > 25
        ORDER BY vote_average DESC LIMIT :n
    """, {"n": n}, ensemble)


# ─── Serve HTML UI ───────────────────────────────────────────────────

@router.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")
