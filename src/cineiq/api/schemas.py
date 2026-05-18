from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    n_users: int
    n_items: int
    n_sentiment_scored: int


class RecommendationItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: list[str]
    score: float
    poster_url: str | None
    explanation: list[str]
    debug: dict


class RecommendationResponse(BaseModel):
    user_id: int | None
    model_version: str
    recommendations: list[RecommendationItem]


class ColdStartRequest(BaseModel):
    seed_movie_ids: list[int] = Field(
        ..., min_length=1, max_length=20,
        description="List of movie IDs the user likes"
    )
    top_k: int = Field(default=10, ge=1, le=50)


class CatalogItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    overview: str | None
    release_year: int | None
    vote_average: float | None
    poster_url: str | None
    sentiment_score: float | None


class MovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    overview: str | None
    release_year: int | None
    vote_average: float | None
    vote_count: int | None
    poster_url: str | None
    sentiment_score: float | None
