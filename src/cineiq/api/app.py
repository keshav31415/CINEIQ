from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cineiq.config import get_config
from cineiq.db import get_session
from cineiq.training.ensemble import HybridEnsemble
from cineiq.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts into memory once at server startup."""
    cfg = get_config()

    print("Loading ML artifacts...")
    ensemble = HybridEnsemble(
        models_dir=cfg['models']['output_dir'],
        alpha=cfg['ensemble']['alpha'],
        beta=cfg['ensemble']['beta'],
        gamma=cfg['ensemble']['gamma'],
        candidate_pool=cfg['ensemble']['candidate_pool'],
        top_k=cfg['ensemble']['final_top_k'],
    )
    app.state.ensemble = ensemble
    app.state.db_session = get_session()

    print("CINEIQ API ready.")
    yield

    # Cleanup
    app.state.db_session.close()


app = FastAPI(
    title="CINEIQ",
    description="Hybrid movie recommendation engine combining SVD collaborative filtering, "
                "content-based embeddings, and VADER sentiment re-ranking.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
