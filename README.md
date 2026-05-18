# CINEIQ — Intelligent Movie Recommendation Engine

A hybrid recommender system that blends collaborative filtering, content-based embeddings, and aspect-based sentiment analysis into a unified ensemble. Ships with a Netflix-style web UI and a user taste intelligence dashboard.

---

## Features

- **Hybrid ensemble** — SVD collaborative filtering + MiniLM-L6 content embeddings + VADER/ABSA sentiment, weights tuned with Optuna (NDCG@10 +88% over baseline)
- **Aspect-Based Sentiment** — per-movie scores for acting, plot, visuals, pacing, and music derived from Stanford aclImdb reviews
- **Explainability** — every recommendation surfaces a human-readable reason (taste match, content similarity, audience reception)
- **Netflix-style UI** — FastAPI-served HTML frontend with personalised rows, hero banner, and search
- **Taste Intelligence Dashboard** — Streamlit app with genre radar, decade heatmap, Bayesian-adjusted preferences, and ABSA profile

---

## Architecture

```
Raw Data  (MovieLens 100K · TMDB 5000 · Stanford aclImdb)
     │
     ▼
SQLite Database  ─────────────────────────────────────────────────────────────
     │                                                                        │
     ├── SVD (scipy sparse)  ──► svd_*.npy                                   │
     ├── Content embeddings  ──► movie_embeddings.npy   (MiniLM-L6-v2)       │
     └── Sentiment scoring   ──► *_sentiment_cache.json (VADER + pyABSA)     │
                                           │                                  │
                    Ensemble  (α·SVD + β·Content + γ·Sentiment)              │
                    Optuna-tuned  ─── writes back to config.yaml             │
                                           │                                  │
                         ┌─────────────────┴──────────────────┐              │
                         ▼                                      ▼             │
               FastAPI REST API                      Streamlit Dashboard ◄────┘
             localhost:8000                            localhost:8501
          (index.html Netflix UI)               (Taste Intelligence Report)
```

---

## Data Sources

Download these datasets and place them as shown in the [Project Structure](#project-structure) section.

| Dataset | Source |
|---------|--------|
| MovieLens 100K | https://grouplens.org/datasets/movielens/100k/ |
| TMDB 5000 Movies + Credits | https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata |
| Stanford aclImdb | http://ai.stanford.edu/~amaas/data/sentiment/ |

---

## Setup

### Prerequisites

- Python 3.10+
- A free TMDB API key — https://www.themoviedb.org/settings/api

### 1. Clone and install

```bash
git clone https://github.com/keshav31415/cineiq.git
cd cineiq
```

**Using `uv` (recommended — faster):**
```bash
pip install uv
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set TMDB_API_KEY=your_key_here
```

---

## Quick Start — Download Pre-trained Artifacts

> Skip the training pipeline entirely. One command fetches the trained models and
> database (~50 MB) from the GitHub Release.

```bash
python scripts/00_download_artifacts.py
```

This places all artifacts in `models/` and `data/processed/`. Then jump straight to
[Starting the Services](#starting-the-services).

---

## Full Training Pipeline

Only needed if you want to retrain from scratch (raw datasets required).

### Place raw datasets

```
data/raw/
├── ml-100k/                  ← unzip ml-100k.zip here
│   ├── u.data
│   ├── u.item
│   └── ...
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── aclImdb/                  ← unzip aclimdb_v1.tar.gz here
    ├── train/
    └── test/
```

### Run the pipeline

```bash
python scripts/01_ingest_data.py          # Build SQLite DB from raw files
python scripts/02_train_svd.py            # Train SVD collaborative filter
python scripts/03_generate_embeddings.py  # Generate MiniLM-L6 content embeddings
python scripts/04_precompute_sentiment.py # Precompute VADER + ABSA sentiment scores
python scripts/05_tune_ensemble.py        # Tune ensemble weights with Optuna (~5 min)
```

Outputs land in `models/` and `data/processed/cineiq.db`.
`config.yaml` is updated automatically with the best ensemble weights.

---

## Starting the Services

Open two terminals from the project root:

```bash
# Terminal 1 — REST API + Netflix UI
python run_api.py
# → http://localhost:8000

# Terminal 2 — Taste Intelligence Dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
```

The Netflix UI is at `http://localhost:8000`. Click **Taste Dashboard** in the navbar to open the Streamlit report.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status and model version |
| `GET` | `/recommend/{user_id}?top_k=10` | Personalised recommendations |
| `GET` | `/movie/{movie_id}` | Movie metadata |
| `GET` | `/search?q=query` | Title search |

---

## Configuration

All runtime settings live in `config.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `svd.n_factors` | 100 | Latent dimensions for SVD |
| `svd.n_epochs` | 20 | Training epochs |
| `ensemble.alpha` | 0.765 | Collaborative filtering weight |
| `ensemble.beta` | 0.101 | Content similarity weight |
| `ensemble.gamma` | 0.088 | Sentiment weight |
| `ensemble.candidate_pool` | 30 | Stage-1 candidate count |
| `ensemble.final_top_k` | 10 | Final recommendations returned |
| `api.port` | 8000 | FastAPI port |

---

## Project Structure

```
cineiq/
├── src/cineiq/               # Core library (installable package)
│   ├── api/                  # FastAPI app, routes, schemas
│   ├── data/                 # Data ingestion & SQLAlchemy ORM
│   ├── features/             # Content embeddings & sentiment scoring
│   └── training/             # SVD, content model, hybrid ensemble
│
├── scripts/                  # One-time pipeline scripts (run in order)
│   ├── 01_ingest_data.py
│   ├── 02_train_svd.py
│   ├── 03_generate_embeddings.py
│   ├── 04_precompute_sentiment.py
│   └── 05_tune_ensemble.py
│
├── dashboard/
│   ├── app.py                # Streamlit taste intelligence dashboard
│   └── index.html            # Netflix-style UI (served by FastAPI)
│
├── data/
│   ├── raw/                  # Source datasets (git-ignored)
│   └── processed/            # SQLite database (git-ignored)
│
├── models/                   # Trained artifacts (git-ignored)
│
├── config.yaml               # Runtime configuration
├── run_api.py                # API server entry point
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Collaborative filtering | SVD via `scipy.sparse.linalg.svds` |
| Content embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Sentiment (scalar) | VADER (`vaderSentiment`) |
| Sentiment (aspect) | `pyABSA` (ATEPC multilingual checkpoint) |
| Hyperparameter tuning | `optuna` (TPE sampler, NDCG@10) |
| REST API | `FastAPI` + `uvicorn` |
| Database | `SQLite` via `SQLAlchemy` |
| Dashboard | `Streamlit` + `Plotly` |
| Experiment tracking | `MLflow` + DagsHub |
