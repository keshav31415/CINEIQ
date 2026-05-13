import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent  # dashboard/app.py → project root
sys.path.insert(0, str(ROOT / "src"))

from cineiq.config import get_config
from cineiq.training.ensemble import HybridEnsemble


# ─── Cached resources ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models...")
def load_ensemble() -> HybridEnsemble:
    cfg = get_config()
    return HybridEnsemble(
        models_dir=str(ROOT / cfg["models"]["output_dir"]),
        alpha=cfg["ensemble"]["alpha"],
        beta=cfg["ensemble"]["beta"],
        gamma=cfg["ensemble"]["gamma"],
        candidate_pool=cfg["ensemble"]["candidate_pool"],
        top_k=cfg["ensemble"]["final_top_k"],
    )


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    cfg = get_config()
    return sqlite3.connect(str(ROOT / cfg["data"]["database"]), check_same_thread=False)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _extract_year(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None


@st.cache_data(show_spinner=False)
def load_user_ratings(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT r.movie_id, r.rating, m.title, m.genres
           FROM ratings r
           JOIN movies m ON r.movie_id = m.movie_id
           WHERE r.user_id = ?""",
        conn,
        params=[user_id],
    )
    df["year"] = df["title"].apply(_extract_year)
    df["decade"] = df["year"].apply(
        lambda y: f"{(y // 10) * 10}s" if y is not None else None
    )
    return df


def compute_genre_affinity(df: pd.DataFrame, min_count: int = 2) -> dict[str, float]:
    rows = []
    for _, row in df.iterrows():
        if not row["genres"]:
            continue
        for g in row["genres"].split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                rows.append({"genre": g, "rating": row["rating"]})
    if not rows:
        return {}
    gdf = pd.DataFrame(rows)
    agg = gdf.groupby("genre").agg(avg=("rating", "mean"), cnt=("rating", "count"))
    return agg[agg["cnt"] >= min_count]["avg"].to_dict()


# ─── Chart builders ──────────────────────────────────────────────────────────

def _radar_chart(affinities: dict) -> go.Figure:
    genres = sorted(affinities.keys())
    vals = [affinities[g] for g in genres]
    theta = genres + [genres[0]]
    r = vals + [vals[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line_color="#7C3AED",
            fillcolor="rgba(124,58,237,0.2)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont_size=9),
            angularaxis=dict(tickfont_size=11),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=30, b=30),
        height=380,
    )
    return fig


def _decade_bar(df: pd.DataFrame) -> go.Figure:
    ddf = (
        df.dropna(subset=["decade"])
        .groupby("decade")
        .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
        .reset_index()
        .sort_values("decade")
    )
    fig = px.bar(
        ddf,
        x="decade",
        y="avg_rating",
        color="avg_rating",
        color_continuous_scale=["#374151", "#7C3AED"],
        text=ddf["count"].apply(lambda c: f"{c}"),
        labels={"avg_rating": "Avg Rating", "decade": "Decade"},
        range_y=[0, 5.5],
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=10, b=10),
        height=380,
        xaxis=dict(type="category"),
        yaxis_title="Avg Rating",
    )
    return fig


def _rating_dist(df: pd.DataFrame) -> go.Figure:
    counts = df["rating"].value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color="#7C3AED",
            opacity=0.8,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        height=220,
        xaxis_title="Rating",
        yaxis_title="Count",
        xaxis=dict(tickvals=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
    )
    return fig


# ─── App layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="CINEIQ", page_icon="🎬", layout="wide")

st.markdown(
    """
    <h1 style='font-family:monospace;letter-spacing:6px;margin-bottom:2px'>CINEIQ</h1>
    <p style='color:#9CA3AF;margin-top:0;font-size:0.9em'>
        Hybrid Recommendation Engine &nbsp;·&nbsp; User Taste Dashboard
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

ensemble = load_ensemble()
conn = get_conn()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    user_id = st.number_input(
        "User ID", min_value=1, max_value=610, value=42, step=1,
        help="MovieLens 100K users: 1–610",
    )
    top_k = st.slider("Recommendations", min_value=5, max_value=20, value=10)
    st.divider()
    st.markdown("**Ensemble weights**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("α SVD", ensemble.alpha)
    col_b.metric("β Content", ensemble.beta)
    col_c.metric("γ Sentiment", ensemble.gamma)
    st.divider()
    st.caption("Model: SVD (100 factors) + MiniLM-L6 + VADER")

# ─── Load user data ───────────────────────────────────────────────────────────

df = load_user_ratings(user_id)

if df.empty:
    st.warning(f"No ratings found for user {user_id}.")
    st.stop()

affinities = compute_genre_affinity(df)
top_genre = max(affinities, key=affinities.get) if affinities else "N/A"
fav_decade_series = (
    df.dropna(subset=["decade"]).groupby("decade")["rating"].mean()
)
fav_decade = fav_decade_series.idxmax() if not fav_decade_series.empty else "N/A"

# ─── Stat cards ──────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies Rated", f"{len(df):,}")
c2.metric("Avg Rating", f"{df['rating'].mean():.2f} / 5")
c3.metric("Favourite Decade", fav_decade)
c4.metric("Top Genre", top_genre)

st.divider()

# ─── Taste charts ─────────────────────────────────────────────────────────────

tab_radar, tab_decade, tab_dist = st.tabs(
    ["Genre Radar", "Decade Preferences", "Rating Distribution"]
)

with tab_radar:
    if len(affinities) >= 3:
        st.plotly_chart(_radar_chart(affinities), width="stretch")
    elif affinities:
        st.info("Rate movies across at least 3 genres to see the radar chart.")
    else:
        st.info("Not enough data — need ≥2 ratings per genre.")

with tab_decade:
    if not df.dropna(subset=["decade"]).empty:
        st.plotly_chart(_decade_bar(df), width="stretch")
    else:
        st.info("No year data available for this user's movies.")

with tab_dist:
    st.plotly_chart(_rating_dist(df), width="stretch")

st.divider()

# ─── Recommendations ─────────────────────────────────────────────────────────

st.subheader(f"🎯 Top {top_k} Recommendations for User {user_id}")

with st.spinner("Running hybrid ensemble..."):
    original_k = ensemble.top_k
    ensemble.top_k = top_k
    recs = ensemble.recommend(user_id, df["movie_id"].tolist())
    ensemble.top_k = original_k

if not recs:
    st.info("No recommendations generated.")
    st.stop()

movie_ids = [r["movie_id"] for r in recs]
placeholders = ",".join("?" * len(movie_ids))
meta_df = pd.read_sql_query(
    f"SELECT movie_id, title, genres, overview FROM movies WHERE movie_id IN ({placeholders})",
    conn,
    params=movie_ids,
)
meta = {int(row.movie_id): row for row in meta_df.itertuples()}

for i, rec in enumerate(recs):
    mid = int(rec["movie_id"])
    info = meta.get(mid)
    title = info.title if info else f"Movie {mid}"
    genres_str = (
        info.genres.replace("|", " · ")
        if (info and info.genres and str(info.genres) != "nan")
        else "N/A"
    )
    overview = (
        (str(info.overview)[:220] + "…")
        if (info and info.overview and str(info.overview) not in ("nan", "None"))
        else ""
    )

    with st.container(border=True):
        left_col, right_col = st.columns([4, 1])

        with left_col:
            st.markdown(
                f"**#{i + 1} &nbsp; {title}**",
            )
            st.caption(genres_str)
            if overview:
                st.caption(overview)
            for reason in rec["explanation"]:
                st.markdown(f"› {reason}")

        with right_col:
            st.metric("Score", f"{rec['score']:.3f}")
            st.caption(
                f"SVD `{rec['svd_score']:.2f}`  \n"
                f"Content `{rec['content_score']:.2f}`  \n"
                f"Sentiment `{rec['sentiment_score']:.2f}`"
            )
