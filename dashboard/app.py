"""
CINEIQ · User Taste Intelligence Dashboard
"""
import re, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cineiq.config import get_config
from cineiq.training.ensemble import HybridEnsemble

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CINEIQ · Taste Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURPLE   = "#7C3AED"
PURPLE_L = "#A78BFA"
BG_CARD  = "#1f2937"
BG_DARK  = "#111827"
TEXT_DIM = "#9CA3AF"

st.markdown(f"""
<style>
  .main {{ background-color: {BG_DARK}; }}
  section[data-testid="stSidebar"] {{ background-color: #0f1623; }}
  .badge-wrap {{
    display:inline-flex; align-items:center; gap:12px;
    background:{BG_CARD}; border:1px solid {PURPLE};
    border-radius:12px; padding:14px 22px; margin:8px 0 18px 0;
  }}
  .badge-icon {{ font-size:2.2em; }}
  .badge-title {{ font-size:1.15em; font-weight:800; color:#F9FAFB; }}
  .badge-sub   {{ font-size:0.82em; color:{TEXT_DIM}; margin-top:2px; }}
  .kpi-card {{
    background:{BG_CARD}; border-radius:10px; padding:16px 20px;
    border-left:3px solid {PURPLE}; margin-bottom:4px;
  }}
  .kpi-label {{ font-size:0.72em; color:{TEXT_DIM}; text-transform:uppercase; letter-spacing:.06em; }}
  .kpi-value {{ font-size:1.6em; font-weight:800; color:#F9FAFB; line-height:1.15; }}
  .kpi-sub   {{ font-size:0.72em; color:{PURPLE_L}; margin-top:2px; }}
  h2, h3 {{ color:#E5E7EB !important; }}
  .section-title {{
    font-size:0.75em; font-weight:700; color:{PURPLE_L};
    text-transform:uppercase; letter-spacing:.1em; margin:28px 0 10px 0;
  }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#D1D5DB", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=30, b=20),
)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models…")
def load_ensemble():
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
def get_db():
    import sqlite3
    cfg = get_config()
    return sqlite3.connect(str(ROOT / cfg["data"]["database"]), check_same_thread=False)

@st.cache_resource
def load_absa():
    p = ROOT / "models" / "absa_sentiment_cache.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _year(title):
    m = re.search(r"\((\d{4})\)", str(title))
    return int(m.group(1)) if m else None

@st.cache_data(show_spinner=False, ttl=300)
def user_ratings(uid: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT r.movie_id, r.rating, r.timestamp,
                  m.title, m.genres, m.vote_average, m.vote_count, m.release_year
           FROM ratings r JOIN movies m USING(movie_id)
           WHERE r.user_id = ?""",
        get_db(), params=[uid],
    )
    df["year"]   = df.apply(lambda r: r["release_year"] if r["release_year"] else _year(r["title"]), axis=1)
    df["decade"] = df["year"].apply(lambda y: f"{(int(y)//10)*10}s" if pd.notnull(y) else None)
    return df

def _enrich_explanations(recs: list, df: pd.DataFrame, ens, absa_cache: dict, conn):
    """Replace generic explanation strings with specific, data-driven reasons."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # User's rated movies: {movie_id: (rating, short_title)}
    rated_info = {}
    for row in df.itertuples():
        short = str(row.title).split("(")[0].strip()
        rated_info[int(row.movie_id)] = (row.rating, short)

    # Batch query: count of users who rated each rec 4★+
    movie_ids = [r["movie_id"] for r in recs]
    ph = ",".join("?" * len(movie_ids))
    fan_df = pd.read_sql_query(
        f"SELECT movie_id, COUNT(*) as cnt FROM ratings "
        f"WHERE movie_id IN ({ph}) AND rating >= 4.0 GROUP BY movie_id",
        conn, params=movie_ids,
    )
    fan_counts = dict(zip(fan_df["movie_id"].astype(int), fan_df["cnt"]))

    # Pre-stack rated-movie embeddings for vectorized similarity
    rated_entries = [
        (mid, ens.content_id_to_idx[mid], info[0], info[1])
        for mid, info in rated_info.items()
        if mid in ens.content_id_to_idx
    ]
    if rated_entries:
        rated_embs = ens.embeddings[[e[1] for e in rated_entries]]

    ASPECTS = ["acting", "plot", "visuals", "pacing", "music"]

    for rec in recs:
        mid      = rec["movie_id"]
        svd_raw  = rec["svd_score"]
        reasons  = []

        try:
            # ── 1. Collaborative signal ───────────────────────────────────────
            fans = fan_counts.get(mid, 0)
            if fans > 0:
                reasons.append(f"Rated 4★+ by {fans:,} users with similar viewing history")
            if svd_raw > 3.5:
                reasons.append(f"Collaborative model predicts {svd_raw:.1f}★ for you")

            # ── 2. Content signal — most similar rated movie ──────────────────
            if mid in ens.content_id_to_idx and rated_entries:
                movie_emb = ens.embeddings[ens.content_id_to_idx[mid]].reshape(1, -1)
                sims      = cos_sim(movie_emb, rated_embs)[0]
                best_i    = int(sims.argmax())
                best_sim  = float(sims[best_i])
                if best_sim > 0.25:
                    best_rating = rated_entries[best_i][2]
                    best_title  = rated_entries[best_i][3]
                    reasons.append(
                        f"{best_sim * 100:.0f}% content match with \"{best_title}\" "
                        f"(your {best_rating:.0f}★)"
                    )

            # ── 3. ABSA signal — top praised aspect ──────────────────────────
            absa = absa_cache.get(str(mid))
            if absa:
                asp_scores = {a: absa[a] for a in ASPECTS if absa.get(a) is not None}
                if asp_scores:
                    top_asp = max(asp_scores, key=asp_scores.get)
                    top_val = asp_scores[top_asp]
                    if top_val > 0.1:
                        reasons.append(
                            f"Reviewers consistently praise its {top_asp} "
                            f"(sentiment +{top_val:.2f})"
                        )
                    elif top_val < -0.15:
                        reasons.append(f"Note: mixed reviews on {top_asp} ({top_val:.2f})")
        except Exception:
            pass  # fall through to default reason

        rec["explanation"] = reasons if reasons else [
            "Recommended based on your overall taste profile"
        ]


@st.cache_data(show_spinner=False, ttl=300)
def run_recs(uid: int, top_k: int):
    ens      = load_ensemble()
    df       = user_ratings(uid)
    absa     = load_absa()
    conn     = get_db()
    seen     = df["movie_id"].tolist()
    ens.top_k = top_k
    recs     = ens.recommend(uid, seen)
    if not recs:
        return []

    # Enrich with movie metadata
    ids  = [r["movie_id"] for r in recs]
    ph   = ",".join("?" * len(ids))
    meta = pd.read_sql_query(
        f"SELECT movie_id,title,genres,overview,vote_average,poster_path "
        f"FROM movies WHERE movie_id IN ({ph})",
        conn, params=ids,
    )
    meta_map = {int(r.movie_id): r for r in meta.itertuples()}
    for rec in recs:
        m = meta_map.get(rec["movie_id"])
        rec["title"]    = m.title if m else f"Movie {rec['movie_id']}"
        rec["genres"]   = (m.genres.split("|") if m and m.genres and str(m.genres) != "nan" else [])
        rec["poster"]   = m.poster_path if m and str(getattr(m, "poster_path", "nan")) != "nan" else None
        rec["tmdb_avg"] = m.vote_average if m else None

    # Enrich with specific, data-driven explanations
    _enrich_explanations(recs, df, ens, absa, conn)
    return recs


# ── Personality badge ─────────────────────────────────────────────────────────

def _bayesian_fav_decade(df: pd.DataFrame, C: int = 8):
    """Return the favourite decade by Bayesian-adjusted avg (same C as the chart)."""
    sub = df.dropna(subset=["decade"])
    if sub.empty:
        return None, pd.Series(dtype=float)
    stats = sub.groupby("decade").agg(avg=("rating", "mean"), n=("rating", "count"))
    global_mean = df["rating"].mean()
    stats["bayes"] = (C * global_mean + stats["n"] * stats["avg"]) / (C + stats["n"])
    return stats["bayes"].idxmax(), stats["avg"]   # fav decade + raw avgs for display


def personality(df: pd.DataFrame, affinities: dict, niche_score: float):
    top_g = max(affinities, key=affinities.get) if affinities else None
    avg_r = df["rating"].mean()
    fav_dec, _ = _bayesian_fav_decade(df)
    n = len(df)

    if n > 400:
        return "🎥", "The Completionist", f"You've catalogued {n:,} films — few match your dedication"
    if niche_score > 0.40:
        return "🔍", "Hidden Gem Hunter", "You consistently surface films the mainstream overlooks"
    if fav_dec in ("1940s", "1950s", "1960s"):
        return "🎞", "Classic Cinema Devotee", "Your heart belongs to Hollywood's golden age"
    if fav_dec in ("1970s", "1980s"):
        return "📼", "New Hollywood Purist", "You trust the era when directors had final cut"
    if avg_r >= 4.2:
        return "🌟", "Cinema Enthusiast", "You find joy in almost everything you watch"
    if avg_r <= 2.5:
        return "🎭", "The Discerning Critic", "You hold films to an exceptionally high standard"
    if top_g in ("Drama", "Thriller", "Crime"):
        return "🎬", "Drama Connoisseur", f"You gravitate toward tension, depth and moral complexity"
    if top_g in ("Documentary",):
        return "📡", "Documentary Devotee", "You value truth over fiction — reality is compelling enough"
    if top_g in ("Action", "Adventure", "Sci-Fi"):
        return "🚀", "Spectacle Seeker", "You love the visceral energy of big-screen storytelling"
    if top_g in ("Animation", "Family"):
        return "✨", "Animation Aficionado", "You appreciate the craft of bringing imagination to life"
    return "🎪", "Eclectic Viewer", "Your taste spans genres and eras with equal curiosity"


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_radar(affinities: dict):
    genres = sorted(affinities, key=affinities.get, reverse=True)[:12]
    vals   = [affinities[g] for g in genres]
    theta  = genres + [genres[0]]
    r      = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta, fill="toself",
        line=dict(color=PURPLE, width=2),
        fillcolor="rgba(124,58,237,0.18)",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 5], tickfont_size=9,
                            gridcolor="#374151", linecolor="#374151"),
            angularaxis=dict(tickfont_size=11, gridcolor="#374151", linecolor="#374151"),
        ),
        showlegend=False, height=360,
    )
    return fig


def chart_decade_bar(df: pd.DataFrame):
    ddf = (
        df.dropna(subset=["decade"])
          .groupby("decade")
          .agg(avg=("rating", "mean"), n=("rating", "count"))
          .reset_index().sort_values("decade")
    )

    # Bayesian average: shrink low-count decades toward the global mean.
    # Formula: (C * global_mean + n * raw_avg) / (C + n)
    # C = confidence threshold — a decade needs ~this many films to "earn" its rating.
    global_mean = df["rating"].mean()
    C = 8
    ddf["bayes_avg"] = (C * global_mean + ddf["n"] * ddf["avg"]) / (C + ddf["n"])

    fig = go.Figure()

    # Bars: height = Bayesian avg, color = Bayesian avg
    fig.add_bar(
        name="Adjusted avg",
        x=ddf["decade"], y=ddf["bayes_avg"],
        marker=dict(
            color=ddf["bayes_avg"],
            colorscale=[[0, "#374151"], [0.5, PURPLE], [1.0, "#C4B5FD"]],
            cmin=1, cmax=5,
        ),
        text=[f"{int(n)} film{'s' if n != 1 else ''}" for n in ddf["n"]],
        textposition="outside", textfont_size=10,
        customdata=list(zip(ddf["avg"], ddf["n"], ddf["bayes_avg"])),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Bayesian avg: <b>%{customdata[2]:.2f}</b><br>"
            "Raw avg: %{customdata[0]:.2f}  (%{customdata[1]} films)<br>"
            "<i>Low-count decades pulled toward " + f"{global_mean:.2f}" + " global mean</i>"
            "<extra></extra>"
        ),
        showlegend=False,
    )

    # Dots: raw average — shows how much shrinkage was applied
    fig.add_scatter(
        name="Raw avg",
        x=ddf["decade"], y=ddf["avg"],
        mode="markers",
        marker=dict(color="#F9FAFB", size=7, symbol="diamond",
                    line=dict(color=PURPLE, width=1.5)),
        hovertemplate="<b>%{x}</b> raw avg: %{y:.2f}<extra></extra>",
    )

    # Reference line: global mean
    fig.add_hline(
        y=global_mean, line_dash="dot", line_color="#6B7280", line_width=1,
        annotation_text=f"Global mean {global_mean:.2f}",
        annotation_font_size=10, annotation_font_color="#9CA3AF",
        annotation_position="bottom right",
    )

    fig.update_layout(
        **PLOT_LAYOUT, height=320,
        yaxis=dict(range=[0, 5.9], title="Rating", gridcolor="#374151"),
        xaxis=dict(type="category", title=""),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08,
                    font_size=11, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_rating_dist(df: pd.DataFrame):
    counts = df["rating"].value_counts().sort_index()
    colors = [PURPLE if v == df["rating"].mode()[0] else "#4B5563" for v in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        text=counts.values, textposition="outside", textfont_size=10,
        hovertemplate="Rating %{x}: %{y} films<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=260,
        xaxis=dict(title="Rating", tickvals=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],
                   gridcolor="#374151"),
        yaxis=dict(title="Films", gridcolor="#374151"),
    )
    return fig


def chart_genre_decade_heatmap(df: pd.DataFrame):
    rows = []
    for _, r in df.dropna(subset=["decade"]).iterrows():
        if not r["genres"] or str(r["genres"]) in ("nan", "None", ""):
            continue
        for g in str(r["genres"]).split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                rows.append({"genre": g, "decade": r["decade"], "rating": r["rating"]})
    if not rows:
        return None

    gdf = pd.DataFrame(rows)
    pivot = (
        gdf.groupby(["genre", "decade"])["rating"].mean()
           .unstack(fill_value=None)
    )
    # Keep genres with enough data
    genre_counts = gdf.groupby("genre").size()
    valid_genres = genre_counts[genre_counts >= 5].index.intersection(pivot.index)
    pivot = pivot.loc[valid_genres]
    if pivot.empty:
        return None

    pivot = pivot.sort_index()
    decades = sorted(pivot.columns.tolist())
    pivot = pivot[decades]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#1f2937"],[0.3,"#4C1D95"],[0.6,PURPLE],[1.0,"#C4B5FD"]],
        zmin=1, zmax=5,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> · %{x}<br>Avg rating: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Avg Rating", tickfont_size=10, len=0.8),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=max(300, len(pivot) * 28 + 60),
        xaxis=dict(title="", side="bottom"),
        yaxis=dict(title="", autorange="reversed"),
    )
    fig.update_layout(margin=dict(l=100, r=80, t=30, b=40))
    return fig


def chart_absa_profile(df: pd.DataFrame, absa_cache: dict):
    ASPECTS = ["acting", "plot", "visuals", "pacing", "music"]
    sums  = {a: [] for a in ASPECTS}
    weights = {a: [] for a in ASPECTS}

    for _, row in df.iterrows():
        absa = absa_cache.get(str(int(row["movie_id"])))
        if not absa:
            continue
        w = row["rating"] / 5.0
        for a in ASPECTS:
            v = absa.get(a)
            if v is not None:
                sums[a].append(v * w)
                weights[a].append(w)

    scores = {}
    for a in ASPECTS:
        if weights[a]:
            scores[a] = sum(sums[a]) / sum(weights[a])

    if not scores:
        return None

    labels = list(scores.keys())
    vals   = list(scores.values())
    colors = [PURPLE if v >= 0 else "#EF4444" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=[l.capitalize() for l in labels],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside", textfont_size=11,
        hovertemplate="<b>%{y}</b><br>Audience score: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#6B7280", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT, height=260,
        xaxis=dict(range=[-0.6, 0.6], title="Weighted Sentiment Score", gridcolor="#374151",
                   zeroline=False),
        yaxis=dict(title=""),
    )
    return fig


def chart_mainstream_scatter(df: pd.DataFrame):
    sdf = df.dropna(subset=["vote_count", "vote_average"]).copy()
    sdf = sdf[sdf["vote_count"] > 0].copy()
    if sdf.empty:
        return None
    sdf["log_pop"] = np.log10(sdf["vote_count"] + 1)
    sdf["genre1"]  = sdf["genres"].apply(
        lambda g: str(g).split("|")[0] if g and str(g) not in ("nan","None","") else "Other"
    )
    sdf["vs_crowd"] = sdf["rating"] - sdf["vote_average"]

    fig = px.scatter(
        sdf, x="log_pop", y="rating",
        color="genre1", color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_name="title",
        hover_data={"vote_count": True, "vote_average": ":.1f",
                    "vs_crowd": ":.2f", "log_pop": False, "genre1": False},
        labels={"log_pop": "Popularity (log₁₀ votes)", "rating": "Your Rating"},
        opacity=0.75,
    )
    # Quadrant annotations
    fig.add_hline(y=df["rating"].mean(), line_dash="dot",
                  line_color="#6B7280", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT, height=380,
        xaxis=dict(title="← Niche  ·  Popularity (log₁₀ votes)  ·  Mainstream →",
                   gridcolor="#374151"),
        yaxis=dict(title="Your Rating", range=[0, 5.5], gridcolor="#374151"),
        legend=dict(title="Genre", font_size=10,
                    bgcolor="rgba(0,0,0,0)", bordercolor="#374151"),
    )
    return fig


def chart_score_breakdown(recs: list):
    if not recs:
        return None

    def _norm(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    titles  = [r["title"][:34] + ("…" if len(r["title"]) > 34 else "") for r in recs]
    svd_raw  = [r["svd_score"]      for r in recs]
    cont_raw = [r["content_score"]  for r in recs]
    sent_raw = [r["sentiment_score"] for r in recs]

    # Normalize each signal to 0-1 independently, then apply tuned weights
    svd_n  = [v * 0.765 for v in _norm(svd_raw)]
    cont_n = [v * 0.101 for v in _norm(cont_raw)]
    sent_n = [v * 0.088 for v in _norm(sent_raw)]

    # Scale segments so total bar width = actual final score
    final_scores = [r["score"] for r in recs]
    contrib_totals = [s + c + g for s, c, g in zip(svd_n, cont_n, sent_n)]
    def _scale(component, i):
        t = contrib_totals[i]
        return (component / t * final_scores[i]) if t > 0 else 0

    svd_pct  = [_scale(svd_n[i],  i) for i in range(len(recs))]
    cont_pct = [_scale(cont_n[i], i) for i in range(len(recs))]
    sent_pct = [_scale(sent_n[i], i) for i in range(len(recs))]

    fig = go.Figure()
    fig.add_bar(
        name="Viewers like you", y=titles, x=svd_pct, orientation="h",
        marker_color="#7C3AED",
        customdata=list(zip(svd_raw, svd_pct)),
        hovertemplate="<b>%{y}</b><br>Taste match: %{customdata[1]:.3f}<extra></extra>",
    )
    fig.add_bar(
        name="Content fit", y=titles, x=cont_pct, orientation="h",
        marker_color="#0EA5E9",
        customdata=list(zip(cont_raw, cont_pct)),
        hovertemplate="<b>%{y}</b><br>Content fit: %{customdata[1]:.3f}<extra></extra>",
    )
    fig.add_bar(
        name="Audience reception", y=titles, x=sent_pct, orientation="h",
        marker_color="#10B981",
        customdata=list(zip(sent_raw, sent_pct)),
        hovertemplate="<b>%{y}</b><br>Audience reception: %{customdata[1]:.3f}<extra></extra>",
    )
    for i, rec in enumerate(recs):
        fig.add_annotation(
            x=102, y=titles[i],
            text=f"<b>{rec['score']:.3f}</b>",
            showarrow=False, xanchor="left",
            font=dict(color=PURPLE_L, size=11),
        )

    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="stack",
        height=max(350, len(recs) * 34 + 80),
        xaxis=dict(title="Score", range=[0, max(final_scores) * 1.18],
                   gridcolor="#374151"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    fig.update_layout(margin=dict(l=230, r=100, t=20, b=30))
    return fig


# ── KPI card helper ───────────────────────────────────────────────────────────

def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ── App ───────────────────────────────────────────────────────────────────────

ens  = load_ensemble()
absa = load_absa()

# Sidebar
with st.sidebar:
    st.markdown(
        "<a href='http://localhost:8000' target='_self' style='"
        "display:inline-flex;align-items:center;gap:6px;"
        "font-size:0.82em;color:#A78BFA;text-decoration:none;"
        "padding:6px 12px;border:1px solid #374151;border-radius:6px;"
        "margin-bottom:4px;'>&#8592; Back to Home</a>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='font-family:monospace;letter-spacing:5px;font-size:1.4em'>CINEIQ</h2>"
        "<p style='color:#6B7280;font-size:0.78em;margin-top:-8px'>Taste Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    user_id = st.number_input("User ID", min_value=1, max_value=610, value=42,
                               help="MovieLens users 1–610")
    top_k   = st.slider("Recommendations", 5, 20, 10)
    st.divider()
    st.caption("Powered by CINEIQ Intelligence Engine")
    if st.button("Clear Cache", help="Force-refresh all recommendations & ratings"):
        st.cache_data.clear()
        st.rerun()

# Load data
df = user_ratings(user_id)

if df.empty:
    st.warning(f"No ratings found for user {user_id}.")
    st.stop()

# Derived stats
affinities = {}
for _, row in df.iterrows():
    if not row["genres"] or str(row["genres"]) in ("nan", "None", ""):
        continue
    for g in str(row["genres"]).split("|"):
        g = g.strip()
        if g and g != "(no genres listed)":
            affinities.setdefault(g, []).append(row["rating"])
affinities = {g: np.mean(v) for g, v in affinities.items() if len(v) >= 2}

top_genre = max(affinities, key=affinities.get) if affinities else "N/A"

fav_decade, fav_decade_s = _bayesian_fav_decade(df)
if fav_decade is None:
    fav_decade = "N/A"

rated_with_pop = df[df["vote_count"].notna() & (df["vote_count"] > 0)]
niche_score    = (rated_with_pop["vote_count"] < 100).mean() if len(rated_with_pop) > 0 else 0.0

crowd_delta = None
if len(rated_with_pop) > 0:
    crowd_delta = (rated_with_pop["rating"] - rated_with_pop["vote_average"] / 2).mean()

icon, badge, badge_sub = personality(df, affinities, niche_score)

# Header
st.markdown(
    f"<h1 style='font-family:monospace;letter-spacing:5px;font-size:1.9em;"
    f"margin-bottom:0'>CINEIQ</h1>"
    f"<p style='color:{TEXT_DIM};margin:0 0 12px'>User {user_id} · Taste Intelligence Report</p>",
    unsafe_allow_html=True,
)

# Badge
st.markdown(
    f'<div class="badge-wrap">'
    f'<span class="badge-icon">{icon}</span>'
    f'<div><div class="badge-title">{badge}</div>'
    f'<div class="badge-sub">{badge_sub}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi("Films Rated", f"{len(df):,}", f"across {df['decade'].nunique()} decades")
with k2:
    avg_r = df["rating"].mean()
    label = "generous" if avg_r > 3.8 else ("critical" if avg_r < 2.8 else "calibrated")
    kpi("Avg Rating", f"{avg_r:.2f}", label)
with k3:
    kpi("Favourite Decade", fav_decade,
        f"avg {fav_decade_s[fav_decade]:.2f}/5" if fav_decade != "N/A" else "")
with k4:
    kpi("Top Genre", top_genre,
        f"{affinities[top_genre]:.2f}/5 avg" if top_genre != "N/A" and top_genre in affinities else "")
with k5:
    kpi("Niche Score", f"{niche_score*100:.0f}%", "of films are under-the-radar")
with k6:
    if crowd_delta is not None:
        sign = "+" if crowd_delta >= 0 else ""
        kpi("vs. Crowd", f"{sign}{crowd_delta:.2f}", "higher than avg voter")
    else:
        kpi("vs. Crowd", "N/A", "")

st.divider()

# Tabs
tab_profile, tab_recs = st.tabs(["🧠  Taste Profile", "🎬  Recommendations"])

# ─── TAB 1: TASTE PROFILE ────────────────────────────────────────────────────
with tab_profile:

    # Row 1: Radar + Decade bar
    section("TASTE SIGNATURE")
    c_radar, c_decade = st.columns([1, 1])

    with c_radar:
        st.markdown("**Genre Affinity**")
        st.caption("Average rating per genre (min. 2 films)")
        if len(affinities) >= 3:
            st.plotly_chart(chart_radar(affinities), use_container_width=True)
        else:
            st.info("Not enough genre data.")

    with c_decade:
        st.markdown("**Decade Preference**")
        st.caption(
            "Bayesian-adjusted avg rating by decade — bars are confidence-weighted "
            "(shrunk toward global mean when few films). Diamonds show raw avg."
        )
        if not df.dropna(subset=["decade"]).empty:
            st.plotly_chart(chart_decade_bar(df), use_container_width=True)

    # Rating distribution
    st.markdown("**Rating Distribution**")
    st.caption("How you distribute your scores — reveals rating habits")
    st.plotly_chart(chart_rating_dist(df), use_container_width=True)

    # Row 2: Genre × Decade Heatmap
    section("GENRE × ERA MATRIX")
    st.markdown("**Which genres do you love from which eras?**")
    st.caption("Average personal rating — darker = higher rated. Blank = no data.")
    heat = chart_genre_decade_heatmap(df)
    if heat:
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough data for matrix.")

    # Row 3: ABSA + Mainstream scatter
    section("DEEPER SIGNALS")
    c_absa, c_scatter = st.columns([1, 1])

    with c_absa:
        st.markdown("**What Your Favourite Films Excel At**")
        st.caption(
            "Aspect-Based Sentiment Analysis weighted by your ratings — "
            "reveals what film qualities you unconsciously value"
        )
        absa_fig = chart_absa_profile(df, absa)
        if absa_fig:
            st.plotly_chart(absa_fig, use_container_width=True)
        else:
            st.info("Not enough review data for this user's films.")

    with c_scatter:
        st.markdown("**Mainstream vs. Niche Taste Map**")
        st.caption(
            "X = film popularity · Y = your rating. "
            "Top-left = you love hidden gems. Bottom-right = you're tough on blockbusters."
        )
        scatter = chart_mainstream_scatter(df)
        if scatter:
            st.plotly_chart(scatter, use_container_width=True)
        else:
            st.info("Not enough popularity data.")

# ─── TAB 2: RECOMMENDATIONS ──────────────────────────────────────────────────
with tab_recs:
    with st.spinner("Running hybrid ensemble…"):
        recs = run_recs(user_id, top_k)

    if not recs:
        st.info("No recommendations generated.")
        st.stop()

    # Score breakdown chart
    section("WHY THESE PICKS? · MATCH BREAKDOWN")
    st.caption(
        "Each bar shows what drove the recommendation — "
        "viewers like you (purple), content fit (blue), and audience reception (green)."
    )
    st.plotly_chart(chart_score_breakdown(recs), use_container_width=True)

    st.divider()
    section(f"TOP {top_k} RECOMMENDATIONS")

    for i, rec in enumerate(recs):
        with st.container(border=True):
            lc, rc = st.columns([5, 1])
            with lc:
                genres_str = " · ".join(rec["genres"][:4]) if rec["genres"] else "N/A"
                st.markdown(f"**#{i+1} &nbsp; {rec['title']}**")
                st.caption(genres_str)
                explanations = rec.get("explanation", [])
                if explanations:
                    st.markdown(
                        f"<span style='font-size:.7em;color:{PURPLE_L};"
                        f"text-transform:uppercase;letter-spacing:.07em'>Why this?</span>",
                        unsafe_allow_html=True,
                    )
                    for reason in explanations:
                        st.markdown(
                            f"<span style='color:{TEXT_DIM};font-size:.85em'>› {reason}</span>",
                            unsafe_allow_html=True,
                        )
            with rc:
                st.metric("Match", f"{rec['score'] * 100:.1f}%")
