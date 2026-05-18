"""
Track E: Optuna hyperparameter tuning for the HybridEnsemble.
Tunes alpha (SVD weight), beta (Content weight), gamma (Sentiment weight),
and candidate_pool size. Metric: NDCG@10 on 500 held-out users.

Pre-computes per-user SVD and content score arrays once, then each Optuna
trial is just a fast vectorized blend + sort — no model re-loading.

Run: python scripts/05_tune_ensemble.py
"""
import sys, yaml
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cineiq.config import get_config
from cineiq.db import get_session
from cineiq.training.ensemble import HybridEnsemble

# ── Settings ──────────────────────────────────────────────────────────────────
N_USERS     = 500   # sampled users for evaluation
MIN_RATINGS = 30    # minimum ratings per user
N_TRIALS    = 100
TOP_K       = 10
SEED        = 42
np.random.seed(SEED)

cfg = get_config()

# ── Load ensemble ─────────────────────────────────────────────────────────────
print("Loading ensemble artifacts...")
ensemble = HybridEnsemble(
    models_dir=cfg['models']['output_dir'],
    alpha=0.5, beta=0.3, gamma=0.2,
    candidate_pool=100,
    top_k=TOP_K,
)

all_ids  = np.array(ensemble.all_movie_ids)
n_movies = len(all_ids)

# Pre-build sentiment array aligned to all_ids, normalized once
sent_arr = np.array([
    float(ensemble.sentiment_cache.get(str(mid), 0.0))
    for mid in all_ids
])
sent_n_global = ensemble._normalize(sent_arr)

# ── Load ratings ──────────────────────────────────────────────────────────────
print("Loading ratings from DB...")
session = get_session()
ratings = pd.read_sql(text("SELECT user_id, movie_id, rating FROM ratings"), session.bind)
session.close()

valid_movie_set = set(ensemble.all_movie_ids)
ratings = ratings[ratings['movie_id'].isin(valid_movie_set)].copy()

# ── Sample users ──────────────────────────────────────────────────────────────
counts   = ratings.groupby('user_id').size()
eligible = counts[counts >= MIN_RATINGS].index.tolist()
sampled  = np.random.choice(eligible, min(N_USERS, len(eligible)), replace=False)
print(f"Sampled {len(sampled)} users from {len(eligible)} eligible (>={MIN_RATINGS} ratings)")

# ── Build per-user hold-out split ─────────────────────────────────────────────
# test  = movies rated >= 4.0  (held out — we want the model to surface these)
# train = all other rated movies  (used as the "already seen" exclusion set)
# NB: test items must NOT be in the seen mask, otherwise they can never be recommended
user_data = {}
for uid in sampled:
    user_r = ratings[ratings['user_id'] == uid]
    pos    = set(user_r[user_r['rating'] >= 4.0]['movie_id'].tolist())
    train  = set(user_r[user_r['rating'] <  4.0]['movie_id'].tolist())
    if not pos or not train:
        continue
    user_data[uid] = {
        'test':  pos,    # items to hit
        'seen':  train,  # items to exclude (does NOT include test items)
    }

print(f"Users with held-out positives: {len(user_data)}")

# ── Pre-compute per-user score arrays ─────────────────────────────────────────
print("Pre-computing SVD + content scores for sampled users (one-time cost)...")
user_svd     = {}
user_cont    = {}
user_has_svd = {}

for uid in user_data:
    rated_list = list(user_data[uid]['seen'])

    raw_svd = ensemble._svd_scores_for_user(uid)
    if raw_svd is not None:
        svd_sc = np.array([
            raw_svd[ensemble.svd_item_map[mid]] if mid in ensemble.svd_item_map else 0.0
            for mid in all_ids
        ])
        user_has_svd[uid] = True
    else:
        svd_sc = np.zeros(n_movies)
        user_has_svd[uid] = False

    raw_cont = ensemble._content_scores_for_user(rated_list)
    cont_sc  = np.array([raw_cont[ensemble.content_id_to_idx[mid]] for mid in all_ids])

    user_svd[uid]  = ensemble._normalize(svd_sc)
    user_cont[uid] = ensemble._normalize(cont_sc)

print("Pre-computation done.\n")

# ── Evaluation helper ─────────────────────────────────────────────────────────
def ndcg_at_k(ranked_ids, test_set, k=TOP_K):
    hits = [1.0 if mid in test_set else 0.0 for mid in ranked_ids[:k]]
    dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    n_pos = min(len(test_set), k)
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(n_pos))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(alpha, beta, gamma, candidate_pool):
    ndcgs = []
    for uid, data in user_data.items():
        seen   = data['seen']
        test   = data['test']
        svd_n  = user_svd[uid]
        cont_n = user_cont[uid]

        s1 = (alpha * svd_n + beta * cont_n) if user_has_svd[uid] else cont_n.copy()

        # Mask seen items
        seen_mask = np.array([mid in seen for mid in all_ids])
        s1[seen_mask] = -np.inf

        # Stage 1: top candidate pool (argpartition is O(n) instead of O(n log n))
        pool_idx = np.argpartition(s1, -candidate_pool)[-candidate_pool:]
        pool_idx = pool_idx[np.argsort(s1[pool_idx])[::-1]]

        # Stage 2: sentiment re-rank
        s2       = s1[pool_idx] + gamma * sent_n_global[pool_idx]
        top_idx  = pool_idx[np.argsort(s2)[::-1][:TOP_K]]
        ranked   = all_ids[top_idx].tolist()

        ndcgs.append(ndcg_at_k(ranked, test))

    return float(np.mean(ndcgs))


# ── Optuna study ──────────────────────────────────────────────────────────────
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.2, 0.9)
    beta  = trial.suggest_float('beta',  0.1, 0.8)
    gamma = trial.suggest_float('gamma', 0.0, 0.5)
    pool  = trial.suggest_categorical('candidate_pool', [30, 50, 75, 100])
    return evaluate(alpha, beta, gamma, pool)


print(f"Starting Optuna ({N_TRIALS} trials) ...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print(f"\n{'='*50}")
print(f"Best NDCG@{TOP_K}: {study.best_value:.4f}")
print(f"Best params:")
for k, v in best.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

baseline = evaluate(0.5, 0.3, 0.2, 50)
print(f"\nBaseline NDCG@{TOP_K} (alpha=0.5 beta=0.3 gamma=0.2, pool=50): {baseline:.4f}")
print(f"Improvement: {(study.best_value - baseline) * 100:+.2f}%")

# ── Write best params back to config.yaml ─────────────────────────────────────
config_path = Path(__file__).parent.parent / 'config.yaml'
with open(config_path) as f:
    cfg_raw = yaml.safe_load(f)

cfg_raw['ensemble']['alpha']          = round(best['alpha'], 3)
cfg_raw['ensemble']['beta']           = round(best['beta'],  3)
cfg_raw['ensemble']['gamma']          = round(best['gamma'], 3)
cfg_raw['ensemble']['candidate_pool'] = int(best['candidate_pool'])

with open(config_path, 'w') as f:
    yaml.dump(cfg_raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"\nconfig.yaml updated:")
print(f"  alpha (SVD):       {cfg_raw['ensemble']['alpha']}")
print(f"  beta  (Content):   {cfg_raw['ensemble']['beta']}")
print(f"  gamma (Sentiment): {cfg_raw['ensemble']['gamma']}")
print(f"  candidate_pool:    {cfg_raw['ensemble']['candidate_pool']}")
