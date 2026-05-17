# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CineIQ — ABSA Precomputation Notebook (Kaggle GPU)                     ║
# ║  Model : yangheng/deberta-v3-base-absa-v1.1                             ║
# ║  Input : /kaggle/input/cineiq-reviews/cineiq_reviews.csv                ║
# ║  Output: /kaggle/working/absa_sentiment_cache.json                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── CELL 1: Install & imports ───────────────────────────────────────────────
# !pip install transformers sentencepiece -q

import re
import json
import torch
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── CELL 2: Aspect config ───────────────────────────────────────────────────
ASPECTS = {
    "acting":  ["acting", "actor", "actress", "performance", "cast",
                "played", "role", "character", "portrayal"],
    "plot":    ["plot", "story", "script", "writing", "narrative",
                "twist", "ending", "screenplay", "storyline"],
    "visuals": ["cinematography", "visual", "shot", "camera", "cgi",
                "effects", "scene", "photography", "beautiful"],
    "pacing":  ["pacing", "pace", "slow", "fast", "boring", "dragged",
                "rushed", "length", "long", "minutes"],
    "music":   ["soundtrack", "music", "score", "song", "sound", "audio"],
}

def split_sentences(text: str) -> list[str]:
    text = re.sub(r'<br\s*/?>', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 15]

def detect_aspects(sentence: str) -> list[str]:
    s = sentence.lower()
    return [asp for asp, kws in ASPECTS.items() if any(kw in s for kw in kws)]


# ── CELL 3: Load model ──────────────────────────────────────────────────────
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model ready.")


# ── CELL 4: Batch inference helper ──────────────────────────────────────────
def batch_score(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
    """
    Score a list of (sentence, aspect) pairs.
    Returns a scalar per pair: Positive_prob - Negative_prob  ∈ [-1, +1]
    """
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        sentences = [p[0] for p in batch]
        aspects   = [p[1] for p in batch]

        enc = tokenizer(
            sentences, aspects,
            padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits                    # [B, 3]
        probs = F.softmax(logits, dim=-1).cpu()            # [B, 3]  Neg/Neu/Pos

        scores.extend((probs[:, 2] - probs[:, 0]).tolist())
    return scores


# ── CELL 5: Load data ────────────────────────────────────────────────────────
df = pd.read_csv("/kaggle/input/cineiq-reviews/cineiq_reviews.csv")
df["review_text"] = df["review_text"].fillna("").astype(str)
print(f"Loaded {len(df):,} reviews for {df['movie_id'].nunique():,} movies.")

# Group reviews by movie
movie_reviews = df.groupby("movie_id")["review_text"].apply(list).to_dict()


# ── CELL 6: Build (sentence, aspect) pairs per movie ─────────────────────────
# Structure: movie_pairs[movie_id][aspect] = [(sentence, global_pair_idx), ...]
print("Splitting sentences and detecting aspects...")

movie_aspect_sentences: dict[int, dict[str, list[str]]] = {}
all_pairs: list[tuple[str, str]] = []          # (sentence, aspect) — for batching
pair_index: list[tuple[int, str, int]] = []    # (movie_id, aspect, local_pos)

for movie_id, reviews in tqdm(movie_reviews.items()):
    asp_sentences: dict[str, list[str]] = defaultdict(list)

    for review in reviews:
        for sent in split_sentences(review):
            matched = detect_aspects(sent)
            for asp in matched:
                asp_sentences[asp].append(sent)
            if not matched:
                # Unmatched sentences contribute to 'overall'
                asp_sentences["overall"].append(sent)

    # Always compute overall from every sentence too
    all_sents = [s for r in reviews for s in split_sentences(r)]
    asp_sentences["overall"] = all_sents[:50]  # cap to 50 to avoid excess calls

    movie_aspect_sentences[movie_id] = dict(asp_sentences)

    for asp, sents in asp_sentences.items():
        for sent in sents:
            pair_index.append((movie_id, asp, len(all_pairs)))
            all_pairs.append((sent, asp))

print(f"Total (sentence, aspect) pairs: {len(all_pairs):,}")


# ── CELL 7: Run inference ────────────────────────────────────────────────────
print("Running DeBERTa ABSA inference...")
all_scores = batch_score(all_pairs, batch_size=64)
print("Inference complete.")


# ── CELL 8: Aggregate per movie per aspect ───────────────────────────────────
print("Aggregating scores...")

movie_aspect_scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

for movie_id, asp, idx in pair_index:
    movie_aspect_scores[movie_id][asp].append(all_scores[idx])

# Average and build final cache
cache: dict[str, dict[str, float | None]] = {}

for movie_id, asp_scores in movie_aspect_scores.items():
    entry = {}
    for asp in list(ASPECTS.keys()) + ["overall"]:
        vals = asp_scores.get(asp, [])
        entry[asp] = round(sum(vals) / len(vals), 4) if vals else None
    cache[str(movie_id)] = entry

print(f"Cache built for {len(cache):,} movies.")


# ── CELL 9: Save output ──────────────────────────────────────────────────────
OUT = "/kaggle/working/absa_sentiment_cache.json"
with open(OUT, "w") as f:
    json.dump(cache, f)

size_kb = __import__('os').path.getsize(OUT) / 1024
print(f"Saved → {OUT}  ({size_kb:.0f} KB)")
print("Download this file and place it in your local models/ directory.")

# Quick sanity check — print a few entries
sample = list(cache.items())[:3]
for mid, scores in sample:
    print(f"\nmovie_id={mid}: {scores}")
