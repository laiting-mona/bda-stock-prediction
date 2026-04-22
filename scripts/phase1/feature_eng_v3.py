"""
Feature engineering v3: extends v2 with 14 new article-level features.
New features: hour cyclical, session dummies, article length,
McDonald n-gram sentiment scores, derived tech ratios.
Same TF-IDF vectoriser as v2 (no re-fitting, no leakage).
Output: data/processed/tsmc_features_v3.csv  (36834 rows, 20 tech cols)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# ── 1. Load v2 base data ───────────────────────────────────────────────────────
print("Loading v2 feature data...")
feat = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
feat["post_time"] = pd.to_datetime(feat["post_time"])
feat["text"] = (feat["title"].fillna("") + " " + feat["content"].fillna("")).str.strip()
print(f"  Rows: {len(feat)}")

# ── 2. McDonald n-gram dictionaries ───────────────────────────────────────────
print("Loading McDonald n-gram dictionaries...")
up_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_up.csv",   encoding="utf-8-sig")
dn_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_down.csv", encoding="utf-8-sig")
up_df.columns = ["ngram","TF","DF","TFIDF","TF_n","DF_n","TF_a","DF_a","MI","Lift"]
dn_df.columns = up_df.columns
up_dict = dict(zip(up_df["ngram"].head(200), up_df["TFIDF"].head(200)))
dn_dict = dict(zip(dn_df["ngram"].head(200), dn_df["TFIDF"].head(200)))
print(f"  UP n-grams: {len(up_dict)}  DOWN n-grams: {len(dn_dict)}")

def _mcd_scores(text: str):
    u = float(sum(text.count(ng) * w for ng, w in up_dict.items()))
    d = float(sum(text.count(ng) * w for ng, w in dn_dict.items()))
    net = (u - d) / (u + d + 1e-9)
    return u, d, net

# ── 3. Article-level new features ─────────────────────────────────────────────
print("Computing article-level features...")
hour        = feat["post_time"].dt.hour.values
dow         = feat["post_time"].dt.dayofweek.values   # 0=Mon

feat["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
feat["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

# session: 0=pre-market (0-8), 1=market (9-13), 2=after-market (14-23)
session = np.where(hour <= 8, 0, np.where(hour <= 13, 1, 2))
feat["session_pre"]  = (session == 0).astype(np.float32)
feat["session_mkt"]  = (session == 1).astype(np.float32)
feat["session_aft"]  = (session == 2).astype(np.float32)

feat["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
feat["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

# article length (normalised by 95th percentile to cap outliers)
art_len = feat["text"].str.len().values.astype(np.float32)
p95     = np.percentile(art_len, 95)
feat["article_len_norm"] = np.clip(art_len / (p95 + 1e-9), 0, 1).astype(np.float32)

# McDonald sentiment scores (per article — non-negative up/down, bounded net)
print("  Computing McDonald scores (this may take ~30 s)...")
scores = [_mcd_scores(t) for t in feat["text"]]
feat["mcd_up"]  = [s[0] for s in scores]
feat["mcd_dn"]  = [s[1] for s in scores]
feat["mcd_net"] = [s[2] for s in scores]   # in [-1, +1]; negative when Down-biased

# ── 4. Derived technical features (same per day, row-duplicated) ──────────────
print("Computing derived technical features...")
r1 = feat["prev_ret_1d"].fillna(0).values
r5 = feat["prev_ret_5d"].fillna(0).values
v5 = feat["vol_5d"].fillna(0).values
v20= feat["vol_20d"].fillna(0).values
rsi= feat["rsi_14"].fillna(50).values

feat["ret1_std"]  = (r1 / (v5  + 1e-8)).astype(np.float32)
feat["ret5_std"]  = (r5 / (v20 + 1e-8)).astype(np.float32)
feat["rsi_norm"]  = ((rsi - 50) / 25.0).astype(np.float32)
feat["vol_ratio"] = (v5  / (v20 + 1e-8)).astype(np.float32)
feat["sign_ret1"] = np.sign(r1).astype(np.float32)

# ── 5. Drop helper columns and save ───────────────────────────────────────────
NEW_TECH_COLS = [
    "hour_sin", "hour_cos",
    "session_pre", "session_mkt", "session_aft",
    "dow_sin", "dow_cos",
    "article_len_norm",
    "mcd_up", "mcd_dn", "mcd_net",
    "ret1_std", "ret5_std", "rsi_norm", "vol_ratio", "sign_ret1",
]

BASE_COLS = ["post_time", "title", "content", "price_0", "price_1",
             "return_rate", "label",
             "prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]

out = feat[BASE_COLS + NEW_TECH_COLS].copy()
out.to_csv(DATA_DIR / "processed" / "tsmc_features_v3.csv", index=False, encoding="utf-8-sig")

print(f"\ntsmc_features_v3.csv saved  shape={out.shape}")
print(f"  Original tech: 6  |  New tech: {len(NEW_TECH_COLS)}  |  Total tech: {6 + len(NEW_TECH_COLS)}")
print(f"  Combined with 300 TF-IDF -> {300 + 6 + len(NEW_TECH_COLS)}-dim input for tree models")
print(f"  ComplementNB input: 300 TF-IDF + mcd_up + mcd_dn + n_articles = 303-dim")
print("Done.")
