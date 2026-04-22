"""
Feature engineering v2:
  - sigma = 0.8%  (was 1.5%) → 36,834 non-neutral samples (was 21,068)
  - TF-IDF + chi2 fitted on TRAIN only (no leakage into test)
  - Technical price features: prev_ret, rolling_vol, RSI-14, article_count
  - TimeSeriesSplit-compatible output
Outputs (all new files, originals untouched):
  data/processed/tsmc_features_v2.csv
  data/processed/tsmc_vector_space_v2.csv
  data/features/top_300_features_v2.csv
  models/_shared/vectorizer_v2.pkl
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEAT_DIR = DATA_DIR / "features"
SHARED   = ROOT / "models" / "_shared"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)

SIGMA       = 0.008   # 0.8 %
SPLIT       = 0.8
CHI2_K      = 300
NGRAM       = (1, 2)
TFIDF_MAX   = 4000
TFIDF_MINDF = 2


# ── 1. Load & relabel ─────────────────────────────────────────────────────────
print("Loading tsmc_clean.csv …")
df = pd.read_csv(DATA_DIR / "processed" / "tsmc_clean.csv", encoding="utf-8-sig")
df["post_time"] = pd.to_datetime(df["post_time"])
df = df.sort_values("post_time").reset_index(drop=True)

# Re-label with sigma = 0.8 %
df["label_v2"] = 2  # neutral / discard
df.loc[df["return_rate"] >  SIGMA, "label_v2"] = 1  # Up
df.loc[df["return_rate"] < -SIGMA, "label_v2"] = 0  # Down

n_up   = (df["label_v2"] == 1).sum()
n_dn   = (df["label_v2"] == 0).sum()
n_neu  = (df["label_v2"] == 2).sum()
print(f"  sigma={SIGMA*100:.1f}%  →  Up={n_up}  Down={n_dn}  Neutral(drop)={n_neu}")

# Keep only non-neutral rows
keep = df[df["label_v2"] != 2].reset_index(drop=True)
print(f"  Kept {len(keep)} rows for training/test")


# ── 2. Technical price features ───────────────────────────────────────────────
print("Computing technical features …")

# Day-level price series from ALL rows (full 63 k) to avoid gaps in rolling
all_days = (
    df.groupby(df["post_time"].dt.date)["price_0"]
    .first()
    .reset_index()
    .rename(columns={"post_time": "date", "price_0": "close"})
    .sort_values("date")
    .reset_index(drop=True)
)
# Daily return = (close_t - close_{t-1}) / close_{t-1}
all_days["day_ret"]     = all_days["close"].pct_change()
all_days["vol_5d"]      = all_days["day_ret"].rolling(5,  min_periods=2).std()
all_days["vol_20d"]     = all_days["day_ret"].rolling(20, min_periods=5).std()

# RSI-14
def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta).clip(lower=0).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)   # fill warm-up with neutral 50

all_days["rsi_14"]      = _rsi(all_days["close"])
all_days["prev_ret_1d"] = all_days["day_ret"].shift(1)          # yesterday's return
all_days["prev_ret_5d"] = all_days["close"].pct_change(5).shift(1)  # 5-day prior momentum

# Article count per day (from full dataset)
art_count = df.groupby(df["post_time"].dt.date).size().reset_index(name="n_articles")
art_count.columns = ["date", "n_articles"]
all_days = all_days.merge(art_count, on="date", how="left").fillna({"n_articles": 0})
all_days["n_articles"] = all_days["n_articles"].astype(int)

# Join tech features to keep (filtered articles)
keep["date"] = keep["post_time"].dt.date
keep = keep.merge(
    all_days[["date","prev_ret_1d","prev_ret_5d","vol_5d","vol_20d","rsi_14","n_articles"]],
    on="date", how="left",
)
keep[["prev_ret_1d","prev_ret_5d","vol_5d","vol_20d","rsi_14"]] = (
    keep[["prev_ret_1d","prev_ret_5d","vol_5d","vol_20d","rsi_14"]].fillna(0)
)


# ── 3. Text column ────────────────────────────────────────────────────────────
keep["title"]   = keep["title"].fillna("").astype(str)
keep["content"] = keep["content"].fillna("").astype(str)
keep["text"]    = (keep["title"] + " " + keep["content"]).str.strip()


# ── 4. Train / test split (chronological) ────────────────────────────────────
split_idx = int(len(keep) * SPLIT)
train_df  = keep.iloc[:split_idx]
test_df   = keep.iloc[split_idx:]
print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
print(f"  Train dates: {train_df['post_time'].min().date()} → "
      f"{train_df['post_time'].max().date()}")
print(f"  Test  dates: {test_df['post_time'].min().date()} → "
      f"{test_df['post_time'].max().date()}")


# ── 5. TF-IDF + chi2 (fit on TRAIN only) ─────────────────────────────────────
print("Fitting TF-IDF + chi2 on train data only …")
vec_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char", ngram_range=NGRAM,
        max_features=TFIDF_MAX, min_df=TFIDF_MINDF, sublinear_tf=True,
    )),
    ("chi2", SelectKBest(score_func=chi2, k=CHI2_K)),
])
X_train_vec = vec_pipe.fit_transform(
    train_df["text"].tolist(), train_df["label_v2"].values
)
X_test_vec  = vec_pipe.transform(test_df["text"].tolist())

# Feature names
tfidf_names = vec_pipe.named_steps["tfidf"].get_feature_names_out()
chi2_mask   = vec_pipe.named_steps["chi2"].get_support()
selected    = tfidf_names[chi2_mask]
print(f"  Selected {len(selected)} chi2 features (train-only fit)")

# Save chi2 feature names
pd.DataFrame({"feature": selected}).to_csv(
    FEAT_DIR / "top_300_features_v2.csv", index=False, encoding="utf-8-sig"
)

# Save vectorizer pipeline
with open(SHARED / "vectorizer_v2.pkl", "wb") as f:
    pickle.dump(vec_pipe, f)
print(f"  vectorizer_v2.pkl saved → {SHARED / 'vectorizer_v2.pkl'}")


# ── 6. Build full vector_space_v2.csv (all 36 k rows) ────────────────────────
import scipy.sparse as sp
X_all_vec = sp.vstack([X_train_vec, X_test_vec])
col_names  = [f"tfidf_{n}" for n in selected]
vec_df     = pd.DataFrame(X_all_vec.toarray(), columns=col_names)
vec_df.to_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv",
              index=False, encoding="utf-8-sig")
print(f"  tsmc_vector_space_v2.csv saved  shape={vec_df.shape}")


# ── 7. Save features_v2.csv ───────────────────────────────────────────────────
TECH_COLS = ["prev_ret_1d","prev_ret_5d","vol_5d","vol_20d","rsi_14","n_articles"]
out_cols  = ["post_time","title","content","price_0","price_1",
             "return_rate","label_v2"] + TECH_COLS
keep[out_cols].rename(columns={"label_v2":"label"}).to_csv(
    DATA_DIR / "processed" / "tsmc_features_v2.csv",
    index=False, encoding="utf-8-sig",
)
print(f"  tsmc_features_v2.csv saved  rows={len(keep)}")
print("Done.")
