"""
Feature engineering day-level v4:
  - Aggregates article-level data to ONE ROW PER TRADING DAY
  - X_text  : mean TF-IDF of all articles that day (300 dims)
  - X_tech  : technical indicators (same per day — prev_ret_1d/5d, vol_5d/20d, rsi_14, n_articles)
  - label   : majority-vote direction for the day (Up=1 / Down=0)
  - 3 feature sets saved separately for ablation study:
      text_only  (300 cols)
      tech_only  (6 cols)
      combined   (306 cols)
Outputs:
  data/processed/tsmc_day_features_v4.csv     (meta + label + return_rate)
  data/processed/tsmc_day_text_v4.csv         (300 mean-TF-IDF cols)
  data/processed/tsmc_day_tech_v4.csv         (6 tech cols)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8

print("Loading v2 article-level data…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

feat_df["post_time"] = pd.to_datetime(feat_df["post_time"])
feat_df["date"]      = feat_df["post_time"].dt.date

print(f"  Article rows: {len(feat_df)}  |  Unique dates: {feat_df['date'].nunique()}")

# ── Aggregate to day level ────────────────────────────────────────────────────
tfidf_arr  = vec_df.values                    # (N_articles, 300)
tfidf_cols = vec_df.columns.tolist()

day_rows, day_text, day_tech = [], [], []

for date, grp in feat_df.groupby("date"):
    idx      = grp.index
    mean_vec = tfidf_arr[idx].mean(axis=0)    # (300,) mean TF-IDF
    tech_row = grp[TECH_COLS].iloc[0].values  # same for all articles on a day
    label_votes = grp["label"].mode()
    label    = int(label_votes.iloc[0])       # majority vote (tie → lower class)
    ret      = float(grp["return_rate"].iloc[0])
    n_art    = len(grp)

    day_rows.append({
        "date":        date,
        "n_articles":  n_art,
        "return_rate": ret,
        "label":       label,
    })
    day_text.append(mean_vec)
    day_tech.append(tech_row)

day_meta_df = pd.DataFrame(day_rows).sort_values("date").reset_index(drop=True)
day_text_df = pd.DataFrame(day_text, columns=tfidf_cols)
day_tech_df = pd.DataFrame(day_tech, columns=TECH_COLS)

print(f"  Day-level rows: {len(day_meta_df)}")
print(f"  Label distribution: Down={( day_meta_df['label']==0).sum()}  "
      f"Up={(day_meta_df['label']==1).sum()}")

n_train = int(len(day_meta_df) * SPLIT)
train_dates = day_meta_df['date'].iloc[:n_train]
test_dates  = day_meta_df['date'].iloc[n_train:]
print(f"  Train: {len(train_dates)} days ({train_dates.iloc[0]} -> {train_dates.iloc[-1]})")
print(f"  Test : {len(test_dates)} days ({test_dates.iloc[0]} -> {test_dates.iloc[-1]})")

# ── Save outputs ──────────────────────────────────────────────────────────────
day_meta_df.to_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv",
                   index=False, encoding="utf-8-sig")
day_text_df.to_csv(DATA_DIR / "processed" / "tsmc_day_text_v4.csv",
                   index=False, encoding="utf-8-sig")
day_tech_df.to_csv(DATA_DIR / "processed" / "tsmc_day_tech_v4.csv",
                   index=False, encoding="utf-8-sig")

print("Saved:")
print(f"  tsmc_day_features_v4.csv  rows={len(day_meta_df)}")
print(f"  tsmc_day_text_v4.csv      shape={day_text_df.shape}")
print(f"  tsmc_day_tech_v4.csv      shape={day_tech_df.shape}")
print("Done.")
