"""
Day-level BERT feature engineering.
Aggregates article-level BERT embeddings (768-dim) to day level via mean pooling.
Combines with technical features from v4 day-level data.
Output:
  data/processed/tsmc_bert_day_text.csv   (136 days x 768)
  data/processed/tsmc_bert_day_tech.csv   (136 days x 13)
  data/processed/tsmc_bert_day_meta.csv   (136 days: date, label, return_rate)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]

print("Loading BERT features and article metadata...")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features.csv", encoding="utf-8-sig")
bert    = np.load(DATA_DIR / "processed" / "tsmc_bert_features.npy")
assert len(feat_df) == len(bert), f"Mismatch: {len(feat_df)} vs {len(bert)}"

feat_df["post_time"] = pd.to_datetime(feat_df["post_time"])
feat_df["date"]      = feat_df["post_time"].dt.date

print(f"  Articles: {len(feat_df)}  Dates: {feat_df['date'].nunique()}")

# Load v4 tech features to get consistent tech columns
feat_v2 = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
feat_v2["post_time"] = pd.to_datetime(feat_v2["post_time"])
feat_v2["date"]      = feat_v2["post_time"].dt.date

# Day-level tech features from v2 (same per day)
day_tech_map = {}
for date, grp in feat_v2.groupby("date"):
    r0 = grp.iloc[0]
    pr1 = r0["prev_ret_1d"]; pr5 = r0["prev_ret_5d"]
    v5  = r0["vol_5d"];       v20 = r0["vol_20d"]
    rsi = r0["rsi_14"];       n   = len(grp)
    ret1_std  = pr1 / (v5 + 1e-8)
    ret5_std  = pr5 / (v20 + 1e-8)
    rsi_norm  = (rsi - 50) / 25.0
    vol_ratio = v5 / (v20 + 1e-8)
    sign_ret1 = float(np.sign(pr1)) if pr1 != 0 else 0.0
    dow       = pd.Timestamp(str(date)).dayofweek
    dow_sin   = np.sin(2 * np.pi * dow / 5)
    dow_cos   = np.cos(2 * np.pi * dow / 5)
    day_tech_map[date] = [pr1, pr5, v5, v20, rsi, n,
                          ret1_std, ret5_std, rsi_norm, vol_ratio,
                          sign_ret1, dow_sin, dow_cos]

text_rows, tech_rows, meta_rows = [], [], []

for date, grp in feat_df.groupby("date"):
    idx      = grp.index.tolist()
    mean_emb = bert[idx].mean(axis=0)   # (768,)
    label    = int(grp["label"].mode().iloc[0])
    ret      = float(grp["return_rate"].iloc[0])

    if date not in day_tech_map:
        continue

    text_rows.append(mean_emb)
    tech_rows.append(day_tech_map[date])
    meta_rows.append({"date": date, "return_rate": ret, "label": label})

meta_df = pd.DataFrame(meta_rows).sort_values("date").reset_index(drop=True)
text_df = pd.DataFrame(text_rows, columns=[f"bert_{i}" for i in range(768)])
tech_cols = (TECH_COLS +
             ["ret1_std", "ret5_std", "rsi_norm", "vol_ratio",
              "sign_ret1", "dow_sin", "dow_cos"])
tech_df = pd.DataFrame(tech_rows, columns=tech_cols)

print(f"  Day rows: {len(meta_df)}")
print(f"  Text shape: {text_df.shape}  Tech shape: {tech_df.shape}")
print(f"  Label: Down={( meta_df['label']==0).sum()}  Up={(meta_df['label']==1).sum()}")

text_df.to_csv(DATA_DIR / "processed" / "tsmc_bert_day_text.csv",
               index=False, encoding="utf-8-sig")
tech_df.to_csv(DATA_DIR / "processed" / "tsmc_bert_day_tech.csv",
               index=False, encoding="utf-8-sig")
meta_df.to_csv(DATA_DIR / "processed" / "tsmc_bert_day_meta.csv",
               index=False, encoding="utf-8-sig")
print("Done — saved tsmc_bert_day_text.csv  tsmc_bert_day_tech.csv  tsmc_bert_day_meta.csv")
