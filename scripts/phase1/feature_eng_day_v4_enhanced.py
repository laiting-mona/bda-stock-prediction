"""
Enhanced day-level features for v4:
  - Text: mean TF-IDF (300) + max TF-IDF (300) = 600 dims
    mean captures "average daily sentiment"
    max  captures "strongest signal in any article that day"
  - Tech (original 6): prev_ret_1d, prev_ret_5d, vol_5d, vol_20d, rsi_14, n_articles
  - Tech (derived 5):
      ret1_std = prev_ret_1d / (vol_5d + 1e-8)   standardised return
      ret5_std = prev_ret_5d / (vol_20d + 1e-8)  standardised 5-day momentum
      rsi_norm = (rsi_14 - 50) / 25              centered RSI (-2 oversold … +2 overbought)
      vol_ratio = vol_5d / (vol_20d + 1e-8)      recent vs long-term vol
      sign_ret1 = sign of prev_ret_1d            yesterday's direction (+1/-1)
  - Temporal (2): sin/cos day-of-week (cyclical encoding)
  Total: 613 features
Outputs:
  data/processed/tsmc_day_text_enhanced_v4.csv    (600 cols)
  data/processed/tsmc_day_tech_enhanced_v4.csv    (13 cols)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]

print("Loading article-level v2 data…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

feat_df["post_time"] = pd.to_datetime(feat_df["post_time"])
feat_df["date"]      = feat_df["post_time"].dt.date
tfidf_arr  = vec_df.values
tfidf_cols = vec_df.columns.tolist()

print(f"  Articles: {len(feat_df)}  Dates: {feat_df['date'].nunique()}")

day_text_rows, day_tech_rows, day_meta_rows = [], [], []

for date, grp in feat_df.groupby("date"):
    idx     = grp.index
    mean_v  = tfidf_arr[idx].mean(axis=0)    # (300,)
    max_v   = tfidf_arr[idx].max(axis=0)     # (300,)  strongest signal per term

    tech0   = grp[TECH_COLS].iloc[0]         # same per day
    pr1     = tech0["prev_ret_1d"]
    pr5     = tech0["prev_ret_5d"]
    v5      = tech0["vol_5d"]
    v20     = tech0["vol_20d"]
    rsi     = tech0["rsi_14"]
    n_art   = len(grp)

    ret1_std  = pr1 / (v5 + 1e-8)
    ret5_std  = pr5 / (v20 + 1e-8)
    rsi_norm  = (rsi - 50) / 25.0
    vol_ratio = v5 / (v20 + 1e-8)
    sign_ret1 = np.sign(pr1) if pr1 != 0 else 0.0

    dow       = pd.Timestamp(str(date)).dayofweek          # 0=Mon … 4=Fri
    dow_sin   = np.sin(2 * np.pi * dow / 5)
    dow_cos   = np.cos(2 * np.pi * dow / 5)

    label = int(grp["label"].mode().iloc[0])
    ret   = float(grp["return_rate"].iloc[0])

    day_meta_rows.append({"date": date, "n_articles": n_art,
                          "return_rate": ret, "label": label})
    day_text_rows.append(np.concatenate([mean_v, max_v]))   # (600,)
    day_tech_rows.append([pr1, pr5, v5, v20, rsi, n_art,
                          ret1_std, ret5_std, rsi_norm, vol_ratio,
                          sign_ret1, dow_sin, dow_cos])

meta_df  = pd.DataFrame(day_meta_rows).sort_values("date").reset_index(drop=True)
mean_cols = tfidf_cols
max_cols  = [f"max_{c}" for c in tfidf_cols]
text_df   = pd.DataFrame(day_text_rows, columns=mean_cols + max_cols)
tech_cols_enh = (TECH_COLS +
                 ["ret1_std", "ret5_std", "rsi_norm", "vol_ratio",
                  "sign_ret1", "dow_sin", "dow_cos"])
tech_df   = pd.DataFrame(day_tech_rows, columns=tech_cols_enh)

print(f"  Day rows: {len(meta_df)}")
print(f"  Text shape: {text_df.shape}  Tech shape: {tech_df.shape}")
print(f"  Label: Down={( meta_df['label']==0).sum()}  Up={(meta_df['label']==1).sum()}")

text_df.to_csv(DATA_DIR / "processed" / "tsmc_day_text_enhanced_v4.csv",
               index=False, encoding="utf-8-sig")
tech_df.to_csv(DATA_DIR / "processed" / "tsmc_day_tech_enhanced_v4.csv",
               index=False, encoding="utf-8-sig")
# overwrite meta (same as before but regenerated cleanly)
meta_df.to_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv",
               index=False, encoding="utf-8-sig")
print("Done — saved tsmc_day_text_enhanced_v4.csv  tsmc_day_tech_enhanced_v4.csv")
