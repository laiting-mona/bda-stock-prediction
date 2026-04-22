"""
Article-level prediction + day-level aggregation backtest.

Two models:
  Route A — Text-only XGBoost (trained here, TF-IDF 300-dim)
  Route B — Combined XGBoost_v2 (text 300 + tech 6, pre-trained)

Five daily aggregation strategies (applied to both models):
  1. mean       — simple mean of prob_up across daily articles
  2. max        — most confident article wins
  3. length_wt  — weight by article text length
  4. attention  — softmax(confidence) as weights
  5. mcdonald   — n-gram sentiment score as per-article weight

Route A also has Stage-2 tech blend:
  6. tech_blend — 0.6*text_signal + 0.4*tech_signal (RSI + momentum)

Output: results/backtest/article_agg/summary.csv + console table
No transaction costs. Focus: trade rate + prediction direction.
"""
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR  = ROOT / "results" / "backtest" / "article_agg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = 0.8
TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading article-level data...")
meta  = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv",   encoding="utf-8-sig")
vec   = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

meta["post_time"] = pd.to_datetime(meta["post_time"])
meta["date"]      = meta["post_time"].dt.date
meta["text"]      = (meta["title"].fillna("") + " " + meta["content"].fillna("")).str.strip()

X_tfidf = vec.values.astype(np.float32)          # (36834, 300)
X_tech  = meta[TECH_COLS].fillna(0).values.astype(np.float32)  # (36834, 6)
X_comb  = np.hstack([X_tfidf, X_tech])            # (36834, 306)
y       = meta["label"].astype(int).values

n_split = int(len(meta) * SPLIT)
X_tr_txt, X_te_txt = X_tfidf[:n_split], X_tfidf[n_split:]
X_tr_cmb, X_te_cmb = X_comb[:n_split],  X_comb[n_split:]
y_train, y_test    = y[:n_split], y[n_split:]
meta_train         = meta.iloc[:n_split].reset_index(drop=True)
meta_test          = meta.iloc[n_split:].reset_index(drop=True)

print(f"  Train: {len(meta_train)} articles | Test: {len(meta_test)} articles")
print(f"  Test date range: {meta_test['date'].min()} -> {meta_test['date'].max()}")
print(f"  Unique test days: {meta_test['date'].nunique()}")

# ── 2. Train Route A: text-only XGBoost ───────────────────────────────────────
route_a_path = MODEL_DIR / "XGBoost" / "XGBoost_article_text_model.pkl"
if route_a_path.exists():
    print("\nLoading cached Route A model...")
    route_a = pickle.load(open(route_a_path, "rb"))
else:
    print("\nTraining Route A (text-only, article-level XGBoost)...")
    tscv = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        "n_estimators":     [100],
        "max_depth":        [3, 5],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss",
                      use_label_encoder=False, n_jobs=1, verbosity=0,
                      tree_method="hist", scale_pos_weight=1),
        param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1, verbose=0, refit=False,
    )
    grid.fit(X_tr_txt, y_train)
    best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
    print(f"  Best params: {best_p}  CV F1={grid.best_score_:.4f}")
    route_a = XGBClassifier(**best_p, n_estimators=200, random_state=42,
                            eval_metric="logloss", use_label_encoder=False,
                            n_jobs=-1, verbosity=0, tree_method="hist")
    route_a.fit(X_tr_txt, y_train)
    pickle.dump(route_a, open(route_a_path, "wb"))
    print(f"  Route A saved -> {route_a_path}")

# ── 3. Load Route B: XGBoost_v2 combined ──────────────────────────────────────
print("\nLoading Route B (XGBoost_v2, text+tech)...")
route_b = pickle.load(open(MODEL_DIR / "XGBoost" / "XGBoost_v2_model.pkl", "rb"))

# ── 4. Article-level probabilities on test set ────────────────────────────────
print("Running article-level inference...")
prob_a = route_a.predict_proba(X_te_txt)[:, 1]   # (n_test,)
prob_b = route_b.predict_proba(X_te_cmb)[:, 1]   # (n_test,)

# ── 5. McDonald n-gram sentiment dictionary ───────────────────────────────────
print("Loading McDonald n-gram dictionaries...")
up_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_up.csv",   encoding="utf-8-sig")
dn_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_down.csv", encoding="utf-8-sig")
up_df.columns = ["ngram","TF","DF","TFIDF","TF_norm","DF_norm","TF_adj","DF_adj","MI","Lift"]
dn_df.columns = up_df.columns

# Keep top-200 by TF-IDF weight for speed
up_dict = dict(zip(up_df["ngram"].head(200), up_df["TFIDF"].head(200)))
dn_dict = dict(zip(dn_df["ngram"].head(200), dn_df["TFIDF"].head(200)))

def mcdonald_score(text: str) -> float:
    """Returns sentiment score in [0,1]: >0.5 = bullish signal."""
    up_s = dn_s = 0.0
    for ng, w in up_dict.items():
        up_s += text.count(ng) * w
    for ng, w in dn_dict.items():
        dn_s += text.count(ng) * w
    total = up_s + dn_s + 1e-9
    return float(up_s / total)   # 0..1, higher = more bullish signals

print("Computing McDonald sentiment scores...")
mcd_scores = np.array([mcdonald_score(t) for t in meta_test["text"].tolist()],
                       dtype=np.float32)

# ── 6. Day-level aggregation ──────────────────────────────────────────────────
def aggregate_daily(prob_up, meta_df, mcd_sc, strategy):
    """
    Returns DataFrame with columns: date, prob_up_agg, confidence, pred, label, return_rate
    """
    rows = []
    for date, grp_idx in meta_df.groupby("date").groups.items():
        grp_idx = list(grp_idx)
        p   = prob_up[grp_idx]          # prob_up for each article
        txt = meta_df.loc[grp_idx, "text"].tolist()
        mcd = mcd_sc[grp_idx]
        lbl = int(meta_df.loc[grp_idx, "label"].mode().iloc[0])
        ret = float(meta_df.loc[grp_idx, "return_rate"].iloc[0])

        # Article confidence = max(p, 1-p)
        conf = np.maximum(p, 1 - p)

        if strategy == "mean":
            agg = p.mean()

        elif strategy == "max":
            # Most confident article's prediction
            agg = p[np.argmax(conf)]

        elif strategy == "length_wt":
            lengths = np.array([len(t) for t in txt], dtype=np.float32)
            lengths = lengths / (lengths.sum() + 1e-9)
            agg = (p * lengths).sum()

        elif strategy == "attention":
            weights = softmax(conf * 3)   # temperature=3 sharpens the distribution
            agg = (p * weights).sum()

        elif strategy == "mcdonald":
            # Weight by McDonald sentiment score; neutral (0.5) gets low weight
            mcd_w = np.abs(mcd - 0.5) + 1e-6     # how opinionated each article is
            mcd_w = mcd_w / mcd_w.sum()
            # Blend model prob with McDonald signal
            mcd_signal = mcd                       # McDonald prob estimate
            agg = 0.5 * (p * mcd_w).sum() + 0.5 * (mcd_signal * mcd_w).sum()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rows.append({
            "date":       date,
            "prob_up":    float(agg),
            "confidence": max(float(agg), 1 - float(agg)),
            "pred":       int(agg >= 0.5),
            "label":      lbl,
            "return_rate": ret,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def tech_blend(daily_a: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 for Route A: blend text prob with RSI+momentum tech signal."""
    day_tech = meta_df.groupby("date").first()[TECH_COLS].reset_index()
    merged = daily_a.merge(day_tech, on="date", how="left")

    rsi_sig = (50 - merged["rsi_14"].fillna(50)) / 50          # +: oversold, -: overbought
    mom_sig = np.tanh(merged["prev_ret_1d"].fillna(0) * 50)    # tanh of yesterday's return
    tech_prob = 0.5 + 0.3 * rsi_sig + 0.2 * mom_sig           # approx 0..1

    out = daily_a.copy()
    out["prob_up"]    = (0.6 * daily_a["prob_up"] + 0.4 * tech_prob).clip(0, 1)
    out["confidence"] = out["prob_up"].apply(lambda p: max(p, 1 - p))
    out["pred"]       = (out["prob_up"] >= 0.5).astype(int)
    return out

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(daily_df, label=""):
    n_days = len(daily_df)
    acc_all = accuracy_score(daily_df["label"], daily_df["pred"])
    f1_all  = f1_score(daily_df["label"], daily_df["pred"], average="macro", zero_division=0)

    up_mask  = daily_df["label"] == 1
    dn_mask  = daily_df["label"] == 0
    acc_up   = accuracy_score(daily_df.loc[up_mask, "label"], daily_df.loc[up_mask, "pred"]) if up_mask.sum() else float("nan")
    acc_dn   = accuracy_score(daily_df.loc[dn_mask, "label"], daily_df.loc[dn_mask, "pred"]) if dn_mask.sum() else float("nan")

    results = {"label": label, "n_days": n_days,
               "acc": acc_all, "f1": f1_all,
               "acc_up": acc_up, "acc_dn": acc_dn}
    for thr in [0.55, 0.60, 0.65]:
        mask = daily_df["confidence"] >= thr
        n_sel = mask.sum()
        if n_sel > 0:
            results[f"trade_{thr}"] = n_sel
            results[f"rate_{thr}"]  = n_sel / n_days
            results[f"acc_{thr}"]   = accuracy_score(daily_df.loc[mask, "label"],
                                                       daily_df.loc[mask, "pred"])
        else:
            results[f"trade_{thr}"] = 0
            results[f"rate_{thr}"]  = 0.0
            results[f"acc_{thr}"]   = float("nan")
    return results


print("\n" + "="*80)
print("Running all aggregation strategies...")
print("="*80)

STRATEGIES = ["mean", "max", "length_wt", "attention", "mcdonald"]

all_results = []
daily_frames = {}

for strat in STRATEGIES:
    for route_name, probs in [("RouteA_text", prob_a), ("RouteB_combined", prob_b)]:
        daily = aggregate_daily(probs, meta_test.reset_index(), mcd_scores, strat)
        key = f"{route_name}|{strat}"
        daily_frames[key] = daily
        ev = evaluate(daily, label=key)
        all_results.append(ev)
        print(f"  {key:40s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
              f"trade@0.60={ev['trade_0.6']:2d}/{ev['n_days']}({ev['rate_0.6']:.0%})")

# Route A + tech blend (5 strategies × tech blend)
for strat in STRATEGIES:
    daily_base = daily_frames[f"RouteA_text|{strat}"]
    daily_tb   = tech_blend(daily_base, meta_test.reset_index())
    key = f"RouteA_TechBlend|{strat}"
    daily_frames[key] = daily_tb
    ev = evaluate(daily_tb, label=key)
    all_results.append(ev)
    print(f"  {key:40s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
          f"trade@0.60={ev['trade_0.6']:2d}/{ev['n_days']}({ev['rate_0.6']:.0%})")

# ── 8. Print summary table ────────────────────────────────────────────────────
summary = pd.DataFrame(all_results)
for col in ["acc","f1","acc_up","acc_dn","rate_0.55","rate_0.6","rate_0.65",
            "acc_0.55","acc_0.6","acc_0.65"]:
    if col in summary.columns:
        summary[col] = summary[col].round(4)

print("\n" + "="*80)
print("SUMMARY  (all test days)")
print("="*80)
disp_cols = ["label","n_days","acc","f1","acc_up","acc_dn",
             "trade_0.55","rate_0.55","acc_0.55",
             "trade_0.6", "rate_0.6", "acc_0.6",
             "trade_0.65","rate_0.65","acc_0.65"]
print(summary[disp_cols].to_string(index=False))

summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT_DIR / 'summary.csv'}")

# ── 9. Plot: accuracy by strategy ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall accuracy by strategy+model
pivot_acc = summary.copy()
pivot_acc[["route","strategy"]] = pivot_acc["label"].str.split("|", expand=True)

for ax, col, title in [(axes[0], "acc", "Overall Accuracy"),
                        (axes[1], "acc_0.6", "Accuracy @ confidence>=0.60")]:
    for route in pivot_acc["route"].unique():
        sub = pivot_acc[pivot_acc["route"] == route]
        ax.bar([f"{r}_{s[:4]}" for r, s in zip(sub["route"], sub["strategy"])],
               sub[col].fillna(0),
               label=route, alpha=0.7)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="random baseline")
    ax.set_title(title); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_by_strategy.png", dpi=150)
plt.close()
print(f"Plot saved -> {OUT_DIR / 'accuracy_by_strategy.png'}")

# ── 10. Best strategy detail ──────────────────────────────────────────────────
best = summary.loc[summary["acc"].idxmax()]
print(f"\nBest overall accuracy: {best['label']}  acc={best['acc']:.4f}  f1={best['f1']:.4f}")
best_thr = summary.loc[summary["acc_0.6"].fillna(0).idxmax()]
print(f"Best acc@0.60:         {best_thr['label']}  acc={best_thr['acc_0.6']:.4f}  "
      f"n_trade={int(best_thr['trade_0.6'])}/{int(best_thr['n_days'])}")
