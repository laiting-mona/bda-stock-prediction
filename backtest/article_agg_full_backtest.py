"""
Full article-level + day-level aggregation backtest.

Models (all article-level, 306-dim input):
  NB_v2, NB_v3, RF_v2, RF_v3, XGB_v2, XGB_v3, XGB_text (Route A, 300-dim)

5 Aggregation strategies per model:
  mean / max / length_wt / attention / mcdonald

Ensemble combinations (per strategy):
  ens_all       — unweighted mean of all 7 models
  ens_cvf1      — weighted by each model's CV F1
  ens_type      — one of each type: NB_v3 + RF_v3 + XGB_v2
  ens_xgb_nb    — XGB_v2 + NB_v2 (historically best pair)

Output: results/backtest/article_agg_full/summary.csv
No transaction costs. Focus: accuracy, Up/Down accuracy, trade rate.
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

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "results" / "backtest" / "article_agg_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT     = 0.8
TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]

# ── Wrapper class stubs (needed for pickle.load) ──────────────────────────────
class NBv3Wrapper:
    def __init__(self, model, n_tfidf, cv_f1):
        self.model = model; self.n_tfidf = n_tfidf; self.cv_f1 = cv_f1
    def predict_proba(self, X):
        return self.model.predict_proba(X[:, :self.n_tfidf])

class RFv3Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class XGBv3Wrapper:
    def __init__(self, model, cv_f1):
        self.model = model; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading article-level data...")
meta = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv",    encoding="utf-8-sig")
vec  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

meta["post_time"] = pd.to_datetime(meta["post_time"])
meta["date"]      = meta["post_time"].dt.date
meta["text"]      = (meta["title"].fillna("") + " " + meta["content"].fillna("")).str.strip()

X_tfidf = vec.values.astype(np.float32)
X_tech  = meta[TECH_COLS].fillna(0).values.astype(np.float32)
X_306   = np.hstack([X_tfidf, X_tech])   # all v2/v3 models
y       = meta["label"].astype(int).values

n_split    = int(len(meta) * SPLIT)
X_te       = X_306[n_split:]
X_te_300   = X_tfidf[n_split:]           # Route A text-only
y_test     = y[n_split:]
meta_test  = meta.iloc[n_split:].reset_index(drop=True)

print(f"  Test: {len(meta_test)} articles | {meta_test['date'].nunique()} days "
      f"({meta_test['date'].min()} -> {meta_test['date'].max()})")

# ── 2. Load all models ─────────────────────────────────────────────────────────
print("\nLoading models...")
MODEL_SPECS = {
    "NB_v2":   (MODEL_DIR / "naive-bayes/NB_v2_model.pkl",       X_te,     0.44),
    "NB_v3":   (MODEL_DIR / "naive-bayes/NB_v3_model.pkl",       X_te,     0.44),
    "RF_v2":   (MODEL_DIR / "RF/RF_v2_model.pkl",                X_te,     0.37),
    "RF_v3":   (MODEL_DIR / "RF/RF_v3_model.pkl",                X_te,     0.45),
    "XGB_v2":  (MODEL_DIR / "XGBoost/XGBoost_v2_model.pkl",      X_te,     0.50),
    "XGB_v3":  (MODEL_DIR / "XGBoost/XGBoost_v3_model.pkl",      X_te,     0.51),
    "XGB_txt": (MODEL_DIR / "XGBoost/XGBoost_article_text_model.pkl", X_te_300, 0.53),
}

loaded_models = {}
for name, (path, X_input, default_cvf1) in MODEL_SPECS.items():
    if not path.exists():
        print(f"  {name}: MISSING {path}"); continue
    m = pickle.load(open(path, "rb"))
    cvf1 = getattr(m, "cv_f1", default_cvf1)
    loaded_models[name] = {"model": m, "X": X_input, "cv_f1": float(cvf1)}
    print(f"  {name:10s} loaded  cv_f1={cvf1:.4f}  type={type(m).__name__}")

# ── 3. McDonald n-gram dictionary ──────────────────────────────────────────────
print("\nLoading McDonald n-gram dictionaries...")
up_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_up.csv",   encoding="utf-8-sig")
dn_df = pd.read_csv(DATA_DIR / "features" / "tsmc_n-gram_down.csv", encoding="utf-8-sig")
up_df.columns = ["ngram","TF","DF","TFIDF","TF_n","DF_n","TF_a","DF_a","MI","Lift"]
dn_df.columns = up_df.columns
up_dict = dict(zip(up_df["ngram"].head(200), up_df["TFIDF"].head(200)))
dn_dict = dict(zip(dn_df["ngram"].head(200), dn_df["TFIDF"].head(200)))

def mcd_score(text: str) -> float:
    u = sum(text.count(ng) * w for ng, w in up_dict.items())
    d = sum(text.count(ng) * w for ng, w in dn_dict.items())
    return float(u / (u + d + 1e-9))

print("  Computing McDonald scores for test articles...")
mcd_arr = np.array([mcd_score(t) for t in meta_test["text"]], dtype=np.float32)

# ── 4. Article-level inference ─────────────────────────────────────────────────
print("\nRunning article-level inference...")
article_probs = {}
for name, spec in loaded_models.items():
    p = spec["model"].predict_proba(spec["X"])[:, 1]
    article_probs[name] = p
    print(f"  {name:10s}  prob mean={p.mean():.3f}  std={p.std():.3f}")

# ── 5. Day-level aggregation ───────────────────────────────────────────────────
def aggregate_daily(prob_up, strategy):
    rows = []
    for date, idx in meta_test.groupby("date").groups.items():
        idx = list(idx)
        p    = prob_up[idx]
        conf = np.maximum(p, 1 - p)
        txt  = meta_test.loc[idx, "text"].tolist()
        mcd  = mcd_arr[idx]
        lbl  = int(meta_test.loc[idx, "label"].mode().iloc[0])
        ret  = float(meta_test.loc[idx, "return_rate"].iloc[0])

        if strategy == "mean":
            agg = p.mean()
        elif strategy == "max":
            agg = p[np.argmax(conf)]
        elif strategy == "length_wt":
            w = np.array([max(len(t), 1) for t in txt], dtype=np.float32)
            agg = (p * w / w.sum()).sum()
        elif strategy == "attention":
            w = softmax(conf * 3)
            agg = (p * w).sum()
        elif strategy == "mcdonald":
            opinion = np.abs(mcd - 0.5) + 1e-6
            w = opinion / opinion.sum()
            agg = 0.5 * (p * w).sum() + 0.5 * (mcd * w).sum()
        else:
            raise ValueError(strategy)

        rows.append({"date": date, "prob_up": float(agg),
                     "confidence": max(float(agg), 1 - float(agg)),
                     "pred": int(agg >= 0.5), "label": lbl, "return_rate": ret})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ── 6. Ensemble combos ────────────────────────────────────────────────────────
ENSEMBLE_DEFS = {
    "ens_all":    list(loaded_models.keys()),
    "ens_cvf1":   list(loaded_models.keys()),          # same members, weighted below
    "ens_type":   ["NB_v3", "RF_v3", "XGB_v2"],
    "ens_xgb_nb": ["XGB_v2", "NB_v2"],
}

def ensemble_daily(daily_map, members, strategy, weighting="equal"):
    """
    daily_map: {model_name: daily_df}
    weighting: 'equal' or 'cvf1'
    """
    rows = []
    dates = daily_map[members[0]]["date"].tolist()
    weights = np.array([loaded_models[m]["cv_f1"] if weighting == "cvf1" else 1.0
                        for m in members], dtype=np.float32)
    weights /= weights.sum()

    for i, date in enumerate(dates):
        probs_row = [daily_map[m].iloc[i]["prob_up"] for m in members]
        agg = float(np.dot(probs_row, weights))
        lbl = daily_map[members[0]].iloc[i]["label"]
        ret = daily_map[members[0]].iloc[i]["return_rate"]
        rows.append({"date": date, "prob_up": agg,
                     "confidence": max(agg, 1 - agg),
                     "pred": int(agg >= 0.5), "label": lbl, "return_rate": ret})
    return pd.DataFrame(rows)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(daily_df, label=""):
    n   = len(daily_df)
    acc = accuracy_score(daily_df["label"], daily_df["pred"])
    f1  = f1_score(daily_df["label"], daily_df["pred"], average="macro", zero_division=0)
    up_m = daily_df["label"] == 1; dn_m = daily_df["label"] == 0
    acc_up = accuracy_score(daily_df.loc[up_m,"label"], daily_df.loc[up_m,"pred"]) if up_m.any() else np.nan
    acc_dn = accuracy_score(daily_df.loc[dn_m,"label"], daily_df.loc[dn_m,"pred"]) if dn_m.any() else np.nan
    res = {"label": label, "n_days": n, "acc": acc, "f1": f1,
           "acc_up": acc_up, "acc_dn": acc_dn}
    for thr in [0.55, 0.60, 0.65]:
        mask = daily_df["confidence"] >= thr
        k = mask.sum()
        key = str(thr)
        res[f"n@{key}"] = k
        res[f"rate@{key}"] = round(k / n, 3) if n else 0
        res[f"acc@{key}"] = accuracy_score(daily_df.loc[mask,"label"],
                                            daily_df.loc[mask,"pred"]) if k else np.nan
    return res

# ── 8. Run everything ─────────────────────────────────────────────────────────
STRATEGIES = ["mean", "max", "length_wt", "attention", "mcdonald"]
all_results = []

# Per-model results
daily_by_model_strat = {}   # (model, strat) -> daily_df

print("\n" + "="*80)
print("Individual models")
print("="*80)
for strat in STRATEGIES:
    for mname in loaded_models:
        daily = aggregate_daily(article_probs[mname], strat)
        daily_by_model_strat[(mname, strat)] = daily
        ev = evaluate(daily, label=f"{mname}|{strat}")
        all_results.append(ev)
        print(f"  {mname:10s} | {strat:12s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
              f"up={ev['acc_up']:.3f}  dn={ev['acc_dn']:.3f}  "
              f"n@0.6={ev['n@0.6']:2d}({ev['rate@0.6']:.0%})  acc@0.6={ev.get('acc@0.6', float('nan')):.3f}")

# Ensemble results
print("\n" + "="*80)
print("Ensembles")
print("="*80)
for strat in STRATEGIES:
    daily_map = {m: daily_by_model_strat[(m, strat)] for m in loaded_models}
    for ens_name, members in ENSEMBLE_DEFS.items():
        avail = [m for m in members if m in loaded_models]
        if len(avail) < 2:
            continue
        weighting = "cvf1" if ens_name == "ens_cvf1" else "equal"
        daily_ens = ensemble_daily(daily_map, avail, strat, weighting)
        key = f"{ens_name}({weighting})|{strat}"
        ev  = evaluate(daily_ens, label=key)
        all_results.append(ev)
        print(f"  {ens_name:15s} | {strat:12s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
              f"up={ev['acc_up']:.3f}  dn={ev['acc_dn']:.3f}  "
              f"n@0.6={ev['n@0.6']:2d}({ev['rate@0.6']:.0%})  acc@0.6={ev.get('acc@0.6', float('nan')):.3f}")

# ── 9. Summary table ──────────────────────────────────────────────────────────
summary = pd.DataFrame(all_results)
for col in [c for c in summary.columns if c not in ("label","n_days")]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)

print("\n" + "="*80)
print("TOP 10 by overall accuracy")
print("="*80)
print(summary.sort_values("acc", ascending=False).head(10)[
    ["label","n_days","acc","f1","acc_up","acc_dn",
     "n@0.55","rate@0.55","acc@0.55","n@0.6","rate@0.6","acc@0.6"]
].to_string(index=False))

print("\nTOP 10 by acc@0.60 (high-confidence days)")
print("="*80)
print(summary.dropna(subset=["acc@0.6"]).sort_values("acc@0.6", ascending=False).head(10)[
    ["label","n_days","acc","f1","acc_up","acc_dn",
     "n@0.6","rate@0.6","acc@0.6","n@0.65","rate@0.65","acc@0.65"]
].to_string(index=False))

summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT_DIR / 'summary.csv'}")

# ── 10. Heatmap: acc by model × strategy ─────────────────────────────────────
individual = summary[~summary["label"].str.startswith("ens")].copy()
individual[["model","strategy"]] = individual["label"].str.split("|", n=1, expand=True)
pivot = individual.pivot(index="model", columns="strategy", values="acc").fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
import matplotlib.colors as mcolors

for ax, col, title in [(axes[0], "acc", "Overall Accuracy"),
                        (axes[1], "acc@0.6", "Accuracy @ conf>=0.60")]:
    if col == "acc@0.6":
        pv = individual.pivot(index="model", columns="strategy", values="acc@0.6").fillna(np.nan)
    else:
        pv = pivot
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=0.35, vmax=0.75)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
    ax.set_title(title)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.4 < v < 0.65 else "white")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_accuracy.png", dpi=150)
plt.close()
print(f"Heatmap saved -> {OUT_DIR / 'heatmap_accuracy.png'}")

# ── 11. Bar chart: ensemble vs best individual ────────────────────────────────
ens_rows  = summary[summary["label"].str.startswith("ens")]
best_ind  = summary[~summary["label"].str.startswith("ens")].nlargest(5, "acc")
plot_df   = pd.concat([best_ind, ens_rows[ens_rows["label"].str.contains("mean")]]).drop_duplicates("label")

fig, ax = plt.subplots(figsize=(12, 4))
colors = ["steelblue" if not l.startswith("ens") else "darkorange"
          for l in plot_df["label"]]
bars = ax.bar(range(len(plot_df)), plot_df["acc"], color=colors)
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=7)
ax.axhline(0.5, color="red", ls="--", lw=1, label="50% baseline")
ax.set_ylabel("Accuracy"); ax.set_title("Individual Models vs Ensemble (mean strategy)")
ax.legend(); ax.set_ylim(0.3, 0.7)
for bar, v in zip(bars, plot_df["acc"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "ensemble_vs_individual.png", dpi=150)
plt.close()
print(f"Bar chart saved -> {OUT_DIR / 'ensemble_vs_individual.png'}")
