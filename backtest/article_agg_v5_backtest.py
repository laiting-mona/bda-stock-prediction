"""
Article-level + day-level aggregation backtest — v5 models.

Models (article-level):
  NB_v5  (TF-IDF 300 + mcd_up + mcd_dn + n_articles = 303-dim)
  RF_v5  (TF-IDF 300 + 22 tech_v3 = 322-dim)
  XGB_v5 (322-dim)
  LGBM_v1(322-dim)
  Stack_v1 (NB+RF+XGB+LGB -> LR meta, 322/303-dim)

5 aggregation strategies per model:
  mean / max / length_wt / attention / mcdonald

Ensemble combos:
  ens_all   — unweighted mean of all 5 v5 models
  ens_cvf1  — weighted by CV F1
  ens_tree  — RF_v5 + XGB_v5 + LGBM_v1
  ens_nb_xgb— NB_v5 + XGB_v5

Output: results/backtest/article_agg_v5/summary.csv + charts
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
OUT_DIR   = ROOT / "results" / "backtest" / "article_agg_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT    = 0.8
TECH_V2  = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
TECH_NEW = [
    "hour_sin", "hour_cos",
    "session_pre", "session_mkt", "session_aft",
    "dow_sin", "dow_cos",
    "article_len_norm",
    "mcd_up", "mcd_dn", "mcd_net",
    "ret1_std", "ret5_std", "rsi_norm", "vol_ratio", "sign_ret1",
]
TECH_COLS   = TECH_V2 + TECH_NEW
NB_EXTRA    = ["mcd_up", "mcd_dn", "n_articles"]

# ── Wrapper stubs for pickle loading ─────────────────────────────────────────
class XGBv5Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class RFv5Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class NBv5Wrapper:
    def __init__(self, model, n_tfidf, n_extra, optimal_threshold, cv_f1):
        self.model = model; self.n_tfidf = n_tfidf; self.n_extra = n_extra
        self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X[:, :self.n_tfidf + self.n_extra])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class LGBMv1Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class StackingV1Wrapper:
    def __init__(self, base_models, meta_model, cv_f1):
        self.base_models = base_models; self.meta_model = meta_model; self.cv_f1 = cv_f1
    def _base_probs(self, X_full, X_nb):
        cols = []
        for name, wrapper, use_nb in self.base_models:
            X_in = X_nb if use_nb else X_full
            cols.append(wrapper.predict_proba(X_in)[:, 1:2])
        return np.hstack(cols)
    def predict_proba(self, X_full, X_nb=None):
        if X_nb is None: X_nb = X_full
        return self.meta_model.predict_proba(self._base_probs(X_full, X_nb))
    def predict(self, X_full, X_nb=None):
        return (self.predict_proba(X_full, X_nb)[:, 1] >= 0.5).astype(int)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading v3 features...")
feat = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v3.csv", encoding="utf-8-sig")
vec  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

feat["post_time"] = pd.to_datetime(feat["post_time"])
feat["date"]      = feat["post_time"].dt.date
feat["text"]      = (feat["title"].fillna("") + " " + feat["content"].fillna("")).str.strip()

X_tfidf = vec.values.astype(np.float32)
X_tech  = feat[TECH_COLS].fillna(0).values.astype(np.float32)
X_extra = feat[NB_EXTRA].fillna(0).clip(lower=0).values.astype(np.float32)
X_322   = np.hstack([X_tfidf, X_tech])
X_303   = np.hstack([X_tfidf, X_extra])
y       = feat["label"].astype(int).values

n_split     = int(len(feat) * SPLIT)
X_te_322    = X_322[n_split:]
X_te_303    = X_303[n_split:]
y_test      = y[n_split:]
meta_test   = feat.iloc[n_split:].reset_index(drop=True)

print(f"  Test: {len(meta_test)} articles | {meta_test['date'].nunique()} days "
      f"({meta_test['date'].min()} -> {meta_test['date'].max()})")

# ── 2. Load models ────────────────────────────────────────────────────────────
print("\nLoading models...")
MODEL_SPECS = {
    "NB_v5":    (MODEL_DIR / "naive-bayes/NB_v5_model.pkl",        X_te_303, 0.53),
    "RF_v5":    (MODEL_DIR / "RF/RF_v5_model.pkl",                 X_te_322, 0.55),
    "XGB_v5":   (MODEL_DIR / "XGBoost/XGBoost_v5_model.pkl",       X_te_322, 0.55),
    "LGBM_v1":  (MODEL_DIR / "lgbm/LightGBM_v1_model.pkl",         X_te_322, 0.55),
    "Stack_v1": (MODEL_DIR / "stacking/stacking_v1_model.pkl",     X_te_322, 0.55),
}
loaded_models = {}
for name, (path, X_input, default_cvf1) in MODEL_SPECS.items():
    if not path.exists():
        print(f"  {name}: MISSING — skipping"); continue
    m = pickle.load(open(path, "rb"))
    cvf1 = getattr(m, "cv_f1", default_cvf1)
    loaded_models[name] = {"model": m, "X": X_input, "X_nb": X_te_303, "cv_f1": float(cvf1)}
    print(f"  {name:12s} loaded  cv_f1={cvf1:.4f}")

if not loaded_models:
    raise RuntimeError("No models loaded. Train v5 models first.")

# ── 3. McDonald n-gram scores ─────────────────────────────────────────────────
print("\nLoading McDonald dictionaries...")
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

print("  Computing McDonald scores...")
mcd_arr = np.array([mcd_score(t) for t in meta_test["text"]], dtype=np.float32)

# ── 4. Article-level inference ────────────────────────────────────────────────
print("\nRunning article-level inference...")
article_probs = {}
for name, spec in loaded_models.items():
    m = spec["model"]
    if name == "Stack_v1":
        p = m.predict_proba(spec["X"], spec["X_nb"])[:, 1]
    else:
        p = m.predict_proba(spec["X"])[:, 1]
    article_probs[name] = p
    print(f"  {name:12s}  prob mean={p.mean():.3f}  std={p.std():.3f}")

# ── 5. Day-level aggregation ──────────────────────────────────────────────────
def aggregate_daily(prob_up, strategy):
    rows = []
    for date, idx in meta_test.groupby("date").groups.items():
        idx  = list(idx)
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

# ── 6. Ensemble ────────────────────────────────────────────────────────────────
ENSEMBLE_DEFS = {
    "ens_all":     list(loaded_models.keys()),
    "ens_cvf1":    list(loaded_models.keys()),
    "ens_tree":    [m for m in ["RF_v5", "XGB_v5", "LGBM_v1"] if m in loaded_models],
    "ens_nb_xgb":  [m for m in ["NB_v5", "XGB_v5"] if m in loaded_models],
}

def ensemble_daily(daily_map, members, weighting="equal"):
    rows  = []
    dates = daily_map[members[0]]["date"].tolist()
    wts   = np.array([loaded_models[m]["cv_f1"] if weighting == "cvf1" else 1.0
                      for m in members], dtype=np.float32)
    wts  /= wts.sum()
    for i, date in enumerate(dates):
        probs_row = [daily_map[m].iloc[i]["prob_up"] for m in members]
        agg = float(np.dot(probs_row, wts))
        lbl = daily_map[members[0]].iloc[i]["label"]
        ret = daily_map[members[0]].iloc[i]["return_rate"]
        rows.append({"date": date, "prob_up": agg,
                     "confidence": max(agg, 1 - agg),
                     "pred": int(agg >= 0.5), "label": lbl, "return_rate": ret})
    return pd.DataFrame(rows)

# ── 7. Evaluate ────────────────────────────────────────────────────────────────
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
        k    = mask.sum()
        key  = str(thr)
        res[f"n@{key}"]    = int(k)
        res[f"rate@{key}"] = round(k / n, 3) if n else 0
        res[f"acc@{key}"]  = accuracy_score(daily_df.loc[mask,"label"],
                                             daily_df.loc[mask,"pred"]) if k else np.nan
    return res

# ── 8. Run all ─────────────────────────────────────────────────────────────────
STRATEGIES = ["mean", "max", "length_wt", "attention", "mcdonald"]
all_results = []
daily_by_model_strat = {}

print("\n" + "="*80)
print("Individual models")
print("="*80)
for strat in STRATEGIES:
    for mname in loaded_models:
        daily = aggregate_daily(article_probs[mname], strat)
        daily_by_model_strat[(mname, strat)] = daily
        ev = evaluate(daily, label=f"{mname}|{strat}")
        all_results.append(ev)
        print(f"  {mname:12s} | {strat:12s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
              f"up={ev['acc_up']:.3f}  dn={ev['acc_dn']:.3f}  "
              f"n@0.6={ev['n@0.6']:2d}({ev['rate@0.6']:.0%})  "
              f"acc@0.6={ev.get('acc@0.6', float('nan')):.3f}")

print("\n" + "="*80)
print("Ensembles")
print("="*80)
for strat in STRATEGIES:
    daily_map = {m: daily_by_model_strat[(m, strat)] for m in loaded_models}
    for ens_name, members in ENSEMBLE_DEFS.items():
        avail = [m for m in members if m in loaded_models]
        if len(avail) < 2: continue
        weighting = "cvf1" if ens_name == "ens_cvf1" else "equal"
        daily_ens = ensemble_daily(daily_map, avail, weighting)
        ev = evaluate(daily_ens, label=f"{ens_name}({weighting})|{strat}")
        all_results.append(ev)
        print(f"  {ens_name:15s} | {strat:12s}  acc={ev['acc']:.3f}  f1={ev['f1']:.3f}  "
              f"n@0.6={ev['n@0.6']:2d}({ev['rate@0.6']:.0%})  "
              f"acc@0.6={ev.get('acc@0.6', float('nan')):.3f}")

# ── 9. Summary table ───────────────────────────────────────────────────────────
summary = pd.DataFrame(all_results)
for col in [c for c in summary.columns if c not in ("label", "n_days")]:
    summary[col] = pd.to_numeric(summary[col], errors="coerce").round(4)

print("\n" + "="*80)
print("TOP 10 by overall accuracy")
print("="*80)
print(summary.sort_values("acc", ascending=False).head(10)[
    ["label", "n_days", "acc", "f1", "acc_up", "acc_dn",
     "n@0.55", "rate@0.55", "acc@0.55", "n@0.6", "rate@0.6", "acc@0.6"]
].to_string(index=False))

print("\nTOP 10 by acc@0.60 (high-confidence days)")
print("="*80)
print(summary.dropna(subset=["acc@0.6"]).sort_values("acc@0.6", ascending=False).head(10)[
    ["label", "n_days", "acc", "f1", "acc_up", "acc_dn",
     "n@0.6", "rate@0.6", "acc@0.6", "n@0.65", "rate@0.65", "acc@0.65"]
].to_string(index=False))

# ── 10. Best per model type ────────────────────────────────────────────────────
print("\n" + "="*80)
print("BEST CONFIG per requested model type")
print("="*80)
targets = {
    "All-5-Weighted": "ens_cvf1",
    "XGBoost":        "XGB_v5",
    "RF":             "RF_v5",
    "NaiveBayes":     "NB_v5",
}
for label, prefix in targets.items():
    subset = summary[summary["label"].str.startswith(prefix)]
    if subset.empty:
        print(f"  {label}: no results"); continue
    best_overall = subset.nlargest(1, "acc").iloc[0]
    best_conf    = subset.dropna(subset=["acc@0.6"]).nlargest(1, "acc@0.6")
    print(f"\n  [{label}]")
    print(f"    Best overall:   {best_overall['label']}  "
          f"acc={best_overall['acc']:.3f}  f1={best_overall['f1']:.3f}")
    if not best_conf.empty:
        bc = best_conf.iloc[0]
        print(f"    Best @conf0.6:  {bc['label']}  "
              f"acc@0.6={bc['acc@0.6']:.3f}  n={bc['n@0.6']:.0f}({bc['rate@0.6']:.0%})")

summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT_DIR / 'summary.csv'}")

# ── 11. Heatmap ───────────────────────────────────────────────────────────────
individual = summary[~summary["label"].str.startswith("ens")].copy()
individual[["model", "strategy"]] = individual["label"].str.split("|", n=1, expand=True)
pivot = individual.pivot(index="model", columns="strategy", values="acc").fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in [(axes[0], "acc", "Overall Accuracy"),
                        (axes[1], "acc@0.6", "Accuracy @ conf>=0.60")]:
    pv = individual.pivot(index="model", columns="strategy",
                          values=col).fillna(np.nan) if col != "acc" else pivot
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=0.35, vmax=0.80)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
    ax.set_title(title)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.4 < v < 0.70 else "white")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_v5.png", dpi=150)
plt.close()
print(f"Heatmap saved -> {OUT_DIR / 'heatmap_v5.png'}")

# ── 12. Bar chart: top 15 configurations ─────────────────────────────────────
top15 = summary.nlargest(15, "acc")
fig, ax = plt.subplots(figsize=(14, 4))
colors = ["darkorange" if l.startswith("ens") else "steelblue" for l in top15["label"]]
bars   = ax.bar(range(len(top15)), top15["acc"], color=colors)
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15["label"], rotation=45, ha="right", fontsize=7)
ax.axhline(0.5, color="red", ls="--", lw=1, label="50% baseline")
ax.set_ylabel("Accuracy"); ax.set_title("Top 15 Configurations (v5)")
ax.legend(); ax.set_ylim(0.3, 0.85)
for bar, v in zip(bars, top15["acc"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "top15_v5.png", dpi=150)
plt.close()
print(f"Bar chart saved -> {OUT_DIR / 'top15_v5.png'}")
