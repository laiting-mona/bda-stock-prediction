"""
Stacking v1: OOF (Out-of-Fold) meta-learner stacking.
Base learners: NB_v5, RF_v5, XGB_v5, LightGBM_v1 (article-level, 322-dim).
Meta-learner: LogisticRegression on stacked OOF probabilities (4-dim input).
TimeSeriesSplit ensures no future leakage in OOF generation.
Output: stacking_v1_model.pkl
"""
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = 0.8
TECH_V2 = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
TECH_NEW = [
    "hour_sin", "hour_cos",
    "session_pre", "session_mkt", "session_aft",
    "dow_sin", "dow_cos",
    "article_len_norm",
    "mcd_up", "mcd_dn", "mcd_net",
    "ret1_std", "ret5_std", "rsi_norm", "vol_ratio", "sign_ret1",
]
TECH_COLS = TECH_V2 + TECH_NEW

# NB v5 uses only: TF-IDF (300) + mcd_up + mcd_dn + n_articles
NB_EXTRA_COLS = ["mcd_up", "mcd_dn", "n_articles"]


# ── Wrapper stubs (required for pickle loading) ───────────────────────────────
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
        self.base_models = base_models   # list of (name, wrapper, X_selector_fn)
        self.meta_model  = meta_model
        self.cv_f1       = cv_f1

    def _base_probs(self, X_full, X_nb):
        cols = []
        for name, wrapper, use_nb in self.base_models:
            X_in = X_nb if use_nb else X_full
            cols.append(wrapper.predict_proba(X_in)[:, 1:2])
        return np.hstack(cols)

    def predict_proba(self, X_full, X_nb=None):
        if X_nb is None:
            X_nb = X_full
        meta_X = self._base_probs(X_full, X_nb)
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X_full, X_nb=None):
        return (self.predict_proba(X_full, X_nb)[:, 1] >= 0.5).astype(int)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading v3 features...")
feat = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v3.csv", encoding="utf-8-sig")
vec  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y       = feat["label"].astype(int).values
X_tech  = feat[TECH_COLS].fillna(0).values
X_extra = feat[NB_EXTRA_COLS].fillna(0).clip(lower=0).values
X_tfidf = vec.values
X_322   = np.hstack([X_tfidf, X_tech])
X_303   = np.hstack([X_tfidf, X_extra])   # NB v5 input

n = int(len(X_322) * SPLIT)
X_train_322, X_test_322 = X_322[:n], X_322[n:]
X_train_303, X_test_303 = X_303[:n], X_303[n:]
y_train, y_test         = y[:n], y[n:]
print(f"Train: {X_train_322.shape}  Test: {X_test_322.shape}")

# ── Load pre-trained base models ───────────────────────────────────────────────
print("\nLoading pre-trained base models...")
base_dir = ROOT / "models"
base_paths = {
    "XGB_v5":  base_dir / "XGBoost/XGBoost_v5_model.pkl",
    "RF_v5":   base_dir / "RF/RF_v5_model.pkl",
    "NB_v5":   base_dir / "naive-bayes/NB_v5_model.pkl",
    "LGBM_v1": base_dir / "lgbm/LightGBM_v1_model.pkl",
}
base_models_loaded = {}
for name, path in base_paths.items():
    if not path.exists():
        print(f"  MISSING: {path}")
        print("  Run XGBoost_v5_train.py, RF_v5_train.py, NB_v5_train.py, LightGBM_v1_train.py first.")
        sys.exit(1)
    m = pickle.load(open(path, "rb"))
    base_models_loaded[name] = m
    print(f"  {name} loaded  cv_f1={getattr(m, 'cv_f1', '?'):.4f}")

# ── OOF meta-feature generation ────────────────────────────────────────────────
print("\nGenerating OOF predictions (TimeSeriesSplit, 5 folds)...")
tscv  = TimeSeriesSplit(n_splits=5)
oof   = np.zeros((len(X_train_322), 4), dtype=np.float32)

model_order = ["XGB_v5", "RF_v5", "NB_v5", "LGBM_v1"]
is_nb       = [False, False, True, False]   # NB uses X_303

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_322)):
    Xtr_322, Xval_322 = X_train_322[tr_idx], X_train_322[val_idx]
    Xtr_303, Xval_303 = X_train_303[tr_idx], X_train_303[val_idx]
    ytr = y_train[tr_idx]

    for i, (mname, use_nb) in enumerate(zip(model_order, is_nb)):
        Xtr_in  = Xtr_303  if use_nb else Xtr_322
        Xval_in = Xval_303 if use_nb else Xval_322
        m = base_models_loaded[mname]
        m.model.fit(Xtr_in, ytr)   # refit base estimator on fold
        oof[val_idx, i] = m.predict_proba(Xval_in)[:, 1]

    print(f"  Fold {fold+1}/5 done")

print(f"OOF shape: {oof.shape}")

# ── Train meta-learner ─────────────────────────────────────────────────────────
print("\nTraining meta-learner (LogisticRegression)...")
meta_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                                  class_weight="balanced", solver="lbfgs")),
])
meta_pipe.fit(oof, y_train)
cv_f1_meta = f1_score(y_train, meta_pipe.predict(oof), average="macro")
print(f"Meta-learner OOF Macro F1 (in-sample): {cv_f1_meta:.4f}")

# ── Generate test meta-features using full-train base models ──────────────────
print("\nBuilding test meta-features from full-train base models...")
test_meta = np.zeros((len(X_test_322), 4), dtype=np.float32)
for i, (mname, use_nb) in enumerate(zip(model_order, is_nb)):
    Xte_in = X_test_303 if use_nb else X_test_322
    m = base_models_loaded[mname]
    m.model.fit(X_train_303 if use_nb else X_train_322, y_train)
    test_meta[:, i] = m.predict_proba(Xte_in)[:, 1]

y_pred   = meta_pipe.predict(test_meta)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\n=== Stacking v1 — All {len(y_test)} articles ===")
print(f"Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"], zero_division=0))

prob_test = meta_pipe.predict_proba(test_meta)[:, 1]
print(f"Prob stats: mean={prob_test.mean():.3f}  std={prob_test.std():.3f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Stacking v1 (NB+RF+XGB+LGB -> LR) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "stacking_v1_confusion_matrix.png", dpi=150)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
base_list = [(mname, base_models_loaded[mname], is_nb[i])
             for i, mname in enumerate(model_order)]
wrapper = StackingV1Wrapper(base_list, meta_pipe, cv_f1_meta)
with open(MODEL_DIR / "stacking_v1_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"\nstacking_v1_model.pkl saved -> {MODEL_DIR / 'stacking_v1_model.pkl'}")
