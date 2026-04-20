"""
Ensemble v3: weighted soft voting by CV Macro F1.
Models with CV F1 below MIN_CV_F1 are excluded automatically.
Output: models/ensemble/ensemble_v3_model.pkl
"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)

# ── Wrapper stubs for pickle deserialization ──────────────────────────────────
class NBv3Wrapper:
    def __init__(self, model, n_tfidf, cv_f1):
        self.model = model; self.n_tfidf = n_tfidf; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X[:, :self.n_tfidf])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

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


ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS   = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT       = 0.8
MIN_CV_F1   = 0.45      # drop models with CV F1 below this

BASE_MODELS = {
    "RF_v3":      ROOT / "models/RF/RF_v3_model.pkl",
    "NB_v3":      ROOT / "models/naive-bayes/NB_v3_model.pkl",
    "XGBoost_v3": ROOT / "models/XGBoost/XGBoost_v3_model.pkl",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading v2 features…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y = feat_df["label"].astype(int).values
X_tech  = feat_df[TECH_COLS].fillna(0).values
X_tfidf = vec_df.values
X_all   = np.hstack([X_tfidf, X_tech])   # shape (36834, 306)

n = int(len(X_all) * SPLIT)
X_test  = X_all[n:]
y_test  = y[n:]

# ── Load base models, collect probs and CV F1 ─────────────────────────────────
models, probs_test, cv_f1s = {}, {}, {}
for name, path in BASE_MODELS.items():
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun {name}_train.py first.")
    m = pickle.load(open(path, "rb"))
    models[name] = m
    p = m.predict_proba(X_test)[:, 1]
    probs_test[name] = p
    cv_f1s[name] = getattr(m, "cv_f1", 0.5)   # RFv3Wrapper / XGBv3Wrapper have .cv_f1
    ind_pred = (p >= 0.5).astype(int)
    ind_acc  = accuracy_score(y_test, ind_pred)
    ind_f1   = f1_score(y_test, ind_pred, average="macro")
    print(f"  {name:12s}  Acc={ind_acc:.4f}  F1={ind_f1:.4f}  CV_F1={cv_f1s[name]:.4f}  "
          f"prob_mean={p.mean():.3f}  prob_std={p.std():.3f}")

# ── Select and weight models ───────────────────────────────────────────────────
selected = {n: cv_f1s[n] for n in models if cv_f1s[n] >= MIN_CV_F1}
if not selected:
    print(f"No model meets MIN_CV_F1={MIN_CV_F1}. Using all models.")
    selected = cv_f1s.copy()

excluded = [n for n in models if n not in selected]
if excluded:
    print(f"Excluded (CV F1 < {MIN_CV_F1}): {excluded}")

# Softmax-normalised weights
cv_arr = np.array(list(selected.values()))
weights = np.exp(cv_arr) / np.exp(cv_arr).sum()
print("\nEnsemble composition:")
for nm, w, cv in zip(selected, weights, cv_arr):
    print(f"  {nm:12s}  CV_F1={cv:.4f}  weight={w:.3f}")

# ── Weighted soft vote ─────────────────────────────────────────────────────────
prob_matrix = np.vstack([probs_test[nm] for nm in selected])  # (n_models, n_samples)
avg_prob = (prob_matrix * weights[:, None]).sum(axis=0)
y_pred   = (avg_prob >= 0.5).astype(int)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nEnsemble v3 (weighted) — Accuracy:{acc:.4f}  Macro F1:{macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))
print(f"Ensemble prob — mean:{avg_prob.mean():.3f}  std:{avg_prob.std():.3f}  "
      f"min:{avg_prob.min():.3f}  max:{avg_prob.max():.3f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Ensemble v3 (Weighted Soft Vote) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "ensemble_v3_confusion_matrix.png", dpi=150)
plt.close()


class WeightedSoftVotingEnsemble:
    """Weighted soft voting ensemble; predict_proba accepts full X_combined (306 cols)."""
    def __init__(self, named_models, weights):
        self.named_models = named_models  # [(name, model), ...]
        self.weights      = np.array(weights)

    def predict_proba(self, X):
        probs = np.vstack([m.predict_proba(X)[:, 1] for _, m in self.named_models])
        avg   = (probs * self.weights[:, None]).sum(axis=0)
        return np.column_stack([1 - avg, avg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


ensemble = WeightedSoftVotingEnsemble(
    [(nm, models[nm]) for nm in selected],
    weights.tolist(),
)
with open(MODEL_DIR / "ensemble_v3_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)
print(f"ensemble_v3_model.pkl saved -> {MODEL_DIR / 'ensemble_v3_model.pkl'}")
