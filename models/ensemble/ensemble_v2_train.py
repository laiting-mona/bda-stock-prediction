"""
Ensemble v2: soft-voting average of RF_v2, NB_v2, XGBoost_v2.
No retraining needed — combines calibrated probabilities from the three v2 models.
Output: models/ensemble/ensemble_v2_model.pkl (wraps the three base models)
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

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8

BASE_MODELS = {
    "RF_v2":      ROOT / "models/RF/RF_v2_model.pkl",
    "NB_v2":      ROOT / "models/naive-bayes/NB_v2_model.pkl",
    "XGBoost_v2": ROOT / "models/XGBoost/XGBoost_v2_model.pkl",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading v2 features…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y = feat_df["label"].astype(int).values
X_tech  = feat_df[TECH_COLS].fillna(0).values
X_tfidf = vec_df.values
X_all   = np.hstack([X_tfidf, X_tech])

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Test: {X_test.shape}")

# ── Load base models and collect probabilities ─────────────────────────────────
models, probs_test = {}, {}
for name, path in BASE_MODELS.items():
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun {name}_train.py first.")
    m = pickle.load(open(path, "rb"))
    models[name] = m
    p = m.predict_proba(X_test)[:, 1]
    probs_test[name] = p
    ind_pred = (p >= 0.5).astype(int)
    ind_acc  = accuracy_score(y_test, ind_pred)
    ind_f1   = f1_score(y_test, ind_pred, average="macro")
    print(f"  {name:12s}  Acc={ind_acc:.4f}  F1={ind_f1:.4f}  "
          f"prob_mean={p.mean():.3f}  prob_std={p.std():.3f}")

# ── Soft-voting ensemble (equal weights) ──────────────────────────────────────
avg_prob = np.mean(list(probs_test.values()), axis=0)
y_pred   = (avg_prob >= 0.5).astype(int)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nEnsemble (soft vote) — Accuracy:{acc:.4f}  Macro F1:{macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Ensemble v2 (Soft Vote) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "ensemble_v2_confusion_matrix.png", dpi=150)
plt.close()

# ── Probability calibration check ─────────────────────────────────────────────
print(f"Ensemble prob — mean:{avg_prob.mean():.3f}  std:{avg_prob.std():.3f}  "
      f"min:{avg_prob.min():.3f}  max:{avg_prob.max():.3f}")


class SoftVotingEnsemble:
    """Lightweight wrapper so the ensemble behaves like a sklearn estimator."""
    def __init__(self, named_models):
        self.named_models = named_models  # [(name, model), ...]

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for _, m in self.named_models], axis=0)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


ensemble = SoftVotingEnsemble(list(models.items()))
with open(MODEL_DIR / "ensemble_v2_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)
print(f"ensemble_v2_model.pkl saved -> {MODEL_DIR / 'ensemble_v2_model.pkl'}")
