"""
NB v5: ComplementNB on TF-IDF (300) + non-negative sentiment scores (mcd_up, mcd_dn)
+ n_articles = 303-dim.  ComplementNB requires non-negative inputs; mcd_net is excluded.
Adds decision threshold optimisation on held-out validation fold.
Output: NB_v5_model.pkl
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import ComplementNB

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

SPLIT  = 0.8
# Non-negative extra features safe for ComplementNB
EXTRA_COLS = ["mcd_up", "mcd_dn", "n_articles"]


class NBv5Wrapper:
    """Stores first n_tfidf TF-IDF cols + appended non-negative cols."""
    def __init__(self, model, n_tfidf, n_extra, optimal_threshold, cv_f1):
        self.model             = model
        self.n_tfidf           = n_tfidf
        self.n_extra           = n_extra       # columns appended right after TF-IDF
        self.optimal_threshold = optimal_threshold
        self.cv_f1             = cv_f1

    def predict_proba(self, X):
        # X layout: [300 TF-IDF | 22 tech_v3] — extra cols are last 3 of tech_v3
        # We need cols 0:300 (TF-IDF) + the 3 extra cols embedded in X
        # In v5 backtest X is pre-assembled as [tfidf | extra_only]
        return self.model.predict_proba(X[:, :self.n_tfidf + self.n_extra])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading v3 features...")
feat = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v3.csv", encoding="utf-8-sig")
vec  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y       = feat["label"].astype(int).values
X_extra = feat[EXTRA_COLS].fillna(0).clip(lower=0).values   # ensure non-negative
X_tfidf = vec.values
X_all   = np.hstack([X_tfidf, X_extra])   # (N, 303) — all non-negative

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}
grid = GridSearchCV(ComplementNB(), param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=True)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_["alpha"]
cv_f1      = grid.best_score_
print(f"Best alpha: {best_alpha}  CV F1={cv_f1:.4f}")

best_model = grid.best_estimator_

# ── Find optimal decision threshold ────────────────────────────────────────────
val_n   = int(len(X_train) * 0.8)
X_tr2, X_val = X_train[:val_n], X_train[val_n:]
y_tr2, y_val = y_train[:val_n], y_train[val_n:]
nb_tmp = ComplementNB(alpha=best_alpha)
nb_tmp.fit(X_tr2, y_tr2)
val_prob   = nb_tmp.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.75, 0.01)
f1s        = [f1_score(y_val, (val_prob >= t).astype(int), average="macro", zero_division=0)
              for t in thresholds]
best_t = float(thresholds[np.argmax(f1s)])
print(f"Optimal threshold: T={best_t:.2f}  val F1={max(f1s):.4f}")

# ── Evaluate ───────────────────────────────────────────────────────────────────
prob_test   = best_model.predict_proba(X_test)[:, 1]
y_pred_opt  = (prob_test >= best_t).astype(int)
y_pred_half = (prob_test >= 0.5).astype(int)

for name, yp in [("T=0.50", y_pred_half), (f"T={best_t:.2f}", y_pred_opt)]:
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp, average="macro")
    print(f"\n=== All {len(y_test)} articles, {name} ===")
    print(f"Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    print(classification_report(y_test, yp, target_names=["Down(0)", "Up(1)"], zero_division=0))

print(f"Prob stats: mean={prob_test.mean():.3f}  std={prob_test.std():.3f}  "
      f"min={prob_test.min():.3f}  max={prob_test.max():.3f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_opt)
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"NB v5 (ComplementNB+sentiment, T={best_t:.2f})")
plt.tight_layout()
plt.savefig(CM_DIR / "nb_v5_confusion_matrix.png", dpi=150)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
wrapper = NBv5Wrapper(best_model, n_tfidf=300, n_extra=len(EXTRA_COLS),
                      optimal_threshold=best_t, cv_f1=cv_f1)
with open(MODEL_DIR / "NB_v5_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"\nNB_v5_model.pkl saved  (T={best_t:.2f}) -> {MODEL_DIR / 'NB_v5_model.pkl'}")
