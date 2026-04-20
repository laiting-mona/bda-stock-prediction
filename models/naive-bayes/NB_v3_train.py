"""
NB v3: ComplementNB on TF-IDF features only (no tech features).
ComplementNB is designed for text — non-negative TF-IDF input, wider probability spread.
No CalibratedClassifierCV (it was compressing probabilities into a narrow band).
Wraps result in NBv3Wrapper so predict_proba(X_combined) auto-slices to TF-IDF columns.
Output: NB_v3_model.pkl
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

N_TFIDF = 300
SPLIT   = 0.8


class NBv3Wrapper:
    """Slices first N_TFIDF columns so the backtest can pass the full combined X."""
    def __init__(self, model, n_tfidf, cv_f1):
        self.model   = model
        self.n_tfidf = n_tfidf
        self.cv_f1   = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, :self.n_tfidf])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading v2 features…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y       = feat_df["label"].astype(int).values
X_tfidf = vec_df.values          # shape (36834, 300) — all non-negative

n = int(len(X_tfidf) * SPLIT)
X_train, X_test = X_tfidf[:n], X_tfidf[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
grid = GridSearchCV(ComplementNB(), param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=True)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_["alpha"]
cv_f1      = grid.best_score_
print(f"Best alpha: {best_alpha}  CV F1={cv_f1:.4f}")

best_model = grid.best_estimator_

y_pred   = best_model.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

prob_test = best_model.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}  "
      f"min:{prob_test.min():.3f}  max:{prob_test.max():.3f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("NB v3 (ComplementNB, TF-IDF only) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "nb_v3_confusion_matrix.png", dpi=150)
plt.close()

# ── Save wrapped model ────────────────────────────────────────────────────────
wrapper = NBv3Wrapper(best_model, N_TFIDF, cv_f1)
with open(MODEL_DIR / "NB_v3_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"NB_v3_model.pkl saved -> {MODEL_DIR / 'NB_v3_model.pkl'}")
