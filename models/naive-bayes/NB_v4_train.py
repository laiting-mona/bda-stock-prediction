"""
NB v4: ComplementNB on DAY-LEVEL mean TF-IDF (text_only, 300 dims).
Day-level aggregation eliminates extreme consensus issue.
Output: NB_v4_model.pkl
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

SPLIT = 0.8

print("Loading day-level v4 features…")
meta_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv", encoding="utf-8-sig")
text_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_text_v4.csv",     encoding="utf-8-sig")

y       = meta_df["label"].astype(int).values
X_text  = text_df.values    # (267, 300) — all non-negative (mean TF-IDF)

n = int(len(X_text) * SPLIT)
X_train, X_test = X_text[:n], X_text[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
grid = GridSearchCV(ComplementNB(), param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=True)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_["alpha"]
cv_f1      = grid.best_score_
print(f"Best alpha: {best_alpha}  CV F1={cv_f1:.4f}")

model    = grid.best_estimator_
y_pred   = model.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

prob_test = model.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}  "
      f"min:{prob_test.min():.3f}  max:{prob_test.max():.3f}")

disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("NB v4 (ComplementNB, day-level TF-IDF) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "nb_v4_confusion_matrix.png", dpi=150)
plt.close()


class NBv4Model:
    """Day-level wrapper: predict_proba accepts full combined X (306 cols), slices to text."""
    def __init__(self, model, n_text, cv_f1):
        self.model  = model
        self.n_text = n_text
        self.cv_f1  = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, :self.n_text])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


wrapper = NBv4Model(model, n_text=300, cv_f1=cv_f1)
with open(MODEL_DIR / "NB_v4_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"NB_v4_model.pkl saved -> {MODEL_DIR / 'NB_v4_model.pkl'}")
