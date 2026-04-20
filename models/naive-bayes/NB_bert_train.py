"""
Gaussian Naive Bayes trained on BERT (bert-base-chinese) 768-dim embeddings.
GaussianNB is used instead of ComplementNB because BERT features are
continuous and can be negative (ComplementNB requires non-negative input).
Prerequisite: run scripts/bert_features/extract_bert.py first.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

# ── Load BERT features ────────────────────────────────────────────────────────
bert_path = DATA_DIR / "processed" / "tsmc_bert_features.npy"
if not bert_path.exists():
    raise FileNotFoundError(
        f"BERT features not found: {bert_path}\n"
        "Run: python scripts/bert_features/extract_bert.py"
    )
X = np.load(bert_path)

feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features.csv", encoding="utf-8-sig")
y = feat_df["label"].astype(int).values

assert len(X) == len(y), f"Feature/label mismatch: {len(X)} vs {len(y)}"

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Grid search on var_smoothing ──────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"var_smoothing": np.logspace(-12, 0, 13)}
grid = GridSearchCV(
    GaussianNB(), param_grid, cv=cv,
    scoring="f1_macro", n_jobs=-1, verbose=1, refit=True,
)
grid.fit(X_train, y_train)
print(f"Best var_smoothing: {grid.best_params_['var_smoothing']:.2e}  "
      f"CV F1={grid.best_score_:.4f}")

model = grid.best_estimator_
y_pred   = model.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("NB-Gaussian (BERT) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "nb_bert_confusion_matrix.png", dpi=150)
plt.close()

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / "NB_bert_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"NB_bert_model.pkl saved → {MODEL_DIR / 'NB_bert_model.pkl'}")
