"""
XGBoost trained on BERT (bert-base-chinese) 768-dim CLS embeddings.
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
from xgboost import XGBClassifier

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

from collections import Counter
cnt = Counter(y_train.tolist())
scale_pos_weight = round(cnt[0] / cnt[1], 4)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"scale_pos_weight: {scale_pos_weight}")

# ── Stage 1: Grid search (n_estimators=100 fixed) ─────────────────────────────
param_grid = {
    "n_estimators":     [100],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, scale_pos_weight],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss",
                  use_label_encoder=False, n_jobs=1, verbosity=0),
    param_grid, cv=cv, scoring="f1_macro",
    n_jobs=-1, verbose=1, refit=False,
)
grid.fit(X_train, y_train)
best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
print(f"Stage1 best: {best_p}  CV F1={grid.best_score_:.4f}")

# ── Stage 2: Refit with 300 trees ─────────────────────────────────────────────
model = XGBClassifier(**best_p, n_estimators=300, random_state=42,
                      eval_metric="logloss", use_label_encoder=False,
                      n_jobs=-1, verbosity=0)
model.fit(X_train, y_train)

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
ax.set_title("XGBoost (BERT) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "xgboost_bert_confusion_matrix.png", dpi=150)
plt.close()

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / "XGBoost_bert_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"XGBoost_bert_model.pkl saved → {MODEL_DIR / 'XGBoost_bert_model.pkl'}")
