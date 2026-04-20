"""
LR v4-BERT: L2-regularised on day-level BERT (768) + tech (13) = 781-dim features.
136 days total (108 train, 28 test). BERT embeddings capture richer semantics than TF-IDF.
Output: models/LR_bert_day_model.pkl
"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = 0.8

print("Loading day-level BERT features...")
meta_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_bert_day_meta.csv",  encoding="utf-8-sig")
text_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_bert_day_text.csv",  encoding="utf-8-sig")
tech_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_bert_day_tech.csv",  encoding="utf-8-sig")

y     = meta_df["label"].astype(int).values
X_all = np.hstack([text_df.values, tech_df.values])   # (136, 781)

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train label dist: Down={( y_train==0).sum()}  Up={(y_train==1).sum()}")
print(f"Test  label dist: Down={( y_test==0).sum()}   Up={(y_test==1).sum()}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight="balanced", solver="lbfgs")),
])
param_grid = {
    "lr__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
}
grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=True)
grid.fit(X_train, y_train)
cv_f1 = grid.best_score_
print(f"Best params: {grid.best_params_}  CV F1={cv_f1:.4f}")

model = grid.best_estimator_

# ── Find optimal threshold on validation fold ─────────────────────────────────
val_n   = int(len(X_train) * 0.8)
X_tr2, X_val = X_train[:val_n], X_train[val_n:]
y_tr2, y_val = y_train[:val_n], y_train[val_n:]
model.fit(X_tr2, y_tr2)
val_prob = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.75, 0.01)
f1s        = [f1_score(y_val, (val_prob >= t).astype(int), average="macro", zero_division=0)
              for t in thresholds]
best_t     = float(thresholds[np.argmax(f1s)])
print(f"Optimal threshold on val: T={best_t:.2f}  val F1={max(f1s):.4f}")

# Refit on full train
model.fit(X_train, y_train)
prob       = model.predict_proba(X_test)
prob_up    = prob[:, 1]
confidence = np.maximum(prob_up, 1 - prob_up)

y_pred_half = (prob_up >= 0.5).astype(int)
y_pred_opt  = (prob_up >= best_t).astype(int)

for thr_name, y_pred in [("T=0.50", y_pred_half), (f"T={best_t:.2f}", y_pred_opt)]:
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== All {len(y_test)} days, {thr_name} ===")
    print(f"Accuracy : {acc:.4f}  ({int(acc*len(y_test))}/{len(y_test)})")
    print(f"Macro F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"], zero_division=0))

# ── Confidence-stratified accuracy ────────────────────────────────────────────
print(f"=== Confidence-stratified accuracy (T={best_t:.2f}) ===")
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = confidence >= thr
    n_sel = mask.sum()
    if n_sel == 0:
        print(f"  thr>={thr:.2f}: 0 days selected"); continue
    pred_sel = y_pred_opt[mask]
    acc_sel  = accuracy_score(y_test[mask], pred_sel)
    f1_sel   = f1_score(y_test[mask], pred_sel, average="macro", zero_division=0)
    print(f"  thr>={thr:.2f}: {n_sel:2d} days  Acc={acc_sel:.4f} ({int(acc_sel*n_sel)}/{n_sel})  F1={f1_sel:.4f}")

print(f"\nProb stats: mean={prob_up.mean():.3f}  std={prob_up.std():.3f}  "
      f"min={prob_up.min():.3f}  max={prob_up.max():.3f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_opt = confusion_matrix(y_test, y_pred_opt)
disp   = ConfusionMatrixDisplay(cm_opt, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"LR-BERT Day-level (T={best_t:.2f}) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "lr_bert_day_confusion_matrix.png", dpi=150)
plt.close()


class LRBertDayModel:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model             = model
        self.optimal_threshold = optimal_threshold
        self.cv_f1             = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)


wrapper = LRBertDayModel(model, best_t, cv_f1)
with open(MODEL_DIR / "LR_bert_day_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"\nLR_bert_day_model.pkl saved  (T={best_t:.2f})")
