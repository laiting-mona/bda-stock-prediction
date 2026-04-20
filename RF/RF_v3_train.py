"""
RF v3: class_weight='balanced_subsample' (per-tree resampling) + optimal decision threshold.
After grid search, finds threshold T on held-out validation fold that maximises Macro F1.
Backtest uses T instead of 0.5 for signal direction, fixing the Up-bias.
Output: RF_v3_model.pkl
"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8


class RFv3Wrapper:
    """Stores optimal decision threshold alongside the model."""
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model              = model
        self.optimal_threshold  = optimal_threshold
        self.cv_f1              = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)


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
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":     [200],
    "max_depth":        [None, 20, 40],
    "min_samples_leaf": [1, 5],
    "max_features":     ["sqrt", 0.3],
    "class_weight":     ["balanced_subsample"],   # per-tree resampling
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=tscv, scoring="f1_macro",
    n_jobs=1, verbose=1, refit=False,
)
grid.fit(X_train, y_train)
best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
cv_f1  = grid.best_score_
print(f"Best params: {best_p}  CV F1={cv_f1:.4f}")

# ── Find optimal threshold on held-out validation fold ────────────────────────
# Train on first 80% of train, validate on last 20% of train (chronological)
val_n   = int(len(X_train) * 0.8)
X_tr2, X_val = X_train[:val_n], X_train[val_n:]
y_tr2, y_val = y_train[:val_n], y_train[val_n:]

rf_tmp = RandomForestClassifier(**best_p, n_estimators=300, random_state=42, n_jobs=-1)
rf_tmp.fit(X_tr2, y_tr2)
val_probs = rf_tmp.predict_proba(X_val)[:, 1]

# Sweep thresholds to find max Macro F1
thresholds  = np.arange(0.30, 0.75, 0.01)
f1_scores   = [f1_score(y_val, (val_probs >= t).astype(int), average="macro", zero_division=0)
               for t in thresholds]
best_t      = float(thresholds[np.argmax(f1_scores)])
best_val_f1 = float(np.max(f1_scores))
print(f"Optimal threshold on val: T={best_t:.2f}  val Macro F1={best_val_f1:.4f}")

# ── Refit on full train with best params + 300 trees ──────────────────────────
rf_final = RandomForestClassifier(**best_p, n_estimators=300, random_state=42, n_jobs=-1)
rf_final.fit(X_train, y_train)

y_pred_opt  = (rf_final.predict_proba(X_test)[:, 1] >= best_t).astype(int)
y_pred_half = rf_final.predict(X_test)  # default 0.5 for comparison

acc_opt  = accuracy_score(y_test, y_pred_opt)
f1_opt   = f1_score(y_test, y_pred_opt, average="macro")
acc_half = accuracy_score(y_test, y_pred_half)
f1_half  = f1_score(y_test, y_pred_half, average="macro")
cm_opt   = confusion_matrix(y_test, y_pred_opt)

print(f"\nThreshold=0.50  Acc={acc_half:.4f}  F1={f1_half:.4f}")
print(f"Threshold={best_t:.2f}  Acc={acc_opt:.4f}  F1={f1_opt:.4f}  ← optimal")
print(f"Confusion Matrix (threshold={best_t:.2f}):\n{cm_opt}")
print(classification_report(y_test, y_pred_opt, target_names=["Down(0)", "Up(1)"]))

prob_test = rf_final.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm_opt, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"RF v3 (balanced_subsample, T={best_t:.2f}) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "rf_v3_confusion_matrix.png", dpi=150)
plt.close()

# ── Save wrapped model ────────────────────────────────────────────────────────
wrapper = RFv3Wrapper(rf_final, best_t, cv_f1)
with open(MODEL_DIR / "RF_v3_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"RF_v3_model.pkl saved  (optimal_threshold={best_t:.2f}) -> {MODEL_DIR / 'RF_v3_model.pkl'}")
