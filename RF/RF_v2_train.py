"""
RF v2: TF-IDF (train-only chi2) + technical features, TimeSeriesSplit CV,
CalibratedClassifierCV (Platt scaling) to fix probability miscalibration.
Separate from original RF_model.pkl — outputs RF_v2_model.pkl.
"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
SHARED    = ROOT / "models" / "_shared"
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading v2 features…")
feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

assert len(feat_df) == len(vec_df), "Row mismatch between feature and vector files"

y = feat_df["label"].astype(int).values
X_tech  = feat_df[TECH_COLS].fillna(0).values
X_tfidf = vec_df.values
X_all   = np.hstack([X_tfidf, X_tech])

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train class balance — Down:{(y_train==0).sum()}  Up:{(y_train==1).sum()}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [200],
    "max_depth":    [None, 20, 40],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.3],
    "class_weight": [None, "balanced"],
}
base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(base_rf, param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=False)
grid.fit(X_train, y_train)
best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
print(f"Best params: {best_p}  CV F1={grid.best_score_:.4f}")

# ── Refit with 300 trees, then calibrate ──────────────────────────────────────
rf = RandomForestClassifier(**best_p, n_estimators=300, random_state=42, n_jobs=-1)
# Calibrate with Platt scaling using TimeSeriesSplit to avoid leakage
cal_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=TimeSeriesSplit(n_splits=5))
cal_rf.fit(X_train, y_train)

y_pred   = cal_rf.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

# ── Calibration check ─────────────────────────────────────────────────────────
prob_test = cal_rf.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}  "
      f"min:{prob_test.min():.3f}  max:{prob_test.max():.3f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("RF v2 (TF-IDF+Tech, Calibrated) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "rf_v2_confusion_matrix.png", dpi=150)
plt.close()

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / "RF_v2_model.pkl", "wb") as f:
    pickle.dump(cal_rf, f)
print(f"RF_v2_model.pkl saved -> {MODEL_DIR / 'RF_v2_model.pkl'}")
