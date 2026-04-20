"""
XGBoost v3: expanded grid (adds min_child_weight, gamma), tree_method='hist' for speed.
Same calibration approach as v2. Output: XGBoost_v3_model.pkl
"""
from collections import Counter
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8


class XGBv3Wrapper:
    """Stores CV F1 alongside model for weighted ensemble."""
    def __init__(self, model, cv_f1):
        self.model  = model
        self.cv_f1  = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


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

cnt = Counter(y_train.tolist())
scale_pos_weight = round(cnt[0] / cnt[1], 4)
print(f"scale_pos_weight: {scale_pos_weight}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":     [100],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "gamma":            [0, 0.1],
    "scale_pos_weight": [1, scale_pos_weight],
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss",
                  use_label_encoder=False, n_jobs=1,
                  verbosity=0, tree_method="hist"),
    param_grid, cv=tscv, scoring="f1_macro",
    n_jobs=-1, verbose=1, refit=False,
)
grid.fit(X_train, y_train)
best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
cv_f1  = grid.best_score_
print(f"Best params: {best_p}  CV F1={cv_f1:.4f}")

# ── Refit with 300 trees + calibrate ──────────────────────────────────────────
xgb = XGBClassifier(**best_p, n_estimators=300, random_state=42,
                    eval_metric="logloss", use_label_encoder=False,
                    n_jobs=-1, verbosity=0, tree_method="hist")
cal_xgb = CalibratedClassifierCV(xgb, method="sigmoid",
                                  cv=TimeSeriesSplit(n_splits=5))
cal_xgb.fit(X_train, y_train)

y_pred   = cal_xgb.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

prob_test = cal_xgb.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost v3 (hist, gamma, min_child_weight) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "xgboost_v3_confusion_matrix.png", dpi=150)
plt.close()

# ── Save wrapped model ────────────────────────────────────────────────────────
wrapper = XGBv3Wrapper(cal_xgb, cv_f1)
with open(MODEL_DIR / "XGBoost_v3_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"XGBoost_v3_model.pkl saved -> {MODEL_DIR / 'XGBoost_v3_model.pkl'}")
