"""
XGBoost v5: 322-dim features (300 TF-IDF + 22 tech from v3).
Improvements over v3:
  - Richer feature set (hour, session, sentiment, derived tech)
  - scale_pos_weight grid search to address class imbalance
  - Decision threshold optimised on held-out validation fold
  - CalibratedClassifierCV for better probability estimates
Output: XGBoost_v5_model.pkl
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

SPLIT = 0.8
TECH_V2 = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
TECH_NEW = [
    "hour_sin", "hour_cos",
    "session_pre", "session_mkt", "session_aft",
    "dow_sin", "dow_cos",
    "article_len_norm",
    "mcd_up", "mcd_dn", "mcd_net",
    "ret1_std", "ret5_std", "rsi_norm", "vol_ratio", "sign_ret1",
]
TECH_COLS = TECH_V2 + TECH_NEW


class XGBv5Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model             = model
        self.optimal_threshold = optimal_threshold
        self.cv_f1             = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading v3 features...")
feat = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v3.csv", encoding="utf-8-sig")
vec  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")

y       = feat["label"].astype(int).values
X_tech  = feat[TECH_COLS].fillna(0).values
X_tfidf = vec.values
X_all   = np.hstack([X_tfidf, X_tech])

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

cnt = Counter(y_train.tolist())
scale_pos_weight = round(cnt[0] / cnt[1], 4)
print(f"scale_pos_weight (natural): {scale_pos_weight}")

# ── TimeSeriesSplit grid search ────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":     [100],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8],
    "min_child_weight": [1, 3],
    "gamma":            [0, 0.1],
    "scale_pos_weight": [1, scale_pos_weight, 1.5],
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

# ── Find optimal decision threshold on held-out validation fold ────────────────
val_n   = int(len(X_train) * 0.8)
X_tr2, X_val = X_train[:val_n], X_train[val_n:]
y_tr2, y_val = y_train[:val_n], y_train[val_n:]
xgb_tmp = XGBClassifier(**best_p, n_estimators=300, random_state=42,
                        eval_metric="logloss", use_label_encoder=False,
                        n_jobs=-1, verbosity=0, tree_method="hist")
xgb_tmp.fit(X_tr2, y_tr2)
val_prob = xgb_tmp.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.75, 0.01)
f1s        = [f1_score(y_val, (val_prob >= t).astype(int), average="macro", zero_division=0)
              for t in thresholds]
best_t = float(thresholds[np.argmax(f1s)])
print(f"Optimal threshold: T={best_t:.2f}  val F1={max(f1s):.4f}")

# ── Refit with 300 trees + calibrate ──────────────────────────────────────────
xgb_final = XGBClassifier(**best_p, n_estimators=300, random_state=42,
                          eval_metric="logloss", use_label_encoder=False,
                          n_jobs=-1, verbosity=0, tree_method="hist")
cal_xgb = CalibratedClassifierCV(xgb_final, method="sigmoid",
                                  cv=TimeSeriesSplit(n_splits=5))
cal_xgb.fit(X_train, y_train)

prob_test = cal_xgb.predict_proba(X_test)[:, 1]
y_pred_opt  = (prob_test >= best_t).astype(int)
y_pred_half = (prob_test >= 0.5).astype(int)

for name, yp in [("T=0.50", y_pred_half), (f"T={best_t:.2f}", y_pred_opt)]:
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp, average="macro")
    print(f"\n=== All {len(y_test)} articles, {name} ===")
    print(f"Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    print(classification_report(y_test, yp, target_names=["Down(0)", "Up(1)"], zero_division=0))

print(f"Prob stats: mean={prob_test.mean():.3f}  std={prob_test.std():.3f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_opt)
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"XGBoost v5 (T={best_t:.2f}) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "xgboost_v5_confusion_matrix.png", dpi=150)
plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────
wrapper = XGBv5Wrapper(cal_xgb, best_t, cv_f1)
with open(MODEL_DIR / "XGBoost_v5_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"\nXGBoost_v5_model.pkl saved -> {MODEL_DIR / 'XGBoost_v5_model.pkl'}")
