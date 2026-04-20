"""
NB v2: GaussianNB on combined TF-IDF (v2, train-only chi2) + technical features.
TimeSeriesSplit CV for var_smoothing search. CalibratedClassifierCV (sigmoid).
Separate from original NB_model.pkl — outputs NB_v2_model.pkl.
"""
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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

TECH_COLS = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
SPLIT     = 0.8

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
# GaussianNB benefits from standard-scaled input for numerical stability
tscv = TimeSeriesSplit(n_splits=5)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gnb",    GaussianNB()),
])
param_grid = {
    "gnb__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
}
grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="f1_macro",
                    n_jobs=-1, verbose=1, refit=False)
grid.fit(X_train, y_train)
best_vs = grid.best_params_["gnb__var_smoothing"]
print(f"Best var_smoothing: {best_vs}  CV F1={grid.best_score_:.4f}")

# ── Refit best pipeline, then calibrate ───────────────────────────────────────
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gnb",    GaussianNB(var_smoothing=best_vs)),
])
cal_nb = CalibratedClassifierCV(best_pipe, method="sigmoid",
                                cv=TimeSeriesSplit(n_splits=5))
cal_nb.fit(X_train, y_train)

y_pred   = cal_nb.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

prob_test = cal_nb.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("NB v2 (TF-IDF+Tech, Calibrated) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "nb_v2_confusion_matrix.png", dpi=150)
plt.close()

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / "NB_v2_model.pkl", "wb") as f:
    pickle.dump(cal_nb, f)
print(f"NB_v2_model.pkl saved -> {MODEL_DIR / 'NB_v2_model.pkl'}")
