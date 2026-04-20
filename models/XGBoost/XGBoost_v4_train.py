"""
XGBoost v4: day-level combined features.
Ablation study: text_only vs tech_only vs combined.
Output: XGBoost_v4_model.pkl  (combined, best for backtest)
        XGBoost_v4_text_model.pkl
        XGBoost_v4_tech_model.pkl
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
from xgboost import XGBClassifier

ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data"
MODEL_DIR = Path(__file__).parent
CM_DIR    = ROOT / "results" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = 0.8

print("Loading day-level v4 features…")
meta_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv", encoding="utf-8-sig")
text_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_text_v4.csv",     encoding="utf-8-sig")
tech_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_tech_v4.csv",     encoding="utf-8-sig")

y       = meta_df["label"].astype(int).values
X_text  = text_df.values                        # (267, 300)
X_tech  = tech_df.values                        # (267, 6)
X_comb  = np.hstack([X_text, X_tech])           # (267, 306)

n = int(len(X_comb) * SPLIT)
y_train, y_test = y[:n], y[n:]
print(f"Train: {n}  Test: {len(y_test)}")

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":     [100],
    "max_depth":        [2, 3, 5],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

def train_xgb(X_tr, X_te, y_tr, y_te, label):
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss",
                      use_label_encoder=False, n_jobs=1, verbosity=0,
                      tree_method="hist"),
        param_grid, cv=tscv, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=False,
    )
    grid.fit(X_tr, y_tr)
    best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
    cv_f1  = grid.best_score_
    print(f"[{label}] Best: {best_p}  CV F1={cv_f1:.4f}")

    xgb = XGBClassifier(**best_p, n_estimators=200, random_state=42,
                        eval_metric="logloss", use_label_encoder=False,
                        n_jobs=-1, verbosity=0, tree_method="hist")
    xgb.fit(X_tr, y_tr)

    y_pred   = xgb.predict(X_te)
    acc      = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    cm       = confusion_matrix(y_te, y_pred)
    prob     = xgb.predict_proba(X_te)[:, 1]
    print(f"[{label}] Acc={acc:.4f}  F1={macro_f1:.4f}  "
          f"prob mean={prob.mean():.3f}  std={prob.std():.3f}")
    print(classification_report(y_te, y_pred, target_names=["Down(0)", "Up(1)"],
                                zero_division=0))

    disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"XGBoost v4 ({label}) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_DIR / f"xgboost_v4_{label}_confusion_matrix.png", dpi=150)
    plt.close()

    return xgb, cv_f1


# ── Ablation study ────────────────────────────────────────────────────────────
print("\n=== text_only (300 TF-IDF features) ===")
xgb_text, cv_f1_text = train_xgb(X_text[:n], X_text[n:], y_train, y_test, "text_only")

print("\n=== tech_only (6 technical features) ===")
xgb_tech, cv_f1_tech = train_xgb(X_tech[:n], X_tech[n:], y_train, y_test, "tech_only")

print("\n=== combined (306 features) ===")
xgb_comb, cv_f1_comb = train_xgb(X_comb[:n], X_comb[n:], y_train, y_test, "combined")

# ── Ablation summary ──────────────────────────────────────────────────────────
print("\n=== Ablation Summary ===")
print(f"  text_only  CV F1={cv_f1_text:.4f}  Test F1={f1_score(y_test, xgb_text.predict(X_text[n:]), average='macro'):.4f}")
print(f"  tech_only  CV F1={cv_f1_tech:.4f}  Test F1={f1_score(y_test, xgb_tech.predict(X_tech[n:]), average='macro'):.4f}")
print(f"  combined   CV F1={cv_f1_comb:.4f}  Test F1={f1_score(y_test, xgb_comb.predict(X_comb[n:]), average='macro'):.4f}")

# ── Save wrappers ─────────────────────────────────────────────────────────────
class XGBv4Model:
    """n_text=300: slices first 300 cols for text_only model; -1 = use all cols."""
    def __init__(self, model, n_text, cv_f1):
        self.model  = model
        self.n_text = n_text   # -1 → use all 306; 300 → text only; 0 → tech only (cols 300:)
        self.cv_f1  = cv_f1

    def _slice(self, X):
        if self.n_text == -1:
            return X
        if self.n_text == 0:
            return X[:, 300:]
        return X[:, :self.n_text]

    def predict_proba(self, X):
        return self.model.predict_proba(self._slice(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


for fname, mdl, nt, cv in [
    ("XGBoost_v4_model.pkl",      xgb_comb, -1,  cv_f1_comb),
    ("XGBoost_v4_text_model.pkl", xgb_text, 300, cv_f1_text),
    ("XGBoost_v4_tech_model.pkl", xgb_tech, 0,   cv_f1_tech),
]:
    with open(MODEL_DIR / fname, "wb") as f:
        pickle.dump(XGBv4Model(mdl, nt, cv), f)
    print(f"Saved: {MODEL_DIR / fname}")
