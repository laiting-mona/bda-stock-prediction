"""
RF v4: day-level combined features (mean TF-IDF + tech), balanced_subsample,
optimal decision threshold found on validation fold.
Output: RF_v4_model.pkl
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

SPLIT = 0.8

print("Loading day-level v4 features…")
meta_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv", encoding="utf-8-sig")
text_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_text_v4.csv",     encoding="utf-8-sig")
tech_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_tech_v4.csv",     encoding="utf-8-sig")

y    = meta_df["label"].astype(int).values
X_all = np.hstack([text_df.values, tech_df.values])   # (267, 306)

n = int(len(X_all) * SPLIT)
X_train, X_test = X_all[:n], X_all[n:]
y_train, y_test = y[:n], y[n:]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":     [100],
    "max_depth":        [None, 15, 30],
    "min_samples_leaf": [1, 3],
    "max_features":     ["sqrt", 0.3],
    "class_weight":     ["balanced_subsample"],
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

# Optimal threshold on held-out validation (last 20% of train, chronological)
val_n   = int(len(X_train) * 0.8)
X_tr2, X_val = X_train[:val_n], X_train[val_n:]
y_tr2, y_val = y_train[:val_n], y_train[val_n:]

rf_tmp = RandomForestClassifier(**best_p, n_estimators=200, random_state=42, n_jobs=-1)
rf_tmp.fit(X_tr2, y_tr2)
val_probs  = rf_tmp.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.75, 0.01)
f1_scores  = [f1_score(y_val, (val_probs >= t).astype(int), average="macro", zero_division=0)
               for t in thresholds]
best_t     = float(thresholds[np.argmax(f1_scores)])
print(f"Optimal threshold on val: T={best_t:.2f}  val F1={max(f1_scores):.4f}")

# Final model on full train
rf_final = RandomForestClassifier(**best_p, n_estimators=200, random_state=42, n_jobs=-1)
rf_final.fit(X_train, y_train)

y_pred   = (rf_final.predict_proba(X_test)[:, 1] >= best_t).astype(int)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)
print(f"\nThreshold={best_t:.2f}  Accuracy:{acc:.4f}  Macro F1:{macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))

prob_test = rf_final.predict_proba(X_test)[:, 1]
print(f"Prob stats — mean:{prob_test.mean():.3f}  std:{prob_test.std():.3f}")

disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"RF v4 (day-level, T={best_t:.2f}) Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_DIR / "rf_v4_confusion_matrix.png", dpi=150)
plt.close()


class RFv4Model:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model             = model
        self.optimal_threshold = optimal_threshold
        self.cv_f1             = cv_f1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)


wrapper = RFv4Model(rf_final, best_t, cv_f1)
with open(MODEL_DIR / "RF_v4_model.pkl", "wb") as f:
    pickle.dump(wrapper, f)
print(f"RF_v4_model.pkl saved  (T={best_t:.2f}) -> {MODEL_DIR / 'RF_v4_model.pkl'}")
