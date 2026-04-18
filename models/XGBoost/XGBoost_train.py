from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"
model_dir = Path(__file__).parent
cm_dir = project_root / "results" / "confusion_matrices"
cm_dir.mkdir(parents=True, exist_ok=True)

# ── 讀取資料 ──────────────────────────────────────────────────────────────────
X = pd.read_csv(
    data_dir / "processed" / "tsmc_vector_space.csv", encoding="utf-8-sig"
).values

feat_df = pd.read_csv(
    data_dir / "processed" / "tsmc_features.csv", encoding="utf-8-sig"
)
y = feat_df["label"].astype(int).values

# ── 時序切分 80 / 20 ──────────────────────────────────────────────────────────
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
unique, counts = np.unique(y_train, return_counts=True)
print(f"Train label dist: { {int(k): int(v) for k, v in zip(unique, counts)} }")

# 類別不平衡權重
scale_pos_weight = int(counts[0]) / int(counts[1])  # neg / pos

# ── Grid Search ───────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, scale_pos_weight],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=1,
        verbosity=0,
    ),
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2,
    refit=True,
)
grid.fit(X_train, y_train)

print(f"\nBest params : {grid.best_params_}")
print(f"Best CV Macro F1 : {grid.best_score_:.4f}")

# ── 測試集評估 ────────────────────────────────────────────────────────────────
y_pred = grid.best_estimator_.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm       = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy  : {acc:.4f}")
print(f"Test Macro F1  : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Down(0)', 'Up(1)'])}")

# ── 儲存 Confusion Matrix PNG ─────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=["Down(0)", "Up(1)"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost Confusion Matrix")
plt.tight_layout()
cm_path = cm_dir / "xgboost_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion Matrix PNG saved → {cm_path}")

# ── 儲存模型 ──────────────────────────────────────────────────────────────────
with open(model_dir / "XGBoost_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
print(f"XGBoost_model.pkl saved → {model_dir / 'XGBoost_model.pkl'}")

# ── 建立並儲存 vectorizer ─────────────────────────────────────────────────────
text_df = pd.read_csv(
    data_dir / "processed" / "tsmc_clean_filtered.csv", encoding="utf-8-sig"
)
text_df["title"]   = text_df["title"].fillna("").astype(str)
text_df["content"] = text_df["content"].fillna("").astype(str)
text_df["text"]    = (text_df["title"] + " " + text_df["content"]).str.strip()

X_text_train  = text_df["text"].values[:split_idx]
y_text_train  = text_df["label"].astype(int).values[:split_idx]

vectorizer_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=4000,
        min_df=2,
        sublinear_tf=True,
    )),
    ("chi2", SelectKBest(score_func=chi2, k=300)),
])
vectorizer_pipeline.fit(X_text_train, y_text_train)

with open(model_dir / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer_pipeline, f)
print(f"vectorizer.pkl saved → {model_dir / 'vectorizer.pkl'}")
