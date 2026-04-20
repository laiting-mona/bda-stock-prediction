from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"
model_dir = Path(__file__).parent

# ── 讀取資料 ──────────────────────────────────────────────────────────────────
# X：feature_eng.py 產出的 300 維 TF-IDF + 卡方向量空間
X = pd.read_csv(
    data_dir / "processed" / "tsmc_vector_space.csv", encoding="utf-8-sig"
).values

# y：由 tsmc_features.csv 取 label（列數與 vector_space 對齊）
feat_df = pd.read_csv(
    data_dir / "processed" / "tsmc_features.csv", encoding="utf-8-sig"
)
y = feat_df["label"].astype(int).values

# ── 時序切分 80 / 20（保留發文時間順序，避免未來資料洩漏）──────────────────
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
unique, counts = np.unique(y_train, return_counts=True)
print(f"Train label dist: { {int(k): int(v) for k, v in zip(unique, counts)} }")

# ── Stage 1：用 100 棵樹快速掃最佳超參數組合 ──────────────────────────────────
param_grid = {
    "n_estimators": [100],           # stage1 固定 100，速度快
    "max_depth": [None, 20, 30],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced", None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),  # RF 單線程
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,      # CV fold 並行，避免雙層並行
    verbose=2,
    refit=False,    # stage1 不 refit，後面手動加樹
)
grid.fit(X_train, y_train)

best_params = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
print(f"\nStage1 best params (excluding n_estimators): {best_params}")
print(f"Stage1 best CV Macro F1: {grid.best_score_:.4f}")

# ── Stage 2：用最佳超參數 + 300 棵樹訓練最終模型 ──────────────────────────────
print("\nStage2: refit with n_estimators=300 ...")
best_model = RandomForestClassifier(
    **best_params,
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
best_model.fit(X_train, y_train)

print(f"\nFinal params: {best_model.get_params()}")

# ── 測試集評估 ────────────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy  : {acc:.4f}")
print(f"Test Macro F1  : {macro_f1:.4f}")
print(f"Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['跌(0)', '漲(1)'])}")

# ── 儲存 RF 模型 ──────────────────────────────────────────────────────────────
with open(model_dir / "RF_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"RF_model.pkl saved → {model_dir / 'RF_model.pkl'}")

# ── 建立並儲存 vectorizer（供推論新文章使用）─────────────────────────────────
# 依照 feature_eng.py 相同參數，只用 train 資料 fit，避免資料洩漏
text_df = pd.read_csv(
    data_dir / "processed" / "tsmc_clean_filtered.csv", encoding="utf-8-sig"
)
text_df["title"] = text_df["title"].fillna("").astype(str)
text_df["content"] = text_df["content"].fillna("").astype(str)
text_df["text"] = (text_df["title"] + " " + text_df["content"]).str.strip()

X_text_train = text_df["text"].values[:split_idx]
y_text_train = text_df["label"].astype(int).values[:split_idx]

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
