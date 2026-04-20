from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB
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
# X：feature_eng.py 產出的 300 維 TF-IDF + 卡方向量空間（值非負，適合 NB）
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

# ── Grid Search（MultinomialNB vs ComplementNB × alpha）────────────────────
# TF-IDF 值非負 → 兩者皆適用；ComplementNB 對不平衡資料通常更穩
results = []

for ModelClass, name in [(MultinomialNB, "MultinomialNB"), (ComplementNB, "ComplementNB")]:
    print(f"\n── {name} ──")
    param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    if ModelClass is ComplementNB:
        param_grid["norm"] = [True, False]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        ModelClass(),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Best params  : {grid.best_params_}")
    print(f"CV Macro F1  : {grid.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    results.append({
        "name": name,
        "model": grid.best_estimator_,
        "params": grid.best_params_,
        "cv_f1": grid.best_score_,
        "test_acc": acc,
        "test_f1": macro_f1,
    })

# ── 選出最佳模型 ──────────────────────────────────────────────────────────────
best = max(results, key=lambda r: r["test_f1"])
print(f"\n{'='*50}")
print(f"最佳模型: {best['name']}")
print(f"Best params : {best['params']}")
print(f"Test Accuracy  : {best['test_acc']:.4f}")
print(f"Test Macro F1  : {best['test_f1']:.4f}")

y_pred_best = best["model"].predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
print(f"Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred_best, target_names=['跌(0)', '漲(1)'])}")

# ── 儲存最佳 NB 模型 ──────────────────────────────────────────────────────────
with open(model_dir / "NB_model.pkl", "wb") as f:
    pickle.dump(best["model"], f)
print(f"NB_model.pkl saved → {model_dir / 'NB_model.pkl'}")

# ── 建立並儲存 vectorizer（供推論新文章使用）─────────────────────────────────
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
