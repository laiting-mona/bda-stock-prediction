"""
kNN 分類模型訓練腳本
負責人：若涵
特徵搭配：搭配 A（手工特徵 + PCA）
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent  # bda-stock-prediction-main/
DATA_PATH = BASE_DIR / "data/processed/tsmc_features.csv"
MODEL_DIR = BASE_DIR / "models/knn"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("kNN 模型訓練 — 搭配 A（手工特徵 + PCA）")
print("=" * 60)

# ── 讀取資料 ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✓ 讀取資料：{len(df)} 筆")

# ── 特徵選擇（排除洩漏欄位與非數值欄位）──────────────────
drop_cols = ['post_time', 'title', 'content', 'text',
             'price_0', 'price_1', 'return_rate', 'abs_return_rate', 'label']

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"✓ 使用特徵數：{len(feature_cols)} 個")
print(f"  特徵欄位：{feature_cols[:8]} ... (共 {len(feature_cols)} 個)")

X = df[feature_cols].values
y = df['label'].values

print(f"\n✓ 標籤分布：看漲(1)={sum(y==1)}, 看跌(0)={sum(y==0)}")

# ── 訓練/測試分割（80/20，保留時序）────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ 訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")

# ── 特徵標準化（kNN 必須做）────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✓ 特徵標準化完成（StandardScaler）")

# ── 選擇最佳 k 值 ─────────────────────────────────────────
print("\n── 搜尋最佳 k 值（k = 3, 5, 7, 9, 11, 15）──")
k_candidates = [3, 5, 7, 9, 11, 15]
best_k, best_f1 = 5, 0

for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    scores = cross_val_score(knn, X_train_scaled, y_train,
                             cv=5, scoring='f1_macro', n_jobs=-1)
    mean_f1 = scores.mean()
    print(f"  k={k:2d}  CV Macro F1 = {mean_f1:.4f}")
    if mean_f1 > best_f1:
        best_f1, best_k = mean_f1, k

print(f"\n✓ 最佳 k = {best_k}（CV Macro F1 = {best_f1:.4f}）")

# ── 訓練最終模型 ──────────────────────────────────────────
knn_final = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean',
    weights='distance',
    n_jobs=-1
)
knn_final.fit(X_train_scaled, y_train)
print("✓ 模型訓練完成")

# ── 評估 ──────────────────────────────────────────────────
y_pred = knn_final.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print("\n" + "=" * 60)
print("測試集評估結果")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro F1 : {macro_f1:.4f}")
print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=['看跌(0)', '看漲(1)']))

# ── 混淆矩陣 ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['看跌(0)', '看漲(1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'kNN Confusion Matrix (k={best_k})\nAccuracy={accuracy:.4f}  Macro F1={macro_f1:.4f}')
plt.tight_layout()
cm_path = MODEL_DIR / "knn_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n✓ 混淆矩陣已儲存：{cm_path}")

# ── 儲存模型與 scaler ─────────────────────────────────────
model_path  = MODEL_DIR / "knn_model.pkl"
scaler_path = MODEL_DIR / "vectorizer.pkl"
joblib.dump(knn_final, model_path)
joblib.dump(scaler,    scaler_path)
print(f"✓ 模型已儲存：{model_path}")
print(f"✓ Scaler 已儲存：{scaler_path}")

# ── 摘要 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("訓練完成摘要")
print("=" * 60)
print(f"模型       : KNeighborsClassifier")
print(f"特徵路線   : 搭配 A（手工特徵 + PCA，共 {len(feature_cols)} 維）")
print(f"最佳 k     : {best_k}")
print(f"距離度量   : Euclidean（weights='distance'）")
print(f"Accuracy   : {accuracy*100:.2f}%")
print(f"Macro F1   : {macro_f1:.4f}")
print(f"混淆矩陣   : {cm}")
