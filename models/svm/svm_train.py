"""
SVM 分類模型訓練腳本
負責人：侑芸
特徵搭配：搭配 A（手工特徵 + PCA）、搭配 B（TF-IDF 向量）、搭配 C（A+B 組合）
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models/svm"
RESULT_DIR = BASE_DIR / "results/confusion_matrices"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SVM 模型訓練 — 搭配 A / B / C 比較")
print("=" * 60)

# ── 讀取資料 ──────────────────────────────────────────────
df_feat = pd.read_csv(BASE_DIR / "data/tsmc_features.csv")
df_vec  = pd.read_csv(BASE_DIR / "data/tsmc_vector_space.csv")
print(f"\n✓ 讀取資料：{len(df_feat)} 筆")

# ── 特徵選擇（排除洩漏欄位與非數值欄位）──────────────────
# 注意：price_0, price_1, return_rate 含未來資訊，必須排除
DROP_COLS = ['post_time', 'title', 'content', 'text',
             'price_0', 'price_1', 'return_rate', 'label']

feat_cols = [c for c in df_feat.columns if c not in DROP_COLS]
print(f"✓ 搭配 A 特徵數：{len(feat_cols)} 個（手工特徵 + PCA）")
print(f"✓ 搭配 B 特徵數：{len(df_vec.columns)} 個（TF-IDF 向量）")

X_A     = df_feat[feat_cols].values
X_B     = df_vec.values.astype(np.float32)
X_C     = np.hstack([X_A, X_B])
y       = df_feat['label'].values

print(f"✓ 搭配 C 特徵數：{X_C.shape[1]} 個（A + B 組合）")
print(f"\n✓ 標籤分布：看漲(1)={sum(y==1)}, 看跌(0)={sum(y==0)}")

# ── 訓練/測試分割（80/20）────────────────────────────────
(X_A_tr, X_A_te,
 X_B_tr, X_B_te,
 X_C_tr, X_C_te,
 y_train, y_test) = train_test_split(
    X_A, X_B, X_C, y,
    test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ 訓練集：{len(y_train)} 筆 | 測試集：{len(y_test)} 筆")

# ── 正規化 ────────────────────────────────────────────────
scaler_A = StandardScaler()   # 手工特徵有正負值
scaler_B = MaxAbsScaler()     # TF-IDF 非負稀疏向量
scaler_C = MaxAbsScaler()     # 組合也用 MaxAbsScaler

X_A_tr_s = scaler_A.fit_transform(X_A_tr); X_A_te_s = scaler_A.transform(X_A_te)
X_B_tr_s = scaler_B.fit_transform(X_B_tr); X_B_te_s = scaler_B.transform(X_B_te)
X_C_tr_s = scaler_C.fit_transform(X_C_tr); X_C_te_s = scaler_C.transform(X_C_te)
print("✓ 特徵標準化完成")

# ── 搜尋最佳 C 值（用 cross_val_score，每種搭配分開找）────
CONFIGS = [
    ("A（手工特徵+PCA）", X_A_tr_s, X_A_te_s, scaler_A),
    ("B（TF-IDF向量）",   X_B_tr_s, X_B_te_s, scaler_B),
    ("C（A+B組合）",      X_C_tr_s, X_C_te_s, scaler_C),
]
C_CANDIDATES = [0.01, 0.1, 1.0, 5.0, 10.0]

all_results  = {}
best_configs = {}   # 每種搭配的最佳 C

for feat_name, X_tr_s, X_te_s, scaler in CONFIGS:
    print(f"\n── 搜尋最佳 C 值（搭配 {feat_name}）──")
    best_C, best_cv_f1 = 1.0, 0.0

    for C in C_CANDIDATES:
        clf = LinearSVC(C=C, max_iter=5000, random_state=42)
        scores = cross_val_score(clf, X_tr_s, y_train,
                         cv=5, scoring='f1_macro', n_jobs=1)
        mean_f1 = scores.mean()
        print(f"  C={C:<5}  CV Macro F1 = {mean_f1:.4f}")
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_C = mean_f1, C

    print(f"✓ 最佳 C = {best_C}（CV Macro F1 = {best_cv_f1:.4f}）")
    best_configs[feat_name] = (best_C, X_tr_s, X_te_s, scaler)

# ── 用最佳 C 訓練最終模型，評估測試集 ────────────────────
print("\n" + "=" * 60)
print("各搭配最終測試集結果")
print("=" * 60)

for feat_name, (best_C, X_tr_s, X_te_s, scaler) in best_configs.items():
    clf = LinearSVC(C=best_C, max_iter=5000, random_state=42)
    clf.fit(X_tr_s, y_train)
    y_pred = clf.predict(X_te_s)

    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average='macro')
    cm  = confusion_matrix(y_test, y_pred)

    all_results[feat_name] = {
        "C": best_C, "acc": acc, "mf1": mf1,
        "cm": cm, "clf": clf, "scaler": scaler, "y_pred": y_pred,
    }

    print(f"\n[搭配 {feat_name}]  C={best_C}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1 : {mf1:.4f}")
    print(f"  混淆矩陣 (Confusion Matrix):")
    print(f"                   [預測為跌(0)]  [預測為漲(1)]")
    print(f"  [實際為跌(0)]       {cm[0][0]:<13} {cm[0][1]}")
    print(f"  [實際為漲(1)]       {cm[1][0]:<13} {cm[1][1]}")

# ── 找出整體最佳 ──────────────────────────────────────────
best_feat = max(all_results, key=lambda k: all_results[k]['mf1'])
best_r    = all_results[best_feat]
cm        = best_r['cm']

print("\n" + "=" * 60)
print("🎯 Phase 2: SVM 最終評估結果")
print("=" * 60)
print(f"最佳特徵搭配 : {best_feat}")
print(f"最佳 C       : {best_r['C']}")
print(f"Accuracy     : {best_r['acc']:.4f} ({best_r['acc']*100:.2f}%)")
print(f"Macro F1     : {best_r['mf1']:.4f}")
print("-" * 60)
print("混淆矩陣 (Confusion Matrix):")
print("                 [預測為跌(0)]  [預測為漲(1)]")
print(f"[實際為跌(0)]       {cm[0][0]:<13} {cm[0][1]}")
print(f"[實際為漲(1)]       {cm[1][0]:<13} {cm[1][1]}")
print("=" * 60)
print("\n分類報告：")
print(classification_report(y_test, best_r['y_pred'],
                             target_names=['看跌(0)', '看漲(1)']))

# ── 畫圖：三種搭配比較 ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("SVM — 三種特徵搭配比較（各取最佳 C）", fontsize=13)

for i, (feat_name, r) in enumerate(all_results.items()):
    star = "  ★ 最佳" if feat_name == best_feat else ""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=r['cm'],
        display_labels=['看跌(0)', '看漲(1)']
    )
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(
        f"搭配 {feat_name}{star}\n"
        f"C={r['C']}  Acc={r['acc']:.4f}  F1={r['mf1']:.4f}",
        fontsize=9
    )

plt.tight_layout()
compare_path = RESULT_DIR / "svm_feature_comparison.png"
plt.savefig(compare_path, dpi=150)
plt.close()
print(f"\n✓ 比較圖已儲存：{compare_path}")

# ── 最佳結果單張 CM ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=best_r['cm'],
    display_labels=['看跌(0)', '看漲(1)']
)
disp2.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title(
    f"SVM 最佳結果\n搭配 {best_feat}  C={best_r['C']}\n"
    f"Accuracy={best_r['acc']:.4f}  Macro F1={best_r['mf1']:.4f}"
)
plt.tight_layout()
best_cm_path = RESULT_DIR / "svm_confusion_matrix.png"
plt.savefig(best_cm_path, dpi=150)
plt.close()
print(f"✓ 最佳 CM 已儲存：{best_cm_path}")

# ── 儲存最佳模型 ──────────────────────────────────────────
joblib.dump(best_r['clf'],    MODEL_DIR / "svm_model.pkl")
joblib.dump(best_r['scaler'], MODEL_DIR / "svm_scaler.pkl")
print(f"✓ 模型已儲存：{MODEL_DIR / 'svm_model.pkl'}")
print(f"✓ Scaler 已儲存：{MODEL_DIR / 'svm_scaler.pkl'}")

# ── 儲存結果摘要 CSV ──────────────────────────────────────
rows = []
for feat_name, r in all_results.items():
    m = r['cm']
    rows.append({
        "特徵搭配": feat_name, "C": r['C'],
        "accuracy": round(r['acc'], 4),
        "macro_f1": round(r['mf1'], 4),
        "TP": int(m[1,1]), "FN": int(m[1,0]),
        "FP": int(m[0,1]), "TN": int(m[0,0]),
        "is_best": feat_name == best_feat,
    })
pd.DataFrame(rows).to_csv(
    MODEL_DIR / "svm_results.csv",
    index=False, encoding="utf-8-sig"
)

# ── 訓練完成摘要 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("訓練完成摘要")
print("=" * 60)
print(f"模型         : LinearSVC")
print(f"最佳特徵搭配 : {best_feat}")
print(f"最佳 C       : {best_r['C']}")
print(f"Accuracy     : {best_r['acc']*100:.2f}%")
print(f"Macro F1     : {best_r['mf1']:.4f}")
print(f"混淆矩陣     : {best_r['cm']}")

print("\n" + "=" * 60)
print("📋 請將以下資訊填入 README.md 的 Phase 2 表格：")
print(f"   模型     : SVM (LinearSVC, C={best_r['C']})")
print(f"   負責人   : 侑芸")
print(f"   特徵搭配 : {best_feat}")
print(f"   Accuracy : {best_r['acc']:.4f}")
print(f"   Macro F1 : {best_r['mf1']:.4f}")
print("=" * 60)