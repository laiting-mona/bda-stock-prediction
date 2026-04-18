"""
LLM 直接判斷模型 - Phase 2
負責人：芝伶
說明：不訓練模型，直接用 Claude LLM prompt 判斷文章是看漲(1)或看跌(0)，作為對照組
"""

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import anthropic
import json

# ========== 設定 ==========
DATA_PATH = "data/processed/tsmc_clean_filtered.csv"
OUTPUT_DIR = "models/llm"
CM_DIR = "results/confusion_matrices"
SAMPLE_SIZE = 100  # 抽樣筆數（API 有費用，建議先用 100 筆）
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# ========== 1. 讀取資料 ==========
df = pd.read_csv(DATA_PATH)
sample = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"抽樣 {SAMPLE_SIZE} 筆，label 分布：")
print(sample['label'].value_counts())

# ========== 2. LLM 判斷函數 ==========
client = anthropic.Anthropic()  # 需設定 ANTHROPIC_API_KEY 環境變數

def llm_predict(title: str, content: str) -> int:
    """用 Claude LLM 判斷文章是看漲(1)或看跌(0)"""
    prompt = f"""以下是一篇關於台積電(2330)的文章，請判斷這篇文章對台積電股價是「看漲」還是「看跌」。

標題：{title}
內容：{str(content)[:300]}

請只回答數字：1（看漲）或 0（看跌），不要有其他文字。"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.content[0].text.strip()
    return int(result) if result in ["0", "1"] else -1

# ========== 3. 執行預測 ==========
print("\n開始 LLM 判斷...")
predictions = []
for i, row in sample.iterrows():
    pred = llm_predict(row['title'], row['content'])
    predictions.append(pred)
    if (i + 1) % 10 == 0:
        print(f"  已處理 {i+1}/{SAMPLE_SIZE} 筆")

# ========== 4. 評估 ==========
valid_idx = [i for i, p in enumerate(predictions) if p != -1]
y_true = [sample.loc[i, 'label'] for i in valid_idx]
y_pred = [predictions[i] for i in valid_idx]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
cm = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# ========== 5. 儲存 Confusion Matrix 圖片 ==========
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['看跌(0)', '看漲(1)'])
ax.set_yticklabels(['看跌(0)', '看漲(1)'])
ax.set_xlabel('預測值'); ax.set_ylabel('真實值')
ax.set_title(f'LLM Confusion Matrix\nAccuracy={acc:.4f}, Macro F1={f1:.4f}')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
cm_path = f"{CM_DIR}/llm.png"
plt.savefig(cm_path, dpi=150)
print(f"Confusion Matrix 圖片已儲存至 {cm_path}")

# ========== 6. 儲存預測結果 ==========
result_df = sample.copy()
result_df['llm_pred'] = predictions
result_df.to_csv(f"{OUTPUT_DIR}/llm_predictions.csv", index=False, encoding='utf-8-sig')

# ========== 7. 儲存模型 (LLM 無需訓練，儲存參數資訊) ==========
model_info = {
    "model": "claude-sonnet-4-20250514",
    "method": "zero-shot prompt",
    "sample_size": SAMPLE_SIZE,
    "accuracy": acc,
    "macro_f1": f1,
    "confusion_matrix": cm.tolist()
}
joblib.dump(model_info, f"{OUTPUT_DIR}/llm_model.pkl")
joblib.dump(None, f"{OUTPUT_DIR}/vectorizer.pkl")  # LLM 不需要 vectorizer
print("模型資訊已儲存")
