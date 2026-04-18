# LLM 直接判斷模型

## 負責人
芝伶

## 使用的特徵
搭配 B（文字向量路線）：直接使用文章標題 + 內容原文
- 輸入檔案：`data/processed/tsmc_clean_filtered.csv`
- 欄位：`title`、`content`、`label`

## 方法說明
不進行任何模型訓練，直接將文章標題與內容傳入 Claude LLM，
透過 zero-shot prompt 要求 LLM 判斷文章對台積電股價是看漲(1)或看跌(0)。
作為其他分類模型的對照組使用。

## 參數
- model = claude-sonnet-4-20250514
- method = zero-shot prompt
- sample_size = 100（隨機抽樣，random_state=42）
- prompt 策略 = 提供標題 + 內容前300字，要求只回答 0 或 1

## 結果
- Accuracy: 81.00%
- Macro F1: 0.8098
- Confusion Matrix: 見 results/confusion_matrices/llm.png

|  | 預測看跌(0) | 預測看漲(1) |
|--|------------|------------|
| 真實看跌(0) | 39 | 10 |
| 真實看漲(1) | 9 | 42 |

## Phase 3 載入方式
LLM 模型無需載入，直接呼叫 API 即可預測。
模型資訊可用 joblib 載入查看：

```python
import joblib
model_info = joblib.load("models/llm/llm_model.pkl")
print(model_info)
```

如需對新文章預測：
```python
import anthropic
client = anthropic.Anthropic()

def llm_predict(title, content):
    prompt = f\"\"\"以下是一篇關於台積電(2330)的文章，請判斷這篇文章對台積電股價是「看漲」還是「看跌」。

標題：{title}
內容：{str(content)[:300]}

請只回答數字：1（看漲）或 0（看跌），不要有其他文字。\"\"\"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.content[0].text.strip()
    return int(result) if result in ["0", "1"] else -1
```
