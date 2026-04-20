# Naive Bayes 模型

## 負責人
亭穎

## 使用的特徵
路徑 B（TF-IDF 文字向量，300 維）
路徑：data/processed/tsmc_vector_space.csv

特徵說明：char n-gram TF-IDF（ngram_range=(1,2), max_features=4000, sublinear_tf=True）→ 卡方選取 Top-300

## 標記參數

| 參數 | 值 | 說明 |
|------|-----|------|
| n | 1 | 預測 D+1 日收盤價相對 D 日的漲跌 |
| σ（sigma） | 0.015（1.5%） | 漲跌門檻：return > +1.5% → 漲(1)；< -1.5% → 跌(0) |

## 樣本統計

| 類別 | 筆數 |
|------|------|
| 漲（label=1） | 11,908 |
| 跌（label=0） | 9,160 |
| 持平丟棄（label=2） | 41,943 |
| **使用樣本** | **21,068** |

持平樣本（\|return\| ≤ 1.5%）全數丟棄，僅保留漲/跌二元分類資料。
類別不平衡（漲56.5% / 跌43.5%）由 ComplementNB 的特性自然處理（對不平衡資料較 MultinomialNB 穩定）。

## 參數（Grid Search 最佳）

比較 MultinomialNB vs ComplementNB，各自對 alpha 做 Grid Search（5-fold CV）。

- 模型：ComplementNB（對不平衡資料優於 MultinomialNB）
- alpha = 0.5
- norm = True

| 模型 | Best alpha | CV Macro F1 |
|------|-----------|-------------|
| MultinomialNB | 0.1 | 0.5439 |
| **ComplementNB** | **0.5** | **0.5847** |

## 結果

- Accuracy: 47.32%
- Macro F1: 0.4700
- Confusion Matrix: 見 results/confusion_matrices/naive_bayes_confusion_matrix.png

|  | 預測 Down(0) | 預測 Up(1) |
|--|-------------|-----------|
| **實際 Down(0)** | 833 | 965 |
| **實際 Up(1)** | 1,255 | 1,161 |

## 模型存檔說明

| 檔案 | 格式 | 說明 |
|------|------|------|
| `NB_model.pkl` | pickle | 訓練好的 ComplementNB |
| `vectorizer.pkl` | pickle | TF-IDF + chi2 Pipeline（推論新文章用） |
| `NB_results.csv` | CSV | 全資料集標記結果（含 n、σ、is_discarded、pred_label） |

## Phase 3 預想問題

1. **每日重新建向量空間**：vectorizer.pkl 只需 transform（不需重新 fit），可直接對新文章產生特徵。
2. **文章數太少**：NB 在樣本極少時預測不穩定，建議文章數 < 5 篇時不出手。
3. **類別偏移**：NB 假設特徵條件獨立且分布固定，時序分布偏移時降幅較大。
4. **推估回測時間**：NB 極快，全回測估計 < 1 分鐘。

## Phase 3 載入方式

```python
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
with open(BASE_DIR / "models/naive-bayes/NB_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "models/naive-bayes/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 推論新文章（回傳 0/1）
texts = ["台積電法說會上修目標價，外資看好先進製程需求"]
X = vec.transform(texts)
y_pred = model.predict(X)           # [1] = 漲, [0] = 跌
y_prob = model.predict_proba(X)     # [[p_down, p_up]]
```
