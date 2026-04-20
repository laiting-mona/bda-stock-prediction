# Random Forest 模型

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
類別不平衡（漲56.5% / 跌43.5%）由 `class_weight='balanced'` 處理。

## 參數（Grid Search 最佳）

Stage 1：24 組超參數（n_estimators=100）× 5-fold CV
Stage 2：最佳超參數搭配 n_estimators=300 重新訓練

- 模型：RandomForestClassifier
- n_estimators = 300
- max_depth = None（不限深度）
- max_features = sqrt
- min_samples_leaf = 2
- class_weight = balanced

## 結果

- Accuracy: 51.71%
- Macro F1: 0.5046
- Confusion Matrix: 見 results/confusion_matrices/rf_confusion_matrix.png

|  | 預測 Down(0) | 預測 Up(1) |
|--|-------------|-----------|
| **實際 Down(0)** | 755 | 1,043 |
| **實際 Up(1)** | 992 | 1,424 |

## 模型存檔說明

| 檔案 | 格式 | 說明 |
|------|------|------|
| `RF_model.pkl` | pickle | 訓練好的 RandomForestClassifier |
| `vectorizer.pkl` | pickle | TF-IDF + chi2 Pipeline（推論新文章用） |
| `RF_results.csv` | CSV | 全資料集標記結果（含 n、σ、is_discarded、pred_label） |

## Phase 3 預想問題

1. **每日重新建向量空間**：vectorizer.pkl 只需 transform（不需重新 fit），可直接對新文章產生特徵。若採滾動訓練則需重新 fit。
2. **文章數太少**：建議設出手門檻（當日文章數 < 5 篇時不出手）。
3. **類別偏移**：牛市期間訓練資料看漲偏多，`class_weight='balanced'` 可緩解，但若整體市場方向轉變仍可能失效。
4. **推估回測時間**：RF 單次 predict 約 0.1s，300 棵樹全回測估計 < 10 分鐘。

## Phase 3 載入方式

```python
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
with open(BASE_DIR / "models/RF/RF_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "models/RF/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 推論新文章（回傳 0/1）
texts = ["台積電法說會上修目標價，外資看好先進製程需求"]
X = vec.transform(texts)
y_pred = model.predict(X)           # [1] = 漲, [0] = 跌
y_prob = model.predict_proba(X)     # [[p_down, p_up]]
```
