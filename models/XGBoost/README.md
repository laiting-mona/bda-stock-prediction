# XGBoost 模型

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
類別不平衡（漲56.5% / 跌43.5%）由 `scale_pos_weight = neg/pos ≈ 0.776` 處理。

## 參數（Grid Search 最佳）

Stage 1：72 組超參數（n_estimators=100）× 5-fold CV
Stage 2：最佳超參數搭配 n_estimators=300 重新訓練

- 模型：XGBClassifier
- n_estimators = 300
- max_depth = 7
- learning_rate = 0.2
- colsample_bytree = 0.8
- subsample = 1.0
- scale_pos_weight = 0.776

## 結果

- Accuracy: 50.76%
- Macro F1: 0.5045
- Confusion Matrix: 見 results/confusion_matrices/xgboost_confusion_matrix.png

|  | 預測 Down(0) | 預測 Up(1) |
|--|-------------|-----------|
| **實際 Down(0)** | 903 | 895 |
| **實際 Up(1)** | 1,180 | 1,236 |

## 模型存檔說明

| 檔案 | 格式 | 說明 |
|------|------|------|
| `XGBoost_model.pkl` | pickle | 訓練好的 XGBClassifier |
| `vectorizer.pkl` | pickle | TF-IDF + chi2 Pipeline（推論新文章用） |
| `XGBoost_results.csv` | CSV | 全資料集標記結果（含 n、σ、is_discarded、pred_label） |

## Phase 3 預想問題

1. **每日重新建向量空間**：vectorizer.pkl（TF-IDF + chi2）需對新文章重新 transform，不需重新 fit。若要滾動訓練則需重新 fit。
2. **文章數太少**：建議設出手門檻（當日文章數 < 5 篇時不出手）。
3. **類別偏移**：牛市期間訓練資料看漲偏多，bearish 期間反轉，`scale_pos_weight` 可動態調整。
4. **推估時間**：XGBoost 單次 predict 極快（< 1s），全回測估計 < 5 分鐘。

## Phase 3 載入方式

```python
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
with open(BASE_DIR / "models/XGBoost/XGBoost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "models/XGBoost/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 推論新文章（回傳 0/1，無 predict_proba 門檻需求可直接用 predict）
texts = ["台積電法說會上修目標價，外資看好先進製程需求"]
X = vec.transform(texts)
y_pred = model.predict(X)           # [1] = 漲, [0] = 跌
y_prob = model.predict_proba(X)     # [[p_down, p_up]]
```
