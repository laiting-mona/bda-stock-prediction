# Random Forest 模型

## 負責人
亭穎

## 使用的特徵
路徑 B（TF-IDF 文字向量，300 維）
路徑：data/processed/tsmc_vector_space.csv

特徵說明：char n-gram TF-IDF（ngram_range=(1,2), max_features=4000, sublinear_tf=True）→ 卡方選取 Top-300

## 參數（Grid Search 最佳）

Stage 1：24 組超參數 × 5-fold CV（n_estimators=100 快速掃描）
Stage 2：以最佳超參數搭配 n_estimators=300 重新訓練

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
| **實際 Down(0)** | 755 | 1043 |
| **實際 Up(1)** | 992 | 1424 |

## Phase 3 載入方式

```python
import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
with open(BASE_DIR / "models/RF/RF_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "models/RF/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 推論新文章
texts = ["台積電法說會上修目標價，外資看好先進製程需求"]
X = vec.transform(texts)
y_pred = model.predict(X)  # 1=漲, 0=跌
```
