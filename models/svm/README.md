# SVM 模型

## 負責人
侑芸

## 使用的特徵
搭配 A（手工特徵 + PCA，共 66 維）
路徑：data/processed/tsmc_features.csv

## 參數
- 模型：LinearSVC
- C = 5.0
- max_iter = 5000

## 結果
- Accuracy: 68.91%
- Macro F1: 0.6759
- Confusion Matrix: 見 results/confusion_matrices/svm_confusion_matrix.png

## Phase 3 載入方式

```python
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
model  = joblib.load(BASE_DIR / "models/svm/svm_model.pkl")
scaler = joblib.load(BASE_DIR / "models/svm/svm_scaler.pkl")

df = pd.read_csv(BASE_DIR / "data/processed/tsmc_features.csv")

DROP_COLS = ['post_time', 'title', 'content', 'text',
             'price_0', 'price_1', 'return_rate', 'label']
feat_cols = [c for c in df.columns if c not in DROP_COLS]
X = scaler.transform(df[feat_cols].values)
y_pred = model.predict(X)
```
# 排除非特徵欄位
```python
DROP_COLS = ['post_time', 'title', 'content', 'text',
             'price_0', 'price_1', 'return_rate', 'label']
feat_cols = [c for c in df.columns if c not in DROP_COLS]
X = scaler.transform(df[feat_cols].values)
y_pred = model.predict(X)
```