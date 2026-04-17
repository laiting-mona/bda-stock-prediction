# kNN 模型

## 負責人
若涵

## 使用的特徵
搭配 A（手工特徵路線），檔案路徑：`data/processed/tsmc_features.csv`

使用欄位（共 65 維）：
- 文字長度與符號密度：`title_len`, `content_len`, `text_len`, `digit_count`, `exclamation_count`, `question_count`
- 關鍵字與情緒詞：`keyword_hits`, `positive_hits`, `negative_hits`, `sentiment_score`
- 價格特徵：`log_price_0`
- 時間拆解特徵：`post_year`, `post_month`, `post_weekday`, `post_hour`
- PCA 降維成分：`pca_1` ~ `pca_50`

排除欄位（避免資料洩漏）：`return_rate`, `abs_return_rate`, `price_0`, `price_1`

## 參數
- n_neighbors = 9（5-fold CV 搜尋最佳 k，候選：3, 5, 7, 9, 11, 15）
- metric = euclidean
- weights = distance
- 前處理：StandardScaler 標準化

## 結果
- Accuracy: 67.42%
- Macro F1: 0.6641
- Confusion Matrix: 見 `results/confusion_matrices/knn.png`

```
              看跌(0)  看漲(1)
預測看跌(0)   1055     777
預測看漲(1)    596    1786
```

## Phase 3 載入方式
```python
import joblib
model = joblib.load("models/knn/knn_model.pkl")
scaler = joblib.load("models/knn/vectorizer.pkl")
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
```
