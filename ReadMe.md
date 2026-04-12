# bda-stock-prediction

## Phase 1
by 妍禎

1. 先裝套件：
    ```bash
    pip install pandas numpy scikit-learn mysql-connector-python
    ```
2. 打開 `scripts/fetch_data.py`，確認 MySQL 連線資訊，要改 [mySQL 密碼]。
3. 先跑資料抓取：
    ```bash
    python scripts/fetch_data.py
    ```
4. 再跑資料前處理：
    ```bash
    python scripts/preprocess.py
    ```
5. 最後跑特徵工程：
    ```bash
    python scripts/feature_eng.py
    ```
6. 會產生的檔案
    - `data/tsmc_data.csv`：原始抓下來的資料
    - `data/tsmc_clean.csv`：清理完的資料
    - `data/tsmc_clean_filtered.csv`：把 `label = 2` 拿掉後的資料
    - `data/tsmc_features.csv`：做完手工特徵 + PCA 的資料
    - `data/tsmc_vector_space.csv`：N-gram TF-IDF 經卡方篩選後的向量空間

### 使用方法

1. `data/tsmc_data.csv`（原始資料）
    - 用途：資料來源備份、回溯抓取內容是否正確。
    - 什麼時候用：你要檢查原始欄位、比對清理前後差異時。

2. `data/tsmc_clean.csv`（清理後 + 三分類標籤）
    - 用途：EDA、標籤分布檢查（漲/跌/中性）。
    - 什麼時候用：你要看資料品質、缺值、類別不平衡時。

3. `data/tsmc_clean_filtered.csv`（移除 label=2）
    - 用途：二分類資料集（漲=1、跌=0）的標準標籤來源。
    - 什麼時候用：你要做二分類訓練與評估時。

4. `data/tsmc_features.csv`（手工特徵 + PCA）
    - 用途：可直接餵給樹模型或線性模型做訓練。
    - 建議模型：RandomForest、XGBoost、LogisticRegression。
    - 備註：可把文字欄位（如 `title`、`content`、`text`）排除，只留數值欄位進模型。

5. `data/tsmc_vector_space.csv`（TF-IDF + 卡方）
    - 用途：純文字向量特徵，適合文字分類基線模型。
    - 建議模型：LogisticRegression、LinearSVC、Naive Bayes。
    - 備註：此檔通常不含 `label`，訓練時請搭配 `data/tsmc_clean_filtered.csv` 的 `label` 欄。

### 訓練搭配

- 搭配 A（手工特徵路線）：`X = tsmc_features.csv` 的數值欄位、`y = tsmc_features.csv` 的 `label`
- 搭配 B（文字向量路線）：`X = tsmc_vector_space.csv`、`y = tsmc_clean_filtered.csv` 的 `label`
- 評估指標建議：`Accuracy` + `Macro F1` + `confusion matrix`

### 特徵工程方法

#### 手工特徵

- 文字長度與符號密度：`title_len`、`content_len`、`text_len`、`digit_count`、`exclamation_count`、`question_count`
- 關鍵字與情緒詞：`keyword_hits`、`positive_hits`、`negative_hits`、`sentiment_score`
- 價格連續特徵：`return_rate`、`abs_return_rate`、`log_price_0`
- 時間拆解特徵：`post_year`、`post_month`、`post_weekday`、`post_hour`

#### 向量空間建構（N-gram / TF-IDF / 卡方 / PCA）

- N-gram 表示：使用中文字元 n-gram，`ngram_range=(1, 2)`
- TF-IDF 參數：`max_features=4000`、`min_df=2`、`sublinear_tf=True`
- 卡方特徵選擇：`SelectKBest(chi2)`，最多保留 `k=300` 維（若詞彙不足則自動調整）
- PCA 降維：最多 `50` 維（實際維度為 `min(50, 樣本數, 特徵數)`）

#### 輸出說明

- `data/tsmc_vector_space.csv`：TF-IDF 經卡方後的向量，可直接做分類模型輸入
- `data/tsmc_features.csv`：原始欄位 + 手工特徵 + PCA 特徵
