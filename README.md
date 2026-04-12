# bda-stock-prediction

## 專題總覽

- **課程**：大數據與商業分析（BDA 2026）
- **專題**：用 AI 及社群數據協助投資決策 — 從分類到預測
- **標的**：台積電（2330）
- **語言與平台**：Python、MySQL、GitHub
- **資料集**：24 個月之新聞、論壇、BBS、股價、財報（435MB）
- **方法**：文本特徵擷取 → 監督式分類 → 移動回測
- **標記方式**：自編碼，以 D+n 日與 D 日收盤價漲跌幅是否超過 σ 來分看漲（1）／看跌（0），幅度內為中性（2）丟棄
- **組長（兼 PM）**：亭穎
- **會議主持人**：每次開會輪流擔任

---

## 分工表

### 模型與階段分工

| 負責人 | 負責內容 |
|--------|--------|
| 妍禎 | Phase 1 資料前處理與特徵工程 |
| 位青 | Phase 1 TF-IDF + N-gram、簡報製作 |
| 侑芸 | Phase 1 卡方、Phase 2 SVM |
| 若涵 | Phase 2 kNN、Phase 3 回測 |
| 亭穎 | Phase 2 RF + Naive Bayes + XGBoost、Phase 3 回測 |
| 芝伶 | Phase 2 LLM 驗證、影片錄製 |

### 簡報與影片分工

| 段落 | 簡報製作 | 講解 |
|------|--------|------|
| 開場、目錄、選股說明 | 位青 | 妍禎 |
| Phase 1 講解與 demo | Phase 1 全體負責人 | 妍禎 |
| Phase 2 講解與 demo | Phase 2 全體負責人 | 侑芸 |
| Phase 3 講解與 demo | Phase 3 全體負責人 | 若涵 |
| 優缺點評判、可改進方式、參考資料、隊員分工 | 亭穎 | 若涵 |

影片長度：12 分鐘以內

---

## 時程規劃

| 階段 | 內容 | 截止日期 |
|------|------|---------|
| Phase 1 | 資料前處理 & Req(1) 特徵工程 | 4/12（六） |
| Phase 2 | Req(2) 分類模型訓練 & 評估 | 4/18（五） |
| Phase 3 | Req(3) 移動回測 & 整合結果 | 4/20（日） |
| Phase 4 | 簡報製作 & 影片錄製 | 4/26（六）繳交 |
| 報告日 | 上台報告 | 4/29（二） |

---

## 專案結構
再麻煩大家如以下格式更新！
```
bda-stock-prediction/
├── README.md
├── requirements.txt
├── .env.example
├── .env                              # 自行新增（本機使用，不上傳）
│
├── data/
│   ├── processed/                     # Phase 1 產出
│   │   ├── tsmc_data.csv
│   │   ├── tsmc_clean.csv
│   │   ├── tsmc_clean_filtered.csv
│   │   ├── tsmc_features.csv
│   │   └── tsmc_vector_space.csv
│   └── features/                      # 特徵工程中間產物
│       ├── top_300_features.csv
│       ├── tsmc_n-gram_up.csv
│       └── tsmc_n-gram_down.csv
│
├── scripts/
│   └── phase1/
│       ├── fetch_data.py
│       ├── preprocess.py
│       ├── feature_eng.py
│       └── feature_selection.py
│
├── models/                            # Phase 2 各模型
│   ├── knn/                           # 若涵
│   ├── svm/                           # 侑芸
│   ├── rf/                            # 亭穎
│   ├── nb/                            # 亭穎
│   ├── xgboost/                       # 亭穎
│   └── llm/                           # 芝伶
│
├── backtest/                          # Phase 3 回測
│   ├── backtest_runner.py
│   ├── config.py
│   └── results/
│
├── results/                           # 分類結果匯總
│   ├── model_comparison.csv
│   └── confusion_matrices/
│
├── deliverables/                      # 繳交物
│   ├── slides/
│   └── screenshots/
│
├── docs/                              # 會議紀錄
│
└── .gitignore
```

---

## 環境建置

```bash
pip install -r requirements.txt
```

建立 `.env` 檔案（可參考 `.env.example`）：

```env
BDA_MYSQL_HOST=localhost
BDA_MYSQL_USER=root
BDA_MYSQL_PASSWORD=你的密碼
BDA_MYSQL_DB=bda2026
BDA_MYSQL_CHARSET=utf8mb4
```

---

## Phase 1：資料前處理與特徵工程

> 注意：因 GitHub 檔案大小限制，`data/processed/` 內的大型資料檔不會放在 repo。請先依下列步驟自行執行資料前處理與特徵工程，再進行後續模型訓練。

### 執行步驟

1. 在專案根目錄建立 `.env`（可由 `.env.example` 複製），並填入 MySQL 密碼：
    ```env
    BDA_MYSQL_HOST=localhost
    BDA_MYSQL_USER=root
    BDA_MYSQL_PASSWORD=你的密碼
    BDA_MYSQL_DB=bda2026
    BDA_MYSQL_CHARSET=utf8mb4
    ```
   （`fetch_data.py` 會自動讀取 `.env`）
2. 確認 `data/processed/` 資料夾已存在（腳本不會自動建立資料夾）。
3. 確認 `data/features/` 中已有：`tsmc_n-gram_up.csv`、`tsmc_n-gram_down.csv`。
4. 依序執行：
    ```bash
    python scripts/phase1/fetch_data.py
    python scripts/phase1/preprocess.py
    python scripts/phase1/feature_selection.py
    python scripts/phase1/feature_eng.py
    ```

### 產出檔案

| 檔案 | 用途 | 什麼時候用 |
|------|------|-----------|
| `data/processed/tsmc_data.csv` | 原始資料備份 | 檢查原始欄位、比對清理前後差異 |
| `data/processed/tsmc_clean.csv` | 清理後 + 三分類標籤 | EDA、標籤分布檢查（漲/跌/中性） |
| `data/processed/tsmc_clean_filtered.csv` | 二分類資料集（漲=1、跌=0） | 二分類訓練與評估的標準標籤來源 |
| `data/features/top_300_features.csv` | 卡方篩選後特徵詞表 | 供 `feature_eng.py` 計算 `keyword_hits` 優先使用 |
| `data/processed/tsmc_features.csv` | 手工特徵 + PCA | 餵給樹模型或線性模型（RF、XGB、LR） |
| `data/processed/tsmc_vector_space.csv` | TF-IDF + 卡方 | 文字分類基線（LR、SVC、NB） |

### 訓練搭配

- **搭配 A（手工特徵路線）**：`X = tsmc_features.csv` 的數值欄位、`y = tsmc_features.csv` 的 `label`
- **搭配 B（文字向量路線）**：`X = tsmc_vector_space.csv`、`y = tsmc_clean_filtered.csv` 的 `label`
- **評估指標**：`Accuracy` + `Macro F1` + `Confusion Matrix`

### 特徵工程方法

**手工特徵：**
- 文字長度與符號密度：`title_len`、`content_len`、`text_len`、`digit_count`、`exclamation_count`、`question_count`
- 關鍵字與情緒詞：`keyword_hits`、`positive_hits`、`negative_hits`、`sentiment_score`
- 價格連續特徵：`return_rate`、`abs_return_rate`、`log_price_0`
- 時間拆解特徵：`post_year`、`post_month`、`post_weekday`、`post_hour`

**向量空間建構：**
- N-gram：中文字元 n-gram，`ngram_range=(1, 2)`
- TF-IDF：`max_features=4000`、`min_df=2`、`sublinear_tf=True`
- 卡方特徵選擇：`SelectKBest(chi2)`，保留 `k=300` 維
- PCA 降維：最多 `50` 維

---

## Phase 2：分類模型訓練與評估

### 各模型狀態

| 模型 | 負責人 | 特徵搭配 | 狀態 | Accuracy | Macro F1 |
|------|--------|---------|------|----------|----------|
| kNN | 若涵 | | 待完成 | - | - |
| SVM | 侑芸 | | 待完成 | - | - |
| Random Forest | 亭穎 | | 待完成 | - | - |
| Naive Bayes | 亭穎 | | 待完成 | - | - |
| XGBoost | 亭穎 | | 待完成 | - | - |
| LLM 直接判斷 | 芝伶 | | 待完成 | - | - |

### 模型資料夾規範

每個模型子資料夾至少要有：

```
models/xxx/
├── xxx_train.py          # 訓練腳本（可重現）
├── xxx_model.pkl         # 訓練好的模型
├── vectorizer.pkl        # 對應的 vectorizer（不能漏）
└── README.md             # 模型說明
```

**模型 README.md 範本：**

```markdown
# 模型名稱

## 負責人
XXX

## 使用的特徵
搭配 A or B，檔案路徑

## 參數
- param1 = xxx
- param2 = xxx

## 結果
- Accuracy: XX%
- Macro F1: XX
- Confusion Matrix: 見 results/confusion_matrices/xxx.png

## Phase 3 載入方式
import joblib
model = joblib.load("models/xxx/xxx_model.pkl")
vectorizer = joblib.load("models/xxx/vectorizer.pkl")
X_new = vectorizer.transform(new_texts)
y_pred = model.predict(X_new)
```

---

## Phase 3：移動回測

### 回測設計（待完成）

- **回測期間**：（待定）
- **訓練窗口**：前 30 天
- **預測天數**：D+n 日
- **出手門檻**：看漲／看跌票數差距 > X 且當日文章數 > Y
- **使用模型**：Phase 2 表現最佳的 2~3 個模型做多數決投票

### 回測結果（待完成）

| 參數組合 (n / sigma / 窗口) | 出手率 | 準確率 | 備註 |
|---------------------------|--------|--------|------|
| | | | |

---

## Deliverables

- 簡報檔（尾附影片連結）
- 12 分鐘內說明影片
- 簡報 + 系統截圖 + 程式碼 zip（小於等於 100MB，不含影片與資料）
- 繳交日期：4/26（六）
- 報告日期：4/29（二）
