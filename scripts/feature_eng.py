from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

POSITIVE_WORDS = ["利多", "成長", "創高", "看好", "上修", "買進", "突破", "樂觀"]
NEGATIVE_WORDS = ["利空", "下修", "衰退", "風險", "賣出", "崩跌", "虧損", "悲觀"]
KEYWORDS = ["台積電", "半導體", "晶圓", "先進製程", "AI", "法說會", "外資", "營收"]

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
input_path = data_dir / "tsmc_clean_filtered.csv"
output_path = data_dir / "tsmc_features.csv"
vector_output_path = data_dir / "tsmc_vector_space.csv"

NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 4000
TFIDF_MIN_DF = 2
CHI2_TOP_K = 300
PCA_COMPONENTS = 50

# 讀取前處理後且已排除中性標籤的資料
df = pd.read_csv(input_path, encoding="utf-8-sig")

feature_df = df.copy()

# 建立文字欄位並處理空值
feature_df["title"] = feature_df["title"].fillna("").astype(str)
feature_df["content"] = feature_df["content"].fillna("").astype(str)
feature_df["text"] = (feature_df["title"] + " " + feature_df["content"]).str.strip()

# 文字長度與符號密度特徵
feature_df["title_len"] = feature_df["title"].str.len()
feature_df["content_len"] = feature_df["content"].str.len()
feature_df["text_len"] = feature_df["text"].str.len()
feature_df["digit_count"] = feature_df["text"].str.count(r"\d")
feature_df["exclamation_count"] = feature_df["text"].str.count("!") + feature_df["text"].str.count("！")
feature_df["question_count"] = feature_df["text"].str.count(r"\?") + feature_df["text"].str.count("？")

# 關鍵字與情緒詞命中特徵
feature_df["keyword_hits"] = feature_df["text"].apply(lambda x: sum(x.count(word) for word in KEYWORDS))
feature_df["positive_hits"] = feature_df["text"].apply(lambda x: sum(x.count(word) for word in POSITIVE_WORDS))
feature_df["negative_hits"] = feature_df["text"].apply(lambda x: sum(x.count(word) for word in NEGATIVE_WORDS))
feature_df["sentiment_score"] = feature_df["positive_hits"] - feature_df["negative_hits"]

# 價格相關連續特徵
feature_df["return_rate"] = (feature_df["price_1"] - feature_df["price_0"]) / feature_df["price_0"]
feature_df["abs_return_rate"] = feature_df["return_rate"].abs()
feature_df["log_price_0"] = feature_df["price_0"].apply(lambda x: np.nan if x <= 0 else np.log(x))

# 貼文時間拆解特徵
feature_df["post_time"] = pd.to_datetime(feature_df["post_time"], errors="coerce")
feature_df["post_year"] = feature_df["post_time"].dt.year
feature_df["post_month"] = feature_df["post_time"].dt.month
feature_df["post_weekday"] = feature_df["post_time"].dt.weekday
feature_df["post_hour"] = feature_df["post_time"].dt.hour

# N-gram + TF-IDF 向量空間建構
vectorizer = TfidfVectorizer(
	analyzer="char",
	ngram_range=NGRAM_RANGE,
	max_features=TFIDF_MAX_FEATURES,
	min_df=TFIDF_MIN_DF,
	sublinear_tf=True,
)
tfidf_matrix = vectorizer.fit_transform(feature_df["text"])
feature_names = vectorizer.get_feature_names_out()

# 卡方特徵選擇（需要監督標籤）
if "label" in feature_df.columns and feature_df["label"].nunique() > 1:
	y = feature_df["label"].astype(int)
	k = min(CHI2_TOP_K, tfidf_matrix.shape[1])
	selector = SelectKBest(score_func=chi2, k=k)
	selected_matrix = selector.fit_transform(tfidf_matrix, y)
	selected_names = feature_names[selector.get_support()]
else:
	selected_matrix = tfidf_matrix
	selected_names = feature_names

# 輸出卡方後的向量空間（可以直接用於機器學習模型）
vector_df = pd.DataFrame(
	selected_matrix.toarray(),
	columns=[f"tfidf_{name}" for name in selected_names],
)
vector_df.to_csv(vector_output_path, index=False, encoding="utf-8-sig")

# PCA 降維（以卡方篩選後向量作為輸入）
pca_input = selected_matrix.toarray()
max_pca_components = min(PCA_COMPONENTS, pca_input.shape[0], pca_input.shape[1])
if max_pca_components >= 1:
	pca = PCA(n_components=max_pca_components, random_state=42)
	pca_features = pca.fit_transform(pca_input)
	pca_columns = [f"pca_{i + 1}" for i in range(max_pca_components)]
	pca_df = pd.DataFrame(pca_features, columns=pca_columns)
	feature_df = pd.concat([feature_df.reset_index(drop=True), pca_df], axis=1)

# 輸出特徵工程結果
feature_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"特徵工程完成，輸出 {len(feature_df)} 筆至：{output_path}")
print(f"向量空間完成，輸出 {vector_df.shape[1]} 維至：{vector_output_path}")