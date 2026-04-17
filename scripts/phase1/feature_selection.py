import pandas as pd
from pathlib import Path

# 1. 設定檔案路徑
project_root = Path(__file__).resolve().parents[2]
features_dir = project_root / "data" / "features"
up_path = features_dir / "tsmc_n-gram_up.csv"
down_path = features_dir / "tsmc_n-gram_down.csv"

print("讀取算好的 N-gram 大檔案中...")
up_df = pd.read_csv(up_path)
down_df = pd.read_csv(down_path)

# 2. 設定篩選門檻 (類似查帳的重大性門檻)
# 條件：DF >= 5 (至少出現在 5 篇文章中)，且 TF卡方值必須大於 0
min_df = 5

print("進行卡方值篩選...")
up_filtered = up_df[(up_df['DF'] >= min_df) & (up_df['TF卡方值(保留正負號)'] > 0)]
down_filtered = down_df[(down_df['DF'] >= min_df) & (down_df['TF卡方值(保留正負號)'] > 0)]

# 3. 各取 Top 150 具鑑別力的黃金詞彙
up_top150 = up_filtered.sort_values(by='TF卡方值(保留正負號)', ascending=False).head(150)
down_top150 = down_filtered.sort_values(by='TF卡方值(保留正負號)', ascending=False).head(150)

# 4. 合併成 300 維的特徵字典
top_300_features = pd.concat([up_top150, down_top150])

# 假設第一欄是字詞，我們把它存成一個 List
# 如果你的檔案第一欄有特定的欄位名稱(如 'word' 或 'N-gram')，請把 columns[0] 換成該名稱
word_column = top_300_features.columns[0] 
top_300_words = top_300_features[word_column].tolist()

print("\n成功萃取出 300 個黃金特徵詞！")
print("-" * 40)
print("前 10 個最強烈【看漲】訊號詞：")
print(up_top150[word_column].tolist()[:10])
print("\n 前 10 個最強烈【看跌】訊號詞：")
print(down_top150[word_column].tolist()[:10])
print("-" * 40)

# 將這 300 個詞輸出，方便交接與報告展示
output_path = features_dir / "top_300_features.csv"
top_300_features.to_csv(output_path, index=False)
print(f"檔案已儲存至：{output_path}")
