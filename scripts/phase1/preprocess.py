from pathlib import Path
import pandas as pd

LABEL_UP_THRESHOLD = 0.015
LABEL_DOWN_THRESHOLD = -0.015

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"
raw_path = data_dir / "processed" / "tsmc_data.csv"
clean_path = data_dir / "processed" / "tsmc_clean.csv"
clean_filtered_path = data_dir / "processed" / "tsmc_clean_filtered.csv"

df = pd.read_csv(raw_path, encoding="utf-8-sig")


# 轉換時間與價格欄位型別，無法轉換的值會變成 NaN
df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")
df["price_0"] = pd.to_numeric(df["price_0"], errors="coerce")
df["price_1"] = pd.to_numeric(df["price_1"], errors="coerce")

# 文字欄位做基本清理，避免空值與前後空白
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["content"] = df["content"].fillna("").astype(str).str.strip()

# 建立三分類標籤：1=漲、0=跌、2=中性
df["return_rate"] = (df["price_1"] - df["price_0"]) / df["price_0"]
df["label"] = 2
df.loc[df["return_rate"] > LABEL_UP_THRESHOLD, "label"] = 1
df.loc[df["return_rate"] < LABEL_DOWN_THRESHOLD, "label"] = 0

# 依時間排序後輸出完整清理資料
df = df.sort_values("post_time").reset_index(drop=True)
df.to_csv(clean_path, index=False, encoding="utf-8-sig")

# 排除 label=2
df_filtered = df[df["label"] != 2].copy()
df_filtered.to_csv(clean_filtered_path, index=False, encoding="utf-8-sig")

print(f"前處理完成：{len(df)} 筆 -> {clean_path}")
print(f"過濾 label=2 後：{len(df_filtered)} 筆 -> {clean_filtered_path}")