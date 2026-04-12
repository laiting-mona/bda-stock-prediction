import pandas as pd
import mysql.connector
from pathlib import Path

conn = mysql.connector.connect(
    host='localhost', 
    user='root', 
    password='[mySQL 密碼]', 
    db='bda2026',
    charset='utf8mb4'
)

try:
    # 以程式檔位置為基準
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 進行 sql 查詢，建立 tsmc_dataset 資料表
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS tsmc_prices_mapping")
        cursor.execute(
            """
            CREATE TABLE tsmc_prices_mapping AS
            SELECT 
                trade_date,
                closing_price AS price_0,
                LEAD(closing_price) OVER (ORDER BY trade_date) AS price_1
            FROM stock_prices 
            WHERE company_id = '2330'
            """
        )

        cursor.execute("DROP TABLE IF EXISTS tsmc_dataset")
        cursor.execute(
            """
            CREATE TABLE tsmc_dataset AS
            SELECT 
                t.post_time, 
                t.title, 
                t.content, 
                m.price_0,
                m.price_1,
                CASE 
                    WHEN (m.price_1 - m.price_0) / m.price_0 > 0.015 THEN 1
                    WHEN (m.price_1 - m.price_0) / m.price_0 < -0.015 THEN 0
                    ELSE 2 
                END AS label
            FROM stock_text t
            JOIN tsmc_prices_mapping m ON DATE(t.post_time) = m.trade_date
            WHERE (t.title LIKE '%台積電%' OR t.content LIKE '%台積電%')
            """
        )
    finally:
        cursor.close()

    # 從建立好的資料表讀取
    df = pd.read_sql("SELECT * FROM tsmc_dataset", conn)
    
    print(f"成功讀取 {len(df)} 筆資料！")

    # 儲存成 CSV 格式 (在 data/processed 資料夾)
    save_path = data_dir / "processed" / "tsmc_data.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"tsmc_data.csv 已成功儲存至 {save_path}")

finally:
    conn.close()