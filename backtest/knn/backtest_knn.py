"""
Phase 3 移動回測 — kNN 模型（修正版）
修正內容：
  1. VOTE_THRESHOLD 從 0.55 提高至 0.70（降低出手率）
  2. 回測日期改以 tsmc_clean.csv 為準（含中性日，共 96 天）
負責人：若涵
回測區間：2024-10-01 ~ 2025-02-26
訓練窗口：前 30 天
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR      = Path.home() / "Desktop/bda-stock-prediction-main"
FEAT_PATH     = BASE_DIR / "data/processed/tsmc_features.csv"
CLEAN_PATH    = BASE_DIR / "data/processed/tsmc_clean.csv"
OUTPUT_DIR    = BASE_DIR / "backtest/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_START = "2024-10-01"
TRAIN_DAYS     = 30
MIN_ARTICLES   = 5
VOTE_THRESHOLD = 0.70
K              = 9

print("=" * 60)
print("Phase 3 移動回測 — kNN 模型（修正版）")
print("=" * 60)

feat_df = pd.read_csv(FEAT_PATH)
feat_df['post_time'] = pd.to_datetime(feat_df['post_time'])
feat_df['date']      = feat_df['post_time'].dt.date
feat_df = feat_df.sort_values('post_time').reset_index(drop=True)

clean_df = pd.read_csv(CLEAN_PATH, encoding='utf-8')
clean_df['post_time'] = pd.to_datetime(clean_df['post_time'])
clean_df['date']      = clean_df['post_time'].dt.date
clean_df = clean_df.sort_values('post_time').reset_index(drop=True)

all_dates      = sorted(clean_df['date'].unique())
backtest_dates = [d for d in all_dates if str(d) >= BACKTEST_START]
feat_dates     = set(feat_df['date'].unique())

print(f"✓ 回測總天數    : {len(backtest_dates)} 天（{backtest_dates[0]} ~ {backtest_dates[-1]}）")
print(f"✓ 有特徵的天數  : {sum(1 for d in backtest_dates if d in feat_dates)} 天")
print(f"✓ 中性日（跳過）: {sum(1 for d in backtest_dates if d not in feat_dates)} 天")

drop_cols    = ['post_time', 'title', 'content', 'text', 'date',
                'price_0', 'price_1', 'return_rate', 'abs_return_rate', 'label']
feature_cols = [c for c in feat_df.columns if c not in drop_cols]
print(f"✓ 特徵數        : {len(feature_cols)} 個")
print(f"✓ VOTE_THRESHOLD: {VOTE_THRESHOLD}")

results = []

for test_date in backtest_dates:
    day_clean     = clean_df[clean_df['date'] == test_date]
    actual_return = day_clean['return_rate'].mean() if len(day_clean) > 0 else 0.0

    test_feat = feat_df[feat_df['date'] == test_date].copy()

    if len(test_feat) < MIN_ARTICLES:
        results.append({'date': test_date, 'signal': 'skip',
                        'reason': '中性日或文章不足',
                        'trade_return': 0.0,
                        'actual_return': round(actual_return, 4)})
        continue

    train_end   = pd.Timestamp(test_date)
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    train_df = feat_df[(feat_df['post_time'] >= train_start) &
                       (feat_df['post_time'] <  train_end)].copy()

    if len(train_df) < 10 or len(np.unique(train_df['label'].values)) < 2:
        results.append({'date': test_date, 'signal': 'skip',
                        'reason': '訓練資料不足',
                        'trade_return': 0.0,
                        'actual_return': round(actual_return, 4)})
        continue

    X_train   = train_df[feature_cols].values
    y_train   = train_df['label'].values
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=min(K, len(X_train)),
                               metric='euclidean', weights='distance', n_jobs=-1)
    knn.fit(X_train_s, y_train)

    X_test_s      = scaler.transform(test_feat[feature_cols].values)
    preds         = knn.predict(X_test_s)
    bullish_ratio = np.mean(preds == 1)
    bearish_ratio = np.mean(preds == 0)

    if bullish_ratio >= VOTE_THRESHOLD:
        signal, direction = 'buy', 1
    elif bearish_ratio >= VOTE_THRESHOLD:
        signal, direction = 'sell', -1
    else:
        signal, direction = 'skip', 0

    actual_label = int(test_feat['label'].mode()[0])
    trade_return = direction * actual_return if signal != 'skip' else 0.0

    results.append({
        'date'          : test_date,
        'signal'        : signal,
        'direction'     : direction,
        'bullish_ratio' : round(bullish_ratio, 3),
        'bearish_ratio' : round(bearish_ratio, 3),
        'n_articles'    : len(preds),
        'actual_label'  : actual_label,
        'actual_return' : round(actual_return, 4),
        'trade_return'  : round(trade_return, 4),
        'correct'       : (direction == 1 and actual_label == 1) or
                          (direction == -1 and actual_label == 0)
                          if signal != 'skip' else None
    })

result_df = pd.DataFrame(results)
result_df['date'] = pd.to_datetime(result_df['date'])
traded     = result_df[result_df['signal'] != 'skip']
win_trades = traded[traded['correct'] == True]

print("\n" + "=" * 60)
print("回測結果摘要")
print("=" * 60)
print(f"總天數        : {len(result_df)}")
print(f"實際出手天數  : {len(traded)}（{len(traded)/len(result_df)*100:.1f}%）")
print(f"跳過天數      : {len(result_df) - len(traded)}")
if len(traded) > 0:
    print(f"勝率          : {len(win_trades)/len(traded)*100:.1f}%")
print(f"累積報酬率    : {result_df['trade_return'].sum()*100:.2f}%")
print(f"買進持有報酬率: {result_df['actual_return'].sum()*100:.2f}%")

csv_path = OUTPUT_DIR / "knn_backtest_results.csv"
result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n✓ 結果已儲存：{csv_path}")

result_df['cum_strategy'] = result_df['trade_return'].cumsum() * 100
result_df['cum_bh']       = result_df['actual_return'].cumsum() * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(result_df['date'], result_df['cum_strategy'],
        label='kNN Strategy', color='steelblue', linewidth=2)
ax.plot(result_df['date'], result_df['cum_bh'],
        label='Buy & Hold', color='gray', linewidth=1.5, linestyle='--')
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_title('kNN Backtest (v2) - Cumulative Return vs Buy & Hold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "knn_cumulative_return.png", dpi=150)
plt.close()

if len(traded) > 0:
    traded_copy = traded.copy()
    traded_copy['month'] = traded_copy['date'].dt.to_period('M').astype(str)
    monthly = traded_copy.groupby('month').size().reset_index(name='count')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(monthly['month'], monthly['count'], color='steelblue')
    ax2.set_title('kNN Monthly Trade Count (v2)')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Trade Days')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_monthly_trades.png", dpi=150)
    plt.close()

print("✓ 圖表已儲存\n全部完成！")
