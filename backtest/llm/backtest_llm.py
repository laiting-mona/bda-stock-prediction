# -*- coding: utf-8 -*-
"""
Phase 3 移動回測 — LLM 模型（Claude zero-shot prompt）
負責人：若涵
回測區間：2024-10-01 ~ 2025-02-26
說明：無訓練窗口，直接對每篇文章呼叫 Claude API 判斷看漲/看跌
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import anthropic
import time
import json
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────
BASE_DIR   = Path.home() / "Desktop/bda-stock-prediction-main"
DATA_PATH  = BASE_DIR / "data/processed/tsmc_clean_filtered.csv"
OUTPUT_DIR = BASE_DIR / "backtest/results"
CACHE_PATH = OUTPUT_DIR / "llm_predictions_cache.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 參數設定 ──────────────────────────────────────────────
BACKTEST_START       = "2024-10-01"
MIN_ARTICLES         = 5
MAX_ARTICLES_PER_DAY = 30    # 每天最多取 30 篇，節省 API 費用與時間
VOTE_THRESHOLD       = 0.55
LLM_MODEL            = "claude-haiku-4-5-20251001"  # 最快、最便宜的模型
RETRY_TIMES          = 2
RETRY_WAIT           = 3

# ── 讀取資料 ──────────────────────────────────────────────
print("=" * 60, flush=True)
print("Phase 3 Moving Backtest - LLM Model", flush=True)
print("=" * 60, flush=True)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df['post_time'] = pd.to_datetime(df['post_time'])
df['date']      = df['post_time'].dt.date
df = df.sort_values('post_time').reset_index(drop=True)

all_dates      = sorted(df['date'].unique())
backtest_dates = [d for d in all_dates if str(d) >= BACKTEST_START]
print(f"Backtest days : {len(backtest_dates)} ({backtest_dates[0]} ~ {backtest_dates[-1]})", flush=True)

est_calls = min(MAX_ARTICLES_PER_DAY, 150) * len(backtest_dates)
print(f"Est. API calls: ~{est_calls} (capped {MAX_ARTICLES_PER_DAY} articles/day)", flush=True)

# ── 載入快取（斷點續跑）────────────────────────────────────
if CACHE_PATH.exists():
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        pred_cache = json.load(f)
    print(f"Loaded cache  : {len(pred_cache)} predictions", flush=True)
else:
    pred_cache = {}
    print("No cache, starting fresh.", flush=True)

# ── LLM 預測函數 ──────────────────────────────────────────
client = anthropic.Anthropic()

def safe_ascii(text, max_len=300):
    """把文字轉成純 ASCII 安全字串（去掉無法編碼的字元後截斷）"""
    if not isinstance(text, str):
        text = str(text)
    # 先 encode 成 utf-8，再用 latin-1 safe 的方式傳輸
    return text.encode('utf-8', errors='ignore').decode('utf-8')[:max_len]

def llm_predict(row_id, title, content):
    if row_id in pred_cache:
        return pred_cache[row_id]

    t = safe_ascii(title, 100)
    c = safe_ascii(content, 300)

    prompt = (
        "You are analyzing articles about TSMC (Taiwan Semiconductor, 2330).\n"
        "Based on the title and content below, predict whether the stock will go UP (bullish) or DOWN (bearish).\n\n"
        f"Title: {t}\n"
        f"Content: {c}\n\n"
        "Reply with ONLY the digit 1 (bullish/up) or 0 (bearish/down). No other text."
    )

    for attempt in range(RETRY_TIMES):
        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            pred = int(result) if result in ["0", "1"] else -1
            pred_cache[row_id] = pred
            return pred
        except Exception as e:
            err = repr(e)[:120]
            if attempt < RETRY_TIMES - 1:
                print(f"  [Warn] attempt {attempt+1} failed, retrying: {err}", flush=True)
                time.sleep(RETRY_WAIT)
            else:
                print(f"  [Error] skipping article: {err}", flush=True)
                pred_cache[row_id] = -1
                return -1

# ── 移動回測主迴圈 ────────────────────────────────────────
results = []
total_called = 0

for idx, test_date in enumerate(backtest_dates):
    test_df = df[df['date'] == test_date].copy().reset_index(drop=True)

    if len(test_df) < MIN_ARTICLES:
        results.append({'date': test_date, 'signal': 'skip',
                        'trade_return': 0.0,
                        'actual_return': test_df['return_rate'].mean() if len(test_df) > 0 else 0.0})
        print(f"[{test_date}] skip (n={len(test_df)} < {MIN_ARTICLES})", flush=True)
        continue

    # 每天最多取 MAX_ARTICLES_PER_DAY 篇
    if len(test_df) > MAX_ARTICLES_PER_DAY:
        test_df = test_df.sample(MAX_ARTICLES_PER_DAY, random_state=42).reset_index(drop=True)

    preds = []
    for i, row in test_df.iterrows():
        row_id = f"{test_date}_{i}"
        pred = llm_predict(row_id, str(row.get('title', '')), str(row.get('content', '')))
        if pred != -1:
            preds.append(pred)
        total_called += 1

        if total_called % 20 == 0:
            with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(pred_cache, f, ensure_ascii=False)

    if len(preds) < MIN_ARTICLES:
        results.append({'date': test_date, 'signal': 'skip',
                        'trade_return': 0.0,
                        'actual_return': df[df['date'] == test_date]['return_rate'].mean()})
        print(f"[{test_date}] skip (valid={len(preds)} < {MIN_ARTICLES})", flush=True)
        continue

    bullish_ratio = np.mean([p == 1 for p in preds])
    bearish_ratio = np.mean([p == 0 for p in preds])

    if bullish_ratio >= VOTE_THRESHOLD:
        signal, direction = 'buy', 1
    elif bearish_ratio >= VOTE_THRESHOLD:
        signal, direction = 'sell', -1
    else:
        signal, direction = 'skip', 0

    actual_df     = df[df['date'] == test_date]
    actual_return = actual_df['return_rate'].mean()
    actual_label  = int(actual_df['label'].mode()[0])
    trade_return  = direction * actual_return if signal != 'skip' else 0.0

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
    print(f"[{test_date}] {signal:4s}  bull={bullish_ratio:.2f}  "
          f"ret={actual_return:+.4f}  trade={trade_return:+.4f}  n={len(preds)}", flush=True)

# 最後存快取
with open(CACHE_PATH, 'w', encoding='utf-8') as f:
    json.dump(pred_cache, f, ensure_ascii=False)
print(f"\nCache saved: {CACHE_PATH}", flush=True)

# ── 整理結果 ──────────────────────────────────────────────
result_df = pd.DataFrame(results)
result_df['date'] = pd.to_datetime(result_df['date'])

traded     = result_df[result_df['signal'] != 'skip']
win_trades = traded[traded['correct'] == True]

print("\n" + "=" * 60, flush=True)
print("Backtest Summary", flush=True)
print("=" * 60, flush=True)
print(f"Total days   : {len(result_df)}", flush=True)
print(f"Days traded  : {len(traded)} ({len(traded)/len(result_df)*100:.1f}%)", flush=True)
print(f"Days skipped : {len(result_df) - len(traded)}", flush=True)
if len(traded) > 0:
    print(f"Win rate     : {len(win_trades)/len(traded)*100:.1f}%", flush=True)
print(f"Cumul. return: {result_df['trade_return'].sum()*100:.2f}%", flush=True)
print(f"Buy & Hold   : {result_df['actual_return'].sum()*100:.2f}%", flush=True)

# ── 儲存 CSV ──────────────────────────────────────────────
csv_path = OUTPUT_DIR / "llm_backtest_results.csv"
result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nResults saved: {csv_path}", flush=True)

# ── 視覺化 ────────────────────────────────────────────────
result_df['cum_strategy'] = result_df['trade_return'].cumsum() * 100
result_df['cum_bh']       = result_df['actual_return'].cumsum() * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(result_df['date'], result_df['cum_strategy'],
        label='LLM Strategy', color='steelblue', linewidth=2)
ax.plot(result_df['date'], result_df['cum_bh'],
        label='Buy & Hold', color='gray', linewidth=1.5, linestyle='--')
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_title('LLM Backtest - Cumulative Return vs Buy & Hold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "llm_cumulative_return.png", dpi=150)
plt.close()
print("Chart saved: llm_cumulative_return.png", flush=True)

if len(traded) > 0:
    traded_copy = traded.copy()
    traded_copy['month'] = traded_copy['date'].dt.to_period('M').astype(str)
    monthly = traded_copy.groupby('month').size().reset_index(name='count')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(monthly['month'], monthly['count'], color='steelblue')
    ax2.set_title('LLM Monthly Trade Count')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Trade Days')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "llm_monthly_trades.png", dpi=150)
    plt.close()
    print("Chart saved: llm_monthly_trades.png", flush=True)

print("\nDone!", flush=True)
