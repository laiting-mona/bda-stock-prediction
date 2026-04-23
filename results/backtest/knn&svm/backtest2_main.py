"""
回測 v2 — 最終版（sklearn + TF-IDF 版本）
==========================================
與 Phase 2 特徵一致：
  技術指標 6 維 + jieba TF-IDF Chi-Square 100 維
  kNN：TF-IDF 再壓 SVD 30 維，避免維度詛咒
  SVM：TF-IDF 直接用（稠密 100 維）

模型（與 Phase 2 一致）：
  ① kNN  (k=9, euclidean, weights='distance')
  ② SVM  (LinearSVC, C=5.0)

預測目標：D+3，訓練窗口滾動 30 日曆天
門檻：0.5 / 0.6 / 0.65 / 0.7

依賴：pip3 install scikit-learn jieba
"""

import pandas as pd
import numpy as np
import jieba
import warnings
from datetime import timedelta
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

warnings.filterwarnings('ignore')

# ============================================================
# 0. 參數
# ============================================================
BACKTEST_START    = '2024-10-01'
TRAIN_DAYS        = 30
BERT_TRAIN_DAYS   = 90    # BERT 日級別資料每天只有1筆，需要更長窗口
BERT_TRAIN_DAYS   = 90
N_FORWARD         = 3
SIGMA             = 0.015
MIN_ARTICLES      = 5
MIN_TRAIN         = 30
THRESHOLDS        = [0.5, 0.6, 0.65, 0.7]
TFIDF_MAX         = 500     # 每個窗口的詞彙上限
CHI2_K            = 100     # Chi-Square 後保留詞數
SVD_N             = 30      # kNN 用 SVD 壓縮文字維度

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = OUT_DIR

FEAT_PATH  = os.path.join(DATA_DIR, 'tsmc_features_v2.csv')
PRICE_PATH = os.path.join(DATA_DIR, 'tsmc_data.csv')
BERT_META  = os.path.join(DATA_DIR, 'tsmc_bert_day_meta.csv')
BERT_TECH  = os.path.join(DATA_DIR, 'tsmc_bert_day_tech.csv')
BERT_TEXT  = os.path.join(DATA_DIR, 'tsmc_bert_day_text.csv')

COLORS = {'kNN':'#2E86AB','SVM':'#E84855','kNN-BERT':'#3D9970','SVM-BERT':'#FF851B'}
TECH_COLS = ['prev_ret_1d','prev_ret_5d','vol_5d','vol_20d','rsi_14','n_articles']

# ============================================================
# 1. 模型工廠
# ============================================================
def make_knn():
    return KNeighborsClassifier(n_neighbors=7, metric='euclidean', weights='distance')

def make_svm():
    return LinearSVC(C=5.0, max_iter=5000, random_state=42, class_weight='balanced')

def svm_predict_proba(clf, X):
    df_vals = clf.decision_function(X)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(df_vals, -10, 10)))
    return np.column_stack([1-p1, p1])

# ============================================================
# 2. 中文斷詞（只跑一次）
# ============================================================
STOPWORDS = {
    '的','了','在','是','我','有','和','就','不','人','都','一','上','也',
    '很','到','說','要','去','你','會','著','沒有','看','好','這','那','但',
    '從','或','與','及','等','把','被','對','又','以','可','來','他','她',
    '它','們','之','其','所','如','而','則','為','於','已','由','再','能',
    '得','各','每','中','後','前','並','且','因','此','雖','然','即','才',
    '臺','台','灣','公司','股票','股價','投資','市場','表示','指出','今日',
    '今天','昨天','本週','本月','日前','近期','目前','預計',
}

def jieba_tokenize(text):
    if not isinstance(text, str): return ''
    words = jieba.lcut(text)
    return ' '.join([w for w in words if len(w) >= 2 and w not in STOPWORDS])

# ============================================================
# 3. 載入資料
# ============================================================
print("="*65)
print("  回測 v2（TF-IDF + 技術指標，與 Phase 2 一致）")
print("="*65)
print(f"\n資料夾：{DATA_DIR}\n")

print("[1] 載入文章資料...")
df_feat = pd.read_csv(FEAT_PATH)
df_feat['post_time']   = pd.to_datetime(df_feat['post_time'])
df_feat['date']        = df_feat['post_time'].dt.date
df_feat['title_len']   = df_feat['title'].fillna('').str.len()
df_feat['content_len'] = df_feat['content'].fillna('').str.len()

print("[2] jieba 斷詞（僅需執行一次，約 30 秒）...")
df_feat['text_cut'] = (
    df_feat['title'].fillna('') + ' ' + df_feat['content'].fillna('')
).str[:1500].apply(jieba_tokenize)
print(f"  ✓ 斷詞完成，共 {len(df_feat):,} 篇")

print("[3] 載入價格資料...")
df_price = pd.read_csv(PRICE_PATH, usecols=['post_time','price_0'])
df_price['date'] = pd.to_datetime(df_price['post_time']).dt.date
daily_p  = (df_price.groupby('date')
            .agg(price_0=('price_0','first'))
            .reset_index().sort_values('date').reset_index(drop=True))
daily_p['date'] = pd.to_datetime(daily_p['date'])
t_dates   = sorted(daily_p['date'].tolist())
price_map = {str(d.date()): float(p)
             for d,p in zip(daily_p['date'], daily_p['price_0'])}

print("[4] 載入 BERT 資料...")
df_bm = pd.read_csv(BERT_META); df_bm['date'] = pd.to_datetime(df_bm['date'])
df_bt = pd.read_csv(BERT_TECH)
df_bx = pd.read_csv(BERT_TEXT)
bert_arr = df_bx.values.astype(np.float32)
U, S, Vt = np.linalg.svd(bert_arr - bert_arr.mean(0), full_matrices=False)
bert_50  = U[:, :50] * S[:50]
df_bert  = pd.concat([df_bm.reset_index(drop=True),
                      df_bt.reset_index(drop=True),
                      pd.DataFrame(bert_50, columns=[f'bpc_{i}' for i in range(50)])], axis=1)

print(f"\n  文章資料：{len(df_feat):,} 篇  ({len(df_feat['date'].unique())} 天)")
print(f"  價格資料：{len(t_dates)} 交易日")
print(f"  BERT資料：{len(df_bert)} 天")

# ============================================================
# 4. D+3 標籤
# ============================================================
print("\n[5] 計算 D+3 標籤...")

def d3_info(date_str):
    d   = pd.Timestamp(date_str)
    fut = [t for t in t_dates if t > d]
    if len(fut) < N_FORWARD: return None, None, None
    d3  = fut[N_FORWARD-1]; d3s = str(d3.date()); ds = str(d.date())
    p0  = price_map.get(ds); pn = price_map.get(d3s)
    if not p0 or not pn: return d3s, None, None
    ret = (pn-p0)/p0
    lbl = 1 if ret>SIGMA else (0 if ret<-SIGMA else 2)
    return d3s, pn, lbl

df_feat['label_d3'] = df_feat['date'].apply(lambda d: d3_info(str(d))[2])
df_bert['label_d3'] = df_bert['date'].apply(lambda d: d3_info(str(d.date()))[2])
lc = df_feat['label_d3'].value_counts()
print(f"  D+3：漲={lc.get(1,0):,}  跌={lc.get(0,0):,}  中性={lc.get(2,0):,}")

# ============================================================
# 5. 回測日期
# ============================================================
art_dates_all = sorted(df_feat[df_feat['date'] >= pd.Timestamp(BACKTEST_START).date()]['date'].unique())
art_dates     = [d for d in art_dates_all if d3_info(str(d))[2] is not None]
bert_dates_all= sorted(df_bert[df_bert['date'] >= pd.Timestamp(BACKTEST_START)]['date'].dt.date.unique())
bert_dates    = [d for d in bert_dates_all if d3_info(str(d))[2] is not None]

FEAT_BERT = [c for c in df_bert.columns
             if c not in ['date','return_rate','label','label_d3']]

print(f"\n[6] 文章回測日：{len(art_dates)} 天 ({art_dates[0]}→{art_dates[-1]})")
print(f"    BERT回測日：{len(bert_dates)} 天")

# ============================================================
# 6. 工具函數
# ============================================================
def soft_vote(probs, weights):
    w = np.maximum(np.array(weights, float), 1.0)
    return float(np.average(probs, weights=w))

def apply_thr(c, thr):
    if c is None: return None
    if c >= thr:   return 1
    if c <= 1-thr: return 0
    return None

def build_text_features(text_tr, y_tr, text_pr, use_svd=False):
    """
    在訓練集 fit TF-IDF + Chi-Square（+ SVD），transform 預測集
    use_svd=True  → kNN 用（壓縮到 SVD_N 維）
    use_svd=False → SVM 用（保留 CHI2_K 維稠密矩陣）
    回傳 (X_text_tr, X_text_pr)，均為 dense ndarray
    """
    k_actual = min(CHI2_K, len(text_tr) - 1, 50)  # 防止樣本不夠

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX, sublinear_tf=True)
    Xtr   = tfidf.fit_transform(text_tr)
    Xpr   = tfidf.transform(text_pr)

    if Xtr.shape[1] == 0:
        return None, None

    k_actual = min(k_actual, Xtr.shape[1])
    sel = SelectKBest(chi2, k=k_actual)
    Xtr = sel.fit_transform(Xtr, y_tr)
    Xpr = sel.transform(Xpr)

    if use_svd:
        n = min(SVD_N, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
        if n < 2: return None, None
        svd = TruncatedSVD(n_components=n, random_state=42)
        Xtr = svd.fit_transform(Xtr)
        Xpr = svd.transform(Xpr)
    else:
        Xtr = Xtr.toarray()
        Xpr = Xpr.toarray()

    return Xtr, Xpr

# ============================================================
# 7. 文章級別回測（kNN + SVM，含 TF-IDF）
# ============================================================
def run_article_backtest(model_name, make_clf):
    use_svd = model_name.startswith('kNN')
    print(f"\n[7/{model_name}] 文章級別回測（TF-IDF + 技術指標）...")
    recs = []
    for idx, pred_date in enumerate(art_dates):
        pred_dt  = pd.Timestamp(pred_date)
        tr_start = (pred_dt - timedelta(days=BERT_TRAIN_DAYS)).date()
        train_df = df_feat[(df_feat['date'] >= tr_start) &
                           (df_feat['date'] <  pred_date) &
                           (df_feat['label_d3'].isin([0,1]))]
        pred_df  = df_feat[df_feat['date'] == pred_date]
        d3s, d3p, act = d3_info(str(pred_date))
        p0 = price_map.get(str(pred_date))
        ar = ((d3p-p0)/p0) if (p0 and d3p) else None

        rec = {'date':str(pred_date),'n_train':len(train_df),
               'n_articles':len(pred_df),'price_d0':p0,'price_d3':d3p,
               'd3_date':d3s,'actual_d3':act,'actual_return':ar,
               'confidence':None,'skip_reason':None}

        if len(train_df) < MIN_TRAIN or train_df['label_d3'].nunique() < 2:
            rec['skip_reason'] = 'insufficient_training'; recs.append(rec); continue
        if len(pred_df) < MIN_ARTICLES:
            rec['skip_reason'] = 'few_articles'; recs.append(rec); continue

        y_tr       = train_df['label_d3'].astype(int).values
        text_tr    = train_df['text_cut'].tolist()
        text_pr    = pred_df['text_cut'].tolist()

        # ── 文字特徵（TF-IDF）──
        Xtext_tr, Xtext_pr = build_text_features(text_tr, y_tr, text_pr, use_svd=use_svd)

        # ── 技術指標特徵 ──
        Xtech_tr = train_df[TECH_COLS].fillna(0).values
        Xtech_pr = pred_df[TECH_COLS].fillna(0).values
        sc_tech  = StandardScaler()
        Xtech_tr = sc_tech.fit_transform(Xtech_tr)
        Xtech_pr = sc_tech.transform(Xtech_pr)

        # ── 合併特徵 ──
        if Xtext_tr is not None:
            sc_text  = StandardScaler()
            Xtext_tr = sc_text.fit_transform(Xtext_tr)
            Xtext_pr = sc_text.transform(Xtext_pr)
            X_train  = np.hstack([Xtech_tr, Xtext_tr])
            X_pred   = np.hstack([Xtech_pr, Xtext_pr])
        else:
            X_train = Xtech_tr
            X_pred  = Xtech_pr

        # ── 訓練 + 預測 ──
        clf = make_clf()
        clf.fit(X_train, y_tr)
        if hasattr(clf, 'predict_proba'):
            pr = clf.predict_proba(X_pred)[:, 1]
        else:
            pr = svm_predict_proba(clf, X_pred)[:, 1]

        wts  = pred_df['content_len'].fillna(100).values
        conf = soft_vote(pr, wts)
        rec['confidence'] = round(conf, 4)
        rec['prob_mean']  = round(float(pr.mean()), 4)
        rec['prob_std']   = round(float(pr.std()),  4)
        recs.append(rec)

        sym = '↑' if conf>=0.65 else ('↓' if conf<=0.35 else '─')
        tru = '漲' if act==1 else ('跌' if act==0 else '中')
        print(f"  [{idx+1:02d}] {pred_date} n={len(train_df):4d} "
              f"art={len(pred_df):3d} feat={X_train.shape[1]}維 "
              f"conf={conf:.3f}{sym} D+3={tru}")
    return pd.DataFrame(recs)

# ============================================================
# 8. BERT 日級別回測（不變）
# ============================================================
def run_bert_backtest(model_name, make_clf):
    print(f"\n[8/{model_name}] BERT 日級別回測...")
    recs = []
    for idx, pred_date in enumerate(bert_dates):
        pred_dt  = pd.Timestamp(pred_date)
        tr_start = (pred_dt - timedelta(days=BERT_TRAIN_DAYS)).date()
        train_df = df_bert[(df_bert['date'].dt.date >= tr_start) &
                           (df_bert['date'].dt.date <  pred_date) &
                           (df_bert['label_d3'].isin([0,1]))]
        pred_row = df_bert[df_bert['date'].dt.date == pred_date]
        d3s, d3p, act = d3_info(str(pred_date))
        p0 = price_map.get(str(pred_date))
        ar = ((d3p-p0)/p0) if (p0 and d3p) else None
        rec = {'date':str(pred_date),'n_train':len(train_df),'n_articles':1,
               'price_d0':p0,'price_d3':d3p,'d3_date':d3s,'actual_d3':act,
               'actual_return':ar,'confidence':None,'skip_reason':None}
        if len(train_df)<10 or train_df['label_d3'].nunique()<2:
            rec['skip_reason']='insufficient_training'; recs.append(rec); continue
        if len(pred_row)==0:
            rec['skip_reason']='no_data'; recs.append(rec); continue
        X_tr   = train_df[FEAT_BERT].fillna(0).values
        y_tr   = train_df['label_d3'].astype(int).values
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        clf    = make_clf(); clf.fit(X_tr_s, y_tr)
        X_pr_s = sc.transform(pred_row[FEAT_BERT].fillna(0).values)
        pr     = (clf.predict_proba(X_pr_s)[:,1] if hasattr(clf,'predict_proba')
                  else svm_predict_proba(clf, X_pr_s)[:,1])
        conf   = float(pr[0])
        rec['confidence'] = round(conf, 4)
        recs.append(rec)
        sym = '↑' if conf>=0.65 else ('↓' if conf<=0.35 else '─')
        tru = '漲' if act==1 else ('跌' if act==0 else '中')
        print(f"  [{idx+1:02d}] {pred_date} n={len(train_df):3d} conf={conf:.3f}{sym} D+3={tru}")
    return pd.DataFrame(recs)

# ============================================================
# 9. 執行四個模型
# ============================================================
df_results = {
    'kNN':      run_article_backtest('kNN',      make_knn),
    'SVM':      run_article_backtest('SVM',      make_svm),
    'kNN-BERT': run_bert_backtest('kNN-BERT',    make_knn),
    'SVM-BERT': run_bert_backtest('SVM-BERT',    make_svm),
}

# ============================================================
# 10. 績效計算
# ============================================================
print("\n[9] 績效分析...")

def calc_metrics(df_rec, thr, model_name):
    df = df_rec.copy()
    df['pred'] = df['confidence'].apply(lambda c: apply_thr(c, thr))
    total   = len(df)
    traded  = df['pred'].notna().sum()
    eval_df = df[df['pred'].notna() & df['actual_d3'].isin([0,1])].copy()
    if len(eval_df) == 0: return None
    yt = eval_df['actual_d3'].astype(int).values
    yp = eval_df['pred'].astype(int).values
    acc = (yt==yp).mean()
    tp=((yt==1)&(yp==1)).sum(); tn=((yt==0)&(yp==0)).sum()
    fp=((yt==0)&(yp==1)).sum(); fn=((yt==1)&(yp==0)).sum()
    def f1c(p,r): return 2*p*r/(p+r+1e-10)
    f1 = (f1c(tn/(tn+fn+1e-10), tn/(tn+fp+1e-10)) +
          f1c(tp/(tp+fp+1e-10), tp/(tp+fn+1e-10))) / 2
    eval_df['tr']  = eval_df.apply(
        lambda r: r['actual_return'] if r['pred']==1
                  else (-r['actual_return'] if r['pred']==0 else 0), axis=1)
    eval_df['nav'] = (1+eval_df['tr']).cumprod()
    cum = float(eval_df['nav'].iloc[-1]) - 1
    p0f = df.dropna(subset=['price_d0'])['price_d0'].iloc[0]
    pnl = df.dropna(subset=['price_d3'])['price_d3'].iloc[-1]
    bh  = (pnl-p0f)/p0f
    roll_max = eval_df['nav'].cummax()
    mdd = float(((eval_df['nav']-roll_max)/roll_max).min())
    sh  = (eval_df['tr'].mean()/eval_df['tr'].std()*np.sqrt(252)
           if eval_df['tr'].std()>0 else 0.0)
    return {'model':model_name,'threshold':thr,'total_days':total,
            'trade_days':traded,'trade_rate':round(traded/total,4),
            'eval_days':len(eval_df),'accuracy':round(float(acc),4),
            'macro_f1':round(float(f1),4),
            'win_rate':round(float((yt==yp).sum()/len(eval_df)),4),
            'cum_return':round(cum,4),'bh_return':round(bh,4),
            'max_drawdown':round(mdd,4),'sharpe':round(sh,4),
            'cm':f'TN={tn} FP={fp} FN={fn} TP={tp}','_df':eval_df}

all_m=[]; eval_dfs={}
for mn, df_rec in df_results.items():
    for thr in THRESHOLDS:
        m = calc_metrics(df_rec, thr, mn)
        if m:
            eval_dfs[(mn,thr)] = m.pop('_df')
            all_m.append(m)
df_perf = pd.DataFrame(all_m)

print("\n"+"="*88)
print(f"{'模型':<10}{'門檻':>5}{'出手率':>8}{'出手天':>7}{'準確率':>8}"
      f"{'勝率':>7}{'累積報酬':>10}{'B&H':>8}{'MaxDD':>8}{'Sharpe':>7}")
print("-"*88)
for _,r in df_perf.iterrows():
    print(f"{r['model']:<10}{r['threshold']:>5.2f} {r['trade_rate']:>7.0%} "
          f"{r['trade_days']:>6}  {r['accuracy']:>7.0%} {r['win_rate']:>6.0%} "
          f"{r['cum_return']:>9.1%}  {r['bh_return']:>6.1%}  "
          f"{r['max_drawdown']:>6.1%}  {r['sharpe']:>6.2f}")
print("="*88)

# ============================================================
# 11. 儲存 CSV
# ============================================================
print("\n[10] 儲存 CSV...")
for mn, df_rec in df_results.items():
    df_out = df_rec.copy()
    for thr in THRESHOLDS:
        df_out[f'pred_{int(thr*100)}'] = df_out['confidence'].apply(lambda c: apply_thr(c,thr))
    path = os.path.join(OUT_DIR, f'{mn.lower().replace("-","_")}_daily_records.csv')
    df_out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  ✓ {os.path.basename(path)}")
df_perf.to_csv(os.path.join(OUT_DIR,'performance_by_threshold.csv'),
               index=False, encoding='utf-8-sig')
print("  ✓ performance_by_threshold.csv")

# ============================================================
# 12. 視覺化
# ============================================================
print("\n[11] 生成圖表...")
plt.rcParams['axes.unicode_minus'] = False
models_ord = ['kNN','SVM','kNN-BERT','SVM-BERT']

# 圖1：累積報酬
fig, axes = plt.subplots(2, 2, figsize=(15,10))
fig.suptitle('Cumulative Return by Threshold (D+3, TF-IDF + Tech)',
             fontsize=14, fontweight='bold')
for ax, thr in zip(axes.flat, THRESHOLDS):
    bh_vals = df_perf[df_perf['threshold']==thr]['bh_return']
    bh = float(bh_vals.mean()) if len(bh_vals)>0 else 0
    for mn in models_ord:
        key = (mn, thr)
        if key not in eval_dfs: continue
        ev = eval_dfs[key].copy().sort_values('date')
        ev['dt'] = pd.to_datetime(ev['date'])
        cum = (1+ev['tr']).cumprod()-1
        ax.plot(ev['dt'], cum*100, color=COLORS[mn], linewidth=2,
                marker='o', markersize=3.5,
                label=f'{mn} ({int(ev["pred"].notna().sum())}d, {float(cum.iloc[-1]):.1%})')
    ax.axhline(y=bh*100, color='gray', linestyle='--', lw=1.5, label=f'B&H ({bh:.1%})')
    ax.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax.set_title(f'Threshold = {thr}', fontsize=11)
    ax.set_ylabel('Cumulative Return (%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'cumulative_returns.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ cumulative_returns.png")

# 圖2：門檻績效對比
fig, axes = plt.subplots(1, 3, figsize=(16,5))
fig.suptitle('Performance by Threshold: 4 Models', fontsize=13, fontweight='bold')
metrics_ = [('accuracy','Accuracy'),('win_rate','Win Rate'),('trade_rate','Trade Rate')]
x = np.arange(len(THRESHOLDS)); w = 0.18
for ax,(mk,ml) in zip(axes, metrics_):
    for i,mn in enumerate(models_ord):
        vals = []
        for thr in THRESHOLDS:
            row = df_perf[(df_perf['model']==mn)&(df_perf['threshold']==thr)]
            vals.append(float(row[mk].values[0]) if len(row)>0 else 0)
        offset = (i-1.5)*w
        bars = ax.bar(x+offset, [v*100 for v in vals], w,
                      label=mn, color=COLORS[mn], alpha=0.8)
        for bar,val in zip(bars,vals):
            if val>0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5, f'{val:.0%}',
                        ha='center', va='bottom', fontsize=7)
    ax.set_title(ml, fontsize=11); ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in THRESHOLDS])
    ax.set_xlabel('Threshold'); ax.set_ylabel('%')
    ax.set_ylim(0,115); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'threshold_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ threshold_comparison.png")

# 圖3：信心分布
fig, axes = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle('Confidence Score Distribution (Article-level, TF-IDF+Tech)', fontsize=13)
for ax,(mn,df_rec) in zip(axes,[('kNN',df_results['kNN']),('SVM',df_results['SVM'])]):
    confs = df_rec['confidence'].dropna()
    ax.hist(confs, bins=20, color=COLORS[mn], alpha=0.7, edgecolor='white')
    for thr in THRESHOLDS:
        ax.axvline(x=thr,     color='red',  linestyle='--', lw=1.2, alpha=0.7)
        ax.axvline(x=1.0-thr, color='blue', linestyle='--', lw=1.2, alpha=0.7)
    ax.set_title(f'{mn} (n={len(confs)}, μ={confs.mean():.3f}, σ={confs.std():.3f})')
    ax.set_xlabel('P(Bullish)'); ax.set_ylabel('Count'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'confidence_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ confidence_distribution.png")

# 圖4：混淆矩陣
best_thr = float(df_perf.sort_values('accuracy', ascending=False)['threshold'].iloc[0])
fig, axes = plt.subplots(1, 4, figsize=(16,4))
fig.suptitle(f'Confusion Matrices at Best Threshold={best_thr}', fontsize=12)
for ax,mn in zip(axes, models_ord):
    row = df_perf[(df_perf['model']==mn)&(df_perf['threshold']==best_thr)]
    if len(row)==0: ax.axis('off'); continue
    cm_str = row.iloc[0]['cm']
    vals   = {k:int(v) for k,v in [x.split('=') for x in cm_str.split()]}
    cm     = np.array([[vals['TN'],vals['FP']],[vals['FN'],vals['TP']]])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred↓','Pred↑']); ax.set_yticklabels(['True↓','True↑'])
    ax.set_title(f'{mn}\nAcc={row.iloc[0]["accuracy"]:.1%}  F1={row.iloc[0]["macro_f1"]:.3f}')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=13,
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ confusion_matrices.png")

# 圖5：出手頻率
fig, axes = plt.subplots(2, 1, figsize=(16,8))
fig.suptitle('Trade Frequency per Day (TF-IDF + Tech Features)', fontsize=13)
for ax,(mn,df_rec) in zip(axes,[('kNN',df_results['kNN']),('SVM',df_results['SVM'])]):
    dates = [str(d) for d in art_dates]
    cts   = [sum(1 for t in THRESHOLDS if apply_thr(r['confidence'],t) is not None)
             for _,r in df_rec.iterrows()]
    cols  = [COLORS[mn] if c>0 else '#dddddd' for c in cts]
    ax.bar(range(len(dates)), cts, color=cols, alpha=0.85)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=60, ha='right', fontsize=6)
    ax.set_title(f'{mn}: Thresholds Active per Day')
    ax.set_ylabel('# Thresholds'); ax.set_ylim(0,4.8)
    ax.axhline(y=4, color='green', linestyle='--', lw=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for i,(_,r) in enumerate(df_rec.iterrows()):
        if r['actual_d3']==1:   ax.text(i,0.1,'↑',ha='center',fontsize=6,color='green')
        elif r['actual_d3']==0: ax.text(i,0.1,'↓',ha='center',fontsize=6,color='red')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'trade_frequency.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ trade_frequency.png")

# ============================================================
# 13. 錯誤案例
# ============================================================
print("\n[12] 錯誤案例分析（門檻0.65）...")
err_rows = []
for mn,df_rec in df_results.items():
    for _,row in df_rec.iterrows():
        pred = apply_thr(row.get('confidence'), 0.65)
        if pred is None or row['actual_d3'] not in [0,1]: continue
        if int(pred)!=int(row['actual_d3']):
            err_rows.append({'model':mn,'date':row['date'],
                             'confidence':row.get('confidence'),
                             'pred':pred,'actual_d3':row['actual_d3'],
                             'return_pct':round(row['actual_return']*100,2)
                             if row['actual_return'] else None,
                             'n_articles':row['n_articles']})
df_err = pd.DataFrame(err_rows).sort_values('confidence', ascending=False)
if len(df_err)>0:
    df_err.to_csv(os.path.join(OUT_DIR,'error_cases.csv'), index=False, encoding='utf-8-sig')
    print(f"  ✓ error_cases.csv ({len(df_err)} 筆)")

# ============================================================
# 14. 最終摘要
# ============================================================
print("\n"+"="*65)
print("  輸出完成 → backtest2/")
files_ = ['knn_daily_records.csv','svm_daily_records.csv',
          'knn_bert_daily_records.csv','svm_bert_daily_records.csv',
          'performance_by_threshold.csv','error_cases.csv',
          'cumulative_returns.png','threshold_comparison.png',
          'confidence_distribution.png','confusion_matrices.png','trade_frequency.png']
for f in files_:
    fp = os.path.join(OUT_DIR, f)
    if os.path.exists(fp):
        print(f"  ✓ {f} ({os.path.getsize(fp)//1024}KB)")
    else:
        print(f"  ✗ {f}")
print("="*65)
