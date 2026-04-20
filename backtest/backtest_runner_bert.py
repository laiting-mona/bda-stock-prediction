"""
Backtest runner for BERT models: RF, NB (Gaussian), XGBoost
Uses pre-computed BERT embeddings (data/processed/tsmc_bert_features.npy)
Output: results/backtest/bert/
"""
from pathlib import Path
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results" / "backtest" / "bert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RF":      ROOT / "models/RF/RF_bert_model.pkl",
    "NB":      ROOT / "models/naive-bayes/NB_bert_model.pkl",
    "XGBoost": ROOT / "models/XGBoost/XGBoost_bert_model.pkl",
}
WINDOWS             = [15, 30, 45, 60]
THRESHOLDS          = [0.50, 0.60, 0.65]
MIN_ART             = 3
CONSENSUS_THRESHOLD = 0.80
SPLIT               = 0.8
COLORS              = {"RF": "steelblue", "NB": "darkorange", "XGBoost": "forestgreen"}


def load_test_data():
    bert_path = DATA_DIR / "processed" / "tsmc_bert_features.npy"
    if not bert_path.exists():
        raise FileNotFoundError(
            f"BERT features not found: {bert_path}\n"
            "Run: python scripts/bert_features/extract_bert.py"
        )
    X_all = np.load(bert_path)
    df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features.csv", encoding="utf-8-sig")
    df["post_time"] = pd.to_datetime(df["post_time"])
    n = int(len(df) * SPLIT)
    X_test   = X_all[n:]
    test_df  = df.iloc[n:].reset_index(drop=True)
    print(f"  Test: {len(test_df)} articles | "
          f"{test_df['post_time'].min().date()} → {test_df['post_time'].max().date()}")
    return X_test, test_df


def run_inference(model_name, X_test, test_df):
    mp = MODELS[model_name]
    if not mp.exists():
        raise FileNotFoundError(
            f"BERT model not found: {mp}\n"
            f"Run: python models/{'RF' if 'RF' in model_name else model_name.replace('-','/')}/{model_name}_bert_train.py"
        )
    model = pickle.load(open(mp, "rb"))
    prob  = model.predict_proba(X_test)
    out   = test_df[["post_time", "label", "return_rate", "title"]].copy()
    out["prob_up"]    = prob[:, 1]
    out["confidence"] = np.maximum(prob[:, 1], 1 - prob[:, 1])
    out["pred"]       = (prob[:, 1] >= 0.5).astype(int)
    out["trade_date"] = out["post_time"].dt.date
    return out


def build_daily(pred_df, threshold):
    rows = []
    for date, grp in pred_df.groupby("trade_date"):
        n        = len(grp)
        mean_p   = float(grp["prob_up"].mean())
        conf     = max(mean_p, 1 - mean_p)
        true_lbl = int(grp["label"].mode().iloc[0])
        ret      = float(grp["return_rate"].iloc[0])

        up_vote_ratio = float((grp["pred"] == 1).mean())
        is_extreme = (up_vote_ratio > CONSENSUS_THRESHOLD or
                      up_vote_ratio < (1 - CONSENSUS_THRESHOLD))

        if n < MIN_ART:
            sig  = None
            skip = f"n<{MIN_ART}"
        elif conf < threshold:
            sig  = None
            skip = f"conf<{threshold}"
        else:
            sig  = 1 if mean_p >= 0.5 else 0
            skip = None
            if is_extreme:
                skip = f"extreme_consensus({up_vote_ratio:.0%})"
                sig  = 1 if mean_p >= 0.5 else 0

        rows.append({
            "date": date, "n_articles": n,
            "mean_prob_up": round(mean_p, 4), "confidence": round(conf, 4),
            "up_vote_ratio": round(up_vote_ratio, 4),
            "is_extreme_consensus": is_extreme,
            "signal": sig, "skip_reason": skip,
            "true_label": true_lbl, "return_rate": ret,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def simulate(daily_df, window):
    avail = len(daily_df)
    used  = min(window, avail)
    d     = daily_df.head(used).copy()
    d["port_ret"] = d.apply(
        lambda r: (r["return_rate"] if r["signal"] == 1 else -r["return_rate"])
                  if r["signal"] is not None else 0.0,
        axis=1,
    )
    d["cum_port"] = (1 + d["port_ret"]).cumprod()
    d["cum_bnh"]  = (1 + d["return_rate"]).cumprod()

    traded    = d[d["signal"].notna()]
    n_trade   = len(traded)
    n_extreme = int(d.get("is_extreme_consensus", pd.Series(False, index=d.index)).sum())
    win_rate  = float((traded["signal"] == traded["true_label"]).mean()) if n_trade else float("nan")
    tot_ret   = float(d["cum_port"].iloc[-1] - 1) if used else 0.0
    bnh_ret   = float(d["cum_bnh"].iloc[-1] - 1) if used else 0.0
    roll_max  = d["cum_port"].cummax()
    max_dd    = float(((d["cum_port"] - roll_max) / roll_max).min()) if used else 0.0
    std = d["port_ret"].std()
    sharpe = float(d["port_ret"].mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    if n_trade:
        acc = float(accuracy_score(traded["true_label"], traded["signal"]))
        f1  = float(f1_score(traded["true_label"], traded["signal"],
                             average="macro", zero_division=0))
    else:
        acc = f1 = float("nan")

    if "is_extreme_consensus" in d.columns:
        d_no_ext = d.copy()
        d_no_ext.loc[d_no_ext["is_extreme_consensus"], "port_ret"] = 0.0
        d_no_ext["cum_port_no_ext"] = (1 + d_no_ext["port_ret"]).cumprod()
        ret_no_ext = float(d_no_ext["cum_port_no_ext"].iloc[-1] - 1) if used else 0.0
        traded_no_ext = traded[~traded["is_extreme_consensus"]]
        win_no_ext = float((traded_no_ext["signal"] == traded_no_ext["true_label"]).mean()) \
            if len(traded_no_ext) else float("nan")
    else:
        ret_no_ext = float("nan")
        win_no_ext = float("nan")

    return {
        "window_requested": window, "window_used": used,
        "n_total": used, "n_trade": n_trade, "n_skip": used - n_trade,
        "n_extreme_consensus": n_extreme,
        "win_rate": win_rate, "accuracy": acc, "macro_f1": f1,
        "total_ret": tot_ret, "bnh_ret": bnh_ret,
        "total_ret_no_extreme": ret_no_ext,
        "win_rate_no_extreme": win_no_ext,
        "max_dd": max_dd, "sharpe": sharpe,
    }, d


def plot_cumulative(sim_map, window, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bnh_done = False
    for mn, df in sim_map.items():
        dates = pd.to_datetime(df["date"].astype(str))
        ax.plot(dates, df["cum_port"] * 100 - 100,
                label=f"{mn}(BERT)", color=COLORS.get(mn), lw=1.8)
        if not bnh_done:
            ax.plot(dates, df["cum_bnh"] * 100 - 100,
                    label="Buy & Hold", color="gray", ls="--", lw=1.5)
            bnh_done = True
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"BERT Cumulative Return — {window}-Day Window (threshold=0.50)")
    ax.set_ylabel("Return (%)"); ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, fontsize=8); ax.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_monthly_heatmap(daily_list, out_path):
    combined = pd.concat(daily_list, ignore_index=True)
    combined["ym"] = pd.to_datetime(combined["date"].astype(str)).dt.to_period("M")
    combined["traded"] = combined["signal"].notna().astype(int)
    pivot = combined.groupby(["model", "ym"])["traded"].sum().unstack(fill_value=0)
    pivot.columns = [str(c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(13, 3))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("BERT Monthly Trade Signal Count (threshold=0.50)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    print("=" * 60)
    print("BERT Models Backtest")
    print("=" * 60)
    print("Loading BERT test data...")
    X_test, test_df = load_test_data()
    print(f"  Unique trading dates: {test_df['post_time'].dt.date.nunique()}")

    summary, daily_thr050 = [], {}

    for mn in MODELS:
        print(f"\n[{mn}] Running BERT inference...")
        pred = run_inference(mn, X_test, test_df)

        for thr in THRESHOLDS:
            daily = build_daily(pred, thr)
            if thr == 0.50:
                d = daily.copy(); d["model"] = mn
                daily_thr050[mn] = d

            for w in WINDOWS:
                m, sim = simulate(daily, w)
                m.update({"model": mn, "threshold": thr})
                summary.append(m)
                win_str = f"{m['win_rate']:.3f}" if not np.isnan(m["win_rate"]) else "n/a"
                print(f"  thr={thr}  W={w:2d}(used={m['window_used']:2d}) | "
                      f"trades={m['n_trade']:2d} | ret={m['total_ret']:+.2%} | "
                      f"bnh={m['bnh_ret']:+.2%} | win={win_str}")
                sim.to_csv(OUT_DIR / f"{mn}_thr{int(thr*100)}_W{w}_daily.csv",
                           index=False, encoding="utf-8-sig")

    sumdf = pd.DataFrame(summary)
    for col in ["win_rate","accuracy","macro_f1","total_ret","bnh_ret","max_dd","sharpe"]:
        sumdf[col] = sumdf[col].round(4)
    sumdf.to_csv(OUT_DIR / "backtest_bert_summary.csv", index=False, encoding="utf-8-sig")

    for w in WINDOWS:
        sim_map = {}
        for mn in MODELS:
            _, sim = simulate(daily_thr050[mn], w)
            sim_map[mn] = sim
        plot_cumulative(sim_map, w, OUT_DIR / f"bert_cumulative_return_{w}d.png")

    plot_monthly_heatmap(list(daily_thr050.values()),
                         OUT_DIR / "bert_monthly_signal_heatmap.png")

    print("\n" + "=" * 70)
    print("BERT SUMMARY  (threshold=0.50)")
    print("=" * 70)
    disp = sumdf[sumdf["threshold"] == 0.50][
        ["model", "window_requested", "window_used", "n_trade",
         "win_rate", "total_ret", "bnh_ret", "max_dd", "sharpe"]
    ]
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
