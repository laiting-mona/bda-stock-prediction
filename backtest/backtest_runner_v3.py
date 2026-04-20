"""
Backtest runner v3: RF_v3, NB_v3, XGBoost_v3, Ensemble_v3
Key v3 changes vs v2:
  - RF uses stored optimal_threshold (not 0.5) for signal direction
  - NB uses ComplementNB → wider prob spread → trades at thr=0.60
  - Ensemble uses weighted soft voting (weights ∝ CV F1)
Output: results/backtest/v3/
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


# ── Wrapper class stubs needed by pickle deserialization ──────────────────────
class NBv3Wrapper:
    def __init__(self, model, n_tfidf, cv_f1):
        self.model = model; self.n_tfidf = n_tfidf; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X[:, :self.n_tfidf])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class RFv3Wrapper:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class XGBv3Wrapper:
    def __init__(self, model, cv_f1):
        self.model = model; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class WeightedSoftVotingEnsemble:
    def __init__(self, named_models, weights):
        self.named_models = named_models; self.weights = np.array(weights)
    def predict_proba(self, X):
        probs = np.vstack([m.predict_proba(X)[:, 1] for _, m in self.named_models])
        avg   = (probs * self.weights[:, None]).sum(axis=0)
        return np.column_stack([1 - avg, avg])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results" / "backtest" / "v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RF_v3":       ROOT / "models/RF/RF_v3_model.pkl",
    "NB_v3":       ROOT / "models/naive-bayes/NB_v3_model.pkl",
    "XGBoost_v3":  ROOT / "models/XGBoost/XGBoost_v3_model.pkl",
    "Ensemble_v3": ROOT / "models/ensemble/ensemble_v3_model.pkl",
}
TECH_COLS           = ["prev_ret_1d", "prev_ret_5d", "vol_5d", "vol_20d", "rsi_14", "n_articles"]
WINDOWS             = [15, 30, 45, 60]
THRESHOLDS          = [0.50, 0.60, 0.65]
MIN_ART             = 3
CONSENSUS_THRESHOLD = 0.80
SPLIT               = 0.8
COLORS              = {"RF_v3": "steelblue", "NB_v3": "darkorange",
                       "XGBoost_v3": "forestgreen", "Ensemble_v3": "purple"}


def load_test_data():
    feat_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features_v2.csv", encoding="utf-8-sig")
    vec_df  = pd.read_csv(DATA_DIR / "processed" / "tsmc_vector_space_v2.csv", encoding="utf-8-sig")
    feat_df["post_time"] = pd.to_datetime(feat_df["post_time"])
    X_tech  = feat_df[TECH_COLS].fillna(0).values
    X_tfidf = vec_df.values
    X_all   = np.hstack([X_tfidf, X_tech])
    n       = int(len(X_all) * SPLIT)
    X_test  = X_all[n:]
    test_df = feat_df.iloc[n:].reset_index(drop=True)
    print(f"  Test: {len(test_df)} articles | "
          f"{test_df['post_time'].min().date()} -> {test_df['post_time'].max().date()}")
    print(f"  Unique trading dates: {test_df['post_time'].dt.date.nunique()}")
    return X_test, test_df


def run_inference(model_name, model, X_test, test_df):
    # For RF_v3, use its stored optimal_threshold for pred; raw prob_up unchanged
    opt_thr = getattr(model, "optimal_threshold", 0.5)
    prob    = model.predict_proba(X_test)
    out     = test_df[["post_time", "label", "return_rate", "title"]].copy()
    out["prob_up"]    = prob[:, 1]
    out["confidence"] = np.maximum(prob[:, 1], 1 - prob[:, 1])
    out["pred"]       = (prob[:, 1] >= opt_thr).astype(int)
    out["trade_date"] = out["post_time"].dt.date
    return out, opt_thr


def build_daily(pred_df, threshold, model_signal_thr=0.5):
    rows = []
    for date, grp in pred_df.groupby("trade_date"):
        n        = len(grp)
        mean_p   = float(grp["prob_up"].mean())
        conf     = max(mean_p, 1 - mean_p)
        true_lbl = int(grp["label"].mode().iloc[0])
        ret      = float(grp["return_rate"].iloc[0])

        up_vote_ratio = float((grp["pred"] == 1).mean())
        is_extreme    = (up_vote_ratio > CONSENSUS_THRESHOLD or
                         up_vote_ratio < (1 - CONSENSUS_THRESHOLD))

        if n < MIN_ART:
            sig  = None
            skip = f"n<{MIN_ART}"
        elif conf < threshold:
            sig  = None
            skip = f"conf<{threshold}"
        else:
            sig  = 1 if mean_p >= model_signal_thr else 0
            skip = None
            if is_extreme:
                skip = f"extreme_consensus({up_vote_ratio:.0%})"
                sig  = 1 if mean_p >= model_signal_thr else 0

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
    n_extreme = int(d["is_extreme_consensus"].sum()) if "is_extreme_consensus" in d.columns else 0
    win_rate  = float((traded["signal"] == traded["true_label"]).mean()) if n_trade else float("nan")
    tot_ret   = float(d["cum_port"].iloc[-1] - 1) if used else 0.0
    bnh_ret   = float(d["cum_bnh"].iloc[-1] - 1) if used else 0.0
    roll_max  = d["cum_port"].cummax()
    max_dd    = float(((d["cum_port"] - roll_max) / roll_max).min()) if used else 0.0
    std    = d["port_ret"].std()
    sharpe = float(d["port_ret"].mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    if n_trade:
        acc = float(accuracy_score(traded["true_label"], traded["signal"]))
        f1  = float(f1_score(traded["true_label"], traded["signal"],
                             average="macro", zero_division=0))
    else:
        acc = f1 = float("nan")

    d_no_ext = d.copy()
    d_no_ext.loc[d_no_ext["is_extreme_consensus"], "port_ret"] = 0.0
    d_no_ext["cum_port_no_ext"] = (1 + d_no_ext["port_ret"]).cumprod()
    ret_no_ext = float(d_no_ext["cum_port_no_ext"].iloc[-1] - 1) if used else 0.0
    traded_no_ext = traded[~traded["is_extreme_consensus"]]
    win_no_ext = float((traded_no_ext["signal"] == traded_no_ext["true_label"]).mean()) \
        if len(traded_no_ext) else float("nan")

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
                label=mn, color=COLORS.get(mn), lw=1.8)
        if not bnh_done:
            ax.plot(dates, df["cum_bnh"] * 100 - 100,
                    label="Buy & Hold", color="gray", ls="--", lw=1.5)
            bnh_done = True
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"v3 Cumulative Return — {window}-Day Window (threshold=0.50)")
    ax.set_ylabel("Return (%)"); ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, fontsize=8); ax.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_threshold_comparison(sumdf, model_name, out_path):
    sub = sumdf[(sumdf["model"] == model_name) & (sumdf["window_used"] > 0)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes, ["total_ret", "win_rate", "sharpe"], ["Total Return", "Win Rate", "Sharpe"],
    ):
        for thr in THRESHOLDS:
            s = sub[sub["threshold"] == thr].sort_values("window_used")
            ax.plot(s["window_used"], s[metric], marker="o", label=f"thr={thr}")
        ax.set_title(f"{model_name} - {title}"); ax.set_xlabel("Days Used"); ax.legend(fontsize=8)
    plt.suptitle(f"{model_name} v3: Threshold Comparison", y=1.02)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_monthly_heatmap(daily_list, out_path):
    combined = pd.concat(daily_list, ignore_index=True)
    combined["ym"]     = pd.to_datetime(combined["date"].astype(str)).dt.to_period("M")
    combined["traded"] = combined["signal"].notna().astype(int)
    pivot = combined.groupby(["model", "ym"])["traded"].sum().unstack(fill_value=0)
    pivot.columns = [str(c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("v3 Monthly Trade Signal Count (threshold=0.50)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def error_analysis(pred_df, model_name, n=10):
    wrong = pred_df[pred_df["pred"] != pred_df["label"]].copy()
    wrong = wrong.nlargest(n, "confidence")
    wrong["model"] = model_name
    return wrong[["model", "trade_date", "confidence", "pred",
                  "label", "return_rate", "title"]].reset_index(drop=True)


def main():
    print("=" * 60)
    print("v3 Models Backtest")
    print("=" * 60)
    print("Loading test data...")
    X_test, test_df = load_test_data()

    active_models = {k: v for k, v in MODELS.items() if v.exists()}
    if len(active_models) < len(MODELS):
        missing = [k for k, v in MODELS.items() if not v.exists()]
        print(f"  Skipping (not yet trained): {missing}")

    summary, errors, daily_thr050 = [], [], {}

    for mn in active_models:
        print(f"\n[{mn}] Running inference...")
        model      = pickle.load(open(active_models[mn], "rb"))
        pred, opt_t = run_inference(mn, model, X_test, test_df)
        errors.append(error_analysis(pred, mn))
        if opt_t != 0.5:
            print(f"  Using model-stored optimal_threshold={opt_t:.2f} for signal direction")

        for thr in THRESHOLDS:
            daily = build_daily(pred, thr, model_signal_thr=opt_t)
            if thr == 0.50:
                d = daily.copy(); d["model"] = mn
                daily_thr050[mn] = d

            for w in WINDOWS:
                m, sim = simulate(daily, w)
                m.update({"model": mn, "threshold": thr, "signal_threshold": opt_t})
                summary.append(m)
                win_str = f"{m['win_rate']:.3f}" if not np.isnan(m["win_rate"]) else "n/a"
                ext_str = (f"  [extreme={m['n_extreme_consensus']} "
                           f"ret_no_ext={m['total_ret_no_extreme']:+.2%}]"
                           if m["n_extreme_consensus"] > 0 else "")
                print(f"  thr={thr}  W={w:2d}(used={m['window_used']:2d}) | "
                      f"trades={m['n_trade']:2d} | ret={m['total_ret']:+.2%} | "
                      f"bnh={m['bnh_ret']:+.2%} | win={win_str}{ext_str}")
                sim.to_csv(
                    OUT_DIR / f"{mn}_thr{int(thr*100)}_W{w}_daily.csv",
                    index=False, encoding="utf-8-sig",
                )

    sumdf = pd.DataFrame(summary)
    for col in ["win_rate","accuracy","macro_f1","total_ret","bnh_ret",
                "total_ret_no_extreme","win_rate_no_extreme","max_dd","sharpe"]:
        sumdf[col] = pd.to_numeric(sumdf[col], errors="coerce").round(4)
    sumdf.to_csv(OUT_DIR / "backtest_v3_summary.csv", index=False, encoding="utf-8-sig")

    for w in WINDOWS:
        sim_map = {}
        for mn in active_models:
            if mn in daily_thr050:
                _, sim = simulate(daily_thr050[mn], w)
                sim_map[mn] = sim
        if sim_map:
            plot_cumulative(sim_map, w, OUT_DIR / f"v3_cumulative_return_{w}d.png")

    for mn in active_models:
        plot_threshold_comparison(sumdf, mn, OUT_DIR / f"{mn}_threshold_comparison.png")

    plot_monthly_heatmap(list(daily_thr050.values()),
                         OUT_DIR / "v3_monthly_signal_heatmap.png")

    err_df = pd.concat(errors, ignore_index=True)
    err_df.to_csv(OUT_DIR / "v3_error_analysis.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("v3 FINAL SUMMARY  (threshold=0.50)")
    print("=" * 70)
    disp = sumdf[sumdf["threshold"] == 0.50][
        ["model", "signal_threshold", "window_requested", "window_used",
         "n_trade", "win_rate", "total_ret", "bnh_ret", "max_dd", "sharpe"]
    ]
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
