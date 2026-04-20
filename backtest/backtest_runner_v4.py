"""
Backtest runner v4: DAY-LEVEL models (no article aggregation needed).
New in v4:
  - Day-level prediction: 1 model call per day, no extreme-consensus issue
  - Transaction cost: tc=0.002 (0.2% round-trip) deducted per trade
  - Ablation: XGBoost text_only vs tech_only vs combined
Output: results/backtest/v4/
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

TC = 0.002   # 0.2% round-trip transaction cost per trade


# ── Wrapper stubs for pickle ──────────────────────────────────────────────────
class NBv4Model:
    def __init__(self, model, n_text, cv_f1):
        self.model = model; self.n_text = n_text; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X[:, :self.n_text])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class RFv4Model:
    def __init__(self, model, optimal_threshold, cv_f1):
        self.model = model; self.optimal_threshold = optimal_threshold; self.cv_f1 = cv_f1
    def predict_proba(self, X): return self.model.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.optimal_threshold).astype(int)

class XGBv4Model:
    def __init__(self, model, n_text, cv_f1):
        self.model = model; self.n_text = n_text; self.cv_f1 = cv_f1
    def _slice(self, X):
        if self.n_text == -1: return X
        if self.n_text == 0:  return X[:, 300:]
        return X[:, :self.n_text]
    def predict_proba(self, X): return self.model.predict_proba(self._slice(X))
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
OUT_DIR  = ROOT / "results" / "backtest" / "v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "NB_v4":          ROOT / "models/naive-bayes/NB_v4_model.pkl",
    "RF_v4":          ROOT / "models/RF/RF_v4_model.pkl",
    "XGBoost_v4":     ROOT / "models/XGBoost/XGBoost_v4_model.pkl",
    "XGB_text":       ROOT / "models/XGBoost/XGBoost_v4_text_model.pkl",
    "XGB_tech":       ROOT / "models/XGBoost/XGBoost_v4_tech_model.pkl",
}
WINDOWS             = [15, 30, 45, 60]
THRESHOLDS          = [0.50, 0.55, 0.60]
SPLIT               = 0.8
COLORS              = {
    "NB_v4": "darkorange", "RF_v4": "steelblue",
    "XGBoost_v4": "forestgreen", "XGB_text": "orchid", "XGB_tech": "sienna",
}


def load_test_data():
    meta_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_features_v4.csv",
                          encoding="utf-8-sig")
    text_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_text_v4.csv", encoding="utf-8-sig")
    tech_df = pd.read_csv(DATA_DIR / "processed" / "tsmc_day_tech_v4.csv", encoding="utf-8-sig")

    X_all = np.hstack([text_df.values, tech_df.values])   # (267, 306)
    n     = int(len(X_all) * SPLIT)
    X_test   = X_all[n:]
    test_meta = meta_df.iloc[n:].reset_index(drop=True)
    print(f"  Test days: {len(test_meta)} | {test_meta['date'].iloc[0]} -> {test_meta['date'].iloc[-1]}")
    return X_test, test_meta


def run_inference(model_name, model, X_test, test_meta):
    opt_thr = getattr(model, "optimal_threshold", 0.5)
    prob    = model.predict_proba(X_test)   # (n_days, 2)
    out     = test_meta[["date", "label", "return_rate"]].copy()
    out["prob_up"]    = prob[:, 1]
    out["confidence"] = np.maximum(prob[:, 1], 1 - prob[:, 1])
    out["pred"]       = (prob[:, 1] >= opt_thr).astype(int)
    return out, opt_thr


def simulate(pred_df, threshold, model_signal_thr, window):
    """Day-level simulation with transaction cost."""
    # Filter by confidence threshold
    traded_mask = pred_df["confidence"] >= threshold
    daily = pred_df.copy()
    daily["signal"] = np.where(traded_mask, daily["pred"], np.nan)

    avail = len(daily)
    used  = min(window, avail)
    d     = daily.head(used).copy()

    def port_ret(r):
        if pd.isna(r["signal"]):
            return 0.0
        direction = 1.0 if r["signal"] == 1 else -1.0
        return direction * r["return_rate"] - TC

    d["port_ret"] = d.apply(port_ret, axis=1)
    d["cum_port"] = (1 + d["port_ret"]).cumprod()
    d["cum_bnh"]  = (1 + d["return_rate"]).cumprod()

    traded  = d[d["signal"].notna()]
    n_trade = len(traded)
    win_rate = float((traded["signal"] == traded["label"]).mean()) if n_trade else float("nan")
    tot_ret  = float(d["cum_port"].iloc[-1] - 1) if used else 0.0
    bnh_ret  = float(d["cum_bnh"].iloc[-1] - 1) if used else 0.0
    roll_max = d["cum_port"].cummax()
    max_dd   = float(((d["cum_port"] - roll_max) / roll_max).min()) if used else 0.0
    std      = d["port_ret"].std()
    sharpe   = float(d["port_ret"].mean() / std * np.sqrt(252)) if std > 0 else float("nan")
    if n_trade:
        acc = float(accuracy_score(traded["label"], traded["signal"]))
        f1  = float(f1_score(traded["label"], traded["signal"], average="macro", zero_division=0))
    else:
        acc = f1 = float("nan")

    return {
        "window_requested": window, "window_used": used,
        "n_total": used, "n_trade": n_trade, "n_skip": used - n_trade,
        "win_rate": win_rate, "accuracy": acc, "macro_f1": f1,
        "total_ret": tot_ret, "bnh_ret": bnh_ret,
        "tc_paid": n_trade * TC,
        "max_dd": max_dd, "sharpe": sharpe,
    }, d


def plot_cumulative(sim_map, window, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bnh_done = False
    for mn, df in sim_map.items():
        dates = pd.to_datetime(df["date"].astype(str))
        ax.plot(dates, df["cum_port"] * 100 - 100,
                label=mn, color=COLORS.get(mn, "gray"), lw=1.8)
        if not bnh_done:
            ax.plot(dates, df["cum_bnh"] * 100 - 100,
                    label="Buy & Hold", color="black", ls="--", lw=1.5)
            bnh_done = True
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"v4 Cumulative Return (after TC) — {window}-Day Window (thr=0.50)")
    ax.set_ylabel("Return (%)"); ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, fontsize=8); ax.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_ablation(sumdf, out_path):
    """Bar chart: text_only vs tech_only vs combined XGBoost at W=45, thr=0.50."""
    sub = sumdf[
        (sumdf["threshold"] == 0.50) &
        (sumdf["model"].isin(["XGB_text", "XGB_tech", "XGBoost_v4"])) &
        (sumdf["window_requested"] == 45)
    ].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = {"XGB_text": "Text only", "XGB_tech": "Tech only", "XGBoost_v4": "Combined"}
    for ax, col, title in zip(axes, ["total_ret", "win_rate", "macro_f1"],
                              ["Total Return", "Win Rate", "Macro F1"]):
        vals  = [sub[sub["model"] == mn][col].values[0] if not sub[sub["model"] == mn].empty
                 else float("nan") for mn in ["XGB_text", "XGB_tech", "XGBoost_v4"]]
        bars  = ax.bar([labels[m] for m in ["XGB_text", "XGB_tech", "XGBoost_v4"]],
                       vals, color=["orchid", "sienna", "forestgreen"])
        ax.set_title(title); ax.set_ylabel(col)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("XGBoost v4 Ablation Study: Text vs Tech vs Combined (W=45, thr=0.50)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    print("=" * 60)
    print(f"v4 Day-Level Backtest  (TC={TC*100:.1f}% per trade)")
    print("=" * 60)
    X_test, test_meta = load_test_data()

    active = {k: v for k, v in MODELS.items() if v.exists()}
    if len(active) < len(MODELS):
        print(f"  Skipping: {[k for k, v in MODELS.items() if not v.exists()]}")

    summary, daily_thr050 = [], {}

    for mn in active:
        print(f"\n[{mn}] Running inference...")
        model       = pickle.load(open(active[mn], "rb"))
        pred, opt_t = run_inference(mn, model, X_test, test_meta)
        if opt_t != 0.5:
            print(f"  optimal_threshold={opt_t:.2f}")

        for thr in THRESHOLDS:
            for w in WINDOWS:
                m, sim = simulate(pred, thr, opt_t, w)
                m.update({"model": mn, "threshold": thr, "signal_threshold": opt_t})
                summary.append(m)
                win_str = f"{m['win_rate']:.3f}" if not np.isnan(m["win_rate"]) else "n/a"
                print(f"  thr={thr}  W={w:2d}(used={m['window_used']:2d}) | "
                      f"trades={m['n_trade']:2d} | ret={m['total_ret']:+.2%} "
                      f"(after TC={m['tc_paid']:.3f}) | bnh={m['bnh_ret']:+.2%} | win={win_str}")
                sim.to_csv(OUT_DIR / f"{mn}_thr{int(thr*100)}_W{w}_daily.csv",
                           index=False, encoding="utf-8-sig")
            if thr == 0.50:
                d = simulate(pred, thr, opt_t, 999)[1]
                d["model"] = mn; daily_thr050[mn] = d

    sumdf = pd.DataFrame(summary)
    for col in ["win_rate","accuracy","macro_f1","total_ret","bnh_ret","max_dd","sharpe"]:
        sumdf[col] = pd.to_numeric(sumdf[col], errors="coerce").round(4)
    sumdf.to_csv(OUT_DIR / "backtest_v4_summary.csv", index=False, encoding="utf-8-sig")

    for w in WINDOWS:
        sim_map = {}
        for mn in active:
            if mn in daily_thr050:
                sim_map[mn] = simulate(daily_thr050[mn].drop(columns="model"),
                                       0.50, 0.5, w)[1]
        if sim_map:
            plot_cumulative(sim_map, w, OUT_DIR / f"v4_cumulative_return_{w}d.png")

    plot_ablation(sumdf, OUT_DIR / "v4_ablation_study.png")

    print("\n" + "=" * 70)
    print(f"v4 SUMMARY  (thr=0.50, after TC={TC*100:.1f}%)")
    print("=" * 70)
    disp = sumdf[sumdf["threshold"] == 0.50][
        ["model", "signal_threshold", "window_requested", "window_used",
         "n_trade", "tc_paid", "win_rate", "total_ret", "bnh_ret", "max_dd", "sharpe"]
    ]
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
