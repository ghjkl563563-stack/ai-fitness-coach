"""
analyze_hr.py — 實驗 B 離線分析：rPPG vs BLE 心率一致性

以 BLE 心率帶為參考標準 (ground truth)，用滑動視窗重放各種 rPPG 演算法，
計算論文所需的全部統計量：

  一致性指標 : MAE / RMSE / MAPE / Pearson r / Bland-Altman (bias, 95% LoA)
  分層分析   : 依運動強度階段 (rest~vigorous) 分別統計
  演算法比較 : POS / CHROM / GREEN / V1(原始 3R-2G 實作) 四者對照
  品質把關   : SNR 信心值門檻對「準確度 vs 覆蓋率」的取捨曲線
  視窗長度   : 3/5/8/10/15 秒視窗的誤差與解析度取捨

輸出：
    <base>_estimates.csv   每個分析視窗的逐筆估計值
    <base>_metrics.csv     各方法 × 各階段的統計量
    <base>_report.md       可直接貼進論文的表格
    <base>_fig_*.png       圖表

用法：
    python experiments/analyze_hr.py experiments/results/hr_S01_xxx.csv
    python experiments/analyze_hr.py <csv> --step 1.0 --window-sec 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rppg_algorithms import extract_pulse, estimate_hr, HR_BAND_LOW, HR_BAND_HIGH

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                   "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["POS", "CHROM", "GREEN", "V1"]
STAGE_ORDER = ["rest", "light", "moderate", "vigorous", "recovery"]


# ======================================================================
# V1 基準：完整重現原始 rPPG_service.py 的行為
# ======================================================================
def v1_estimate(C, fps_assumed=30.0):
    """
    原始實作：buffer/mean → S = 3R - 2G → 直接 FFT → 取 0.8~3.0Hz 峰值。
    無帶通濾波、無去趨勢、無補零內插，且取樣率寫死 30fps。
    """
    C = np.asarray(C, dtype=np.float64)
    mu = np.mean(C, axis=0)
    mu[np.abs(mu) < 1e-9] = 1e-9
    H = C / mu
    S = 3.0 * H[:, 0] - 2.0 * H[:, 1]

    if len(S) < 8:
        return 0.0
    freqs = np.fft.rfftfreq(len(S), d=1.0 / fps_assumed)
    mags = np.abs(np.fft.rfft(S))
    idx = np.flatnonzero((freqs >= 0.8) & (freqs <= 3.0))
    if idx.size == 0:
        return 0.0
    return float(freqs[idx[np.argmax(mags[idx])]] * 60.0)


# ======================================================================
# 滑動視窗重放
# ======================================================================
def replay(df, window_sec=10.0, step_sec=1.0, v1_window_sec=5.0):
    """
    對錄製的 RGB 軌跡做滑動視窗估計。
    每個視窗取結尾時間點作為代表時刻，與該視窗內的 BLE 中位數比對。
    """
    t = df["t"].to_numpy(float)
    C_all = df[["r", "g", "b"]].to_numpy(float)
    hr_ble = df["hr_ble"].to_numpy(float)
    ble_ok = df["ble_connected"].to_numpy(int)
    face = df["face_found"].to_numpy(int)
    stage = df["stage"].to_numpy(str)
    has_truth = "_hr_true" in df.columns
    hr_true = df["_hr_true"].to_numpy(float) if has_truth else None

    t0, t1 = t[0], t[-1]
    rows = []
    cur = t0 + window_sec

    while cur <= t1:
        m = (t > cur - window_sec) & (t <= cur)
        if m.sum() < 32:
            cur += step_sec
            continue

        seg_t = t[m]
        seg_C = C_all[m]
        span = seg_t[-1] - seg_t[0]
        fps = (len(seg_t) - 1) / span if span > 1e-6 else 30.0

        # 重取樣到均勻時間軸
        grid = np.linspace(seg_t[0], seg_t[-1], len(seg_t))
        Cu = np.stack([np.interp(grid, seg_t, seg_C[:, c]) for c in range(3)], axis=1)

        rec = {
            "t": cur,
            "t_rel": cur - t0,
            "stage": stage[m][-1],
            "fps": fps,
            "face_ratio": float(face[m].mean()),
            "n": int(m.sum()),
        }

        # BLE 參考值：視窗內有連線的樣本取中位數
        bm = m & (ble_ok == 1) & (hr_ble > 0)
        rec["hr_ble"] = float(np.median(hr_ble[bm])) if bm.sum() >= 5 else np.nan
        rec["ble_coverage"] = float((ble_ok[m] == 1).mean())
        if has_truth:
            rec["hr_true"] = float(np.median(hr_true[m]))

        # v2 三種演算法
        for meth in ("POS", "CHROM", "GREEN"):
            pulse = extract_pulse(Cu, fps, meth)
            r = estimate_hr(pulse, fps)
            rec[f"hr_{meth}"] = r["hr_bpm"]
            rec[f"snr_{meth}"] = r["snr_db"] if np.isfinite(r["snr_db"]) else -99.0
            rec[f"conf_{meth}"] = r["conf"]

        # V1 基準：用它原本的 5 秒視窗、假設 30fps
        mv = (t > cur - v1_window_sec) & (t <= cur)
        rec["hr_V1"] = v1_estimate(C_all[mv], 30.0) if mv.sum() >= 32 else 0.0
        rec["snr_V1"] = np.nan
        rec["conf_V1"] = np.nan

        rows.append(rec)
        cur += step_sec

    return pd.DataFrame(rows)


# ======================================================================
# 統計量
# ======================================================================
def agreement(est, ref):
    """計算一致性指標。est/ref 為等長 1D 陣列。"""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref) & (ref > 0) & (est > 0)
    n = int(ok.sum())
    if n < 3:
        return {"n": n, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                "r": np.nan, "bias": np.nan, "LoA_lo": np.nan, "LoA_hi": np.nan,
                "within_5bpm": np.nan, "within_10bpm": np.nan}

    e, g = est[ok], ref[ok]
    d = e - g
    sd = float(np.std(d, ddof=1))
    r = float(stats.pearsonr(e, g)[0]) if n >= 3 and np.std(e) > 0 and np.std(g) > 0 else np.nan

    return {
        "n": n,
        "MAE": float(np.mean(np.abs(d))),
        "RMSE": float(np.sqrt(np.mean(d ** 2))),
        "MAPE": float(np.mean(np.abs(d / g)) * 100),
        "r": r,
        "bias": float(np.mean(d)),
        "LoA_lo": float(np.mean(d) - 1.96 * sd),
        "LoA_hi": float(np.mean(d) + 1.96 * sd),
        "within_5bpm": float(np.mean(np.abs(d) <= 5) * 100),
        "within_10bpm": float(np.mean(np.abs(d) <= 10) * 100),
    }


def build_metrics(est_df, ref_col="hr_ble"):
    """各方法 × 各階段 (含 ALL) 的統計量表。"""
    out = []
    stages = [s for s in STAGE_ORDER if s in set(est_df["stage"])]
    for meth in METHODS:
        for stage in ["ALL"] + stages:
            sub = est_df if stage == "ALL" else est_df[est_df["stage"] == stage]
            m = agreement(sub[f"hr_{meth}"], sub[ref_col])
            m.update({"method": meth, "stage": stage})
            out.append(m)
    cols = ["method", "stage", "n", "MAE", "RMSE", "MAPE", "r",
            "bias", "LoA_lo", "LoA_hi", "within_5bpm", "within_10bpm"]
    return pd.DataFrame(out)[cols]


def conf_sweep(est_df, method="POS", ref_col="hr_ble"):
    """信心門檻掃描：展示「準確度 vs 覆蓋率」的取捨。"""
    rows = []
    total = len(est_df)
    for thr in np.arange(0.0, 0.91, 0.05):
        sub = est_df[est_df[f"conf_{method}"] >= thr]
        m = agreement(sub[f"hr_{method}"], sub[ref_col])
        rows.append({"conf_thr": round(float(thr), 2),
                     "coverage_pct": 100.0 * len(sub) / max(1, total),
                     "MAE": m["MAE"], "RMSE": m["RMSE"],
                     "within_5bpm": m["within_5bpm"], "n": m["n"]})
    return pd.DataFrame(rows)


def window_sweep(df, ref_col="hr_ble", method="POS", step_sec=2.0):
    """視窗長度掃描。"""
    rows = []
    for wsec in (3, 5, 8, 10, 15, 20):
        e = replay(df, window_sec=float(wsec), step_sec=step_sec)
        if e.empty:
            continue
        m = agreement(e[f"hr_{method}"], e[ref_col])
        rows.append({"window_sec": wsec,
                     "freq_res_bpm": 60.0 / wsec,   # 未補零的原始頻率解析度
                     "MAE": m["MAE"], "RMSE": m["RMSE"], "r": m["r"],
                     "mean_conf": float(e[f"conf_{method}"].mean()), "n": m["n"]})
    return pd.DataFrame(rows)


# ======================================================================
# 圖表
# ======================================================================
def fig_timeseries(est_df, base, ref_col="hr_ble"):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(est_df["t_rel"], est_df[ref_col], "k-", lw=2.2, label="BLE 心率帶 (參考)")
    for meth, c in zip(("POS", "CHROM", "GREEN", "V1"),
                       ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")):
        y = est_df[f"hr_{meth}"].replace(0, np.nan)
        ax.plot(est_df["t_rel"], y, lw=1.2, alpha=0.85, color=c, label=f"rPPG-{meth}")

    # 階段分隔線
    st = est_df["stage"].to_numpy()
    for i in range(1, len(st)):
        if st[i] != st[i - 1]:
            x = est_df["t_rel"].iloc[i]
            ax.axvline(x, color="gray", ls=":", lw=1)
            ax.text(x + 1, ax.get_ylim()[1] * 0.98, st[i], fontsize=8,
                    color="gray", va="top")

    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("心率 (BPM)")
    ax.set_title("雙軌心率時序比對")
    ax.legend(ncol=5, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = f"{base}_fig_timeseries.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_bland_altman(est_df, base, ref_col="hr_ble"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, meth in zip(axes, METHODS):
        e = est_df[f"hr_{meth}"].to_numpy(float)
        g = est_df[ref_col].to_numpy(float)
        ok = np.isfinite(e) & np.isfinite(g) & (e > 0) & (g > 0)
        e, g = e[ok], g[ok]
        if len(e) < 3:
            ax.set_title(f"{meth} (資料不足)")
            continue

        mean = (e + g) / 2
        diff = e - g
        bias, sd = np.mean(diff), np.std(diff, ddof=1)

        ax.scatter(mean, diff, s=9, alpha=0.45, color="#1f77b4", edgecolors="none")
        ax.axhline(bias, color="red", lw=1.6, label=f"bias={bias:+.1f}")
        ax.axhline(bias + 1.96 * sd, color="red", ls="--", lw=1.1,
                   label=f"95% LoA ±{1.96*sd:.1f}")
        ax.axhline(bias - 1.96 * sd, color="red", ls="--", lw=1.1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{meth}  (n={len(e)})")
        ax.set_xlabel("(rPPG + BLE)/2  BPM")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rPPG - BLE  (BPM)")
    fig.suptitle("Bland-Altman 一致性分析", y=1.02)
    fig.tight_layout()
    p = f"{base}_fig_bland_altman.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_stage_bars(met_df, base):
    stages = [s for s in STAGE_ORDER if s in set(met_df["stage"])]
    if not stages:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(METHODS)
    x = np.arange(len(stages))
    for i, meth in enumerate(METHODS):
        vals = [met_df[(met_df.method == meth) & (met_df.stage == s)]["MAE"].values
                for s in stages]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("各運動強度階段的 rPPG 誤差 — 強度越高誤差越大")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = f"{base}_fig_stage_mae.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_conf_tradeoff(sweep_df, base, method="POS"):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(sweep_df["conf_thr"], sweep_df["MAE"], "o-", color="#d62728", label="MAE")
    ax1.set_xlabel(f"信心值門檻 (rPPG-{method})")
    ax1.set_ylabel("MAE (BPM)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["conf_thr"], sweep_df["coverage_pct"], "s--",
             color="#1f77b4", label="覆蓋率")
    ax2.set_ylabel("覆蓋率 (%)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("SNR 品質把關：準確度 vs 覆蓋率取捨")
    fig.tight_layout()
    p = f"{base}_fig_conf_tradeoff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ======================================================================
def fmt(df, floatfmt="{:.2f}"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "—" if not np.isfinite(v) else floatfmt.format(v))
    return d.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--skip-sweeps", action="store_true", help="略過視窗長度掃描 (較慢)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    base = os.path.splitext(args.csv)[0]
    has_truth = "_hr_true" in df.columns

    print(f"[*] 載入 {len(df)} 筆，時長 {(df['t'].iloc[-1]-df['t'].iloc[0])/60:.1f} 分鐘")
    print(f"[*] 階段: {list(dict.fromkeys(df['stage']))}")
    print(f"[*] 臉部偵測成功率: {df['face_found'].mean()*100:.1f}%")
    print(f"[*] 藍牙連線率: {df['ble_connected'].mean()*100:.1f}%")
    if has_truth:
        print("[*] 偵測到 _hr_true 欄位 (合成資料)，將同時對照真值")

    print(f"[*] 滑動視窗重放 (window={args.window_sec}s, step={args.step}s) ...")
    est = replay(df, args.window_sec, args.step)
    if est.empty:
        print("[X] 沒有產生任何分析視窗，請檢查資料長度")
        return 1
    est.to_csv(f"{base}_estimates.csv", index=False, encoding="utf-8-sig")
    print(f"[*] 產生 {len(est)} 個分析視窗")

    # BLE 可用的視窗才納入一致性統計
    est_ble = est[np.isfinite(est["hr_ble"])].copy()
    print(f"[*] 其中 {len(est_ble)} 個視窗有 BLE 參考值")

    ref = "hr_ble" if len(est_ble) >= 10 else ("hr_true" if has_truth else "hr_ble")
    src = est_ble if ref == "hr_ble" else est
    if ref != "hr_ble":
        print("[!] BLE 參考值不足，改用合成真值 _hr_true 作為參考")

    met = build_metrics(src, ref)
    met.to_csv(f"{base}_metrics.csv", index=False, encoding="utf-8-sig")

    sweep = conf_sweep(src, "POS", ref)
    sweep.to_csv(f"{base}_conf_sweep.csv", index=False, encoding="utf-8-sig")

    wsweep = pd.DataFrame()
    if not args.skip_sweeps:
        print("[*] 視窗長度掃描 ...")
        wsweep = window_sweep(df, ref, "POS", step_sec=max(2.0, args.step * 2))
        wsweep.to_csv(f"{base}_window_sweep.csv", index=False, encoding="utf-8-sig")

    figs = [fig_timeseries(src, base, ref),
            fig_bland_altman(src, base, ref),
            fig_stage_bars(met, base),
            fig_conf_tradeoff(sweep, base)]
    figs = [f for f in figs if f]

    # --- 報告 ---
    overall = met[met.stage == "ALL"].set_index("method")
    lines = [
        f"# 實驗 B 分析報告 — {os.path.basename(args.csv)}",
        "",
        f"- 樣本數：{len(df)} 影格，時長 {(df['t'].iloc[-1]-df['t'].iloc[0])/60:.1f} 分鐘",
        f"- 分析視窗：{args.window_sec:.0f} 秒，步進 {args.step:.1f} 秒，共 {len(src)} 個視窗",
        f"- 參考標準：{'BLE 心率帶' if ref=='hr_ble' else '合成真值'}",
        f"- 臉部偵測成功率：{df['face_found'].mean()*100:.1f}%",
        f"- 藍牙連線率：{df['ble_connected'].mean()*100:.1f}%",
        "",
        "## 表 1　整體一致性 (各演算法)",
        "",
        fmt(met[met.stage == "ALL"].drop(columns=["stage"])),
        "",
        "## 表 2　依運動強度分層",
        "",
        fmt(met[met.stage != "ALL"]),
        "",
        "## 表 3　SNR 信心門檻：準確度 vs 覆蓋率",
        "",
        fmt(sweep),
        "",
    ]
    if not wsweep.empty:
        lines += ["## 表 4　視窗長度取捨 (POS)", "", fmt(wsweep), ""]

    if {"POS", "V1"}.issubset(set(overall.index)):
        p_mae, v_mae = overall.loc["POS", "MAE"], overall.loc["V1", "MAE"]
        if np.isfinite(p_mae) and np.isfinite(v_mae) and v_mae > 0:
            lines += [
                "## 重點結論",
                "",
                f"- POS (v2) MAE **{p_mae:.2f} BPM** vs V1 原始實作 **{v_mae:.2f} BPM**，"
                f"誤差降低 **{(1 - p_mae/v_mae)*100:.1f}%**",
            ]
            hi = met[(met.method == "POS") & (met.stage == "vigorous")]["MAE"].values
            lo = met[(met.method == "POS") & (met.stage == "rest")]["MAE"].values
            if len(hi) and len(lo) and np.isfinite(hi[0]) and np.isfinite(lo[0]):
                lines.append(
                    f"- 靜止時 POS MAE {lo[0]:.2f} BPM，高強度時升至 {hi[0]:.2f} BPM "
                    f"(惡化 {hi[0]/max(lo[0],1e-6):.1f} 倍) — 這正是需要 BLE 主軌的理由")
            lines.append("")

    lines += ["## 圖檔", ""] + [f"- `{os.path.basename(f)}`" for f in figs]

    rep = f"{base}_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 70)
    print(fmt(met[met.stage == "ALL"].drop(columns=["stage"])))
    print("=" * 70)
    print(f"\n[OK] 報告 → {rep}")
    for f in figs:
        print(f"     圖 → {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
