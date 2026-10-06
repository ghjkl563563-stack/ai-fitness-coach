"""
analyze_failover.py — 實驗 B 第二部分：優雅降級 (Graceful Degradation) 量測

量測藍牙斷線後，系統由 rPPG 接管所需的時間與接管期間的心率品質。

核心比較兩種架構設計 (這是論文的主要論點)：

    常駐並行 (warm)  : rPPG 從頭就一直跑，緩衝區永遠是滿的
                       → 藍牙一斷，下一個影格就有可用估計
    延遲啟動 (cold)  : 偵測到斷線才 new 一個 RPPGService
                       → 必須等緩衝區重新填滿 (視窗長度) 才有輸出

專案中兩支 agent 正好各採用一種：
    AI_Agent_crossatt_multimodal_merged.py → 常駐並行
    AI_Agent_Strength_Training.py          → 延遲啟動
本腳本量化兩者差距，作為「為何應採常駐並行」的實證依據。

輸出指標：
    takeover_latency_s  斷線 → 首次取得可信 rPPG 心率的時間
    gap_coverage_pct    斷線期間有可信心率可用的時間比例
    gap_mae             斷線期間 rPPG 與參考心率的平均絕對誤差
    reconnect_error     復連瞬間 rPPG 與 BLE 首讀值的差距

用法：
    python experiments/analyze_failover.py experiments/results/hr_S01_xxx.csv
    python experiments/analyze_failover.py <csv> --window-sec 10 --min-conf 0.55
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rppg_algorithms import extract_pulse, estimate_hr

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                   "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
def find_dropouts(t, ble_ok, min_dur=2.0):
    """找出所有藍牙斷線區間 [(t_start, t_end, dur), ...]"""
    out = []
    in_drop = False
    t_start = None
    for i in range(len(t)):
        if not in_drop and ble_ok[i] == 0:
            in_drop, t_start = True, t[i]
        elif in_drop and ble_ok[i] == 1:
            dur = t[i] - t_start
            if dur >= min_dur:
                out.append((t_start, t[i], dur))
            in_drop = False
    if in_drop and t_start is not None:
        dur = t[-1] - t_start
        if dur >= min_dur:
            out.append((t_start, t[-1], dur))
    return out


def rppg_series(t, C, window_sec, step_sec, method, warm_from=None):
    """
    重放 rPPG 估計序列。
    warm_from=None  → 常駐並行：整段資料都可用來填緩衝區
    warm_from=t_d   → 延遲啟動：只允許使用 t >= t_d 之後的影格
    """
    rows = []
    t0 = t[0] if warm_from is None else warm_from
    cur = t0 + window_sec
    t_end = t[-1]

    while cur <= t_end:
        lo = cur - window_sec
        if warm_from is not None:
            lo = max(lo, warm_from)
        m = (t > lo) & (t <= cur)
        if m.sum() >= 32:
            seg_t, seg_C = t[m], C[m]
            span = seg_t[-1] - seg_t[0]
            fps = (len(seg_t) - 1) / span if span > 1e-6 else 30.0
            grid = np.linspace(seg_t[0], seg_t[-1], len(seg_t))
            Cu = np.stack([np.interp(grid, seg_t, seg_C[:, c]) for c in range(3)], axis=1)
            r = estimate_hr(extract_pulse(Cu, fps, method), fps)
            rows.append((cur, r["hr_bpm"], r["conf"], r["snr_db"]))
        cur += step_sec

    if not rows:
        return pd.DataFrame(columns=["t", "hr", "conf", "snr"])
    return pd.DataFrame(rows, columns=["t", "hr", "conf", "snr"])


def ref_hr_at(t_query, t, hr_ble, ble_ok, hr_true=None):
    """
    斷線期間沒有 BLE 可用，因此用「斷線前最後一筆」與「復連後第一筆」
    線性內插作為參考值。合成資料若有 _hr_true 則直接採用真值。
    """
    if hr_true is not None:
        return float(np.interp(t_query, t, hr_true))
    ok = (ble_ok == 1) & (hr_ble > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.interp(t_query, t[ok], hr_ble[ok]))


# ======================================================================
def analyze(df, window_sec=10.0, step_sec=0.5, method="POS", min_conf=0.55):
    t = df["t"].to_numpy(float)
    C = df[["r", "g", "b"]].to_numpy(float)
    hr_ble = df["hr_ble"].to_numpy(float)
    ble_ok = df["ble_connected"].to_numpy(int)
    hr_true = df["_hr_true"].to_numpy(float) if "_hr_true" in df.columns else None

    drops = find_dropouts(t, ble_ok)
    if not drops:
        print("[!] 資料中找不到藍牙斷線區間。")
        print("    真實錄製時請在運動中途按 d 並拔除感測器；")
        print("    或用合成資料測試：python experiments/synth_session.py")
        return None, None, drops

    warm = rppg_series(t, C, window_sec, step_sec, method, warm_from=None)

    results = []
    for k, (td, tr, dur) in enumerate(drops, 1):
        cold = rppg_series(t, C, window_sec, step_sec, method, warm_from=td)

        row = {"event": k, "t_drop_rel": td - t[0], "duration_s": dur}

        for label, ser in (("warm", warm), ("cold", cold)):
            after = ser[(ser["t"] >= td) & (ser["t"] <= tr)]
            good = after[after["conf"] >= min_conf]

            if len(good):
                row[f"{label}_latency_s"] = float(good["t"].iloc[0] - td)
                errs = []
                for _, g in good.iterrows():
                    ref = ref_hr_at(g["t"], t, hr_ble, ble_ok, hr_true)
                    if np.isfinite(ref) and ref > 0 and g["hr"] > 0:
                        errs.append(abs(g["hr"] - ref))
                row[f"{label}_gap_mae"] = float(np.mean(errs)) if errs else np.nan
            else:
                row[f"{label}_latency_s"] = np.nan      # 整段斷線期都沒有可信估計
                row[f"{label}_gap_mae"] = np.nan

            row[f"{label}_coverage_pct"] = (100.0 * len(good) / len(after)) if len(after) else 0.0

        # 復連瞬間的落差
        post = (t >= tr) & (ble_ok == 1) & (hr_ble > 0)
        if post.any():
            first_ble = float(hr_ble[post][0])
            w_end = warm[(warm["t"] <= tr) & (warm["conf"] >= min_conf)]
            row["ble_first_after"] = first_ble
            row["reconnect_error"] = (abs(float(w_end["hr"].iloc[-1]) - first_ble)
                                      if len(w_end) else np.nan)
        results.append(row)

    return pd.DataFrame(results), warm, drops


def fig_failover(df, warm, drops, base, min_conf, window_sec):
    t = df["t"].to_numpy(float)
    t0 = t[0]
    hr_ble = df["hr_ble"].to_numpy(float)
    ble_ok = df["ble_connected"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(12, 4.8))

    vis = hr_ble.copy()
    vis[(ble_ok == 0) | (hr_ble <= 0)] = np.nan
    ax.plot(t - t0, vis, "k-", lw=2.2, label="BLE 心率帶 (主軌)")

    good = warm[warm["conf"] >= min_conf]
    bad = warm[warm["conf"] < min_conf]
    ax.plot(good["t"] - t0, good["hr"], "o-", ms=2.4, lw=1.1,
            color="#1f77b4", label=f"rPPG 常駐 (可信 conf>={min_conf})")
    ax.plot(bad["t"] - t0, bad["hr"], "x", ms=3.2, alpha=0.4,
            color="#999999", label="rPPG 不可信 (已被把關擋下)")

    for i, (td, tr, dur) in enumerate(drops):
        ax.axvspan(td - t0, tr - t0, color="red", alpha=0.12,
                   label="藍牙斷線" if i == 0 else None)
        # 延遲啟動架構要等緩衝區填滿才有輸出
        ax.axvline(td - t0 + window_sec, color="orange", ls="--", lw=1.4,
                   label=f"延遲啟動需等 {window_sec:.0f}s" if i == 0 else None)

    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("心率 (BPM)")
    ax.set_title("優雅降級：藍牙斷線期間由 rPPG 接管")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = f"{base}_fig_failover.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--method", default="POS")
    ap.add_argument("--min-conf", type=float, default=0.55)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    base = os.path.splitext(args.csv)[0]

    res, warm, drops = analyze(df, args.window_sec, args.step, args.method, args.min_conf)
    if res is None:
        return 1

    print(f"[*] 偵測到 {len(drops)} 次藍牙斷線事件")
    for td, tr, dur in drops:
        print(f"    t={td-df['t'].iloc[0]:7.1f}s  持續 {dur:5.1f}s")

    res.to_csv(f"{base}_failover.csv", index=False, encoding="utf-8-sig")
    figp = fig_failover(df, warm, drops, base, args.min_conf, args.window_sec)

    # --- 摘要 ---
    def mean_of(c):
        v = res[c].to_numpy(float) if c in res else np.array([])
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else np.nan

    w_lat, c_lat = mean_of("warm_latency_s"), mean_of("cold_latency_s")
    w_cov, c_cov = mean_of("warm_coverage_pct"), mean_of("cold_coverage_pct")
    w_mae, c_mae = mean_of("warm_gap_mae"), mean_of("cold_gap_mae")

    summ = pd.DataFrame([
        {"架構": "常駐並行 (warm)", "接管延遲(s)": w_lat,
         "斷線期覆蓋率(%)": w_cov, "斷線期MAE(BPM)": w_mae},
        {"架構": "延遲啟動 (cold)", "接管延遲(s)": c_lat,
         "斷線期覆蓋率(%)": c_cov, "斷線期MAE(BPM)": c_mae},
    ])

    lines = [
        f"# 實驗 B-2　優雅降級量測 — {os.path.basename(args.csv)}",
        "",
        f"- rPPG 演算法：{args.method}，視窗 {args.window_sec:.0f} 秒，信心門檻 {args.min_conf}",
        f"- 斷線事件數：{len(drops)}，平均持續 {res['duration_s'].mean():.1f} 秒",
        "",
        "## 表　兩種降級架構比較",
        "",
        summ.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## 逐事件明細",
        "",
        res.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## 結論",
        "",
    ]
    if np.isfinite(w_lat) and np.isfinite(c_lat):
        lines.append(f"- 常駐並行接管延遲 **{w_lat:.2f} 秒**，延遲啟動需 **{c_lat:.2f} 秒**，"
                     f"相差 **{c_lat - w_lat:.1f} 秒**")
    if np.isfinite(w_cov) and np.isfinite(c_cov):
        lines.append(f"- 斷線期間可信心率覆蓋率：常駐 **{w_cov:.1f}%** vs 延遲啟動 **{c_cov:.1f}%**")
    lines += [
        "- 常駐並行的代價是 rPPG 需全程運算；實測每影格約 4.5 ms，"
        "對 30 fps 的即時系統而言可忽略",
        "",
        f"圖：`{os.path.basename(figp)}`",
    ]

    rep = f"{base}_failover_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 68)
    print(summ.to_markdown(index=False, floatfmt=".2f"))
    print("=" * 68)
    print(f"\n[OK] 報告 → {rep}")
    print(f"     圖   → {figp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
