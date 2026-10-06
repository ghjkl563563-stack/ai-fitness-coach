"""
aggregate_subjects.py — 跨受試者彙整，產生論文用的統計表

單一受試者的數據不能作為論文結果。本腳本把多位受試者的分析結果彙整成
「平均值 ± 標準差 (n=N)」的形式，並做配對統計檢定。

輸入：results/ 目錄下由 analyze_hr.py 產生的 *_metrics.csv
      以及 analyze_failover.py 產生的 *_failover.csv (若有)

統計檢定：
    以受試者為配對單位，用 Wilcoxon 符號等級檢定比較 POS 與其他方法。
    受試者數通常在 5~10 位，屬小樣本且不保證常態分布，
    因此採用無母數檢定而非成對 t 檢定。

輸出：
    aggregate_metrics.csv      各方法 × 各階段的 mean/SD/n
    aggregate_per_subject.csv  逐受試者明細 (供檢查離群值)
    aggregate_report.md        可直接貼進論文的表格
    aggregate_fig_*.png        含誤差棒的圖表

用法：
    python experiments/aggregate_subjects.py
    python experiments/aggregate_subjects.py --include-synthetic
    python experiments/aggregate_subjects.py --dir experiments/results
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                   "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHODS = ["POS", "CHROM", "GREEN", "V1"]
STAGE_ORDER = ["rest", "light", "moderate", "vigorous", "recovery"]
METRIC_COLS = ["MAE", "RMSE", "MAPE", "r", "bias", "within_5bpm", "within_10bpm"]

_SUBJ_RE = re.compile(r"hr_(?P<subj>.+?)_(?:\d{8}_\d{6}|synthetic)_metrics\.csv$")


def subject_of(path):
    m = _SUBJ_RE.search(os.path.basename(path))
    if m:
        return m.group("subj")
    return os.path.basename(path).replace("_metrics.csv", "")


def _gather(directory, pattern, include_synth):
    """掃描指定目錄；納入合成資料時一併掃 _synthetic_demo 子目錄。"""
    files = glob.glob(os.path.join(directory, pattern))
    if include_synth:
        files += glob.glob(os.path.join(directory, "_synthetic_demo", pattern))
    return sorted(files)


def load_all(directory, include_synth):
    """載入所有受試者的 metrics 檔，合併成一張長表。"""
    files = _gather(directory, "*_metrics.csv", include_synth)
    # 排除本腳本自己的輸出，否則重跑時會把 aggregate_metrics.csv 當成一位受試者
    files = [f for f in files if not os.path.basename(f).startswith("aggregate")]
    if not include_synth:
        files = [f for f in files if "synthetic" not in os.path.basename(f).lower()]

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"[!] 略過 {os.path.basename(f)}: {e}")
            continue
        if "method" not in d.columns or "stage" not in d.columns:
            print(f"[!] 略過 {os.path.basename(f)}: 欄位格式不符")
            continue
        d["subject"] = subject_of(f)
        d["source_file"] = os.path.basename(f)
        frames.append(d)

    if not frames:
        return None, files
    return pd.concat(frames, ignore_index=True), files


def summarise(long_df):
    """各方法 × 各階段跨受試者的 mean / SD / n。"""
    rows = []
    stages = ["ALL"] + [s for s in STAGE_ORDER if s in set(long_df["stage"])]
    for meth in METHODS:
        for stage in stages:
            sub = long_df[(long_df.method == meth) & (long_df.stage == stage)]
            if sub.empty:
                continue
            rec = {"method": meth, "stage": stage, "n_subjects": int(sub["subject"].nunique())}
            for c in METRIC_COLS:
                if c not in sub.columns:
                    continue
                v = pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
                v = v[np.isfinite(v)]
                rec[f"{c}_mean"] = float(np.mean(v)) if len(v) else np.nan
                rec[f"{c}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def min_achievable_p(n):
    """
    雙尾 Wilcoxon 符號等級檢定在樣本數 n 下「理論上可能達到的最小 p 值」= 2/2^n。
    n=5 時為 0.0625 —— 即使所有受試者都朝同一方向，也永遠達不到 p<0.05。
    因此想宣稱統計顯著，至少需要 n=6 (最小 p=0.03125)。
    """
    return 2.0 / (2 ** n) if n >= 1 else np.nan


def paired_tests(long_df, stage="ALL", metric="MAE", baseline="POS"):
    """
    以受試者為配對單位比較 baseline 與其他方法。
    小樣本採 Wilcoxon 符號等級檢定。
    """
    piv = (long_df[long_df.stage == stage]
           .pivot_table(index="subject", columns="method", values=metric, aggfunc="mean"))
    rows = []
    if baseline not in piv.columns:
        return pd.DataFrame(rows)

    for meth in METHODS:
        if meth == baseline or meth not in piv.columns:
            continue
        pair = piv[[baseline, meth]].dropna()
        n = len(pair)
        rec = {"比較": f"{baseline} vs {meth}", "n": n,
               f"{baseline} 平均": float(pair[baseline].mean()) if n else np.nan,
               "對照方法平均": float(pair[meth].mean()) if n else np.nan}

        if n >= 5 and (pair[baseline] - pair[meth]).abs().sum() > 0:
            try:
                stat, p = stats.wilcoxon(pair[baseline], pair[meth])
                rec["W"] = float(stat)
                rec["p"] = float(p)
            except Exception:
                rec["W"], rec["p"] = np.nan, np.nan
            d = pair[baseline] - pair[meth]           # 效果量：配對差的 Cohen's d
            sd = d.std(ddof=1)
            rec["Cohen_d"] = float(d.mean() / sd) if sd > 0 else np.nan
            rec["可達最小 p"] = min_achievable_p(n)
        else:
            rec["W"], rec["p"], rec["Cohen_d"] = np.nan, np.nan, np.nan
            rec["可達最小 p"] = min_achievable_p(n)
            rec["備註"] = f"受試者數不足 (n={n})，未做檢定"
        rows.append(rec)
    return pd.DataFrame(rows)


def load_failover(directory, include_synth):
    files = _gather(directory, "*_failover.csv", include_synth)
    files = [f for f in files if not os.path.basename(f).startswith("aggregate")]
    if not include_synth:
        files = [f for f in files if "synthetic" not in os.path.basename(f).lower()]
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d["subject"] = os.path.basename(f).split("_")[1] if "_" in os.path.basename(f) else "?"
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


# ======================================================================
def fig_stage_errorbar(summ, out):
    stages = [s for s in STAGE_ORDER if s in set(summ["stage"])]
    if not stages:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(METHODS)
    x = np.arange(len(stages))
    for i, meth in enumerate(METHODS):
        means, sds = [], []
        for s in stages:
            r = summ[(summ.method == meth) & (summ.stage == s)]
            means.append(float(r["MAE_mean"].iloc[0]) if len(r) else np.nan)
            sds.append(float(r["MAE_sd"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=sds, capsize=3, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("各運動強度階段的 rPPG 誤差 (跨受試者平均 ± SD)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_method_box(long_df, out):
    sub = long_df[long_df.stage == "ALL"]
    data, labels = [], []
    for meth in METHODS:
        v = pd.to_numeric(sub[sub.method == meth]["MAE"], errors="coerce").dropna().to_numpy()
        if len(v):
            data.append(v)
            labels.append(f"{meth}\n(n={len(v)})")
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    # 疊上逐受試者的點
    for i, v in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.045, len(v)), v, s=16,
                   color="black", alpha=0.6, zorder=3)
    ax.set_ylabel("整體 MAE (BPM)")
    ax.set_title("各演算法的逐受試者誤差分布")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ======================================================================
def pm(summ, meth, stage, metric, fmt="{:.2f}"):
    """格式化為 mean ± SD。"""
    r = summ[(summ.method == meth) & (summ.stage == stage)]
    if r.empty:
        return "—"
    m, s = r[f"{metric}_mean"].iloc[0], r[f"{metric}_sd"].iloc[0]
    if not np.isfinite(m):
        return "—"
    return fmt.format(m) + (f" ± {fmt.format(s)}" if np.isfinite(s) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--include-synthetic", action="store_true",
                    help="把合成資料一起納入 (預設排除)")
    args = ap.parse_args()

    long_df, files = load_all(args.dir, args.include_synthetic)
    if long_df is None:
        print(f"[X] 在 {args.dir} 找不到任何 *_metrics.csv")
        print("    請先對每位受試者執行：python experiments/analyze_hr.py <該受試者的 csv>")
        if not args.include_synthetic:
            print("    (合成資料預設被排除，要納入請加 --include-synthetic —— "
                  "會一併掃描 _synthetic_demo/ 子目錄)")
        return 1

    subjects = sorted(long_df["subject"].unique())
    n_subj = len(subjects)
    print(f"[*] 載入 {len(files)} 個檔案，{n_subj} 位受試者: {', '.join(subjects)}")
    if n_subj < 5:
        print(f"[!] 受試者僅 {n_subj} 位。論文建議至少 5 位，統計檢定至少需 5 位。")

    base = os.path.join(args.dir, "aggregate")
    summ = summarise(long_df)
    summ.to_csv(f"{base}_metrics.csv", index=False, encoding="utf-8-sig")
    long_df.to_csv(f"{base}_per_subject.csv", index=False, encoding="utf-8-sig")

    tests = paired_tests(long_df, "ALL", "MAE", "POS")
    if not tests.empty:
        tests.to_csv(f"{base}_tests.csv", index=False, encoding="utf-8-sig")

    figs = [f for f in (fig_stage_errorbar(summ, f"{base}_fig_stage_mae.png"),
                        fig_method_box(long_df, f"{base}_fig_method_box.png")) if f]

    fo = load_failover(args.dir, args.include_synthetic)

    # ------------------------------------------------------------------
    # 論文表格
    # ------------------------------------------------------------------
    t1 = pd.DataFrame([{
        "方法": m,
        "MAE (BPM)": pm(summ, m, "ALL", "MAE"),
        "RMSE (BPM)": pm(summ, m, "ALL", "RMSE"),
        "MAPE (%)": pm(summ, m, "ALL", "MAPE"),
        "Pearson r": pm(summ, m, "ALL", "r", "{:.3f}"),
        "±5 BPM 內 (%)": pm(summ, m, "ALL", "within_5bpm", "{:.1f}"),
    } for m in METHODS])

    stages = [s for s in STAGE_ORDER if s in set(summ["stage"])]
    t2 = pd.DataFrame([{"階段": s, **{m: pm(summ, m, s, "MAE") for m in METHODS}}
                       for s in stages])

    lines = [
        "# 實驗 B 跨受試者彙整",
        "",
        f"- 受試者數：**n = {n_subj}**（{', '.join(subjects)}）",
        "- 數值格式：平均值 ± 標準差（跨受試者）",
        "- 參考標準：BLE 心率帶",
        "",
        "## 表 1　整體一致性",
        "",
        t1.to_markdown(index=False),
        "",
        "## 表 2　依運動強度分層的 MAE (BPM)",
        "",
        t2.to_markdown(index=False),
        "",
    ]

    if not tests.empty:
        lines += [
            "## 表 3　配對統計檢定 (Wilcoxon 符號等級檢定，以受試者為配對單位)",
            "",
            tests.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
        sig = tests[tests["p"].notna() & (tests["p"] < 0.05)] if "p" in tests else pd.DataFrame()
        if not sig.empty:
            lines.append("達統計顯著 (p < 0.05) 的比較：" +
                         "、".join(sig["比較"].tolist()))
            lines.append("")

        minp = min_achievable_p(n_subj)
        if np.isfinite(minp) and minp >= 0.05:
            lines += [
                f"> ⚠️ **樣本數限制**：雙尾 Wilcoxon 檢定在 n={n_subj} 下可達到的"
                f"最小 p 值為 **{minp:.5f}**，數學上不可能小於 0.05。",
                f"> 即使所有受試者的結果都朝同一方向，也無法宣稱統計顯著。",
                "> 若論文需要顯著性結論，**至少需收滿 6 位受試者** "
                "(n=6 最小 p=0.03125)，建議收 8–10 位以保留餘裕。",
                "",
            ]

    if fo is not None and not fo.empty:
        def mstat(c):
            if c not in fo.columns:
                return np.nan, np.nan
            v = pd.to_numeric(fo[c], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else \
                   ((float(np.mean(v)), np.nan) if len(v) else (np.nan, np.nan))

        fo_tbl = []
        for label, pre in (("常駐並行 (warm)", "warm"), ("延遲啟動 (cold)", "cold")):
            lm, ls = mstat(f"{pre}_latency_s")
            cm, cs = mstat(f"{pre}_coverage_pct")
            em, es = mstat(f"{pre}_gap_mae")
            fo_tbl.append({
                "架構": label,
                "接管延遲 (s)": f"{lm:.2f} ± {ls:.2f}" if np.isfinite(ls) else f"{lm:.2f}",
                "斷線期覆蓋率 (%)": f"{cm:.1f} ± {cs:.1f}" if np.isfinite(cs) else f"{cm:.1f}",
                "斷線期 MAE (BPM)": f"{em:.2f} ± {es:.2f}" if np.isfinite(es) else f"{em:.2f}",
            })
        lines += [
            f"## 表 4　優雅降級 (共 {len(fo)} 次斷線事件)",
            "",
            pd.DataFrame(fo_tbl).to_markdown(index=False),
            "",
        ]

    lines += ["## 圖檔", ""] + [f"- `{os.path.basename(f)}`" for f in figs]

    if n_subj < 5:
        lines += ["", "---", "",
                  f"> ⚠️ 目前僅 {n_subj} 位受試者，樣本數不足以支撐論文結論，"
                  "且無法進行統計檢定。建議收滿 5–10 位。"]

    rep = f"{base}_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 70)
    print(t1.to_markdown(index=False))
    print("=" * 70)
    print(f"\n[OK] 報告 → {rep}")
    for f in figs:
        print(f"     圖   → {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
