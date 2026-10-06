"""
synth_session.py — 產生一段「假的」錄製資料，格式與 collect_hr.py 輸出完全相同。

用途有兩個：
  1. 在還沒收到真實受試者資料前，先驗證整條分析管線可正常運作
  2. 因為真值已知，可用來檢驗分析程式本身的正確性 (sanity check)

刻意模擬的真實現象：
  - 心率隨運動階段上升，恢復期指數衰減
  - 運動強度越高 → 臉部晃動越大 → 帶內運動假影越強 (rPPG 誤差應隨之上升)
  - 高強度時臉部偶爾偵測失敗 (低頭、出框)
  - 藍牙在指定時間點斷線一段時間 (供失效轉移分析)

用法：
    python experiments/synth_session.py --subject SYN01
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from collect_hr import FIELDS

rng = np.random.default_rng(42)   # 由 --seed 覆寫

# (階段, 秒數, 起始HR, 結束HR, 運動假影強度, 臉部遺失率)
PLAN = [
    ("rest",      90,  72,  75, 0.0005, 0.00),
    ("light",     90,  78,  98, 0.0040, 0.02),
    ("moderate",  90, 100, 132, 0.0110, 0.06),
    ("vigorous",  90, 135, 168, 0.0220, 0.14),
    ("recovery",  90, 160,  92, 0.0030, 0.02),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="SYN01")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ble-drop-at", type=float, default=200.0, help="藍牙斷線起始秒數")
    ap.add_argument("--ble-drop-dur", type=float, default=45.0, help="藍牙斷線持續秒數")
    ap.add_argument("--seed", type=int, default=42, help="亂數種子，不同受試者請給不同值")
    args = ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"hr_{args.subject}_synthetic.csv")

    fps = args.fps
    t0 = 1_700_000_000.0
    base = np.array([158.0, 118.0, 108.0])
    gain = np.array([0.6, 1.0, 0.3])
    pulse_amp = 0.012

    rows = []
    t_abs = 0.0
    phase = 0.0          # 脈波相位，需連續累積才不會在階段邊界斷掉
    n = 0

    for stage, dur, hr0, hr1, motion, lost_p in PLAN:
        n_f = int(dur * fps)
        for i in range(n_f):
            frac = i / max(1, n_f - 1)
            if stage == "recovery":
                hr_true = hr1 + (hr0 - hr1) * np.exp(-3.0 * frac)   # 指數恢復
            else:
                hr_true = hr0 + (hr1 - hr0) * frac
            hr_true += rng.normal(0, 0.8)                            # 心率自然變異

            dt = 1.0 / fps
            phase += 2 * np.pi * (hr_true / 60.0) * dt
            p = np.sin(phase) + 0.3 * np.sin(2 * phase)

            # 運動時脈波訊號本身會被削弱 (ROI 不穩定、皮膚形變、血流訊號被雜訊淹沒)
            amp = pulse_amp * (1.0 - 0.55 * min(1.0, motion / 0.022))
            rgb = base * (1.0 + amp * gain * p)
            rgb = rgb + rng.normal(0, 0.35, 3)

            if motion > 0:
                # 帶內運動假影。刻意「不是」完美共模 —— 真實晃動會改變 ROI 內容、
                # 造成非線性光照變化與追蹤抖動，三通道的增益與相位都略有差異。
                # 若寫成完美共模，POS/CHROM 在數學上可完全消除，會高估其效能。
                mf = 1.1 + 1.5 * (motion / 0.022)
                ch_gain = np.array([1.00, 0.82, 1.18])
                ch_phase = np.array([1.30, 1.42, 1.18])
                rgb = rgb + motion * base * ch_gain * np.sin(2 * np.pi * mf * t_abs + ch_phase)

                # ROI 追蹤抖動：每隔數秒臉部重偵測造成的階梯狀偏移
                if int(t_abs * 2) != int((t_abs - dt) * 2):
                    jitter = rng.normal(0, motion * 40, 3)
                    rgb = rgb + jitter

            face_ok = int(rng.random() > lost_p)

            # 藍牙狀態
            in_drop = args.ble_drop_at <= t_abs < args.ble_drop_at + args.ble_drop_dur
            ble_ok = 0 if in_drop else 1
            # 心率帶每秒才更新一次，且有 ±1 量化誤差
            hr_ble = 0 if in_drop else int(round(hr_true + rng.normal(0, 1.0)))
            rr_ms = 0 if in_drop else int(60000.0 / max(40, hr_ble))
            marker = 1 if abs(t_abs - args.ble_drop_at) < (0.5 / fps) else 0

            rows.append({
                "t": f"{t0 + t_abs:.4f}", "frame": n, "stage": stage,
                "r": f"{rgb[0]:.4f}", "g": f"{rgb[1]:.4f}", "b": f"{rgb[2]:.4f}",
                "face_found": face_ok,
                "hr_ble": hr_ble, "ble_connected": ble_ok,
                "rr_ms": rr_ms, "drop_marker": marker,
                "_hr_true": f"{hr_true:.3f}",
            })
            t_abs += dt
            n += 1

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["_hr_true"])
        w.writeheader()
        w.writerows(rows)

    total_s = t_abs
    print(f"[OK] 合成 {n} 筆 ({total_s/60:.1f} 分鐘) → {out}")
    print(f"     心率範圍 72~168 BPM，藍牙於 {args.ble_drop_at:.0f}s 斷線 {args.ble_drop_dur:.0f}s")
    print(f"     注意：_hr_true 欄僅合成資料有，真實錄製不會有此欄")
    print(f"\n     下一步：python experiments/analyze_hr.py {out}")


if __name__ == "__main__":
    main()
