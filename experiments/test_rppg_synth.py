"""
合成訊號驗證：確認三種 rPPG 演算法能從已知頻率的假脈波中還原心率。
這是實驗數據可信度的前提 — 演算法先在受控條件下驗證過，實測才有意義。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rppg_algorithms import hr_from_rgb, extract_pulse, estimate_hr

rng = np.random.default_rng(0)


def synth_rgb(hr_bpm, fps, dur_s, pulse_amp=0.01, noise=0.002,
              motion=0.0, motion_hz=None):
    """
    模擬臉部 ROI 的 RGB 均值時序。

    脈波：G 通道最強、R 次之、B 最弱 (符合血液吸收光譜)，含一次諧波。
    運動假影：三通道「同相同幅」的強度變化 (亮度/晃動造成)，這正是
        CHROM/POS 設計要抵銷的成分 —— GREEN 因為只看單通道無法分離。
        motion_hz 預設落在心率頻帶內 (1.2~2.6 Hz)，帶通濾不掉，
        才是運動情境的真實難點。
    """
    t = np.arange(int(dur_s * fps)) / fps
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)

    base = np.array([160.0, 120.0, 110.0])
    gain = np.array([0.6, 1.0, 0.3])          # 各通道脈波強度
    C = base[None, :] * (1.0 + pulse_amp * gain[None, :] * pulse[:, None])

    C += rng.normal(0, noise * base, size=C.shape)
    if motion > 0:
        mf = motion_hz if motion_hz is not None else rng.uniform(1.2, 2.6)
        art = np.sin(2 * np.pi * mf * t + rng.uniform(0, 6.28))
        C += (motion * base)[None, :] * art[:, None]   # 同相同幅 → 鏡面反射型假影
    return C


def main():
    fps = 30.0
    print(f"{'情境':<22}{'真值':>6}{'GREEN':>18}{'CHROM':>18}{'POS':>18}")
    print("-" * 82)

    scenarios = [
        ("乾淨 / 10秒",        72,  10, 0.002, 0.000, None),
        ("乾淨 / 10秒",       120,  10, 0.002, 0.000, None),
        ("高雜訊 / 10秒",      90,  10, 0.010, 0.000, None),
        ("帶內假影(弱) / 10秒", 140,  10, 0.004, 0.004, 1.60),
        ("帶內假影(中) / 10秒", 140,  10, 0.004, 0.010, 1.60),
        ("帶內假影(強) / 10秒",  90,  10, 0.004, 0.020, 2.10),
        ("短視窗 / 5秒",        72,   5, 0.002, 0.000, None),
        ("短視窗 / 5秒",       150,   5, 0.002, 0.000, None),
    ]

    errs = {m: [] for m in ('GREEN', 'CHROM', 'POS')}
    errs_motion = {m: [] for m in ('GREEN', 'CHROM', 'POS')}
    for name, hr, dur, noise, motion, mhz in scenarios:
        C = synth_rgb(hr, fps, dur, noise=noise, motion=motion, motion_hz=mhz)
        cells = []
        for m in ('GREEN', 'CHROM', 'POS'):
            r = hr_from_rgb(C, fps, m)
            errs[m].append(abs(r['hr_bpm'] - hr))
            if motion > 0:
                errs_motion[m].append(abs(r['hr_bpm'] - hr))
            cells.append(f"{r['hr_bpm']:6.1f} ({r['snr_db']:+5.1f}dB)")
        print(f"{name:<22}{hr:>6}" + "".join(f"{c:>18}" for c in cells))

    print("-" * 82)
    print("全部情境 MAE (BPM)：" + "  ".join(f"{m}={np.mean(v):6.2f}" for m, v in errs.items()))
    print("帶內假影 MAE (BPM)：" + "  ".join(f"{m}={np.mean(v):6.2f}" for m, v in errs_motion.items()))

    # 視窗長度對頻率解析度的影響 — 論文要討論的 tradeoff
    print("\n[視窗長度 vs 誤差]  真值 72 BPM，POS")
    for dur in (3, 5, 8, 10, 15, 20):
        C = synth_rgb(72, fps, dur, noise=0.004)
        r = hr_from_rgb(C, fps, 'POS')
        raw_bin = (fps / (dur * fps)) * 60.0     # 未補零的原始頻率解析度
        print(f"  {dur:>2}s  估計={r['hr_bpm']:6.1f}  誤差={abs(r['hr_bpm']-72):5.2f}  "
              f"SNR={r['snr_db']:+5.1f}dB  原始bin寬={raw_bin:5.1f} BPM")

    assert np.mean(errs['POS']) < 8.0, "POS 平均誤差過大，演算法實作有問題"
    print("\n[OK] 合成訊號驗證通過")


if __name__ == "__main__":
    main()
