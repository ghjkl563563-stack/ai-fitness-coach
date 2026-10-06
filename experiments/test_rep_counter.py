"""
rep_counter 驗證：用合成骨架做出已知次數的動作，檢查計次是否正確。

同時對照原本的計時器作法，量化「計時器 vs 真實計次」的差距 ——
這組數字可直接寫進論文，說明為何必須改掉。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rep_counter import RepCounter, classify

FPS = 30.0


def make_kps():
    """建立一副預設站姿的 COCO-17 關節點 (x, y, conf)。"""
    k = np.zeros((17, 3))
    base = {
        0: (200, 100), 1: (195, 95), 2: (205, 95), 3: (188, 100), 4: (212, 100),
        5: (175, 150), 6: (225, 150),      # 肩
        7: (170, 210), 8: (230, 210),      # 肘
        9: (168, 265), 10: (232, 265),     # 腕
        11: (185, 250), 12: (215, 250),    # 髖
        13: (185, 330), 14: (215, 330),    # 膝
        15: (185, 410), 16: (215, 410),    # 踝
    }
    for i, (x, y) in base.items():
        k[i] = (x, y, 0.9)
    return k


def squat_kps(knee_deg):
    """依指定膝角擺放髖-膝-踝。膝角 180=站直，90=深蹲。"""
    k = make_kps()
    L = 80.0
    theta = np.radians(180.0 - knee_deg)
    for kn, hp, an in ((13, 11, 15), (14, 12, 16)):
        kx, ky = k[kn][0], k[kn][1]
        k[an] = (kx, ky + L, 0.9)
        k[hp] = (kx + L * np.sin(theta), ky - L * np.cos(theta), 0.9)
    return k


def pushup_kps(elbow_deg):
    """依指定肘角擺放肩-肘-腕。"""
    k = make_kps()
    L = 60.0
    theta = np.radians(180.0 - elbow_deg)
    for el, sh, wr in ((7, 5, 9), (8, 6, 10)):
        ex, ey = k[el][0], k[el][1]
        k[wr] = (ex, ey + L, 0.9)
        k[sh] = (ex + L * np.sin(theta), ey - L * np.cos(theta), 0.9)
    return k


def jack_kps(spread):
    """spread 0=合腿, 1=張開到最大。"""
    k = make_kps()
    d = 15 + 65 * spread
    k[15] = (200 - d, 410, 0.9)
    k[16] = (200 + d, 410, 0.9)
    return k


def run(builder, action, n_reps, dur_s, lo, hi, noise=0.0, rng=None):
    """產生 n_reps 個完整循環，回傳 (計次結果, 計時器結果)。"""
    rng = rng or np.random.default_rng(0)
    rc = RepCounter()
    n = int(dur_s * FPS)
    timer_reps, last_t = 0, 0.0

    for i in range(n):
        t = i / FPS
        phase = 2 * np.pi * n_reps * (i / max(1, n - 1))
        # 由 hi 出發下降到 lo 再回來 = 一次完整動作
        v = hi - (hi - lo) * (0.5 - 0.5 * np.cos(phase))
        if noise:
            v += rng.normal(0, noise)
        rc.update(builder(v), action, t=t)

        # 原本的計時器作法：分數達標且距上次 >1.5 秒就 +1
        if (t - last_t) > 1.5:
            timer_reps += 1
            last_t = t

    return rc.count_of(action), timer_reps


def main():
    print(f"{'動作':<16}{'訊號型別':<15}{'真實次數':>9}{'計次器':>8}{'計時器':>8}")
    print("-" * 60)

    cases = [
        ("squat",        squat_kps,  "膝角",  12, 24.0, 95, 175),
        ("squat",        squat_kps,  "膝角",   8, 24.0, 100, 170),
        ("push up",      pushup_kps, "肘角",  15, 25.0, 75, 170),
        ("jumping jacks", jack_kps,  "踝距",  20, 22.0, 0.05, 1.0),
    ]

    errs, timer_errs = [], []
    for action, builder, _sig, n_reps, dur, lo, hi in cases:
        got, timer = run(builder, action, n_reps, dur, lo, hi)
        errs.append(abs(got - n_reps))
        timer_errs.append(abs(timer - n_reps))
        kind = classify(action)
        print(f"{action:<16}{kind:<15}{n_reps:>9}{got:>8}{timer:>8}")

    print("-" * 60)
    print(f"計次器平均誤差 {np.mean(errs):.2f} 次   "
          f"計時器平均誤差 {np.mean(timer_errs):.2f} 次")

    # --- 抗雜訊 ---
    print("\n[抗雜訊] 深蹲 10 下，關節點加入像素抖動")
    rng = np.random.default_rng(1)
    for nz in (0, 2, 5, 10):
        got, _ = run(squat_kps, "squat", 10, 20.0, 95, 175, noise=nz, rng=rng)
        print(f"  雜訊 ±{nz:>2} 度: 計得 {got:>2} 下 (誤差 {abs(got-10)})")

    # --- 靜止不動不該計次 ---
    rc = RepCounter()
    for i in range(int(15 * FPS)):
        rc.update(squat_kps(175), "squat", t=i / FPS)
    print(f"\n[靜止 15 秒] 計得 {rc.count_of('squat')} 下 → 應為 0")

    # --- 靜態支撐改計時 ---
    rc2 = RepCounter()
    for i in range(int(12 * FPS)):
        rc2.update(make_kps(), "high plank", t=i / FPS)
    hold = rc2.hold_seconds.get("high plank", 0)
    print(f"[棒式 12 秒] 計次={rc2.count_of('high plank')} 持續={hold:.1f}s → 應為 0 次 / 約 12 秒")

    assert np.mean(errs) <= 1.0, f"計次誤差過大: {np.mean(errs)}"
    assert rc.count_of("squat") == 0, "靜止不動不應計次"
    assert rc2.count_of("high plank") == 0, "靜態動作不應計次"
    assert 11.0 <= hold <= 13.0, f"持續秒數不正確: {hold}"
    print("\n[OK] rep_counter 測試通過")


if __name__ == "__main__":
    main()
