"""rPPG_service v2 煙霧測試：確認新介面可用、且 v1 呼叫方式仍相容。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rPPG_service import RPPGService
from experiments.test_rppg_synth import synth_rgb

FPS = 30.0

def fake_face_frame(rgb):
    """把指定 RGB 值畫成一張含膚色臉部區域的影格 (BGR)。"""
    f = np.full((480, 640, 3), 30, dtype=np.uint8)
    b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
    f[80:380, 200:440] = (b, g, r)          # 臉
    return f

def main():
    # --- 1. 新介面 update() ---
    s = RPPGService(window_size=300, method='POS', require_face=False)
    print(f"MediaPipe FaceMesh 可用: {s._mesh is not None}")

    C = synth_rgb(78, FPS, 12, pulse_amp=0.06, noise=0.001)
    t0 = time.time()
    st = None
    for i, rgb in enumerate(C):
        st = s.update(fake_face_frame(rgb), timestamp=t0 + i / FPS)

    print(f"\n[update()] 真值 78 BPM")
    print(f"  hr={st['hr']:.1f}  hr_raw={st['hr_raw']:.1f}  snr={st['snr_db']:+.1f}dB "
          f"conf={st['conf']:.2f} valid={st['valid']} fps={st['fps']:.1f} n={st['n_samples']}")

    # --- 2. v1 相容介面 (agent 目前的呼叫方式) ---
    s2 = RPPGService(window_size=300)
    sig = None
    for i, rgb in enumerate(C):
        rgb_mean = s2.extract_roi_mean(fake_face_frame(rgb), None)
        sig = s2.process(rgb_mean, timestamp=t0 + i / FPS)

    assert sig is not None and len(sig) > 0, "v1 相容介面 process() 回傳空值"
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / FPS)
    mags = np.abs(np.fft.rfft(sig))
    idx = np.where((freqs >= 0.8) & (freqs <= 3.0))[0]
    legacy_hr = freqs[idx[np.argmax(mags[idx])]] * 60
    print(f"[v1 相容介面] 舊式自行 FFT 得 {legacy_hr:.1f} BPM (長度 {len(sig)})")

    # --- 3. 無臉輸入 → 必須標記為不可信 ---
    s3 = RPPGService(window_size=300)
    noise_st = None
    for i in range(300):
        noise_st = s3.update((np.random.rand(480, 640, 3) * 255).astype('uint8'),
                             timestamp=t0 + i / FPS)
    print(f"[純雜訊輸入] valid={noise_st['valid']} conf={noise_st['conf']:.2f} "
          f"snr={noise_st['snr_db']:+.1f}dB  → 應為 valid=False")

    # --- 4. 每影格處理耗時 ---
    s4 = RPPGService(window_size=300)
    frame = fake_face_frame(C[0])
    for _ in range(30):
        s4.update(frame)
    t1 = time.time()
    for _ in range(100):
        s4.update(frame)
    ms = (time.time() - t1) / 100 * 1000
    print(f"[效能] 每影格 {ms:.2f} ms  (理論上限 {1000/ms:.0f} fps)")

    # --- 5. 臉部把關：偵測不到臉時即使訊號乾淨也不可信 ---
    s5 = RPPGService(window_size=300, require_face=True)
    gate_st = None
    for i, rgb in enumerate(C):
        gate_st = s5.update(fake_face_frame(rgb), timestamp=t0 + i / FPS)
    print(f"[臉部把關] 色塊假臉 face_found={gate_st['face_found']} "
          f"conf={gate_st['conf']:.2f} valid={gate_st['valid']} → 訊號乾淨但無臉，應為 valid=False")
    s5.close()

    assert abs(st['hr'] - 78) < 12, f"update() 心率誤差過大: {st['hr']}"
    assert not noise_st['valid'], "純雜訊輸入不應被判定為可信"
    assert not gate_st['valid'], "偵測不到臉時不應判定為可信"
    for x in (s, s2, s3, s4):
        x.close()
    print("\n[OK] rPPG_service v2 測試通過")

if __name__ == "__main__":
    main()
