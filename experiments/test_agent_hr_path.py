"""
驗證兩支 agent 的 rPPG 心率路徑在改用 v2 update() 後可正常運作。
不連藍牙、不開攝影機，直接餵合成影格。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rPPG_service import RPPGService

# 固定種子：雜訊影格若不設種子，conf 會在門檻附近浮動而讓測試偶發失敗
_rng = np.random.default_rng(20260911)
from experiments.test_rppg_synth import synth_rgb

FPS = 30.0

def face_frame(rgb):
    f = np.full((480, 640, 3), 30, dtype=np.uint8)
    f[80:380, 200:440] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
    return f

class Stub:
    """複製 agent 改後的心率決策邏輯，驗證把關與寬限期行為。"""
    def __init__(self):
        self.rppg_conf = 0.0
        self._rppg_last_ok = 0.0
        self.RPPG_GRACE_SEC = 3.0
        self.real_heart_rate = 0
        # 視窗 300 影格 = 10 秒；雜訊需餵滿超過視窗長度才會完全洗掉舊訊號
        self.hr_sensor_rppg = RPPGService(window_size=300, require_face=False)

    def step(self, frame, ts=None):
        # 注意：不傳 ts 時服務會用真實時鐘量測影格到達速率 (線上運作的正確行為)。
        # 餵合成訊號時必須傳入對應的合成時間戳，否則量到的是迴圈速度而非訊號取樣率。
        st = self.hr_sensor_rppg.update(frame, timestamp=ts)
        self.rppg_conf = st['conf']
        if st['valid']:
            self.real_heart_rate = int(round(st['hr']))
            self._rppg_last_ok = time.time()
        elif (time.time() - self._rppg_last_ok) > self.RPPG_GRACE_SEC:
            self.real_heart_rate = 0
        return st

def main():
    # 1. 語法與匯入：確認兩支 agent 檔案本身沒問題
    import ast, io
    for p in ('AI_Agent_Strength_Training.py', 'AI_Agent_crossatt_multimodal_merged.py'):
        ast.parse(io.open(p, encoding='utf-8').read())
    print("[1] 兩支 agent 語法正確")

    # 2. 乾淨訊號 → 應取得正確心率
    a = Stub()
    C = synth_rgb(82, FPS, 12, pulse_amp=0.06, noise=0.001)
    t0 = time.time()
    for i, rgb in enumerate(C):
        st = a.step(face_frame(rgb), ts=t0 + i / FPS)
    print(f"[2] 乾淨訊號 (真值 82): 顯示={a.real_heart_rate} conf={a.rppg_conf:.2f} "
          f"估計fps={st['fps']:.1f}")
    assert abs(a.real_heart_rate - 82) < 12, f"心率誤差過大: {a.real_heart_rate}"

    # 3. 不同串流速率 → v2 應自動適應 (舊版寫死 fps 會在此失準)
    for real_fps in (10.0, 15.0, 30.0):
        b = Stub()
        Cx = synth_rgb(75, real_fps, 20, pulse_amp=0.06, noise=0.001)
        t = time.time()
        for i, rgb in enumerate(Cx):
            s2 = b.step(face_frame(rgb), ts=t + i / real_fps)
        err = abs(b.real_heart_rate - 75)
        flag = "OK" if err < 12 else "FAIL"
        print(f"[3] 串流 {real_fps:>4.0f} fps (真值 75): 顯示={b.real_heart_rate:3d} "
              f"誤差={err:5.1f}  估計fps={s2['fps']:5.1f}  [{flag}]")
        assert err < 12, f"{real_fps}fps 下心率失準"
        b.hr_sensor_rppg.close()

    # 4. 訊號轉壞 → 超過寬限期應顯示 0 (UI 顯示「--」)
    t_noise = t0 + len(C) / FPS
    for k in range(int(14 * FPS)):
        a.step((_rng.random((480, 640, 3)) * 255).astype('uint8'),
               ts=t_noise + k / FPS)
    print(f"[4] 訊號轉為雜訊 14 秒後 (超過 10 秒視窗): 顯示={a.real_heart_rate} conf={a.rppg_conf:.2f} "
          f"→ 應為 0 (UI 顯示 --)")
    assert a.real_heart_rate == 0, "不可信訊號應停止顯示心率"

    a.hr_sensor_rppg.close()
    print("\n[OK] agent 心率路徑測試通過")

if __name__ == "__main__":
    main()
