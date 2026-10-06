"""
S2 骨架卸載路徑驗證：直接呼叫 FitnessAIAgent.process_keypoints，
確認伺服器端在「只拿到關節點、沒有影像」的情況下能正常評分與計次。

不需要 fastapi，也不需要啟動伺服器。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.test_rep_counter import squat_kps, make_kps

FPS = 30.0


def main():
    print("[*] 建立 FitnessAIAgent（會載入 ST-GCN 與範本庫）...")
    from AI_Agent_crossatt_multimodal_merged import FitnessAIAgent

    # use_sensors=False：伺服器端不啟動 BLE 與 rPPG
    agent = FitnessAIAgent(
        weight_path="best_model 1.pth",
        stgcn_path="models/gcn_weight.pth",
        exemplar_path="exemplar_bank.pt",
        use_sensors=False,
    )

    # --- 1. 餵入 8 下深蹲的關節點序列 ---
    n_reps, dur = 8, 16.0
    n = int(dur * FPS)
    lat = []
    res = {}
    for i in range(n):
        phase = 2 * np.pi * n_reps * (i / max(1, n - 1))
        knee = 175 - 80 * (0.5 - 0.5 * np.cos(phase))
        kps = squat_kps(knee)
        hr = int(95 + 25 * (i / n))          # 模擬心率隨運動上升

        t0 = time.perf_counter()
        res = agent.process_keypoints(kps, hr=hr)
        lat.append((time.perf_counter() - t0) * 1e3)

    print(f"\n[1] 餵入 {n} 幀（{n_reps} 下深蹲）")
    print(f"    回傳: {json.dumps(res, ensure_ascii=False)}")
    # 注意：這個測試迴圈跑得比真實時間快，assess_quality 的 5 Hz 節流會
    # 把大部分評分跳過，因此「平均 ms/影格」在此不具代表性。
    # 真正該報的是「單次評分成本」與「依影格率攤提後的成本」。
    t0 = time.perf_counter()
    for _ in range(10):
        agent.assess_quality(force=True)
    once_ms = (time.perf_counter() - t0) / 10 * 1e3
    hz = 1.0 / agent.ASSESS_INTERVAL_S

    print(f"    單次評分            {once_ms:6.1f} ms")
    print(f"    評分頻率            {hz:6.1f} Hz (ASSESS_INTERVAL_S={agent.ASSESS_INTERVAL_S})")
    for fps in (10, 30):
        print(f"    攤提 @{fps:>2} fps        {once_ms*min(hz,fps)/fps:6.1f} ms/影格")

    assert res.get("ready"), "緩衝區應已填滿"
    assert res.get("hr") == int(95 + 25 * ((n - 1) / n)), "心率未正確帶入"
    assert isinstance(res.get("score"), int), "分數格式錯誤"
    assert "image" not in res, "S2 回傳不應包含任何影像"

    # --- 2. 回傳大小 ---
    payload_down = json.dumps(res, ensure_ascii=False)
    payload_up = json.dumps({"kps": np.round(squat_kps(120).astype(float), 2).tolist(),
                             "hr": 110})
    print(f"\n[2] 傳輸量")
    print(f"    上行 (關節點+心率) {len(payload_up.encode()):>6,} B")
    print(f"    下行 (結構化結果) {len(payload_down.encode()):>6,} B")
    print(f"    合計              {len(payload_up.encode())+len(payload_down.encode()):>6,} B")
    print(f"    對照 S1 現況約     31,000 B")

    # --- 3. 格式防呆 ---
    bad = agent.process_keypoints(np.zeros((5, 3)), hr=90)
    print(f"\n[3] 錯誤格式輸入 → {bad}")
    assert "error" in bad, "格式錯誤應被擋下"

    # --- 4. 省略信心值欄位也要能吃 ---
    ok2 = agent.process_keypoints(squat_kps(150)[:, :2], hr=88)
    print(f"[4] 省略 conf 欄位 → score={ok2.get('score')} reps={ok2.get('reps')}")
    assert "error" not in ok2, "應接受 (17,2) 輸入"

    # --- 5. 心率確實影響模型輸入 ---
    r_lo = agent.process_keypoints(squat_kps(120), hr=60)
    r_hi = agent.process_keypoints(squat_kps(120), hr=170)
    print(f"[5] 心率帶入檢查: hr=60 → {r_lo['hr']}, hr=170 → {r_hi['hr']}")
    assert r_lo["hr"] == 60 and r_hi["hr"] == 170

    print("\n[OK] S2 骨架卸載路徑測試通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
