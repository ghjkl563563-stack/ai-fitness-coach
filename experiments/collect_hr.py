"""
collect_hr.py — 實驗 B 資料收集：雙軌心率同步錄製

錄製內容是「每影格的臉部 ROI RGB 均值」而非已算好的心率。
這樣一次錄製即可離線重跑任意 rPPG 演算法與任意視窗長度，
所有論文數據皆可重現，不必為了換參數重做實驗。

實驗流程 (依 ACSM 運動強度分級設計，涵蓋靜止到高強度的心率範圍)：
    rest       靜止坐姿      90s
    light      原地踏步      90s
    moderate   深蹲          90s
    vigorous   開合跳        90s
    recovery   坐姿恢復      90s

操作鍵：
    SPACE  手動進入下一階段
    d      標記一次人為藍牙斷線 (拔除感測器時按，供失效轉移分析對齊)
    q      結束並存檔

用法：
    python experiments/collect_hr.py --subject S01
    python experiments/collect_hr.py --subject S01 --no-ble      # 無藍牙裝置時
    python experiments/collect_hr.py --subject S01 --stage-sec 60
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from rPPG_service import RPPGService

try:
    from hr_service import HeartRateService
except ImportError:
    HeartRateService = None

STAGES = [
    ("rest", "靜止坐姿，放鬆不要說話"),
    ("light", "原地踏步 (輕度)"),
    ("moderate", "深蹲 (中度)"),
    ("vigorous", "開合跳 (高強度)"),
    ("recovery", "坐下恢復，保持不動"),
]

FIELDS = ["t", "frame", "stage", "r", "g", "b", "face_found",
          "hr_ble", "ble_connected", "rr_ms", "drop_marker"]


def draw_hud(frame, stage_name, stage_desc, remain, hr_ble, ble_ok, face_ok, n):
    """畫出錄製狀態。用 OpenCV 內建字型避免中文字型相依，說明另外印在終端機。"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (430, 130), (30, 30, 30), -1)
    cv2.putText(frame, f"STAGE: {stage_name}  {remain:5.1f}s", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"BLE {'OK ' if ble_ok else 'OFF'}  HR={hr_ble if hr_ble else '--'}",
                (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 255, 80) if ble_ok else (80, 80, 255), 2)
    cv2.putText(frame, f"FACE {'OK' if face_ok else 'LOST'}   frames={n}", (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 255, 80) if face_ok else (80, 80, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="受試者代號，例如 S01")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--ble-addr", default=os.getenv("BLE_HR_ADDRESS"))
    ap.add_argument("--no-ble", action="store_true", help="不啟用藍牙 (僅錄 rPPG 原始訊號)")
    ap.add_argument("--stage-sec", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.out_dir, f"hr_{args.subject}_{ts}.csv")

    # --- 感測器 ---
    ble = None
    if not args.no_ble:
        if HeartRateService is None:
            print("[!] 找不到 hr_service，改為僅錄 rPPG")
        else:
            print(f"[*] 連線藍牙心率帶 {args.ble_addr} ...")
            ble = HeartRateService(address=args.ble_addr)

    rppg = RPPGService(window_size=300, method="POS", detect_every=3)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[X] 無法開啟攝影機")
        return 1

    print("\n" + "=" * 62)
    print(f"受試者 {args.subject}  每階段 {args.stage_sec:.0f} 秒  輸出 {out_csv}")
    print("  SPACE=下一階段   d=標記藍牙斷線   q=結束存檔")
    print("=" * 62)
    for i, (name, desc) in enumerate(STAGES):
        print(f"  {i+1}. {name:<10} {desc}")
    print("=" * 62 + "\n")

    f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()

    stage_i = 0
    stage_t0 = time.time()
    n = 0
    drop_marker = 0

    try:
        while stage_i < len(STAGES):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()

            # rPPG：只取 ROI 的 RGB 均值，心率留給離線分析
            rgb = rppg.extract_roi_mean(frame)
            face_ok = rppg.face_found

            # BLE 真值
            hr_ble, ble_ok, rr_ms = 0, False, 0
            if ble is not None:
                ble_ok = bool(getattr(ble, "connected", False))
                d = ble.get_data() or {}
                hr_ble = int(d.get("hr", 0) or 0)
                rr_ms = int(d.get("hrv", 0) or 0)

            stage_name, stage_desc = STAGES[stage_i]
            if rgb is not None:
                writer.writerow({
                    "t": f"{now:.4f}", "frame": n, "stage": stage_name,
                    "r": f"{rgb[0]:.4f}", "g": f"{rgb[1]:.4f}", "b": f"{rgb[2]:.4f}",
                    "face_found": int(face_ok),
                    "hr_ble": hr_ble, "ble_connected": int(ble_ok),
                    "rr_ms": rr_ms, "drop_marker": drop_marker,
                })
                n += 1
                drop_marker = 0

            elapsed = now - stage_t0
            remain = max(0.0, args.stage_sec - elapsed)
            draw_hud(frame, stage_name, stage_desc, remain, hr_ble, ble_ok, face_ok, n)
            cv2.imshow("HR Collection (SPACE=next  d=drop  q=quit)", frame)

            if remain <= 0:
                stage_i += 1
                stage_t0 = now
                if stage_i < len(STAGES):
                    print(f"[>] 進入階段 {stage_i+1}/{len(STAGES)}: "
                          f"{STAGES[stage_i][0]} — {STAGES[stage_i][1]}")

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                stage_i += 1
                stage_t0 = now
                if stage_i < len(STAGES):
                    print(f"[>] 手動跳至階段 {stage_i+1}/{len(STAGES)}: "
                          f"{STAGES[stage_i][0]} — {STAGES[stage_i][1]}")
            if k == ord("d"):
                drop_marker = 1
                print(f"[!] t={now:.2f} 標記藍牙斷線事件")
    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        rppg.close()
        if ble is not None and hasattr(ble, "_stop_event"):
            ble._stop_event.set()

    dur = n / 30.0
    print(f"\n[OK] 已寫入 {n} 筆 (約 {dur/60:.1f} 分鐘) → {out_csv}")
    print(f"     下一步：python experiments/analyze_hr.py {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
