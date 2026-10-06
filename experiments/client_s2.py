"""
client_s2.py — S2 骨架卸載架構的參考客戶端

這是實驗 C 所提架構的實際實作（而非推估）：
姿態估計與生理感測都留在端側，只有 17 個關節點與心率會經網路上傳，
伺服器回傳評分等結構化結果，全程沒有任何影像離開本機。

對照組是現況的 S1：瀏覽器上傳 JPEG 影格，伺服器回傳標註後的影格。

用法
    # 先啟動伺服器
    uvicorn app:app --host 0.0.0.0 --port 8000

    # 互動模式（開視窗，可看到評分與次數）
    python experiments/client_s2.py --url ws://localhost:8000/ws_skeleton

    # 經 ngrok
    python experiments/client_s2.py --url wss://xxxx.ngrok-free.app/ws_skeleton

    # 不使用心率感測器
    python experiments/client_s2.py --url ws://localhost:8000/ws_skeleton --no-hr
"""
import argparse
import asyncio
import json
import os
import statistics
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


def _connect(url):
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets.client import connect
    return connect(url, max_size=8 * 1024 * 1024, ping_interval=None, open_timeout=30)


def nbytes(p):
    return len(p) if isinstance(p, (bytes, bytearray)) else len(p.encode("utf-8"))


class LocalSensing:
    """端側生理感測：藍牙為主、rPPG 備援，與伺服器端的雙軌邏輯一致。"""

    GRACE_SEC = 3.0

    def __init__(self, use_hr=True, ble_addr=None):
        self.ble = None
        self.rppg = None
        self.hr = 0
        self.source = "none"
        self._last_ok = 0.0
        if not use_hr:
            return
        if HeartRateService is not None:
            try:
                self.ble = HeartRateService(address=ble_addr or os.getenv("BLE_HR_ADDRESS"))
            except Exception as e:
                print(f"⚠️ 藍牙啟動失敗: {e}")
        try:
            self.rppg = RPPGService(window_size=300)
        except Exception as e:
            print(f"⚠️ rPPG 啟動失敗: {e}")

    def update(self, frame):
        if self.ble is not None and getattr(self.ble, "connected", False):
            d = self.ble.get_data() or {}
            v = int(d.get("hr", 0) or 0)
            if v > 0:
                self.hr, self.source, self._last_ok = v, "BLE", time.time()
                return self.hr
        if self.rppg is not None:
            st = self.rppg.update(frame)
            if st["valid"] and st["hr"] > 0:
                self.hr, self.source, self._last_ok = int(round(st["hr"])), "rPPG", time.time()
                return self.hr
        if time.time() - self._last_ok > self.GRACE_SEC:
            self.hr, self.source = 0, "none"
        return self.hr

    def close(self):
        if self.ble is not None and hasattr(self.ble, "_stop_event"):
            self.ble._stop_event.set()
        if self.rppg is not None:
            self.rppg.close()


def draw_overlay(frame, kps, res, stats):
    for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                 (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
        if kps[a][2] > 0.3 and kps[b][2] > 0.3:
            cv2.line(frame, tuple(kps[a][:2].astype(int)),
                     tuple(kps[b][:2].astype(int)), (0, 255, 255), 2)
    for i in range(5, 17):
        if kps[i][2] > 0.3:
            cv2.circle(frame, (int(kps[i][0]), int(kps[i][1])), 4, (0, 255, 0), -1)

    cv2.rectangle(frame, (10, 10), (330, 150), (35, 30, 25), -1)
    rep = (f"{res.get('hold_s', 0):.0f}s hold" if res.get("hold_s")
           else f"{res.get('reps', 0)} reps")
    rows = [
        (f"S2 SKELETON OFFLOAD", (120, 220, 255)),
        (f"score {res.get('score', 0):>3}   {rep}", (255, 255, 255)),
        (f"HR {res.get('hr', 0) or '--'}  ({stats['hr_src']})", (120, 255, 120)),
        (f"RTT {stats['rtt']:.0f} ms   up {stats['up']:,} B", (200, 200, 200)),
    ]
    for i, (txt, col) in enumerate(rows):
        cv2.putText(frame, txt, (22, 42 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return frame


async def main_async(args):
    from ultralytics import YOLO

    print("[*] 載入 YOLOv8n-pose（端側執行）...")
    yolo = YOLO("yolov8n-pose.pt")

    sens = LocalSensing(use_hr=not args.no_hr, ble_addr=args.ble_addr)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("[X] 無法開啟攝影機")
        return 1

    rtts, ups, downs, poses = [], [], [], []
    t_start = time.time()
    n = 0

    try:
        print(f"[*] 連線 {args.url} ...")
        async with _connect(args.url) as ws:
            await ws.send(json.dumps({"action": "start"}))
            print("[*] 等待伺服器載入模型（首次可能需要數十秒）...")
            ack = await asyncio.wait_for(ws.recv(), timeout=300)
            print(f"[*] 伺服器就緒: {ack}")

            res = {}
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                # --- 端側：姿態估計 ---
                t0 = time.perf_counter()
                out = yolo(frame, verbose=False)
                kps = None
                for r in out:
                    if r.keypoints is not None and len(r.keypoints.data) > 0:
                        kps = r.keypoints.data[0].cpu().numpy()
                        break
                poses.append((time.perf_counter() - t0) * 1e3)
                if kps is None:
                    cv2.imshow("S2 Client (q=quit)", frame)
                    if cv2.waitKey(1) & 0xFF == 113:
                        break
                    continue

                # --- 端側：生理感測 ---
                hr = sens.update(frame)

                # --- 只上傳關節點與心率 ---
                payload = json.dumps({
                    "kps": np.round(kps.astype(float), 2).tolist(),
                    "hr": int(hr),
                })
                t1 = time.perf_counter()
                await ws.send(payload)
                reply = await asyncio.wait_for(ws.recv(), timeout=30)
                rtt = (time.perf_counter() - t1) * 1e3

                res = json.loads(reply)
                if "error" in res:
                    print("[X] 伺服器回報:", res["error"])
                    break

                n += 1
                if n > 3:                      # 前幾幀暖機不計
                    rtts.append(rtt)
                    ups.append(nbytes(payload))
                    downs.append(nbytes(reply))

                stats = {"rtt": rtt, "up": nbytes(payload), "hr_src": sens.source}
                if not args.headless:
                    cv2.imshow("S2 Client (q=quit)",
                               draw_overlay(frame, kps, res, stats))
                    if cv2.waitKey(1) & 0xFF == 113:
                        break
                if args.frames and n >= args.frames:
                    break

            try:
                await ws.send(json.dumps({"action": "stop"}))
            except Exception:
                pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sens.close()

    if not rtts:
        print("[X] 沒有取得有效量測")
        return 1

    elapsed = time.time() - t_start
    fps = n / elapsed if elapsed > 0 else 0
    up, dn = statistics.mean(ups), statistics.mean(downs)

    print("\n" + "=" * 58)
    print("S2 骨架卸載 — 實測結果")
    print("=" * 58)
    print(f"  影格數            {n}  ({elapsed:.1f} 秒, {fps:.1f} fps)")
    print(f"  端側姿態估計      {statistics.mean(poses):7.1f} ms/影格")
    print(f"  伺服器往返 (中位) {statistics.median(rtts):7.1f} ms")
    print(f"  伺服器往返 (P95)  {np.percentile(rtts, 95):7.1f} ms")
    print(f"  上行              {up:7.0f} B/影格")
    print(f"  下行              {dn:7.0f} B/影格")
    print(f"  實際頻寬          {(up+dn)*8*fps/1e6:7.3f} Mbps")
    print("=" * 58)
    print("  對照：S1 現況架構約 31,000 B/影格、7.6 Mbps @30fps")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws_skeleton")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-hr", action="store_true", help="不啟用心率感測")
    ap.add_argument("--ble-addr", default=os.getenv("BLE_HR_ADDRESS"))
    ap.add_argument("--frames", type=int, default=0, help="跑滿 N 幀後自動結束 (0=不限)")
    ap.add_argument("--headless", action="store_true", help="不開視窗，純量測")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
