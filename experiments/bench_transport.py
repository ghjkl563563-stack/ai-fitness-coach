"""
bench_transport.py — 實驗 C：端-雲協同架構的傳輸量與延遲比較

比較三種部署方式，量化「把運算放在哪一端」對頻寬與延遲的影響：

  S1  雲端全幀 (現況)   瀏覽器把 JPEG 影格經 base64 送上雲端，
                        伺服器跑完 YOLO+評分後把「畫好標註的影格」再送回來。
                        → app.py 目前的作法

  S2  骨架卸載          瀏覽器端先跑姿態估計，只上傳 17 個關節點座標，
                        伺服器跑評分後只回傳分數與講評文字。

  S3  全端側運算        所有運算都在裝置上，不使用網路。

量測項目：
    上行/下行每影格位元組數 (含 base64 與 JSON 封裝開銷)
    各段處理耗時：編碼 / 推論 / 解碼
    WebSocket 迴路實測往返時間 (可用 --no-ws 略過)
    不同網路條件下的端到端延遲推估

用法：
    python experiments/bench_transport.py                     # 用攝影機
    python experiments/bench_transport.py --frames 60
    python experiments/bench_transport.py --source synthetic  # 無攝影機時
    python experiments/bench_transport.py --no-ws             # 略過 WebSocket 實測
"""
import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei",
                                   "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 代表性網路條件 (名稱, 上行 Mbps, 下行 Mbps, RTT ms)
NET_PROFILES = [
    ("家用 Wi-Fi", 50.0, 200.0, 15.0),
    ("4G LTE", 10.0, 40.0, 50.0),
    ("5G", 60.0, 200.0, 25.0),
    ("行動網路壅塞", 2.0, 8.0, 120.0),
]


# ======================================================================
# 影格來源
# ======================================================================
def get_frames(source, n, width, height, cam_id=0):
    """取得測試影格。JPEG 大小高度依賴畫面內容，務必優先使用真實攝影機畫面。"""
    frames = []
    if source == "webcam":
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            for _ in range(10):       # 丟掉前幾張曝光未穩定的
                cap.read()
            for _ in range(n):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(f, (width, height)))
            cap.release()
        if frames:
            return frames, "webcam"
        print("[!] 攝影機無法使用，改用合成畫面")

    # 合成畫面：刻意做出接近真實室內場景的紋理複雜度，
    # 純雜訊會高估 JPEG 大小、純色塊會低估。
    rng = np.random.default_rng(7)
    for i in range(n):
        yy, xx = np.mgrid[0:height, 0:width]
        bg = (60 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)).astype(np.float32)
        img = np.dstack([bg * 0.9, bg * 1.0, bg * 1.1])
        cv2.rectangle(img, (180, 90), (460, 470), (130, 140, 155), -1)      # 人形
        cv2.circle(img, (320, 130), 52, (150, 165, 180), -1)                 # 頭
        cv2.rectangle(img, (0, 400), (width, height), (70, 75, 85), -1)      # 地板
        img += rng.normal(0, 4.0, img.shape)                                 # 感測器雜訊
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames, "synthetic"


# ======================================================================
# 各方案的封裝
# ======================================================================
def pack_s1_uplink(frame, quality=40):
    """S1 上行：JPEG → base64 → JSON (完全比照 index.html 的作法)"""
    t0 = time.perf_counter()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    t1 = time.perf_counter()
    b64 = base64.b64encode(buf).decode("ascii")
    payload = json.dumps({"image": f"data:image/jpeg;base64,{b64}"})
    t2 = time.perf_counter()
    return payload, {"jpeg_bytes": len(buf), "encode_ms": (t1 - t0) * 1e3,
                     "pack_ms": (t2 - t1) * 1e3}


def pack_s1_downlink(frame, advice, quality=70):
    """S1 下行：伺服器把標註後的影格再編一次送回"""
    t0 = time.perf_counter()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("ascii")
    payload = json.dumps({"image": f"data:image/jpeg;base64,{b64}", "advice": advice})
    return payload, {"encode_ms": (time.perf_counter() - t0) * 1e3}


def pack_s2_uplink(kps, mode="json"):
    """
    S2 上行：只送 17×3 關節點。
    json   → 可讀性佳，與現有 WebSocket 協定一致
    binary → float32 緊湊封裝，理論下限
    """
    t0 = time.perf_counter()
    if mode == "binary":
        payload = struct.pack(f"<{kps.size}f", *kps.astype(np.float32).ravel())
    else:
        payload = json.dumps({"kps": np.round(kps, 2).tolist()})
    return payload, {"pack_ms": (time.perf_counter() - t0) * 1e3}


def pack_s2_downlink(score, action, reps, advice):
    """S2 下行：只送結構化結果"""
    return json.dumps({"score": score, "action": action,
                       "reps": reps, "advice": advice}), {}


def nbytes(p):
    return len(p) if isinstance(p, (bytes, bytearray)) else len(p.encode("utf-8"))


# ======================================================================
# WebSocket 迴路實測
# ======================================================================
async def _ws_server(port, stop_evt):
    import websockets
    async def handler(ws):
        async for msg in ws:
            await ws.send(msg if isinstance(msg, str) else msg)
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets.server import serve
    async with serve(handler, "127.0.0.1", port, max_size=32 * 1024 * 1024):
        while not stop_evt.is_set():
            await asyncio.sleep(0.05)


async def _ws_client_rt(port, payloads):
    import websockets
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets.client import connect
    out = []
    async with connect(f"ws://127.0.0.1:{port}", max_size=32 * 1024 * 1024) as ws:
        for p in payloads:                       # 暖機
            await ws.send(p); await ws.recv()
            break
        for p in payloads:
            t0 = time.perf_counter()
            await ws.send(p)
            await ws.recv()
            out.append((time.perf_counter() - t0) * 1e3)
    return out


def ws_roundtrip(payloads, port=8799):
    """在本機跑一個真實 WebSocket 迴路，量測框架封裝與序列化開銷。"""
    stop = threading.Event()
    res = {}

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_ws_server(port, stop))

    th = threading.Thread(target=run_server, daemon=True)
    th.start()
    time.sleep(0.8)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res["rt_ms"] = loop.run_until_complete(_ws_client_rt(port, payloads))
        loop.close()
    except Exception as e:
        print(f"[!] WebSocket 迴路測試失敗，略過: {e}")
        res["rt_ms"] = []
    finally:
        stop.set()
    return res.get("rt_ms", [])


# ======================================================================
# 評分成本量測
# ======================================================================
def measure_scoring_ms(n=15):
    """
    實測伺服器端「單次動作評分」的耗時 (ST-GCN backbone + 範本比對)。

    這段成本 S1 與 S2 都要付，先前版本把 S2 的伺服器成本寫死成 3 ms，
    嚴重低估。實測在 CPU 上單次約 50~60 ms，其中 ST-GCN 佔約 45 ms。

    回傳 (單次評分 ms, 範本數)；載入失敗時回傳 (None, 0)。
    """
    try:
        import numpy as _np
        from AI_Agent_crossatt_multimodal_merged import FitnessAIAgent
    except Exception as e:
        print(f"[!] 無法載入評分模型，將沿用 --scoring-ms 給定值: {e}")
        return None, 0

    try:
        agent = FitnessAIAgent(weight_path="best_model 1.pth",
                               stgcn_path="models/gcn_weight.pth",
                               exemplar_path="exemplar_bank.pt",
                               use_sensors=False)
    except Exception as e:
        print(f"[!] 建立 FitnessAIAgent 失敗: {e}")
        return None, 0

    rng = _np.random.default_rng(0)
    for _ in range(55):
        agent.pose_history.append(rng.random((17, 2)) * 300)

    agent.assess_quality(force=True)          # 暖機
    t0 = time.perf_counter()
    for _ in range(n):
        agent.assess_quality(force=True)      # force 繞過節流，量單次真實成本
    ms = (time.perf_counter() - t0) / n * 1e3

    n_ex = 0
    try:
        n_ex = sum(len(v) if isinstance(v, list) else 1
                   for v in agent.exemplar_bank.get("qevd", {}).values())
    except Exception:
        pass
    del agent
    return ms, n_ex


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--source", choices=["webcam", "synthetic"], default="webcam")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--jpeg-up", type=int, default=40, help="上行 JPEG 品質 (index.html 用 0.4)")
    ap.add_argument("--jpeg-down", type=int, default=70, help="下行 JPEG 品質 (app.py 用 70)")
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--scoring-ms", type=float, default=None,
                    help="單次動作評分耗時 (ms)。不給則實際載入模型量測")
    ap.add_argument("--skip-scoring-measure", action="store_true",
                    help="不載入評分模型，改用 --scoring-ms 或預設值")
    ap.add_argument("--assess-hz", type=float, default=5.0,
                    help="評分更新頻率 (Hz)，對應 FitnessAIAgent.ASSESS_INTERVAL_S")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "transport")

    print(f"[*] 取得 {args.frames} 張測試影格 ({args.width}x{args.height}) ...")
    frames, src = get_frames(args.source, args.frames, args.width, args.height)
    print(f"[*] 影格來源：{src}（共 {len(frames)} 張）")
    if src == "synthetic":
        print("    注意：JPEG 大小與畫面內容高度相關，論文數據請務必用真實攝影機重跑")

    print("[*] 載入 YOLOv8n-pose ...")
    from ultralytics import YOLO
    yolo = YOLO("yolov8n-pose.pt")
    yolo(frames[0], verbose=False)          # 暖機

    rows = []
    s1_up_payloads, s2_up_payloads = [], []

    for i, f in enumerate(frames):
        rec = {"frame": i}

        # --- 姿態推論 (S2 在端側、S1 在雲端，耗時相同，量一次即可) ---
        t0 = time.perf_counter()
        res = yolo(f, verbose=False)
        rec["yolo_ms"] = (time.perf_counter() - t0) * 1e3

        kps = None
        for r in res:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kps = r.keypoints.data[0].cpu().numpy()
                break
        if kps is None:
            kps = np.zeros((17, 3), dtype=np.float32)
        rec["has_person"] = int(kps.any())

        # --- S1 上行 ---
        p1u, m1u = pack_s1_uplink(f, args.jpeg_up)
        rec["s1_up_bytes"] = nbytes(p1u)
        rec["s1_jpeg_bytes"] = m1u["jpeg_bytes"]
        rec["s1_encode_ms"] = m1u["encode_ms"]

        # --- S1 下行：伺服器畫完標註再編碼回傳 ---
        annotated = f.copy()
        for (x, y, c) in kps:
            if c > 0.3:
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
        p1d, m1d = pack_s1_downlink(annotated, "保持核心收緊，再深一點", args.jpeg_down)
        rec["s1_down_bytes"] = nbytes(p1d)
        rec["s1_down_encode_ms"] = m1d["encode_ms"]

        # --- S2 上行 / 下行 ---
        p2u, m2u = pack_s2_uplink(kps, "json")
        p2ub, _ = pack_s2_uplink(kps, "binary")
        rec["s2_up_bytes"] = nbytes(p2u)
        rec["s2_up_bytes_binary"] = nbytes(p2ub)
        rec["s2_pack_ms"] = m2u["pack_ms"]

        p2d, _ = pack_s2_downlink(87, "squat", 12, "保持核心收緊，再深一點")
        rec["s2_down_bytes"] = nbytes(p2d)

        # --- 客戶端解碼 (S1 才需要，S2 無影像回傳) ---
        t0 = time.perf_counter()
        raw = base64.b64decode(json.loads(p1d)["image"].split(",", 1)[1])
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        rec["s1_decode_ms"] = (time.perf_counter() - t0) * 1e3

        s1_up_payloads.append(p1u)
        s2_up_payloads.append(p2u)
        rows.append(rec)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(frames)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{base}_raw.csv", index=False, encoding="utf-8-sig")

    # --- 伺服器端評分成本 ---
    n_exemplars = 0
    if args.scoring_ms is not None:
        scoring_ms = float(args.scoring_ms)
        scoring_src = "使用者指定"
    elif args.skip_scoring_measure:
        scoring_ms, scoring_src = 55.0, "預設值 (未實測)"
    else:
        print("[*] 量測伺服器端評分成本 (載入 ST-GCN + 範本庫) ...")
        m, n_exemplars = measure_scoring_ms()
        if m is None:
            scoring_ms, scoring_src = 55.0, "預設值 (量測失敗)"
        else:
            scoring_ms, scoring_src = m, "實測"
    print(f"[*] 單次評分 {scoring_ms:.1f} ms ({scoring_src})"
          + (f"，範本數 {n_exemplars}" if n_exemplars else ""))

    # --- WebSocket 迴路實測 ---
    ws_s1 = ws_s2 = []
    if not args.no_ws:
        print("[*] WebSocket 迴路實測 ...")
        ws_s1 = ws_roundtrip(s1_up_payloads, 8799)
        ws_s2 = ws_roundtrip(s2_up_payloads, 8800)

    # ==================================================================
    # 彙整
    # ==================================================================
    M = df.mean(numeric_only=True)

    s1_up, s1_dn = M["s1_up_bytes"], M["s1_down_bytes"]
    s2_up, s2_dn = M["s2_up_bytes"], M["s2_down_bytes"]
    s2_upb = M["s2_up_bytes_binary"]

    payload_tbl = pd.DataFrame([
        {"方案": "S1 雲端全幀 (現況)", "上行 B/影格": s1_up, "下行 B/影格": s1_dn,
         "合計 B/影格": s1_up + s1_dn},
        {"方案": "S2 骨架卸載 (JSON)", "上行 B/影格": s2_up, "下行 B/影格": s2_dn,
         "合計 B/影格": s2_up + s2_dn},
        {"方案": "S2 骨架卸載 (binary)", "上行 B/影格": s2_upb, "下行 B/影格": s2_dn,
         "合計 B/影格": s2_upb + s2_dn},
        {"方案": "S3 全端側運算", "上行 B/影格": 0.0, "下行 B/影格": 0.0,
         "合計 B/影格": 0.0},
    ])
    payload_tbl["相對 S1"] = payload_tbl["合計 B/影格"] / max(1e-9, s1_up + s1_dn)

    # 頻寬需求
    bw_rows = []
    for fps in (10, 15, 30):
        for name, up, dn in (("S1 雲端全幀", s1_up, s1_dn),
                             ("S2 骨架卸載", s2_up, s2_dn),
                             ("S3 全端側", 0.0, 0.0)):
            bw_rows.append({
                "fps": fps, "方案": name,
                "上行 Mbps": up * 8 * fps / 1e6,
                "下行 Mbps": dn * 8 * fps / 1e6,
                "合計 Mbps": (up + dn) * 8 * fps / 1e6,
            })
    bw_tbl = pd.DataFrame(bw_rows)

    # 伺服器端每影格成本。
    # 評分 (ST-GCN + 範本比對) 兩種方案都要跑，但它以 assess_hz 節流，
    # 因此攤提到每影格的成本是「單次評分 x 評分頻率 / 影格率」。
    # S1 另外還要每影格跑 YOLO 與 JPEG 編碼；S2 這兩項都在客戶端。
    def _amortized_scoring(fps):
        return scoring_ms * min(args.assess_hz, fps) / max(1e-9, fps)

    srv_s1_ms = M["yolo_ms"] + M["s1_down_encode_ms"] + _amortized_scoring(30.0)
    srv_s2_ms = _amortized_scoring(30.0)

    # 端到端延遲推估
    lat_rows = []
    for pname, up_mbps, dn_mbps, rtt in NET_PROFILES:
        # S1：客戶端編碼 → 上行 → 伺服器推論+編碼 → 下行 → 客戶端解碼
        s1 = {
            "網路": pname, "方案": "S1 雲端全幀",
            "客戶端編碼": M["s1_encode_ms"],
            "上行傳輸": s1_up * 8 / (up_mbps * 1e6) * 1e3,
            "伺服器推論": srv_s1_ms,
            "下行傳輸": s1_dn * 8 / (dn_mbps * 1e6) * 1e3,
            "客戶端解碼": M["s1_decode_ms"],
            "RTT": rtt,
        }
        # S2：客戶端推論 → 上行 → 伺服器評分 → 下行
        s2 = {
            "網路": pname, "方案": "S2 骨架卸載",
            "客戶端編碼": M["yolo_ms"] + M["s2_pack_ms"],
            "上行傳輸": s2_up * 8 / (up_mbps * 1e6) * 1e3,
            "伺服器推論": srv_s2_ms,
            "下行傳輸": s2_dn * 8 / (dn_mbps * 1e6) * 1e3,
            "客戶端解碼": 0.0,
            "RTT": rtt,
        }
        s3 = {
            "網路": pname, "方案": "S3 全端側",
            "客戶端編碼": M["yolo_ms"], "上行傳輸": 0.0,
            "伺服器推論": srv_s2_ms, "下行傳輸": 0.0,
            "客戶端解碼": 0.0, "RTT": 0.0,
        }
        for d in (s1, s2, s3):
            d["總計 ms"] = sum(v for k, v in d.items()
                              if k not in ("網路", "方案") and isinstance(v, float))
            lat_rows.append(d)
    lat_tbl = pd.DataFrame(lat_rows)

    # 伺服器承載能力 (攤提成本定義見上方)
    cap_rows = []
    for cores in (1, 4, 8):
        for fps in (10, 30):
            s1_ms = M["yolo_ms"] + M["s1_down_encode_ms"] + _amortized_scoring(fps)
            s2_ms = _amortized_scoring(fps)
            cap_rows.append({
                "CPU 核心數": cores, "串流 fps": fps,
                "S1 伺服器 ms/影格": s1_ms,
                "S1 可承載連線數": cores * 1000.0 / (s1_ms * fps),
                "S2 伺服器 ms/影格": s2_ms,
                "S2 可承載連線數": cores * 1000.0 / (s2_ms * fps),
            })
    cap_tbl = pd.DataFrame(cap_rows)
    cap_tbl.to_csv(f"{base}_capacity.csv", index=False, encoding="utf-8-sig")

    payload_tbl.to_csv(f"{base}_payload.csv", index=False, encoding="utf-8-sig")
    bw_tbl.to_csv(f"{base}_bandwidth.csv", index=False, encoding="utf-8-sig")
    lat_tbl.to_csv(f"{base}_latency.csv", index=False, encoding="utf-8-sig")

    # ==================================================================
    # 圖表
    # ==================================================================
    figs = []

    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = ["S1 雲端全幀\n(現況)", "S2 骨架卸載\n(JSON)", "S2 骨架卸載\n(binary)"]
    vals = [s1_up + s1_dn, s2_up + s2_dn, s2_upb + s2_dn]
    bars = ax.bar(names, vals, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_yscale("log")
    ax.set_ylabel("每影格位元組 (log)")
    ax.set_title("單影格傳輸量比較")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15,
                f"{v:,.0f} B\n({(s1_up+s1_dn)/v:.0f}×)", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = f"{base}_fig_payload.png"; fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, c in (("S1 雲端全幀", "#d62728"), ("S2 骨架卸載", "#1f77b4")):
        sub = bw_tbl[bw_tbl["方案"] == name]
        ax.plot(sub["fps"], sub["合計 Mbps"], "o-", color=c, lw=2, label=name)
    ax.axhline(10.0, color="gray", ls="--", lw=1.2, label="4G 上行典型上限 10 Mbps")
    ax.set_xlabel("串流影格率 (fps)")
    ax.set_ylabel("所需頻寬 (Mbps)")
    ax.set_yscale("log")
    ax.set_title("頻寬需求 vs 影格率")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = f"{base}_fig_bandwidth.png"; fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    parts = ["客戶端編碼", "上行傳輸", "伺服器推論", "下行傳輸", "客戶端解碼", "RTT"]
    colors = ["#8dd3c7", "#fb8072", "#bebada", "#fdb462", "#80b1d3", "#b3de69"]
    sub = lat_tbl[lat_tbl["方案"] != "S3 全端側"]
    labels = [f"{r['網路']}\n{r['方案'].split()[0]}" for _, r in sub.iterrows()]
    bottom = np.zeros(len(sub))
    x = np.arange(len(sub))
    for part, c in zip(parts, colors):
        v = sub[part].to_numpy(float)
        ax.bar(x, v, 0.66, bottom=bottom, label=part, color=c)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("延遲 (ms)")
    ax.set_title("端到端延遲組成 (依網路條件)")
    ax.legend(fontsize=8, ncol=6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = f"{base}_fig_latency.png"; fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # ==================================================================
    # 報告
    # ==================================================================
    ratio = (s1_up + s1_dn) / max(1e-9, s2_up + s2_dn)

    # 取總延遲最大的網路情境，用實際算出的數字而非寫死值
    _w = lat_tbl[lat_tbl["方案"] == "S1 雲端全幀"].nlargest(1, "總計 ms").iloc[0]
    _w2 = lat_tbl[(lat_tbl["方案"] == "S2 骨架卸載") &
                  (lat_tbl["網路"] == _w["網路"])].iloc[0]
    _worst = (_w["網路"], float(_w2["總計 ms"]), float(_w["總計 ms"]))
    lines = [
        "# 實驗 C　端-雲協同傳輸效能比較",
        "",
        f"- 影格來源：**{src}**，解析度 {args.width}×{args.height}，樣本 {len(frames)} 張",
        f"- 上行 JPEG 品質 {args.jpeg_up} (對應 index.html 的 `toDataURL('image/jpeg', 0.4)`)",
        f"- 下行 JPEG 品質 {args.jpeg_down} (對應 app.py 的 `IMWRITE_JPEG_QUALITY, 70`)",
        f"- YOLOv8n-pose 推論：平均 {M['yolo_ms']:.1f} ms/影格 (CPU)",
        f"- 單次動作評分：{scoring_ms:.1f} ms（{scoring_src}），"
        f"以 {args.assess_hz:.0f} Hz 節流，30 fps 下攤提為 "
        f"{_amortized_scoring(30.0):.1f} ms/影格",
        "",
    ]
    if src == "synthetic":
        lines += ["> ⚠️ 目前使用合成畫面，JPEG 大小與真實場景有落差。",
                  "> 論文數據請接上攝影機重跑：`python experiments/bench_transport.py`", ""]

    lines += [
        "## 表 1　單影格傳輸量",
        "",
        payload_tbl.to_markdown(index=False, floatfmt=(".0f", ".0f", ".0f", ".0f", ".4f")),
        "",
        "## 表 2　頻寬需求",
        "",
        bw_tbl.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 表 3　端到端延遲推估",
        "",
        lat_tbl.to_markdown(index=False, floatfmt=".1f"),
        "",
    ]
    if ws_s1 and ws_s2:
        lines += [
            "## 表 4　WebSocket 迴路實測 (本機，已排除廣域網路延遲)",
            "",
            f"| 方案 | 中位數 RTT (ms) | P95 (ms) |",
            f"|---|---|---|",
            f"| S1 雲端全幀 | {np.median(ws_s1):.2f} | {np.percentile(ws_s1, 95):.2f} |",
            f"| S2 骨架卸載 | {np.median(ws_s2):.2f} | {np.percentile(ws_s2, 95):.2f} |",
            "",
            "此項僅反映序列化與框架開銷，真實佈署另需加上廣域網路傳輸時間。",
            "",
        ]

    lines += [
        "## 表 5　伺服器承載能力",
        "",
        cap_tbl.to_markdown(index=False, floatfmt=(".0f", ".0f", ".1f", ".1f", ".1f", ".1f")),
        "",
        "## 重點結論",
        "",
        f"- **頻寬**：S2 使每影格傳輸量由 **{s1_up+s1_dn:,.0f} B** 降至 "
        f"**{s2_up+s2_dn:,.0f} B**，減少 **{ratio:.0f} 倍**；"
        f"30 fps 下由 **{(s1_up+s1_dn)*8*30/1e6:.2f} Mbps** 降至 "
        f"**{(s2_up+s2_dn)*8*30/1e6:.3f} Mbps**。改用 binary 封裝可再降至 "
        f"**{s2_upb+s2_dn:,.0f} B/影格**。",
        f"- **伺服器承載**：S1 每連線每影格佔用伺服器 **{srv_s1_ms:.1f} ms**，"
        f"S2 僅 **{srv_s2_ms:.1f} ms**，單核在 30 fps 下可承載連線數由 "
        f"**{1000.0/(srv_s1_ms*30):.1f}** 提升至 **{1000.0/(srv_s2_ms*30):.1f}**。",
        "- **延遲**：兩者差距有限，因總延遲被姿態推論主導"
        f" (本機 CPU 實測 {M['yolo_ms']:.1f} ms)。S2 的優勢要在網路品質差時才明顯"
        f" ({_worst[0]} 情境下 S2 {_worst[1]:.0f} ms vs S1 {_worst[2]:.0f} ms)。",
        "",
        "## 限制與說明",
        "",
        f"1. 本測試在 CPU 上執行 YOLOv8n-pose ({M['yolo_ms']:.1f} ms/影格)。S2 把這段"
        "移到客戶端，若客戶端是手機瀏覽器，實務上會改用 TensorFlow.js 或 MediaPipe"
        " 的 WebGL 加速模型，耗時與本數據不同，延遲結論需依實際客戶端重新量測。",
        "2. 廣域網路傳輸時間為依頻寬推估，非實地量測。若要取得實測值，"
        "可在 ngrok 佈署上以相同 payload 大小測量往返時間後代入。",
        "3. 頻寬與伺服器承載兩項結論不受上述限制影響 —— 兩者只取決於"
        "傳輸量與伺服器端工作量，與客戶端硬體無關。",
        "",
        "## 圖檔",
        "",
    ] + [f"- `{os.path.basename(f)}`" for f in figs]

    rep = f"{base}_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 72)
    print(payload_tbl.to_markdown(index=False, floatfmt=".0f"))
    print("\n頻寬 @30fps:")
    print(bw_tbl[bw_tbl.fps == 30].to_markdown(index=False, floatfmt=".3f"))
    print("=" * 72)
    print(f"\n[OK] 報告 → {rep}")
    for f in figs:
        print(f"     圖   → {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
