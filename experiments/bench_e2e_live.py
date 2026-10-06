"""
bench_e2e_live.py — 實驗 C 補強：真實佈署的端到端量測與多連線壓力測試

bench_transport.py 量的是「本機處理成本 + 依頻寬推估的網路時間」。
這支則是對**真的跑起來的伺服器**量測，補上兩塊推估不來的數據：

  A. 真實端到端延遲與可達影格率（含廣域網路、ngrok 通道、TLS）
  B. 多連線並發下的延遲劣化曲線（驗證 bench_transport 表 5 的承載推估）

另附一個純回聲伺服器模式，用來隔離出「純網路傳輸時間」——
把 S1 大小與 S2 大小的 payload 各打一輪，就能直接比較兩種方案
在同一條網路路徑上的傳輸成本差異，不受伺服器運算影響。

────────────────────────────────────────────────────────────
用法

 1) 量測真實的 app.py（先啟動 uvicorn 與 ngrok）

    python experiments/bench_e2e_live.py --url ws://localhost:8000/ws --frames 100
    python experiments/bench_e2e_live.py --url wss://xxxx.ngrok-free.app/ws --frames 100

 2) 多連線壓力測試（驗證伺服器承載）

    python experiments/bench_e2e_live.py --url ws://localhost:8000/ws \
        --clients 1,2,4,8 --frames 40 --watch-cpu

 3) 純網路傳輸時間（隔離運算成本）

    在要量測的機器上先跑回聲伺服器：
        python experiments/bench_e2e_live.py --serve-echo --port 8899
    再用 ngrok 開通道：
        ngrok http 8899
    然後從客戶端量：
        python experiments/bench_e2e_live.py --echo-url wss://xxxx.ngrok-free.app \
            --frames 100
────────────────────────────────────────────────────────────
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import struct
import sys
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

try:
    import psutil
except ImportError:
    psutil = None


def _connect(url, **kw):
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets.client import connect
    return connect(url, max_size=64 * 1024 * 1024, **kw)


def _serve(handler, host, port):
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets.server import serve
    return serve(handler, host, port, max_size=64 * 1024 * 1024)


# ======================================================================
# 影格來源
# ======================================================================
def load_frames(n, width, height, cam_id=0, source="webcam"):
    frames = []
    if source == "webcam":
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            for _ in range(8):
                cap.read()
            for _ in range(min(n, 60)):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(f, (width, height)))
            cap.release()
    if not frames:
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:height, 0:width]
        bg = (60 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)).astype(np.float32)
        for _ in range(min(n, 30)):
            img = np.dstack([bg * 0.9, bg, bg * 1.1])
            cv2.rectangle(img, (180, 90), (460, 470), (130, 140, 155), -1)
            cv2.circle(img, (320, 130), 52, (150, 165, 180), -1)
            img += rng.normal(0, 4.0, img.shape)
            frames.append(np.clip(img, 0, 255).astype(np.uint8))
        print("[!] 未取得攝影機畫面，改用合成畫面（JPEG 大小會與真實場景有落差）")
    return frames


def encode_s1(frame, quality=40):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode("ascii")
    return json.dumps({"image": f"data:image/jpeg;base64,{b64}"})


def make_s2_payload(kps=None):
    kps = np.zeros((17, 3), dtype=np.float32) if kps is None else kps
    return json.dumps({"kps": np.round(kps, 2).tolist()})


def nbytes(p):
    return len(p) if isinstance(p, (bytes, bytearray)) else len(p.encode("utf-8"))


# ======================================================================
# A. 量測真實的 app.py
# ======================================================================
async def run_client(url, payloads, mode, client_id, warmup=3):
    """單一客戶端：模擬 index.html 的一次一幀、收到回應才送下一幀。"""
    rec = []
    async with _connect(url, open_timeout=30, ping_interval=None) as ws:
        if mode:
            await ws.send(json.dumps({"action": "switch_mode", "mode": mode}))
            try:
                await asyncio.wait_for(ws.recv(), timeout=180)   # 等模型載入
            except asyncio.TimeoutError:
                pass

        for i, p in enumerate(payloads):
            t0 = time.perf_counter()
            await ws.send(p)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                break
            dt = (time.perf_counter() - t0) * 1e3
            if i >= warmup:                     # 前幾幀含暖機，不計入
                rec.append({"client": client_id, "frame": i,
                            "rtt_ms": dt,
                            "up_bytes": nbytes(p), "down_bytes": nbytes(resp)})
        try:
            await ws.send(json.dumps({"action": "stop"}))
        except Exception:
            pass
    return rec


async def run_concurrent(url, payloads, mode, n_clients):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[run_client(url, payloads, mode, c)
                                 for c in range(n_clients)],
                               return_exceptions=True)
    elapsed = time.perf_counter() - t0

    rows, errs = [], []
    for r in res:
        if isinstance(r, Exception):
            errs.append(repr(r))
        else:
            rows.extend(r)
    return rows, elapsed, errs


def server_cpu_snapshot():
    """找出本機的 uvicorn / app.py 行程並回報 CPU 與記憶體。"""
    if psutil is None:
        return None
    best = None
    for pr in psutil.process_iter(["name", "cmdline", "cpu_percent", "memory_info"]):
        try:
            cl = " ".join(pr.info.get("cmdline") or [])
            if "uvicorn" in cl or "app:app" in cl:
                pr.cpu_percent(None)
                best = pr
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if best is None:
        return None
    time.sleep(1.0)
    try:
        return {"cpu_pct": best.cpu_percent(None),
                "rss_mb": best.memory_info().rss / 1e6}
    except Exception:
        return None


# ======================================================================
# B. 純回聲伺服器（隔離網路傳輸時間）
# ======================================================================
async def _echo_handler(ws):
    async for msg in ws:
        await ws.send(msg)


def serve_echo(host, port):
    async def main():
        async with _serve(_echo_handler, host, port):
            print(f"[*] 回聲伺服器已啟動於 ws://{host}:{port}")
            print("    用 ngrok http %d 開通道後，從客戶端執行：" % port)
            print("      python experiments/bench_e2e_live.py --echo-url wss://<你的網址>")
            print("    Ctrl+C 結束")
            await asyncio.Future()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] 已停止")


async def run_echo_bench(url, s1_payloads, s2_payloads, warmup=3):
    out = []
    async with _connect(url, open_timeout=30, ping_interval=None) as ws:
        for label, payloads in (("S1 雲端全幀", s1_payloads), ("S2 骨架卸載", s2_payloads)):
            for i, p in enumerate(payloads):
                t0 = time.perf_counter()
                await ws.send(p)
                await asyncio.wait_for(ws.recv(), timeout=60)
                dt = (time.perf_counter() - t0) * 1e3
                if i >= warmup:
                    out.append({"scheme": label, "rtt_ms": dt, "bytes": nbytes(p)})
    return out


# ======================================================================
def summarize(rtts):
    if not rtts:
        return {}
    return {
        "n": len(rtts),
        "mean_ms": statistics.mean(rtts),
        "median_ms": statistics.median(rtts),
        "p95_ms": float(np.percentile(rtts, 95)),
        "p99_ms": float(np.percentile(rtts, 99)),
        "max_ms": max(rtts),
        "fps": 1000.0 / statistics.mean(rtts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="app.py 的 WebSocket 位址，例如 ws://localhost:8000/ws")
    ap.add_argument("--echo-url", help="回聲伺服器位址（純網路量測）")
    ap.add_argument("--serve-echo", action="store_true", help="啟動回聲伺服器")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--mode", default="2", help="1=重訓 2=居家 auto=自動決策")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--clients", default="1", help="並發數，可用逗號列出多組如 1,2,4,8")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--jpeg-q", type=int, default=40)
    ap.add_argument("--source", choices=["webcam", "synthetic"], default="webcam")
    ap.add_argument("--watch-cpu", action="store_true", help="量測本機伺服器行程的 CPU")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    if args.serve_echo:
        serve_echo(args.host, args.port)
        return 0

    if not args.url and not args.echo_url:
        ap.error("請指定 --url 或 --echo-url，或用 --serve-echo 啟動回聲伺服器")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "e2e_live")

    print(f"[*] 準備 {args.frames} 幀測試影像 ...")
    frames = load_frames(args.frames, args.width, args.height, source=args.source)
    s1 = [encode_s1(frames[i % len(frames)], args.jpeg_q) for i in range(args.frames)]

    # ---------------- 純網路量測 ----------------
    if args.echo_url:
        s2 = [make_s2_payload() for _ in range(args.frames)]
        print(f"[*] 回聲量測 → {args.echo_url}")
        print(f"    S1 payload {nbytes(s1[0]):,} B   S2 payload {nbytes(s2[0]):,} B")
        rows = asyncio.run(run_echo_bench(args.echo_url, s1, s2))
        df = pd.DataFrame(rows)
        df.to_csv(f"{base}_echo_raw.csv", index=False, encoding="utf-8-sig")

        tbl = []
        for scheme, g in df.groupby("scheme"):
            s = summarize(g["rtt_ms"].tolist())
            s.update({"方案": scheme, "payload B": int(g["bytes"].iloc[0])})
            tbl.append(s)
        t = pd.DataFrame(tbl)[["方案", "payload B", "n", "median_ms",
                               "mean_ms", "p95_ms", "p99_ms", "fps"]]
        print("\n" + t.to_markdown(index=False, floatfmt=".2f"))
        t.to_csv(f"{base}_echo_summary.csv", index=False, encoding="utf-8-sig")

        lines = ["# 實驗 C 補強　純網路傳輸量測（回聲伺服器）", "",
                 f"- 位址：`{args.echo_url}`",
                 f"- 每方案 {args.frames} 幀（前 3 幀暖機不計）",
                 "- 伺服器只做回聲，不含任何運算，因此差異純粹來自傳輸量", "",
                 t.to_markdown(index=False, floatfmt=".2f"), ""]
        if len(t) == 2:
            a, b = t.iloc[0], t.iloc[1]
            big, small = (a, b) if a["payload B"] >= b["payload B"] else (b, a)
            lines += ["## 結論", "",
                      f"- payload 由 {big['payload B']:,} B 降至 {small['payload B']:,} B "
                      f"（{big['payload B']/max(1,small['payload B']):.0f} 倍）",
                      f"- 中位數往返時間由 {big['median_ms']:.1f} ms 降至 "
                      f"{small['median_ms']:.1f} ms",
                      f"- P95 由 {big['p95_ms']:.1f} ms 降至 {small['p95_ms']:.1f} ms", ""]
        with open(f"{base}_echo_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[OK] 報告 → {base}_echo_report.md")
        return 0

    # ---------------- 真實 app.py 量測 ----------------
    client_counts = [int(c) for c in args.clients.split(",") if c.strip()]
    print(f"[*] 目標 {args.url}   模式 {args.mode}   並發 {client_counts}")
    print(f"    S1 payload 平均 {np.mean([nbytes(p) for p in s1]):,.0f} B/幀")

    all_rows, summary = [], []
    for n in client_counts:
        print(f"\n[*] {n} 個並發連線 ...")
        cpu_before = server_cpu_snapshot() if args.watch_cpu else None
        rows, elapsed, errs = asyncio.run(run_concurrent(args.url, s1, args.mode, n))
        cpu_after = server_cpu_snapshot() if args.watch_cpu else None

        if errs:
            print(f"    [!] {len(errs)} 個連線發生錯誤: {errs[0][:110]}")
        if not rows:
            print("    [X] 沒有取得任何有效回應，請確認伺服器是否已啟動")
            continue

        for r in rows:
            r["n_clients"] = n
        all_rows.extend(rows)

        df = pd.DataFrame(rows)
        s = summarize(df["rtt_ms"].tolist())
        s["n_clients"] = n
        s["總吞吐 fps"] = len(df) / elapsed if elapsed > 0 else 0
        s["每連線 fps"] = s["總吞吐 fps"] / n
        s["上行 B/幀"] = float(df["up_bytes"].mean())
        s["下行 B/幀"] = float(df["down_bytes"].mean())
        s["總頻寬 Mbps"] = ((df["up_bytes"].mean() + df["down_bytes"].mean())
                            * 8 * s["總吞吐 fps"] / 1e6)
        if cpu_after:
            s["伺服器 CPU %"] = cpu_after["cpu_pct"]
            s["伺服器 RSS MB"] = cpu_after["rss_mb"]
        summary.append(s)
        print(f"    中位 RTT {s['median_ms']:7.1f} ms   P95 {s['p95_ms']:7.1f} ms   "
              f"每連線 {s['每連線 fps']:.2f} fps   總吞吐 {s['總吞吐 fps']:.2f} fps")

    if not summary:
        print("\n[X] 沒有任何量測結果")
        return 1

    pd.DataFrame(all_rows).to_csv(f"{base}_raw.csv", index=False, encoding="utf-8-sig")
    st = pd.DataFrame(summary)
    cols = ["n_clients", "n", "median_ms", "mean_ms", "p95_ms", "p99_ms",
            "每連線 fps", "總吞吐 fps", "上行 B/幀", "下行 B/幀", "總頻寬 Mbps"]
    cols += [c for c in ("伺服器 CPU %", "伺服器 RSS MB") if c in st.columns]
    st = st[cols]
    st.to_csv(f"{base}_summary.csv", index=False, encoding="utf-8-sig")

    # --- 圖 ---
    figs = []
    if len(st) > 1:
        fig, ax1 = plt.subplots(figsize=(8, 4.4))
        ax1.plot(st["n_clients"], st["median_ms"], "o-", color="#d62728", label="中位 RTT")
        ax1.plot(st["n_clients"], st["p95_ms"], "s--", color="#ff7f0e", label="P95 RTT")
        ax1.set_xlabel("並發連線數")
        ax1.set_ylabel("往返延遲 (ms)")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(st["n_clients"], st["每連線 fps"], "^-", color="#1f77b4", label="每連線 fps")
        ax2.set_ylabel("每連線影格率 (fps)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=9)
        ax1.set_title("並發連線對延遲與影格率的影響")
        fig.tight_layout()
        p = f"{base}_fig_scaling.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    lines = ["# 實驗 C 補強　真實佈署端到端量測", "",
             f"- 目標：`{args.url}`",
             f"- 模式：{args.mode}",
             f"- 每連線 {args.frames} 幀（前 3 幀暖機不計）",
             f"- 影格來源：{args.source}，{args.width}×{args.height}，JPEG 品質 {args.jpeg_q}",
             "",
             "## 量測結果", "",
             st.to_markdown(index=False, floatfmt=".2f"), ""]

    one = st[st["n_clients"] == 1]
    if len(one):
        r = one.iloc[0]
        lines += ["## 單連線基準", "",
                  f"- 中位往返延遲 **{r['median_ms']:.1f} ms**，P95 **{r['p95_ms']:.1f} ms**",
                  f"- 可達影格率 **{r['每連線 fps']:.2f} fps**",
                  f"- 實際頻寬 **{r['總頻寬 Mbps']:.2f} Mbps**", ""]
    if len(st) > 1:
        f0, fl = st.iloc[0], st.iloc[-1]
        lines += ["## 並發劣化", "",
                  f"- 連線數由 {int(f0['n_clients'])} 增至 {int(fl['n_clients'])} 時，"
                  f"中位延遲由 {f0['median_ms']:.1f} ms 升至 {fl['median_ms']:.1f} ms "
                  f"（{fl['median_ms']/max(1e-9,f0['median_ms']):.1f} 倍）",
                  f"- 每連線影格率由 {f0['每連線 fps']:.2f} 降至 {fl['每連線 fps']:.2f} fps",
                  "", "此曲線可用來驗證 `transport_report.md` 表 5 的伺服器承載推估。", ""]

    lines += ["## 說明", "",
              "- 本數據包含真實網路路徑（含 ngrok 通道與 TLS），",
              "  與 `bench_transport.py` 的推估值互為對照。",
              "- 量測的是現況 S1 架構。S2 骨架卸載需在伺服器實作對應端點後才能同法量測；",
              "  若只想比較兩者的網路傳輸成本，請改用 `--serve-echo` / `--echo-url` 模式。", ""]
    if figs:
        lines += ["## 圖檔", ""] + [f"- `{os.path.basename(f)}`" for f in figs]

    with open(f"{base}_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 72)
    print(st.to_markdown(index=False, floatfmt=".2f"))
    print("=" * 72)
    print(f"\n[OK] 報告 → {base}_report.md")
    for f in figs:
        print(f"     圖   → {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
