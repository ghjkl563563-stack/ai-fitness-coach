import asyncio
import threading
import time
from bleak import BleakClient, BleakScanner

HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

class HeartRateService:
    def __init__(self, address=None):
        self.target_addr = address
        self.data = {"hr": 0, "hrv": 0, "rr_intervals": []}
        self.connected = False
        self._stop_event = threading.Event()

        # 啟動背景執行緒 (Background Worker)
        threading.Thread(target=self._run_loop, daemon=True).start()

    def get_data(self):
        return self.data

    def _parse_data(self, sender, data):
        """解析 BLE 標準心率封包 (含 HRV)"""
        flag = data[0]
        offset = 1

        # 1. 解析 BPM (8-bit or 16-bit)
        if flag & 0x01:
            hr = int.from_bytes(data[offset:offset+2], "little")
            offset += 2
        else:
            hr = data[offset]
            offset += 1

        # 2. 解析 RR-Interval (HRV) - 期刊關鍵數據
        rr_ms = 0
        if (flag >> 4) & 0x01: # 檢查 RR-Interval Bit
            # 可能有多個 RR 值，取最後一個
            while offset + 2 <= len(data):
                val = int.from_bytes(data[offset:offset+2], "little")
                rr_ms = int(val / 1024.0 * 1000) # 轉換單位為 ms
                offset += 2

        self.data["hr"] = hr
        if rr_ms > 0:
            self.data["hrv"] = rr_ms # 這裡簡單用 RR 代表 HRV，進階可算 SDNN
            self.data["rr_intervals"].append(rr_ms)
            if len(self.data["rr_intervals"]) > 50: self.data["rr_intervals"].pop(0)

        # print(f"❤️ HR: {hr} | RR: {rr_ms}ms")

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            while not self._stop_event.is_set():
                # 自動掃描 (Auto-Discovery)
                addr = self.target_addr
                if not addr:
                    print("[HR] Scanning for Heart Rate Sensors...")
                    devs = await BleakScanner.discover()
                    for d in devs:
                        if "Heart" in str(d.name) or "HR" in str(d.name):
                            addr = d.address
                            break

                if not addr:
                    await asyncio.sleep(5)
                    continue

                # 自動重連邏輯 (Auto-Reconnect)
                try:
                    print(f"[HR] Connecting to {addr}...")
                    async with BleakClient(addr, timeout=10.0) as client:
                        self.connected = True
                        print("[HR] Connected!")
                        await client.start_notify(HR_UUID, self._parse_data)

                        while client.is_connected and not self._stop_event.is_set():
                            await asyncio.sleep(1)

                except Exception as e:
                    print(f"[HR] Connection Lost: {e}")
                    self.connected = False
                    await asyncio.sleep(3) # 冷卻後重試

        loop.run_until_complete(main())
