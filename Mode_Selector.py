import cv2
import os
import time
import subprocess
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 💡 1. 載入 rPPG 視覺心率微服務
# ---------------------------------------------------------
try:
    from rPPG_service import RPPGService
    print("✅ 成功載入 rPPG 視覺心率微服務 (rPPG_service.py)")
except ImportError:
    print("⚠️ 找不到 rPPG_service.py，請確認檔案在同一層資料夾")

# ---------------------------------------------------------
# 💡 2. 載入 BLE 藍牙心率微服務
# ---------------------------------------------------------
try:
    from hr_service import HeartRateService
    print("✅ 成功載入 BLE 藍牙心率微服務 (hr_service.py)")
except ImportError:
    print("⚠️ 找不到 hr_service.py，請確認檔案在同一層資料夾")


# ==========================================
# 🌐 專為網頁端 (FastAPI) 設計的「無阻擋決策引擎」
# ==========================================
class WebModeSelector:
    HR_THRESHOLD = 85          # 分流門檻 (BPM)：高於此視為已在活動狀態

    def __init__(self, use_hr=True, ble_addr=None):
        print("🔄 初始化網頁版 AI 智能決策引擎...")
        from ultralytics import YOLO
        self.yolo = YOLO('yolov8n-pose.pt')
        self.start_time = None
        self.pose_list = []
        self.decision_made = None

        # 決策引擎自備雙軌心率來源。先前由 app.py 傳入寫死的 85，
        # 使心率條件恆為真，實際上只有姿態在決策。
        self.hr_ble = None
        self.hr_rppg = None
        self.last_hr = 0
        self.hr_source = "none"
        if use_hr:
            if HeartRateService is not None:
                try:
                    self.hr_ble = HeartRateService(address=ble_addr or os.getenv("BLE_HR_ADDRESS"))
                except Exception as e:
                    print(f"⚠️ 決策引擎藍牙啟動失敗: {e}")
            if RPPGService is not None:
                try:
                    self.hr_rppg = RPPGService(window_size=300)
                except Exception as e:
                    print(f"⚠️ 決策引擎 rPPG 啟動失敗: {e}")

    def read_hr(self, frame):
        """讀取當前心率：藍牙優先，其次 rPPG (須通過品質把關)，都沒有則回傳 0。"""
        if self.hr_ble is not None and getattr(self.hr_ble, 'connected', False):
            d = self.hr_ble.get_data() or {}
            hr = int(d.get("hr", 0) or 0)
            if hr > 0:
                self.last_hr, self.hr_source = hr, "BLE"
                return hr
        if self.hr_rppg is not None:
            st = self.hr_rppg.update(frame)
            if st['valid'] and st['hr'] > 0:
                self.last_hr, self.hr_source = int(round(st['hr'])), "rPPG"
                return self.last_hr
        return self.last_hr if self.last_hr > 0 else 0

    def close(self):
        if self.hr_ble is not None and hasattr(self.hr_ble, '_stop_event'):
            self.hr_ble._stop_event.set()
        if self.hr_rppg is not None:
            self.hr_rppg.close()

    def process_frame(self, frame, current_hr=None):
        import time
        import cv2
        import numpy as np

        # current_hr 未指定時由引擎自己的雙軌感測器讀取
        if current_hr is None:
            current_hr = self.read_hr(frame)
        # ✅ 新增 PIL 處理模組，讓系統能畫出完美中文！
        from PIL import Image, ImageDraw, ImageFont

        # 確保在收到第一張照片時，才開始按碼表計時
        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time

        # 1. 抓取骨架判斷姿勢
        results = self.yolo(frame, verbose=False)
        current_pose = "UNKNOWN"
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kps = r.keypoints.data[0].cpu().numpy()
                if kps[5][2] > 0.5 and kps[15][2] > 0.5:
                    h = abs(((kps[15][1]+kps[16][1])/2) - ((kps[5][1]+kps[6][1])/2))
                    w = abs(((kps[15][0]+kps[16][0])/2) - ((kps[5][0]+kps[6][0])/2))
                    current_pose = "STANDING" if h > w else "PRONE"
                    self.pose_list.append(current_pose)

        # ==========================================
        # 2. 畫上科技感的全中文掃描介面 (透過 PIL)
        # ==========================================
        # 先用 OpenCV 畫一個稍微半透明的深色底框
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 160), (40, 30, 20), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # 轉換為 PIL 圖片格式來畫中文字
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 載入中文字體 (附帶 Mac 備援防呆)
        try:
            font_L = ImageFont.truetype("msjh.ttc", 24)
            font_M = ImageFont.truetype("msjh.ttc", 20)
        except:
            try:
                mac_font = "/System/Library/Fonts/STHeiti Light.ttc"
                font_L = ImageFont.truetype(mac_font, 24)
                font_M = ImageFont.truetype(mac_font, 20)
            except:
                font_L = font_M = ImageFont.load_default()

        # 中文化文字與邏輯轉換
        countdown = max(0, 35.0 - elapsed_time)
        if elapsed_time < 5.0:
            status_text = f"🔄 AI 掃描分析中... ({int(elapsed_time)}/5s)"
        else:
            status_text = f"⏳ 決策確認中... {countdown:.1f}s"

        pose_zh = "站姿 (準備重訓)" if current_pose == "STANDING" else ("趴姿 (準備居家)" if current_pose == "PRONE" else "偵測中...")

        # 將文字畫上去
        draw.text((30, 35), status_text, font=font_L, fill=(255, 255, 0))
        draw.text((30, 75), f"🧍 偵測姿態: {pose_zh}", font=font_M, fill=(100, 255, 100))
        draw.text((30, 115), f"❤️ 即時心率: {current_hr} BPM", font=font_M, fill=(255, 200, 200))

        # 畫完後轉回 OpenCV 的 BGR 圖片格式送回網頁
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # ==========================================

        # 3. 決策邏輯
        if elapsed_time > 35.0 or (elapsed_time > 5.0 and current_hr > 0):
            majority_pose = "STANDING"
            if self.pose_list:
                majority_pose = "STANDING" if self.pose_list.count("STANDING") >= self.pose_list.count("PRONE") else "PRONE"

            if majority_pose == "PRONE":
                # 趴姿一律居家，與心率無關
                print("🏠 判定：趴姿 → 【居家模式】")
                self.decision_made = "2"
            elif current_hr <= 0:
                # 沒有可信心率就誠實走純視覺，不要假裝做了多模態決策
                print("⚠️ 未取得可信心率，改採純視覺分流 → 站姿【重訓模式】")
                self.decision_made = "1"
            elif current_hr >= self.HR_THRESHOLD:
                print(f"🚀 判定：站姿 + 心率 {current_hr} >= {self.HR_THRESHOLD} ({self.hr_source}) → 【重訓模式】")
                self.decision_made = "1"
            else:
                print(f"🏠 判定：站姿但心率 {current_hr} < {self.HR_THRESHOLD} ({self.hr_source}) → 【居家模式】")
                self.decision_made = "2"

        return frame, self.decision_made


if __name__ == "__main__":
    # 單獨執行時做一次分流測試。
    # 原本呼叫的 mac_address / cam_id 參數與 run_selection 方法都不存在，
    # 直接執行必然拋 TypeError。
    import cv2
    sel = WebModeSelector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame, decision = sel.process_frame(cv2.flip(frame, 1))
            cv2.imshow("Mode Selector (q=quit)", frame)
            if decision is not None:
                print(f"決策完成：模式 {decision}")
                break
            if cv2.waitKey(1) & 0xFF == 113:      # 'q'
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sel.close()
