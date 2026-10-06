import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import threading
import time
from google import genai

# 🆕 載入藍牙心率微服務 (主線)
try:
    from hr_service import HeartRateService
except ImportError:
    HeartRateService = None

# 🆕 載入視覺心率微服務 (備援)
try:
    from rPPG_service import RPPGService
except ImportError:
    RPPGService = None

class StrengthTrainingAgent:
    def __init__(self, cam_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加上 cv2.CAP_DSHOW 解決 Windows MSMF 鏡頭報錯問題
        self.cam_id = cam_id

        # 初始化真實心率變數與感測器狀態
        self.real_heart_rate = 0
        self.hr_sensor_ble = None
        self.hr_sensor_rppg = None
        self.current_hr_source = "None"
        self._ble_was_connected = False # 紀錄藍牙是否曾經連線過

        # rPPG 訊號品質把關：低於門檻不採用，避免把雜訊當心率顯示
        self.rppg_conf = 0.0
        self._rppg_last_ok = 0.0
        self.RPPG_GRACE_SEC = 3.0     # 品質短暫掉落時的寬限期，避免畫面閃爍

        print("🔍 正在初始化多模態生理感測模組...")

        # 1. 優先嘗試：啟動藍牙心率感測器
        if HeartRateService:
            try:
                self.hr_sensor_ble = HeartRateService(address=os.getenv("BLE_HR_ADDRESS"))
                self.current_hr_source = "BLE"
                print("✅ 系統提示：藍牙心率感測器監聽中 (主線)")
            except Exception as e:
                print(f"⚠️ 藍牙連線失敗，準備切換影像心率: {e}")
                self.current_hr_source = "rPPG"

        # 2. 如果一開始就沒藍牙，直接開 rPPG
        if self.current_hr_source == "rPPG" and RPPGService:
            self.hr_sensor_rppg = RPPGService(window_size=300)  # 10 秒視窗：5 秒版頻率解析度僅 12 BPM/bin，實測誤差偏大
            print("✅ 系統提示：已啟動 rPPG 視覺影像心率 (備援接管)")

        print("⏳ 載入 YOLO 視覺追蹤...")
        self.yolo = YOLO('yolov8n-pose.pt')
        print("✅ 系統初始化完成！")

        # 🌟 多運動模式支援
        self.current_exercise = "深蹲"
        self.rep_counts = {"深蹲": 0, "肩推": 0, "划船": 0}
        self.stages = {"深蹲": "UP", "肩推": "DOWN", "划船": "DOWN"}

        self.pose_buffer = []
        self.last_score = 0
        self.feedback = "請準備開始"
        self.current_angle = 180.0

        # 金鑰一律由環境變數提供，不得寫進原始碼 (原本硬寫的金鑰已移除，請自行撤銷)
        #   PowerShell : $env:GEMINI_API_KEY = "你的金鑰"
        #   CMD        : set GEMINI_API_KEY=你的金鑰
        _api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if _api_key:
            self.gemini_client = genai.Client(api_key=_api_key)
        else:
            self.gemini_client = None
            print("⚠️ 未設定 GEMINI_API_KEY，AI 教練講評停用 (其餘功能不受影響)")
        self.gemini_advice = "準備好就開始吧！"
        self.is_calling_gemini = False

    def _ask_gemini_coach(self, rep, score, min_angle, basic_feedback):
        prompt = f"""
        你是一個充滿活力且嚴格的 AI 健身教練。
        玩家剛完成了第 {rep} 下{self.current_exercise}。
        系統診斷：{basic_feedback} (本次角度 {min_angle:.1f}度，評分 {score}/100)
        目前即時心率：{self.real_heart_rate} BPM

        請根據這個表現，用一句話（限 20 字內）給予熱情鼓勵或精準糾正。如果心率大於 150 且分數較低，請務必提醒注意安全。口吻要像真人教練。
        """
        if self.gemini_client is None:
            self.is_calling_gemini = False
            return
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            self.gemini_advice = response.text.strip()
        except Exception as e:
            self.gemini_advice = "節奏很好，繼續保持！"
        finally:
            self.is_calling_gemini = False

    def get_angle(self, kps, p1, p2, p3):
        a, b, c = kps[p1][:2], kps[p2][:2], kps[p3][:2]
        ba, bc = a - b, c - b
        na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if na < 1e-6 or nc < 1e-6: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc + 1e-6), -1.0, 1.0)))

    def detect_action(self, kps):
        left_wrist_y = kps[9][1]
        left_shoulder_y = kps[5][1]
        left_hip_y = kps[11][1]
        if left_wrist_y < left_shoulder_y: return "肩推"
        elif left_wrist_y > left_hip_y: return "划船"
        else: return "深蹲"

    def assess_rep(self):
        if not self.pose_buffer: return
        valid_angles = [a for a in self.pose_buffer if 15 < a < 175]
        if not valid_angles: return
        target_angle = 180.0

        if self.current_exercise == "深蹲":
            target_angle = min(valid_angles)
            if target_angle <= 110: self.last_score, self.feedback = 100, f"標準深蹲 ({target_angle:.0f}°)"
            elif target_angle <= 140: self.last_score, self.feedback = 80, f"半蹲 ({target_angle:.0f}°)"
            else: self.last_score, self.feedback = 40, "蹲太淺了"
        elif self.current_exercise == "肩推":
            target_angle = max(valid_angles)
            if target_angle >= 160: self.last_score, self.feedback = 100, f"標準肩推 ({target_angle:.0f}°)"
            elif target_angle >= 140: self.last_score, self.feedback = 80, f"手未伸直 ({target_angle:.0f}°)"
            else: self.last_score, self.feedback = 40, "推不夠高"
        elif self.current_exercise == "划船":
            target_angle = min(valid_angles)
            if target_angle <= 80: self.last_score, self.feedback = 100, f"標準划船 ({target_angle:.0f}°)"
            elif target_angle <= 110: self.last_score, self.feedback = 80, f"拉不夠深 ({target_angle:.0f}°)"
            else: self.last_score, self.feedback = 40, "動作不完整"

        if not self.is_calling_gemini:
            self.is_calling_gemini = True
            threading.Thread(
                target=self._ask_gemini_coach,
                args=(self.rep_counts[self.current_exercise], self.last_score, target_angle, self.feedback)
            ).start()
        self.pose_buffer = []

    def count_logic(self, kps):
        is_neutral = (
            (self.current_exercise == "深蹲" and self.stages["深蹲"] == "UP") or
            (self.current_exercise == "肩推" and self.stages["肩推"] == "DOWN") or
            (self.current_exercise == "划船" and self.stages["划船"] == "DOWN")
        )

        if is_neutral:
            detected_action = self.detect_action(kps)
            if detected_action != self.current_exercise:
                self.current_exercise = detected_action
                self.pose_buffer = []

        if self.current_exercise == "深蹲":
            self.current_angle = self.get_angle(kps, 11, 13, 15)
            if self.current_angle < 140 and self.stages["深蹲"] == "UP":
                self.stages["深蹲"] = "DOWN"
                self.pose_buffer = []
            elif self.current_angle >= 160 and self.stages["深蹲"] == "DOWN":
                self.stages["深蹲"] = "UP"
                self.rep_counts["深蹲"] += 1
                self.assess_rep()
            if self.stages["深蹲"] == "DOWN": self.pose_buffer.append(self.current_angle)

        elif self.current_exercise == "肩推":
            self.current_angle = self.get_angle(kps, 5, 7, 9)
            if self.current_angle > 140 and self.stages["肩推"] == "DOWN":
                self.stages["肩推"] = "UP"
                self.pose_buffer = []
            elif self.current_angle < 100 and self.stages["肩推"] == "UP":
                self.stages["肩推"] = "DOWN"
                self.rep_counts["肩推"] += 1
                self.assess_rep()
            if self.stages["肩推"] == "UP": self.pose_buffer.append(self.current_angle)

        elif self.current_exercise == "划船":
            self.current_angle = self.get_angle(kps, 5, 7, 9)
            if self.current_angle < 100 and self.stages["划船"] == "DOWN":
                self.stages["划船"] = "UP"
                self.pose_buffer = []
            elif self.current_angle > 140 and self.stages["划船"] == "UP":
                self.stages["划船"] = "DOWN"
                self.rep_counts["划船"] += 1
                self.assess_rep()
            if self.stages["划船"] == "UP": self.pose_buffer.append(self.current_angle)

    def run(self):
        # 使用 cv2.CAP_DSHOW 解決 Windows 鏡頭錯誤
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # ==========================================
            # 🚀 核心容錯機制：雙軌心率自動路由 (Graceful Degradation)
            # ==========================================
            if self.current_hr_source == "BLE" and self.hr_sensor_ble:
                # 追蹤藍牙連線狀態
                if getattr(self.hr_sensor_ble, 'connected', False):
                    self._ble_was_connected = True
                    hr_data = self.hr_sensor_ble.get_data()
                    if hr_data and hr_data.get("hr", 0) > 0:
                        self.real_heart_rate = hr_data["hr"]
                else:
                    # 💡 判斷是否為「非預期斷線」
                    if self._ble_was_connected:
                        print("⚠️ 藍牙訊號異常中斷，系統零延遲切換至影像心率！")
                        self.current_hr_source = "rPPG"
                        if RPPGService and not self.hr_sensor_rppg:
                            self.hr_sensor_rppg = RPPGService(window_size=300)  # 10 秒視窗：5 秒版頻率解析度僅 12 BPM/bin，實測誤差偏大

            elif self.current_hr_source == "rPPG" and self.hr_sensor_rppg:
                # v2 介面：依影格時間戳估計實際 fps、帶通濾波並回報訊號品質。
                # 舊寫法自行 FFT 且把 fps 寫死，桌面版與網頁版串流速率不同時心率會等比例偏移。
                st = self.hr_sensor_rppg.update(frame)
                self.rppg_conf = st['conf']
                if st['valid']:
                    self.real_heart_rate = int(round(st['hr']))
                    self._rppg_last_ok = time.time()
                elif (time.time() - self._rppg_last_ok) > self.RPPG_GRACE_SEC:
                    # 訊號長時間不可信，寧可顯示「--」也不要報假數字
                    self.real_heart_rate = 0

            # ==========================================

            results = self.yolo(frame, verbose=False)
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kps = r.keypoints.data[0].cpu().numpy()
                    self.count_logic(kps)
                    self._draw_skeleton(frame, kps)

            self._draw_ui(frame)
            cv2.imshow("NCU AI Coach - Dual HR System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release(); cv2.destroyAllWindows()

    def _draw_skeleton(self, frame, kps):
        links = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for p1, p2 in links:
            pt1, pt2 = tuple(kps[p1][:2].astype(int)), tuple(kps[p2][:2].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        for i in range(5, 17):
            cv2.circle(frame, (int(kps[i][0]), int(kps[i][1])), 5, (0, 0, 255), -1)

    def _draw_ui(self, frame):
        import time
        if not hasattr(self, 'start_time'): self.start_time = time.time()
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        W, H = img_pil.size
        scale = min(W / 640.0, H / 480.0) * 0.75
        def S(val): return int(val * scale)

        # 維持你原本完美的字體大小設定
        try:
            font_L = ImageFont.truetype("msjh.ttc", S(26))
            font_M = ImageFont.truetype("msjh.ttc", S(18))
            font_S = ImageFont.truetype("msjh.ttc", S(14))
        except:
            try:
                mac_font = "/System/Library/Fonts/STHeiti Light.ttc"
                font_L = ImageFont.truetype(mac_font, S(26))
                font_M = ImageFont.truetype(mac_font, S(18))
                font_S = ImageFont.truetype(mac_font, S(14))
            except:
                font_L = font_M = font_S = ImageFont.load_default()

        # 處理 Gemini 講評與動態斷行
        advice = self.gemini_advice
        chars_per_line = 10
        lines = [advice[i:i+chars_per_line] for i in range(0, len(advice), chars_per_line)]
        if len(lines) > 3:
            lines = lines[:3]
            lines[2] = lines[2][:-1] + ".."

        # 動態計算橫線位置
        next_y = 52 + len(lines) * 20 + 5

        # 🌟 面板高度也動態拉長，保留你原本的底部留白比例
        panel_height = next_y + 173
        panel_box = [S(15), S(15), S(220), S(panel_height)]
        try:
            draw.rounded_rectangle(panel_box, radius=S(20), fill=(40, 30, 20, 210))
        except AttributeError:
            draw.rectangle(panel_box, fill=(40, 30, 20, 210))

        # 頂部時間
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        draw.ellipse([S(25), S(25), S(40), S(40)], outline=(255, 220, 0, 255), width=max(1, S(2)))
        draw.line([(S(32), S(32)), (S(32), S(27))], fill=(255, 220, 0, 255), width=max(1, S(2)))
        draw.text((S(50), S(21)), f"{mins:02d}:{secs:02d}", font=font_M, fill=(255, 220, 0, 255))

        # 繪製講評
        for i, line_text in enumerate(lines):
            draw.text((S(25), S(52 + i * 20)), line_text, font=font_M, fill=(255, 255, 255, 255))

        # 分隔橫線
        draw.line([(S(25), S(next_y)), (S(210), S(next_y))], fill=(100, 90, 80, 150), width=max(1, S(2)))

        # ==========================================
        # 🌟 完美還原：保留你原本垂直錯開的排版與字體大小！
        # ==========================================
        base_y = next_y

        # 左側：動作名稱與次數 (還原你原本的相對間距)
        draw.text((S(25), S(base_y + 10)), f"{self.current_exercise}", font=font_M, fill=(255, 220, 0, 255))
        current_rep = self.rep_counts.get(self.current_exercise, 0)
        draw.text((S(25), S(base_y + 40)), f"{current_rep:02d} 次", font=font_L, fill=(255, 255, 255, 255))

        # 右側：心率圖示與數值 (還原在你原本設計的 Reps 右下方位置)
        heart_color = (255, 60, 60, 255)
        hy = base_y + 63
        draw.ellipse([S(135), S(hy), S(145), S(hy+10)], fill=heart_color)
        draw.ellipse([S(143), S(hy), S(153), S(hy+10)], fill=heart_color)
        draw.polygon([(S(136), S(hy+7)), (S(152), S(hy+7)), (S(144), S(hy+17))], fill=heart_color)

        hr_val = f"{int(self.real_heart_rate)}" if self.real_heart_rate > 0 else "--"
        hr_w = draw.textlength(hr_val, font=font_L)
        draw.text((S(125) - hr_w, S(base_y + 53)), hr_val, font=font_L, fill=(255, 255, 255, 255))

        source_label = f"({self.current_hr_source})" if self.current_hr_source != "None" else ""
        draw.text((S(130), S(base_y + 68)), source_label, font=font_S, fill=(200, 200, 200, 255))

        # 右側下半部：卡路里圖示與計算 (還原在最底部的位置)
        fy = base_y + 118
        fire_color = (255, 100, 50, 255)
        draw.polygon([(S(140), S(fy)), (S(146), S(fy-5)), (S(149), S(fy+3)), (S(156), S(fy)), (S(148), S(fy+13)), (S(140), S(fy+8))], fill=fire_color)
        draw.text((S(162), S(fy+1)), "大卡", font=font_S, fill=fire_color)

        if not hasattr(self, 'total_cal'): self.total_cal = 0.0; self.last_cal_time = time.time()
        dt = time.time() - self.last_cal_time
        self.last_cal_time = time.time()
        self.total_cal += dt * (8.0 / 60.0)

        cal_str = f"{int(self.total_cal)}"
        cal_w = draw.textlength(cal_str, font=font_L)
        draw.text((S(130) - cal_w, S(base_y + 113)), cal_str, font=font_L, fill=(255, 255, 255, 255))

        img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)[:]

    def process_frame(self, frame):
        """專門給網頁端 (FastAPI) 呼叫的單幀處理邏輯，包含雙軌心率路由"""

        # ==========================================
        # 🚀 1. 核心容錯機制：雙軌心率自動路由 (新增超時切換)
        # ==========================================
        if self.current_hr_source == "BLE" and self.hr_sensor_ble:
            if not hasattr(self, 'ble_wait_frames'): self.ble_wait_frames = 0

            # 追蹤藍牙連線狀態
            if getattr(self.hr_sensor_ble, 'connected', False):
                self.ble_wait_frames = 0 # 連上就重置計時器
                self._ble_was_connected = True
                hr_data = self.hr_sensor_ble.get_data()
                if hr_data and hr_data.get("hr", 0) > 0:
                    self.real_heart_rate = hr_data["hr"]
            else:
                self.ble_wait_frames += 1
                # 💡 如果曾經連上卻斷線，或是「開局等了超過 30 幀 (約3秒)」都連不上
                if self._ble_was_connected or self.ble_wait_frames > 30:
                    print("⚠️ 藍牙未連線或逾時，系統自動切換至影像心率 (rPPG)！")
                    self.current_hr_source = "rPPG"
                    if RPPGService and not self.hr_sensor_rppg:
                        self.hr_sensor_rppg = RPPGService(window_size=300)  # 10 秒視窗：5 秒版頻率解析度僅 12 BPM/bin，實測誤差偏大

        elif self.current_hr_source == "rPPG" and self.hr_sensor_rppg:
            # v2 介面：依影格時間戳估計實際 fps、帶通濾波並回報訊號品質。
            # 舊寫法自行 FFT 且把 fps 寫死，桌面版與網頁版串流速率不同時心率會等比例偏移。
            st = self.hr_sensor_rppg.update(frame)
            self.rppg_conf = st['conf']
            if st['valid']:
                self.real_heart_rate = int(round(st['hr']))
                self._rppg_last_ok = time.time()
            elif (time.time() - self._rppg_last_ok) > self.RPPG_GRACE_SEC:
                # 訊號長時間不可信，寧可顯示「--」也不要報假數字
                self.real_heart_rate = 0

        # ==========================================
        # 2. YOLO 視覺辨識與骨架繪製
        # ==========================================
        results = self.yolo(frame, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kps = r.keypoints.data[0].cpu().numpy()
                self.count_logic(kps)
                self._draw_skeleton(frame, kps)

        # ==========================================
        # 3. 繪製 UI 與回傳
        # ==========================================
        self._draw_ui(frame)
        return frame

if __name__ == "__main__":
    c_id = input("請輸入相機 ID (預設 0): "); c_id = int(c_id) if c_id.isdigit() else 0
    agent = StrengthTrainingAgent(cam_id=c_id)
    agent.run()
