"""
AI_Agent_crossatt_multimodal_merged.py 【V45 優雅降級版 + 最新 Gemini 2.5 API】
- 核心邏輯：執行 L2 歸一化後計算 (User - Exemplar) 殘差
- 數值對齊：將 0~1 的模型輸出映射回 0~100 分
- 防錯機制：修正了先前導致 SyntaxError 的縮排與換行問題
- AI 教練：整合最新 google-genai 套件與 gemini-2.5-flash 模型
- 🚀 新增：雙軌並行生理感測 (Sensor Fusion) 與優雅降級 (Graceful Degradation) 機制
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import threading
import time
from google import genai

# 🆕 載入藍牙心率微服務
try:
    from hr_service import HeartRateService
except ImportError:
    HeartRateService = None

# 🆕 載入視覺心率微服務 (rPPG)
try:
    from rPPG_service import RPPGService
except ImportError:
    RPPGService = None

# --- 導入模型組件 ---
try:
    from models.stgcn_encoder import STGCNFeatureExtractor, remap21_to_25, normalize_ntu_skeleton
    from models.cross_att_pose import PoseCrossAttModel
    from rep_counter import RepCounter
    print("✅ 成功啟動多模態融合引擎 (ST-GCN + Cross-Attention)")
except ImportError as e:
    print(f"⚠️ 找不到模型檔案: {e}")


class FitnessAIAgent:
    # 評分節流間隔 (秒)。assess_quality 分析的是 50 幀滑動視窗，
    # 每一影格都重算等於每秒做 30 次幾乎相同的推論。
    # 實測 ST-GCN backbone 單次約 45 ms，是伺服器端最大的成本來源。
    # 以 5 Hz 更新對教練回饋而言已足夠，可省下約 8 成算力。
    ASSESS_INTERVAL_S = 0.2

    def __init__(self, weight_path, stgcn_path, exemplar_path, use_sensors=True):
        """
        use_sensors : 是否啟動端側生理感測 (BLE + rPPG)。
                      S2 骨架卸載架構下伺服器沒有影像、也不接感測器，
                      心率由客戶端隨關節點一起上傳，此時應設 False —
                      否則會白白開啟藍牙掃描執行緒與 MediaPipe。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================
        # 🌟 生理感測變數與優雅降級狀態機
        # ==========================================
        self.real_heart_rate = 0
        self.hr_ble = 0
        self.hr_rppg = 0
        self.ble_connected = False
        self._ble_was_connected = False
        self.is_graceful_degradation = False
        self.system_start_time = time.time()

        self.hr_sensor_ble = None
        self.hr_sensor_rppg = None

        # rPPG 訊號品質把關：低於門檻不採用，避免把雜訊當心率顯示
        self.rppg_conf = 0.0
        self._rppg_last_ok = 0.0
        self.RPPG_GRACE_SEC = 3.0     # 品質短暫掉落時的寬限期，避免畫面閃爍

        if use_sensors:
            print("🔍 正在初始化居家模組：雙軌並行生理感測...")

            # 1. 啟動藍牙心率感測器 (若有)
            if HeartRateService:
                try:
                    self.hr_sensor_ble = HeartRateService(address=os.getenv("BLE_HR_ADDRESS"))
                    print("✅ 居家模組：藍牙心率感測器監聽中...")
                except Exception as e:
                    print(f"⚠️ 藍牙連線失敗: {e}")

            # 2. 不管有沒有藍牙，一律常駐啟動 rPPG 影像心率！
            if RPPGService:
                try:
                    self.hr_sensor_rppg = RPPGService(window_size=300)  # 10 秒視窗：5 秒版頻率解析度僅 12 BPM/bin，實測誤差偏大
                    print("✅ 居家模組：rPPG 視覺影像心率已常駐啟動！")
                except Exception as e:
                    print(f"❌ 影像心率啟動失敗: {e}")
        else:
            print("🔍 伺服器模式：不啟動端側生理感測 (心率由客戶端提供)")

        # ==========================================

        # 1. 載入 ST-GCN 視覺骨幹
        self.backbone = STGCNFeatureExtractor().to(self.device)
        if os.path.exists(stgcn_path):
            s_dict = torch.load(stgcn_path, map_location=self.device)
            self.backbone.load_state_dict({k.replace('model.', ''): v for k, v in s_dict.items()}, strict=False)

        # 狀態變數
        self.score, self.action, self.feedback = 0, "偵測中", "請運動"
        self.diff_val = 0.0

        # 以骨架動作訊號計次 (取代原本的計時器作法，見 rep_counter.py)
        self.rep_counter = RepCounter()
        self._last_assess_t = 0.0
        self.rep_counts = {}

        # 2. 載入範本庫
        if os.path.exists(exemplar_path):
            self.exemplar_bank = torch.load(exemplar_path, map_location=self.device)
            print("📚 範本庫載入成功")
        else:
            self.exemplar_bank = {}
            print("❌ 找不到 exemplar_bank.pt")

        # 3. 載入融合模型
        self.fusion_model = self._load_model(weight_path)

        self.backbone.eval()
        self.fusion_model.eval()
        self.yolo = YOLO('yolov8n-pose.pt')
        self.pose_history = deque(maxlen=50)

        # 狀態變數
        self.score, self.action, self.feedback = 0, "偵測中", "請運動"
        self.diff_val = 0.0

        # 金鑰一律由環境變數提供，不得寫進原始碼 (原本硬寫的金鑰已移除，請自行撤銷)
        #   PowerShell : $env:GEMINI_API_KEY = "你的金鑰"
        #   CMD        : set GEMINI_API_KEY=你的金鑰
        _api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if _api_key:
            self.gemini_client = genai.Client(api_key=_api_key)
        else:
            self.gemini_client = None
            print("⚠️ 未設定 GEMINI_API_KEY，AI 教練講評停用 (其餘功能不受影響)")
        self.gemini_advice = "等待 AI 教練講評中..."
        self.is_calling_gemini = False

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 找不到權重檔 {path}，使用初始參數")
            return PoseCrossAttModel().to(self.device)

        sd = torch.load(path, map_location=self.device)
        clean_sd = {k.replace('fusion_model.', '').replace('visual_backbone.model.', '').replace('visual_backbone.', ''): v for k, v in sd.items()}

        h_dim = clean_sd.get('score_head.weight', torch.zeros(1, 128)).shape[1]
        a_hid = clean_sd.get('aux_mlp.0.weight', torch.zeros(128, 1)).shape[0]

        model = PoseCrossAttModel(hidden=h_dim, aux_hidden=a_hid).to(self.device)
        model.load_state_dict(clean_sd, strict=False)
        return model

    def assess_quality(self, force=False):
        if len(self.pose_history) < 50: return

        # 節流：距上次評分未達 ASSESS_INTERVAL_S 就跳過。
        # force=True 可強制執行 (供離線量測使用)。
        if not force:
            _now = time.time()
            if _now - getattr(self, '_last_assess_t', 0.0) < self.ASSESS_INTERVAL_S:
                return
            self._last_assess_t = _now

        poses = np.array(list(self.pose_history))
        pose_norm = normalize_ntu_skeleton(remap21_to_25(np.pad(poses, ((0,0),(0,4),(0,1))))[None, ...])[0]
        pose_tensor = torch.from_numpy(pose_norm).permute(2, 0, 1).unsqueeze(-1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_feat = self.backbone(pose_tensor)
            target_domain = 'qevd'
            q_feat_unit = F.normalize(q_feat, p=2, dim=1)

            best_action = "未知動作"
            min_diff = float('inf')
            best_kv_feat_unit = None

            if target_domain in self.exemplar_bank:
                for action_name, exemplars in self.exemplar_bank[target_domain].items():
                    if isinstance(exemplars, list):
                        kv_feat = torch.stack(exemplars).mean(0).to(self.device).view(1, -1)
                    else:
                        kv_feat = exemplars.mean(0).to(self.device).view(1, -1)

                    kv_feat_unit = F.normalize(kv_feat, p=2, dim=1)
                    diff_tensor_temp = q_feat_unit - kv_feat_unit
                    diff_val_temp = torch.abs(diff_tensor_temp).mean().item()

                    if diff_val_temp < min_diff:
                        min_diff = diff_val_temp
                        best_action = action_name
                        best_kv_feat_unit = kv_feat_unit
                        best_diff_tensor = diff_tensor_temp

            if best_kv_feat_unit is None: return

            self.action = best_action
            self.diff_val = min_diff
            diff_tensor = best_diff_tensor
            self.diff_val = torch.abs(diff_tensor).mean().item()
            diff_feat_3d = diff_tensor.unsqueeze(1)

            # 將即時心率正規化後送入融合模型的 aux 分支。
            # 正規化方式必須與訓練時一致 (見 models/mmfit_dataset.py)：
            #     (HR - 100) / 50，再 clamp 到 [-5, 5]
            # 心率取不到時用 0.0 (對應訓練資料的中心值 100 BPM)，
            # 而非沿用先前寫死的 -0.5 (等同假設心率恆為 75)。
            hr_val = float(getattr(self, 'real_heart_rate', 0) or 0)
            hr_scaled = (hr_val - 100.0) / 50.0 if hr_val > 0 else 0.0
            hr_scaled = float(np.clip(hr_scaled, -5.0, 5.0))
            hr_norm = torch.tensor([[hr_scaled]], dtype=torch.float32).to(self.device)

            logits_cls, score, logits_err, _ = self.fusion_model(diff_feat_3d, aux=hr_norm)
            self.score = int(np.clip(score.item() * 100, 0, 100))

            # 次數由 rep_counter 在主迴圈逐影格計算，此處不再用計時器累加。
            # 原作法是「分數>=50 且距上次>1.5 秒就 +1」，只要姿勢維持得好
            # 就會每 1.5 秒加一次，與實際做了幾下無關。

            err_idx = torch.argmax(logits_err).item()
            print(f"📊 [Inference] Score:{self.score} | UnitDiff:{self.diff_val:.6f}")

            if not self.is_calling_gemini:
                self.is_calling_gemini = True
                threading.Thread(
                    target=self._ask_gemini_coach,
                    args=(self.action, self.score, self.feedback)
                ).start()

    def run(self, camera_id=0):
        # 使用 cv2.CAP_DSHOW 解決 Windows 鏡頭讀取錯誤
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # 鏡像翻轉

            # ==========================================
            # 🚀 雙軌並行監測與優雅降級 (Graceful Degradation)
            # ==========================================
            # 1. 讀取實體藍牙心率 (主線)
            if self.hr_sensor_ble and getattr(self.hr_sensor_ble, 'connected', False):
                self.ble_connected = True
                self._ble_was_connected = True
                self.is_graceful_degradation = False

                hr_data = self.hr_sensor_ble.get_data()
                if hr_data and hr_data.get("hr", 0) > 0:
                    self.hr_ble = hr_data["hr"]
            else:
                self.ble_connected = False
                # 觸發優雅降級
                time_elapsed = time.time() - self.system_start_time
                if (self._ble_was_connected or time_elapsed > 10) and not self.is_graceful_degradation:
                    print("\n⚠️ [優雅降級啟動] 居家模式：藍牙異常，系統由 rPPG 接管！\n")
                    self.is_graceful_degradation = True

            # 2. 永遠常駐讀取 rPPG 影像心率
            if self.hr_sensor_rppg:
                # v2 介面：依影格時間戳估計實際 fps、帶通濾波並回報訊號品質。
                # 舊寫法在此自行 FFT 且把 fps 寫死 30，攝影機實際幀率不同時
                # 心率會等比例偏移 (例如實際 25fps 會低估 17%)。
                st = self.hr_sensor_rppg.update(frame)
                self.rppg_conf = st['conf']
                if st['valid']:
                    self.hr_rppg = int(round(st['hr']))
                    self._rppg_last_ok = time.time()

            # 3. 決策最終心率：藍牙為主軌；斷線時才由 rPPG 接管，
            #    且須通過訊號品質把關，超過寬限期仍不可信則顯示「--」
            if self.ble_connected:
                self.real_heart_rate = self.hr_ble
            elif self.hr_rppg > 0 and (time.time() - self._rppg_last_ok) <= self.RPPG_GRACE_SEC:
                self.real_heart_rate = self.hr_rppg
            else:
                self.real_heart_rate = 0
            # ==========================================

            results = self.yolo(frame, verbose=False)
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kps = r.keypoints.data[0].cpu().numpy()
                    self.pose_history.append(kps[:, :2])

                    # 每一影格都餵給計次器 (它自己判斷動作循環)
                    self.rep_counter.update(kps, self.action)

                    self._draw_skeleton(frame, kps)

                    if len(self.pose_history) == 50:
                        self.assess_quality()

            self._display_ui(frame)
            cv2.imshow("NCU Fitness Coach V45 (Dual HR)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release(); cv2.destroyAllWindows()

    def _draw_skeleton(self, frame, kps):
        links = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for link in links:
            p1 = tuple(kps[link[0]][:2].astype(int))
            p2 = tuple(kps[link[1]][:2].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
        for i, kp in enumerate(kps):
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

    def _display_ui(self, frame):
        import time
        if not hasattr(self, 'start_time'): self.start_time = time.time()

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        W, H = img_pil.size
        scale = min(W / 640.0, H / 480.0) * 0.75

        def S(val):
            return int(val * scale)

        # 完美字體設定
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

        # 🌟 動態拉長面板高度，保留底部的完美比例
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
        # 🌟 完美還原：左大右小錯開排版，加上「動作次數」
        # ==========================================
        base_y = next_y

        action_key = self.action.lower().replace("_", " ").replace("-", " ").strip()

        # 把截圖上的動作也加進來了！
        action_dict = {
            "high plank": "高姿棒式",
            "plank": "棒式",
            "plank preparation": "棒式預備",
            "squat": "深蹲",
            "squats": "深蹲",
            "push up": "伏地挺身",
            "lunge": "弓箭步"
        }

        action_name = "分析中" if self.action == "偵測中" else action_dict.get(action_key, self.action)
        draw.text((S(25), S(base_y + 10)), f"{action_name}", font=font_M, fill=(255, 220, 0, 255))

        # 棒式等靜態支撐顯示持續秒數，其餘動作顯示次數
        rc = getattr(self, 'rep_counter', None)
        if rc is not None and rc.hold_seconds.get(self.action):
            hold = rc.hold_seconds[self.action]
            rep_text = f"{int(hold//60):01d}:{int(hold%60):02d}"
        else:
            rep_text = f"{(rc.count_of(self.action) if rc else 0):02d} 次"
        draw.text((S(25), S(base_y + 40)), rep_text, font=font_L, fill=(255, 255, 255, 255))

        # 右側：心率圖示與數值 (排在次數的右下方)
        heart_color = (255, 60, 60, 255)
        hy = base_y + 63
        draw.ellipse([S(135), S(hy), S(145), S(hy+10)], fill=heart_color)
        draw.ellipse([S(143), S(hy), S(153), S(hy+10)], fill=heart_color)
        draw.polygon([(S(136), S(hy+7)), (S(152), S(hy+7)), (S(144), S(hy+17))], fill=heart_color)

        hr_val = f"{int(self.real_heart_rate)}" if self.real_heart_rate > 0 else "--"
        hr_w = draw.textlength(hr_val, font=font_L)
        draw.text((S(125) - hr_w, S(base_y + 53)), hr_val, font=font_L, fill=(255, 255, 255, 255))

        # 右側下半部：卡路里圖示與計算
        fy = base_y + 118
        fire_color = (255, 100, 50, 255)
        draw.polygon([(S(140), S(fy)), (S(146), S(fy-5)), (S(149), S(fy+3)), (S(156), S(fy)), (S(148), S(fy+13)), (S(140), S(fy+8))], fill=fire_color)
        draw.text((S(162), S(fy+1)), "大卡", font=font_S, fill=fire_color)

        if not hasattr(self, 'total_cal'): self.total_cal = 0.0; self.last_cal_time = time.time()
        dt = time.time() - self.last_cal_time
        self.last_cal_time = time.time()
        if self.action != "偵測中":
            self.total_cal += dt * (self._kcal_per_min() / 60.0)

        cal_str = f"{int(self.total_cal)}"
        cal_w = draw.textlength(cal_str, font=font_L)
        draw.text((S(130) - cal_w, S(base_y + 113)), cal_str, font=font_L, fill=(255, 255, 255, 255))

        img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)[:]

    # 各動作的 MET 值 (Ainsworth 2011 Compendium of Physical Activities)
    MET_TABLE = {
        "plank": 3.8, "high plank": 3.8, "elbow plank": 3.8, "side plank": 3.8,
        "plank preparation": 2.8, "plank taps": 4.0,
        "squat": 5.0, "squats": 5.0, "squat jacks": 8.0,
        "push up": 8.0, "lunge": 4.0, "alternating forward lunges": 4.0,
        "high kicks": 6.0, "running in place": 8.0, "quick feet": 8.0,
        "air jump rope": 8.8, "jumping jacks": 8.0,
        "child pose": 2.0, "downward dog (frontal)": 2.5,
        "tree pose": 2.3, "warrior 1 (left)": 2.8, "warrior 2 (left)": 2.8,
    }
    DEFAULT_MET = 4.0

    def _kcal_per_min(self):
        """
        能量消耗估算。有心率時採 Keytel et al. (2005) 的心率迴歸式，
        無心率時退回依動作查 MET 表；兩者都比原本固定 8 kcal/min 合理
        (固定值等於假設全程都在高強度運動)。

        受試者體重/年齡/性別可由環境變數覆寫：
            USER_WEIGHT_KG / USER_AGE / USER_SEX (M 或 F)
        """
        import os as _os
        w = float(_os.getenv("USER_WEIGHT_KG", 65.0))
        age = float(_os.getenv("USER_AGE", 22.0))
        sex = _os.getenv("USER_SEX", "M").upper()

        hr = float(getattr(self, 'real_heart_rate', 0) or 0)
        if hr > 0:
            # Keytel 公式輸出為 kJ/min，除以 4.184 換算為 kcal/min
            if sex == "F":
                kj = -20.4022 + 0.4472 * hr - 0.1263 * w + 0.074 * age
            else:
                kj = -55.0969 + 0.6309 * hr + 0.1988 * w + 0.2017 * age
            kcal = kj / 4.184
            if kcal > 0:
                return min(kcal, 25.0)      # 上限防呆

        key = str(self.action).lower().replace("_", " ").replace("-", " ").strip()
        met = self.MET_TABLE.get(key, self.DEFAULT_MET)
        return met * 3.5 * w / 200.0        # 標準 MET → kcal/min 換算

    def _ask_gemini_coach(self, action, score, basic_feedback):
        prompt = f"""
        你是一個充滿活力的 AI 健身教練。
        目前玩家正在進行的動作：{action}
        目前的動作分數：{score}/100
        系統初步診斷：{basic_feedback}
        目前即時心率：{self.real_heart_rate} BPM

        請用一句話（限 20 字內）給予玩家熱情的鼓勵或精準的姿勢糾正建議。如果心率過高請提醒休息。口吻要像真正的教練。
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
            print(f"Gemini API 錯誤: {e}")
            self.gemini_advice = "繼續保持！"
        finally:
            self.is_calling_gemini = False

    # 確保網頁串流呼叫時，也能吃到雙軌心率邏輯
    def process_keypoints(self, kps, hr=None):
        """
        S2 骨架卸載路徑：只接收關節點，全程不碰影像。

        與 process_frame 的差別：
          - 姿態估計已在客戶端完成，伺服器不跑 YOLO
          - 伺服器沒有影像，因此無法跑 rPPG；心率改由客戶端提供
            (S2 架構下攝影機與心率感測器都在端側，這是架構使然而非限制)
          - 不回傳任何影像，只回傳結構化結果

        這同時是隱私上的優勢：臉部影像完全不離開使用者裝置。

        參數
            kps : (17, 3) 的 COCO-17 關節點 [x, y, conf]
            hr  : 客戶端量到的心率 (BPM)，沒有就傳 None

        回傳 dict，可直接序列化為 JSON。
        """
        import numpy as _np

        kps = _np.asarray(kps, dtype=_np.float64)
        if kps.ndim != 2 or kps.shape[0] < 17:
            return {"error": "keypoints 格式錯誤，需為 (17, 3)"}
        if kps.shape[1] == 2:                       # 允許客戶端省略信心值
            kps = _np.concatenate([kps, _np.ones((kps.shape[0], 1))], axis=1)

        # 心率由客戶端提供；沒有就維持 0，融合模型會用訓練資料中心值代入
        self.real_heart_rate = int(hr) if hr and hr > 0 else 0

        self.pose_history.append(kps[:, :2])
        self.rep_counter.update(kps, self.action)

        if len(self.pose_history) >= 50:
            self.assess_quality()

        rc = self.rep_counter
        hold = rc.hold_seconds.get(self.action, 0.0)
        return {
            "score": int(self.score),
            "action": self.action,
            "reps": rc.count_of(self.action),
            "hold_s": round(float(hold), 1),
            "hr": int(self.real_heart_rate),
            "advice": self.gemini_advice,
            "ready": len(self.pose_history) >= 50,
        }


    def process_frame(self, frame):
        # 1. 雙軌並行監測與優雅降級
        if self.hr_sensor_ble and getattr(self.hr_sensor_ble, 'connected', False):
            self.ble_connected = True
            self._ble_was_connected = True
            self.is_graceful_degradation = False
            hr_data = self.hr_sensor_ble.get_data()
            if hr_data and hr_data.get("hr", 0) > 0:
                self.hr_ble = hr_data["hr"]
        else:
            self.ble_connected = False
            time_elapsed = time.time() - self.system_start_time
            if (self._ble_was_connected or time_elapsed > 10) and not self.is_graceful_degradation:
                print("\n⚠️ [優雅降級啟動] 網頁模式：藍牙異常，系統由 rPPG 接管！\n")
                self.is_graceful_degradation = True

        if self.hr_sensor_rppg:
            # v2 介面：依影格時間戳估計實際 fps、帶通濾波並回報訊號品質。
            # 舊寫法自行 FFT 且把 fps 寫死，桌面版與網頁版串流速率不同時心率會等比例偏移。
            st = self.hr_sensor_rppg.update(frame)
            self.rppg_conf = st['conf']
            if st['valid']:
                self.hr_rppg = int(round(st['hr']))
                self._rppg_last_ok = time.time()

        # 藍牙為主軌；斷線時才由 rPPG 接管，且須通過訊號品質把關
        if self.ble_connected:
            self.real_heart_rate = self.hr_ble
        elif self.hr_rppg > 0 and (time.time() - self._rppg_last_ok) <= self.RPPG_GRACE_SEC:
            self.real_heart_rate = self.hr_rppg
        else:
            self.real_heart_rate = 0
        # ==========================================

        results = self.yolo(frame, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kps = r.keypoints.data[0].cpu().numpy()

                self.pose_history.append(kps[:, :2])
                self.rep_counter.update(kps, self.action)
                self._draw_skeleton(frame, kps)

                if len(self.pose_history) == 50:
                    self.assess_quality()

        self._display_ui(frame)
        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model 1.pth")
    parser.add_argument("--stgcn", default="models/gcn_weight.pth")
    parser.add_argument("--bank", default="exemplar_bank.pt")
    args = parser.parse_args()
    FitnessAIAgent(args.model, args.stgcn, args.bank).run()
