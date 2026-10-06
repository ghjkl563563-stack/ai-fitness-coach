"""
rPPG_service.py — 視覺影像心率微服務 (v2)

相對於 v1 (experiments/rPPG_service_v1_baseline.py) 的修正：
  1. ROI      : 固定矩形 → MediaPipe 臉部關鍵點定位額頭+雙頰，並加膚色遮罩
  2. 取樣率   : 寫死 30 fps → 由影格時間戳動態估計，並重取樣到均勻時間軸
  3. 演算法   : 簡化色差 3R-2G → 標準 POS (可切換 CHROM / GREEN 做比較)
  4. 濾波     : 無 → Butterworth 0.7~3.0 Hz 零相位帶通
  5. 品質指標 : 無 → 頻譜 SNR + 信心值，供雙軌降級機制判斷可信度

v1 的 extract_roi_mean() / process() 介面保留可用，既有 agent 不需修改。
新程式請改用 update(frame)，可一次取得心率與訊號品質。
"""
import time
from collections import deque

import cv2
import numpy as np

from rppg_algorithms import extract_pulse, estimate_hr

try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False

# MediaPipe FaceMesh 關鍵點索引
_FOREHEAD = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]
_CHEEK_L = [116, 117, 118, 119, 100, 142, 205, 123]
_CHEEK_R = [345, 346, 347, 348, 329, 371, 425, 352]


class RPPGService:
    def __init__(self, window_size=300, method='POS', fps_hint=30.0,
                 min_conf=0.55, detect_every=3, use_skin_mask=True,
                 require_face=True):
        """
        window_size : 分析視窗影格數。300 @30fps = 10 秒。
                      v1 用 150 (5秒)，頻率解析度僅 12 BPM/bin，實測誤差偏大。
        method      : 'POS' | 'CHROM' | 'GREEN'
        min_conf    : 低於此信心值時 valid=False，呼叫端應視為不可信。
                      預設 0.55 依實測訂定：純雜訊輸入在 0.35 時有 19.5% 會被
                      誤判為可信，提高到 0.55 後降至 1.0%，而真實脈波的保留率
                      僅由 96.7% 降到 90.0%。正式實驗請依 analyze_hr.py 產出的
                      信心門檻掃描表，用自己的資料重新選定。
        detect_every: 每 N 影格重跑一次臉部偵測，其餘沿用上次 ROI (省算力)
        require_face: True 時必須偵測到臉才判定為可信 (線上運作應維持 True)。
                      離線重放已錄好的 RGB 軌跡時設 False。
        """
        self.window_size = int(window_size)
        self.method = method.upper()
        self.min_conf = float(min_conf)
        self.detect_every = max(1, int(detect_every))
        self.use_skin_mask = use_skin_mask
        self.require_face = bool(require_face)

        self.buffer = deque(maxlen=self.window_size)      # v1 相容：RGB 均值
        self._times = deque(maxlen=self.window_size)      # 對應時間戳
        self._frame_idx = 0
        self._last_boxes = None
        self._fps_hint = float(fps_hint)

        self.current_hr = 0.0
        self.snr_db = -np.inf
        self.conf = 0.0
        self.valid = False
        self.face_found = False

        self._mesh = None
        if _MP_OK:
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
            except Exception as e:
                print(f"[rPPG] MediaPipe 初始化失敗，改用 Haar 偵測: {e}")

        self._haar = None
        if self._mesh is None:
            try:
                self._haar = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                pass

    # ------------------------------------------------------------------
    # ROI 定位
    # ------------------------------------------------------------------
    def _boxes_from_mesh(self, frame):
        h, w = frame.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        boxes = []
        for idxs in (_FOREHEAD, _CHEEK_L, _CHEEK_R):
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs if i < len(lm)])
            if len(pts) < 3:
                continue
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            # 內縮 12% 避開髮際線與臉部邊緣
            dx, dy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
            x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                boxes.append((int(x0), int(y0), int(x1), int(y1)))
        return boxes or None

    def _boxes_from_haar(self, frame):
        if self._haar is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return [(int(x + 0.30 * w), int(y + 0.10 * h),
                 int(x + 0.70 * w), int(y + 0.25 * h))]     # 額頭帶

    def _boxes_fallback(self, frame):
        """完全偵測不到臉時的固定區域 (等同 v1 行為)，僅作最後保險。"""
        h, w = frame.shape[:2]
        return [(int(w * 0.40), int(h * 0.10), int(w * 0.60), int(h * 0.20))]

    def _skin_mean(self, patch):
        """取膚色像素的 BGR 均值；膚色像素太少則退回全區均值。"""
        if patch.size == 0:
            return None
        if not self.use_skin_mask:
            return patch.reshape(-1, 3).mean(axis=0)

        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        px = patch[mask > 0]
        if px.shape[0] < max(20, 0.15 * patch.shape[0] * patch.shape[1]):
            return patch.reshape(-1, 3).mean(axis=0)
        return px.mean(axis=0)

    # ------------------------------------------------------------------
    # 主要 API
    # ------------------------------------------------------------------
    def extract_roi_mean(self, frame, face_landmarks=None):
        """回傳 ROI 的 RGB 均值 (長度 3)。v1 相容介面。"""
        if frame is None or frame.size == 0:
            return None

        if self._frame_idx % self.detect_every == 0 or self._last_boxes is None:
            boxes = None
            if self._mesh is not None:
                try:
                    boxes = self._boxes_from_mesh(frame)
                except Exception:
                    boxes = None
            if boxes is None:
                boxes = self._boxes_from_haar(frame)
            self.face_found = boxes is not None
            self._last_boxes = boxes if boxes is not None else self._boxes_fallback(frame)
        self._frame_idx += 1

        h, w = frame.shape[:2]
        vals = []
        for (x0, y0, x1, y1) in self._last_boxes:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = self._skin_mean(frame[y0:y1, x0:x1])
            if m is not None:
                vals.append(m)
        if not vals:
            return None

        bgr = np.mean(vals, axis=0)
        return np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float64)   # → RGB

    def process(self, rgb_mean, timestamp=None):
        """
        將 RGB 均值推入緩衝並回傳脈波訊號。v1 相容介面。
        注意：v1 回傳的是未濾波的 3R-2G；v2 回傳的是完整帶通後的脈波，
              呼叫端若沿用自行 FFT 的舊寫法仍可運作。
        """
        if rgb_mean is None:
            return None
        self.buffer.append(np.asarray(rgb_mean, dtype=np.float64))
        self._times.append(time.time() if timestamp is None else float(timestamp))

        if len(self.buffer) < max(64, self.window_size // 3):
            return None
        C, fps = self._uniform_series()
        if C is None:
            return None
        return extract_pulse(C, fps, self.method)

    def update(self, frame, timestamp=None):
        """
        v2 主要介面：吃一張影格，回傳完整狀態。
            hr     : 心率 BPM (信心不足時仍回傳最近一次有效值)
            snr_db : 頻譜訊噪比
            conf   : 0~1 信心值
            valid  : conf >= min_conf 且偵測到臉，才可信任
        """
        rgb = self.extract_roi_mean(frame)
        pulse = self.process(rgb, timestamp)

        if pulse is None:
            self.valid = False
            return self._state(filled=False)

        _, fps = self._uniform_series()
        r = estimate_hr(pulse, fps)
        self.snr_db, self.conf = r['snr_db'], r['conf']
        face_ok = self.face_found or not self.require_face
        self.valid = bool(face_ok and r['conf'] >= self.min_conf and r['hr_bpm'] > 0)

        if self.valid:
            # 只在可信時更新，避免雜訊污染顯示值
            self.current_hr = (0.7 * self.current_hr + 0.3 * r['hr_bpm']
                               if self.current_hr > 0 else r['hr_bpm'])
        return self._state(filled=True, raw_hr=r['hr_bpm'], fps=fps)

    def push_rgb(self, rgb, timestamp=None):
        """
        離線重放介面：直接餵入已萃取好的 RGB 均值 (跳過影像處理)。
        供演算法比較實驗重跑同一段錄製資料使用。
        """
        pulse = self.process(rgb, timestamp)
        if pulse is None:
            self.valid = False
            return self._state(filled=False)

        _, fps = self._uniform_series()
        r = estimate_hr(pulse, fps)
        self.snr_db, self.conf = r['snr_db'], r['conf']
        self.valid = bool(r['conf'] >= self.min_conf and r['hr_bpm'] > 0)
        if self.valid:
            self.current_hr = (0.7 * self.current_hr + 0.3 * r['hr_bpm']
                               if self.current_hr > 0 else r['hr_bpm'])
        return self._state(filled=True, raw_hr=r['hr_bpm'], fps=fps)

    def _state(self, filled, raw_hr=0.0, fps=None):
        return {
            'hr': float(self.current_hr),
            'hr_raw': float(raw_hr),
            'snr_db': float(self.snr_db) if np.isfinite(self.snr_db) else -99.0,
            'conf': float(self.conf),
            'valid': bool(self.valid),
            'face_found': bool(self.face_found),
            'fps': float(fps) if fps else float(self._fps_hint),
            'filled': bool(filled),
            'n_samples': len(self.buffer),
        }

    # ------------------------------------------------------------------
    def _uniform_series(self):
        """
        由時間戳估計實際 fps，並把不等間隔的 RGB 序列重取樣到均勻時間軸。
        網路攝影機的影格間隔本來就會抖動，直接假設 30fps 會造成頻率偏移。
        """
        n = len(self.buffer)
        if n < 16:
            return None, self._fps_hint

        t = np.asarray(self._times, dtype=np.float64)
        C = np.asarray(self.buffer, dtype=np.float64)

        span = t[-1] - t[0]
        if span <= 1e-6:
            return C, self._fps_hint

        fps = (n - 1) / span
        if not (1.0 < fps < 240.0):
            fps = self._fps_hint
        self._fps_hint = fps

        grid = np.linspace(t[0], t[-1], n)
        return np.stack([np.interp(grid, t, C[:, c]) for c in range(3)], axis=1), fps

    def reset(self):
        self.buffer.clear()
        self._times.clear()
        self.current_hr, self.snr_db, self.conf = 0.0, -np.inf, 0.0
        self.valid = False

    def close(self):
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
