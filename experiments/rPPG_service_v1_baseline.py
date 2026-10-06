import numpy as np
import cv2

class RPPGService:
    def __init__(self, window_size=150):
        self.window_size = window_size
        self.buffer = [] # 儲存 RGB 平均值
        self.current_hr = 0

    def extract_roi_mean(self, frame, face_landmarks):
        """
        利用您 333.py 中的 MediaPipe 臉部關鍵點鎖定額頭區域
        """
        # 假設選取額頭區域 (ROI)
        # 這裡簡化為固定區域，實務上可根據 landmarks 動態調整
        h, w, _ = frame.shape
        roi = frame[int(h*0.1):int(h*0.2), int(w*0.4):int(w*0.6)]
        return np.mean(roi, axis=(0, 1))

    def process(self, rgb_mean):
        self.buffer.append(rgb_mean)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

            # --- POS 演算法核心 ---
            H = np.array(self.buffer)
            # 正規化
            H = H / np.mean(H, axis=0)
            # 投影 (POS 公式)
            S = 3 * H[:, 0] - 2 * H[:, 1] # 簡化版綠紅差值

            # 帶通濾波與 FFT 取得心率
            # 此處輸出的訊號可直接餵入您的 emg_encoder
            return S
        return None
