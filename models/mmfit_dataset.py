import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from models.stgcn_encoder import remap21_to_25, normalize_ntu_skeleton

class MMFitDataset(Dataset):
    def __init__(self, csv_path, data_root):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到 {csv_path}，請先執行 setup_and_align.py")
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 讀取 Pose
        pose_full_path = os.path.join(self.data_root, row['pose_path'])
        try:
            # 使用 mmap_mode 防止記憶體爆掉
            pose_raw = np.load(pose_full_path, mmap_mode='r')
        except:
            pose_raw = np.zeros((50, 21, 3)) # 讀取失敗給假資料

        # 2. 讀取 Sensor (心率 HR)
        sensor_full_path = os.path.join(self.data_root, row['sensor_path'])
        try:
            # MM-Fit 的心率檔通常是一維陣列 [HR, HR, HR...]
            sensor_raw = np.load(sensor_full_path, mmap_mode='r')
        except:
            sensor_raw = np.zeros((100,)) # 讀失敗給 0 (1D)

        # 3. 時間切割 (Slicing)
        start = int(row['start_frame'])
        end = int(row['end_frame'])

        if end == -1: end = start + 50
        length = end - start
        if length < 10: length = 10

        # 切割 Pose
        if pose_raw.ndim >= 2:
            real_start = min(start, pose_raw.shape[0]-1)
            real_end = min(end, pose_raw.shape[0])
            pose_chunk = pose_raw[real_start:real_end]
        else:
            pose_chunk = np.zeros((50, 21, 3))

        # 切割 Sensor (適應心率 1D 資料)
        if sensor_raw.ndim == 1:
            real_start_s = min(start, sensor_raw.shape[0]-1)
            real_end_s = min(end, sensor_raw.shape[0])
            sensor_chunk = sensor_raw[real_start_s:real_end_s]
        elif sensor_raw.ndim >= 2:
            real_start_s = min(start, sensor_raw.shape[0]-1)
            real_end_s = min(end, sensor_raw.shape[0])
            sensor_chunk = sensor_raw[real_start_s:real_end_s]
        else:
            sensor_chunk = np.zeros((50,))

        # --- 4. 格式轉換與對齊 ---

        # A. Pose 初步映射
        if pose_chunk.ndim == 2 and pose_chunk.shape[1] != 3:
            T = pose_chunk.shape[0]
            fake_pose = np.zeros((T, 21, 3))
            fake_pose[:, 0, 0] = pose_chunk[:, 0] if pose_chunk.shape[1] > 0 else 0
            pose_mapped = remap21_to_25(fake_pose)
        else:
            if pose_chunk.ndim == 3:
                try:
                    pose_mapped = remap21_to_25(pose_chunk)
                except:
                    pose_mapped = pose_chunk
            else:
                 pose_mapped = np.zeros((50, 25, 3))

        # B. Pose 維度強制對齊
        T, V, C = pose_mapped.shape
        if V != 25:
            new_pose = np.zeros((T, 25, C))
            min_V = min(V, 25)
            new_pose[:, :min_V, :] = pose_mapped[:, :min_V, :]
            pose_mapped = new_pose
        if C != 3:
            new_pose_c = np.zeros((T, 25, 3))
            min_C = min(C, 3)
            new_pose_c[:, :, :min_C] = pose_mapped[:, :, :min_C]
            pose_mapped = new_pose_c

        # C. Pose 歸一化
        pose_expanded = pose_mapped[None, ...]
        try:
            pose_norm = normalize_ntu_skeleton(pose_expanded)[0]
        except:
            pose_norm = pose_mapped

        # D. Pose 轉 Tensor
        pose_tensor = torch.from_numpy(pose_norm).permute(2, 0, 1).unsqueeze(-1).float()

        # --- 5. 動態補零 (Dynamic Padding) ---
        target_len = 50
        # Pose Padding
        C_curr, L_curr, V_curr, M_curr = pose_tensor.shape
        if L_curr > target_len:
            pose_tensor = pose_tensor[:, :target_len, :, :]
        elif L_curr < target_len:
            pad = torch.zeros(C_curr, target_len - L_curr, V_curr, M_curr)
            pose_tensor = torch.cat([pose_tensor, pad], dim=1)

        # Sensor 處理
        # 加上 .copy() 解決 writable warning
        sensor_tensor = torch.from_numpy(sensor_chunk.copy()).float()

        # 如果是一維 (Time,)，要變成二維 (Time, 1) 以符合模型輸入要求
        if sensor_tensor.dim() == 1:
            sensor_tensor = sensor_tensor.unsqueeze(-1)

        # Sensor Padding
        if sensor_tensor.shape[0] > target_len:
            sensor_tensor = sensor_tensor[:target_len, :]
        elif sensor_tensor.shape[0] < target_len:
            pad_s = torch.zeros(target_len - sensor_tensor.shape[0], sensor_tensor.shape[1])
            sensor_tensor = torch.cat([sensor_tensor, pad_s], dim=0)

        # --- 【關鍵】心率歸一化 (Normalization) ---
        # 假設心率範圍約 60~180。映射到 -1 ~ 1 之間
        # 公式: (HR - 100) / 50  => 150變1, 50變-1, 100變0
        sensor_tensor = (sensor_tensor - 100.0) / 50.0

        # 保護機制：避免極端值 (例如讀取錯誤變成 0 或異常高)
        sensor_tensor = torch.clamp(sensor_tensor, -5.0, 5.0)

        label = int(row['action_label'])

        return pose_tensor, sensor_tensor, label, 1.0
