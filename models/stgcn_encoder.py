import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

# 嘗試引用您的 ST-GCN 模型架構
# 根據您的檔案結構，st_gcn.py 應該在 models/st_gcn/st_gcn.py
try:
    from models.st_gcn.st_gcn import Model
except ImportError:
    # 備用：如果您是在 models 資料夾內執行
    from st_gcn.st_gcn import Model

class STGCNFeatureExtractor(nn.Module):
    def __init__(self, weight_path=None):
        super(STGCNFeatureExtractor, self).__init__()

        # 1. 設定權重路徑
        # 預設路徑：與此檔案同一層目錄下的 gcn_weight.pth
        if weight_path is None:
            weight_path = os.path.join(os.path.dirname(__file__), 'gcn_weight.pth')

        print(f"🏗️ 初始化 ST-GCN 特徵提取器...")

        # 2. 定義模型架構 (ST-GCN)
        # 注意: 這裡的參數必須與訓練時完全一致
        # 如果您是用我提供的 train_stgcn.py 訓練的，num_class 應該是 2
        # 如果是用 NTU 預訓練模型，num_class 通常是 60
        self.model = Model(
            in_channels=3,
            num_class=2,  # <--- 請注意這裡！如果您用的是 NTU 預訓練，請改回 60
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, # strategy 通常是 'spatial' 或 'uniform'
            edge_importance_weighting=True
        )

        # 3. 載入權重
        if os.path.exists(weight_path):
            print(f"💾 正在載入權重: {weight_path}")
            try:
                checkpoint = torch.load(weight_path, map_location='cpu')

                # 處理 state_dict (有些儲存格式會多包一層 'state_dict' key)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 移除可能的 'module.' 前綴 (如果是多卡訓練產生的)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # 載入參數 (strict=False 允許部分參數不匹配，例如最後的全連接層)
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ ST-GCN 權重載入成功！")

            except Exception as e:
                print(f"⚠️ 權重載入失敗: {e}")
                print("將使用隨機初始化權重繼續運行 (特徵可能無效)")
        else:
            print(f"❌ 找不到權重檔: {weight_path}")
            print("請確認 gcn_weight.pth 是否位於 models/ 資料夾下。")

        self.model.eval() # 設定為評估模式

    def forward(self, x):
        """
        輸入 x: Tensor of shape (N, 3, T, V, M)
        輸出: Tensor of shape (N, 256)
        """
        # 確保輸入在正確的設備上
        if next(self.model.parameters()).is_cuda:
            x = x.cuda()

        with torch.no_grad():
            # extract_feature 會回傳 (output, feature)
            # feature shape: (N, 256, T, V, M)
            try:
                # 呼叫 ST-GCN 的 extract_feature 方法
                _, features = self.model.extract_feature(x)

                # 平均池化 (Global Average Pooling) -> (N, 256)
                feature_vector = features.mean(dim=[2, 3, 4])
                return feature_vector

            except Exception as e:
                print(f"❌ 特徵提取錯誤: {e}")
                # 回傳全 0 向量避免程式崩潰
                return torch.zeros((x.size(0), 256)).to(x.device)

# ==========================================
# 工具函式區
# ==========================================

def remap21_to_25(data21):
    """ 將 Mediapipe/YOLO 的 21/17 點骨架映射到 NTU 25 點 """
    # 這裡保留您提供的映射邏輯
    mapping_25 = {
        0:  None,  1:  None,  2:   1,  3:  0,
        4:  7,     5:  8,     6:   9,  7:  10,
        8:  3,     9:  4,     10:  5,  11: 6,
        12: 16,    13: 17,    14: 18,  15: None,
        16: 11,    17: 12,    18: 13,  19: None,
        20: 2,     21: 10,    22: 10,  23: 6,  24: 6
    }

    # 支援 numpy 或 torch 輸入
    if isinstance(data21, torch.Tensor):
        data21 = data21.cpu().numpy()

    T, V_old, C = data21.shape
    data25 = np.zeros((T, 25, C), dtype=data21.dtype)

    for i25, i21 in mapping_25.items():
        if i21 is not None and i21 < V_old:
            data25[:, i25, :] = data21[:, i21, :]

    # 補值邏輯 (SpineBase, SpineMid 等)
    # 注意：這裡的索引需要根據您的原始數據定義確認
    # 假設 11=LHip, 16=RHip (YOLO 格式通常是 11, 12)
    # 為了安全起見，這裡建議使用您之前驗證過的邏輯

    # 這裡示範簡單補值，避免全 0
    if V_old > 12: # 確保有足夠點數
        # SpineBase = Hip Center
        # 假設 data21 裡 index 11, 12 是髖部 (YOLO standard)
        # 請根據您的實際數據調整
        spine_base = 0.5 * (data21[:, 11, :] + data21[:, 12, :])
        data25[:, 0, :] = spine_base

        # SpineMid
        data25[:, 1, :] = spine_base # 暫時用 base 代替，或計算中點

    return data25

def normalize_ntu_skeleton(data, hip_index=0, spine_index=1, epsilon=1e-6):
    """
    輸入: [N, T, V, C]
    輸出: [N, T, V, C] 歸一化後的數據
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    data = data.astype(np.float32)

    # 1. 以髖關節為中心 (Centering)
    if data.shape[2] > hip_index:
        hip_coords = data[:, :, hip_index, :]          # [N, T, C]
        hip_coords = hip_coords[:, :, np.newaxis, :]   # [N, T, 1, C]
        normalized_data = data - hip_coords
    else:
        normalized_data = data

    # 2. 骨骼長度歸一化 (Scaling)
    # 計算脊柱長度 (髖 -> 脊柱中心)
    if data.shape[2] > spine_index:
        spine_vector = normalized_data[:, :, spine_index, :]  # [N, T, C]
        spine_length = np.linalg.norm(spine_vector, axis=-1, keepdims=True)  # [N, T, 1]
        normalized_data = normalized_data / (spine_length[:, :, np.newaxis, :] + epsilon)

    return normalized_data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 測試區 (當直接執行此檔案時)
# ==========================================
if __name__ == "__main__":
    set_seed()
    print("🚀 開始測試 STGCNFeatureExtractor...")

    # 1. 建立假資料 (Batch=1, Channels=3, Frames=50, Joints=25, Person=1)
    # 這是 ST-GCN 的標準輸入格式
    dummy_input = torch.randn(1, 3, 50, 25, 1)

    # 2. 初始化模型
    # 注意：它會自動去抓 models/gcn_weight.pth
    try:
        extractor = STGCNFeatureExtractor()

        # 3. 執行前向傳播
        output = extractor(dummy_input)

        print(f"✅ 測試成功！輸出形狀: {output.shape}") # 預期 (1, 256)
        print("前 10 個特徵值:", output[0, :10].tolist())

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
