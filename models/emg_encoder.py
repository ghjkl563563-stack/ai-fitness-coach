import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.models as models
import os

class EMG_Spectrogram_ResNet18_Encoder(nn.Module):
    def __init__(self):
        super(EMG_Spectrogram_ResNet18_Encoder, self).__init__()

        # 定義梅爾頻譜轉換 (將時間序列轉為圖片特徵)
        # sample_rate 設為 100Hz (假設), n_mels=64 (頻譜高度)
        self.spectrogram = T.MelSpectrogram(
            sample_rate=100,
            n_fft=64,
            hop_length=8,
            n_mels=64
        )

        # 載入 ResNet18
        # 我們使用 weights 參數來避免 'pretrained' 的警告
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- 處理權重載入 (包含您之前做的 1-channel 修改) ---
        weight_path = os.path.join(os.path.dirname(__file__), 'resnet_weight.pth')

        if os.path.exists(weight_path):
            try:
                # 嘗試載入您製作的單通道權重
                state_dict = torch.load(weight_path, map_location='cpu')

                # 修改第一層卷積以接受單通道 (Grayscale Spectrogram)
                self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

                # 過濾掉不匹配的鍵值 (防呆)
                model_dict = self.base_model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.base_model.load_state_dict(model_dict)
                # print("✅ EMG Encoder: 成功載入自訂權重")
            except Exception as e:
                print(f"⚠️ EMG Encoder 權重載入失敗，使用隨機初始化: {e}")
                self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # 如果沒有權重檔，就直接改第一層結構
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除最後的全連接層 (我們只需要特徵)
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        # --- [關鍵修復] 自動處理多通道輸入 ---
        # 輸入 x 的形狀可能是 (Batch, Time, Channels) 例如 (2, 50, 3)
        if x.dim() == 3:
            # 我們將 Channels 維度 (第 2 維) 取平均，混合成單一訊號
            # 這樣變回 (Batch, Time)，就可以通過 M, N = x.shape 了
            x = x.mean(dim=2)

        # 確保是 float 型態
        x = x.float()

        # 1. 取得形狀
        M, N = x.shape  # 現在這裡是 (Batch, Time)

        # 2. 產生頻譜圖
        # MelSpectrogram 預期輸入 (Batch, Time) -> 輸出 (Batch, n_mels, Time_frames)
        spec = self.spectrogram(x)

        # 3. 調整維度以符合 ResNet 輸入 (Batch, Channels, Height, Width)
        # 這裡增加一個 channel 維度 -> (Batch, 1, 64, Time_frames)
        spec = spec.unsqueeze(1)

        # 4. 透過 ResNet 提取特徵
        feat = self.base_model(spec) # 輸出通常是 (Batch, 512)

        return feat
