"""
rppg_algorithms.py — rPPG 訊號處理演算法核心 (純函式，不依賴攝影機)

實作三種標準 rPPG 色彩訊號萃取法，供論文做演算法比較 (ablation)：
  - GREEN : Verkruysse et al. (2008) 綠通道基準法
  - CHROM : de Haan & Jeanne (2013) 色度差分法
  - POS   : Wang et al. (2017) Plane-Orthogonal-to-Skin

以及心率估計 (含頻譜訊噪比 SNR)，SNR 是雙軌降級機制的信心指標：
只有 rPPG 的 SNR 達標時才允許它接管，否則標示為不可信。

所有函式輸入皆為 (T, 3) 的 RGB 時序均值陣列，輸出為 1D 脈波訊號。
"""
import numpy as np
from scipy import signal as sps

# 生理合理心率範圍 (Hz)：0.7 Hz = 42 BPM, 3.0 Hz = 180 BPM
HR_BAND_LOW = 0.7
HR_BAND_HIGH = 3.0


# ============================================================
# 前處理
# ============================================================
def detrend(x, lam=None):
    """去趨勢。預設用線性去趨勢；給定 lam 則用 Tarvainen 平滑先驗法。"""
    x = np.asarray(x, dtype=np.float64)
    if lam is None:
        return sps.detrend(x, type='linear')

    # Tarvainen (2002) smoothness priors detrending
    T = len(x)
    I = np.eye(T)
    D2 = np.zeros((T - 2, T))
    for i in range(T - 2):
        D2[i, i:i + 3] = [1, -2, 1]
    return (I - np.linalg.inv(I + (lam ** 2) * (D2.T @ D2))) @ x


def bandpass(x, fps, low=HR_BAND_LOW, high=HR_BAND_HIGH, order=3):
    """Butterworth 帶通濾波 (零相位)。訊號太短時原樣返回。"""
    x = np.asarray(x, dtype=np.float64)
    nyq = fps / 2.0
    lo, hi = low / nyq, min(high / nyq, 0.99)
    if lo >= hi or len(x) < 3 * (order + 1):
        return x
    b, a = sps.butter(order, [lo, hi], btype='band')
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return sps.filtfilt(b, a, x, padlen=padlen)


def _temporal_normalize(C):
    """以時間軸均值正規化 (T,3)，避免除以零。"""
    mu = np.mean(C, axis=0)
    mu[np.abs(mu) < 1e-9] = 1e-9
    return C / mu


# ============================================================
# 色彩訊號萃取：三種演算法
# ============================================================
def extract_green(C, fps):
    """GREEN 基準法：只用綠通道 (Verkruysse et al. 2008)。"""
    C = np.asarray(C, dtype=np.float64)
    g = _temporal_normalize(C)[:, 1]
    return bandpass(detrend(g), fps)


def extract_chrom(C, fps):
    """
    CHROM (de Haan & Jeanne 2013)
    Xs = 3Rn - 2Gn ; Ys = 1.5Rn + Gn - 1.5Bn
    S  = Xf - (std(Xf)/std(Yf)) * Yf
    """
    C = np.asarray(C, dtype=np.float64)
    Cn = _temporal_normalize(C)
    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]

    Xs = 3.0 * R - 2.0 * G
    Ys = 1.5 * R + G - 1.5 * B

    Xf = bandpass(Xs, fps)
    Yf = bandpass(Ys, fps)

    sY = np.std(Yf)
    alpha = (np.std(Xf) / sY) if sY > 1e-9 else 0.0
    return Xf - alpha * Yf


def extract_pos(C, fps, window_sec=1.6):
    """
    POS (Wang et al. 2017) — Plane-Orthogonal-to-Skin，滑動視窗 + 重疊相加。

    每個長度 l 的視窗內：
        Cn = C / mean(C)                     時間正規化
        S  = P @ Cn.T,  P = [[0,1,-1],[-2,1,1]]   投影至皮膚正交平面
        h  = S1 + (std(S1)/std(S2)) * S2     自適應合併兩投影軸
    """
    C = np.asarray(C, dtype=np.float64)
    T = C.shape[0]
    l = int(round(window_sec * fps))
    if l < 3 or T < l:
        # 訊號長度不足一個視窗，退化為單一視窗處理
        l = T
        if l < 3:
            return np.zeros(T)

    H = np.zeros(T)
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    for n in range(0, T - l + 1):
        m = n + l
        Cn = _temporal_normalize(C[n:m, :])       # (l, 3)
        S = P @ Cn.T                              # (2, l)

        s2 = np.std(S[1, :])
        alpha = (np.std(S[0, :]) / s2) if s2 > 1e-9 else 0.0
        h = S[0, :] + alpha * S[1, :]

        H[n:m] += (h - np.mean(h))                # 重疊相加

    return bandpass(H, fps)


EXTRACTORS = {
    'GREEN': extract_green,
    'CHROM': extract_chrom,
    'POS': extract_pos,
}


def extract_pulse(C, fps, method='POS'):
    """依名稱取出脈波訊號。method ∈ {'GREEN','CHROM','POS'}"""
    m = method.upper()
    if m not in EXTRACTORS:
        raise ValueError(f"未知的 rPPG 演算法: {method} (可用: {list(EXTRACTORS)})")
    return EXTRACTORS[m](C, fps)


# ============================================================
# 心率估計 + 訊號品質
# ============================================================
def estimate_hr(pulse, fps, low=HR_BAND_LOW, high=HR_BAND_HIGH, zero_pad=8):
    """
    由脈波訊號估計心率。

    重點：短視窗的頻率解析度很差 (5 秒視窗 = 0.2 Hz = 12 BPM/bin)，
    因此補零提高頻譜取樣密度，再用拋物線內插做次格點峰值定位。

    回傳 dict:
        hr_bpm : 心率 (BPM)，估計失敗為 0.0
        snr_db : 頻譜訊噪比 (dB)，越高代表脈波越乾淨
        conf   : 0~1 的信心值 (由 SNR 映射)
    """
    x = np.asarray(pulse, dtype=np.float64)
    n = len(x)
    if n < 16 or not np.isfinite(x).all() or np.std(x) < 1e-12:
        return {'hr_bpm': 0.0, 'snr_db': -np.inf, 'conf': 0.0}

    x = x - np.mean(x)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * zero_pad)))

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    psd = np.abs(np.fft.rfft(x * win, n=nfft)) ** 2

    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return {'hr_bpm': 0.0, 'snr_db': -np.inf, 'conf': 0.0}

    band_idx = np.flatnonzero(band)
    k = band_idx[np.argmax(psd[band_idx])]

    # 拋物線內插取次格點峰值
    f_peak = freqs[k]
    if 0 < k < len(psd) - 1:
        a, b, c = psd[k - 1], psd[k], psd[k + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-20:
            delta = 0.5 * (a - c) / denom
            if abs(delta) <= 1.0:
                f_peak = freqs[k] + delta * (freqs[1] - freqs[0])

    # SNR：基頻 ±0.1Hz + 一次諧波 ±0.2Hz 的能量 vs 帶內其餘能量 (de Haan 定義)
    sig_mask = (np.abs(freqs - f_peak) <= 0.1) | (np.abs(freqs - 2 * f_peak) <= 0.2)
    sig_p = float(np.sum(psd[band & sig_mask]))
    noi_p = float(np.sum(psd[band & ~sig_mask]))
    snr_db = 10.0 * np.log10(sig_p / noi_p) if noi_p > 1e-20 else np.inf

    # SNR -6dB → conf 0，+6dB → conf 1，中間線性
    conf = float(np.clip((snr_db + 6.0) / 12.0, 0.0, 1.0))
    return {'hr_bpm': float(f_peak * 60.0), 'snr_db': float(snr_db), 'conf': conf}


def hr_from_rgb(C, fps, method='POS'):
    """一步到位：RGB 時序 → 心率估計結果。"""
    return estimate_hr(extract_pulse(C, fps, method), fps)
