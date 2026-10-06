"""
rep_counter.py — 以骨架動作訊號計次

取代原本「分數 >= 50 且距上次超過 1.5 秒就 +1」的計時器寫法 —— 那個作法
只要姿勢維持得夠好就會每 1.5 秒加一次，與實際做了幾下無關。

作法：
  1. 依動作類型從 COCO-17 關節點算出一個一維「動作訊號」
     (深蹲看膝角、伏地挺身看肘角、開合跳看踝距…)
  2. 以近期視窗的 min/max 把訊號正規化到 0~1，消除個體與距離差異
  3. 用帶遲滯的雙門檻狀態機偵測「下去再上來」的完整循環
  4. 加上最短週期與最小動作幅度兩道防呆，避免抖動被當成次數

靜態支撐類動作 (棒式、瑜珈體位) 不計次，改回報持續秒數。
"""
import time
from collections import deque

import numpy as np

# COCO-17 關節點索引
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# 靜態支撐類：計時而非計次
_STATIC_HINTS = ("plank", "pose", "dog", "child", "warrior", "tree",
                 "stretch", "side plank", "preparation")


def _angle(a, b, c):
    """回傳 ∠abc (度)。任一邊長為零時回傳 180。"""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cosv = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def _conf_ok(kps, idxs, thr=0.3):
    return all(kps[i][2] >= thr for i in idxs) if kps.shape[1] > 2 else True


def _torso_scale(kps):
    """以肩寬與肩髖距取得身體尺度，用來把像素距離正規化。"""
    sw = np.linalg.norm(kps[L_SHO][:2] - kps[R_SHO][:2])
    sh_mid = (kps[L_SHO][:2] + kps[R_SHO][:2]) / 2
    hip_mid = (kps[L_HIP][:2] + kps[R_HIP][:2]) / 2
    th = np.linalg.norm(sh_mid - hip_mid)
    scale = max(sw, th)
    return scale if scale > 1e-3 else 1.0


def classify(action):
    """把動作名稱歸類成訊號型別。"""
    a = str(action).lower().replace("_", " ").replace("-", " ").strip()
    if any(h in a for h in _STATIC_HINTS) and "tap" not in a:
        return "static"
    if "squat" in a or "lunge" in a or "sit" in a:
        return "knee_angle"
    if "push up" in a or "push-up" in a or "pushup" in a or "tap" in a or "dip" in a:
        return "elbow_angle"
    if "jack" in a or "jump" in a or "rope" in a:
        return "ankle_spread"
    if "kick" in a or "running" in a or "feet" in a or "knee" in a or "march" in a:
        return "knee_height"
    if "circle" in a or "cross" in a or "twist" in a or "punch" in a \
            or "jab" in a or "hook" in a or "uppercut" in a:
        return "wrist_height"
    return "knee_angle"


def signal_of(kps, kind):
    """
    由關節點算出一維動作訊號。回傳 (值, 是否可信)。
    訊號方向統一為「數值變小 = 動作收縮期」，方便共用同一套狀態機。
    """
    try:
        if kind == "knee_angle":
            if not _conf_ok(kps, (L_HIP, L_KNE, L_ANK)):
                if not _conf_ok(kps, (R_HIP, R_KNE, R_ANK)):
                    return 0.0, False
                return _angle(kps[R_HIP][:2], kps[R_KNE][:2], kps[R_ANK][:2]), True
            return _angle(kps[L_HIP][:2], kps[L_KNE][:2], kps[L_ANK][:2]), True

        if kind == "elbow_angle":
            if not _conf_ok(kps, (L_SHO, L_ELB, L_WRI)):
                if not _conf_ok(kps, (R_SHO, R_ELB, R_WRI)):
                    return 0.0, False
                return _angle(kps[R_SHO][:2], kps[R_ELB][:2], kps[R_WRI][:2]), True
            return _angle(kps[L_SHO][:2], kps[L_ELB][:2], kps[L_WRI][:2]), True

        if kind == "ankle_spread":
            if not _conf_ok(kps, (L_ANK, R_ANK, L_SHO, R_SHO)):
                return 0.0, False
            spread = abs(kps[L_ANK][0] - kps[R_ANK][0]) / _torso_scale(kps)
            return -spread, True          # 取負號：合腿(小) → 值大，統一方向

        if kind == "knee_height":
            if not _conf_ok(kps, (L_KNE, L_HIP, R_HIP)):
                return 0.0, False
            hip_y = (kps[L_HIP][1] + kps[R_HIP][1]) / 2
            # 影像 y 軸向下，抬膝時 knee_y 變小 → 差值變大，取負號統一方向
            return -((hip_y - kps[L_KNE][1]) / _torso_scale(kps)), True

        if kind == "wrist_height":
            if not _conf_ok(kps, (L_WRI, L_SHO, R_SHO)):
                return 0.0, False
            sho_y = (kps[L_SHO][1] + kps[R_SHO][1]) / 2
            return -((sho_y - kps[L_WRI][1]) / _torso_scale(kps)), True

    except (IndexError, ValueError):
        return 0.0, False
    return 0.0, False


class RepCounter:
    """
    單一動作的計次器。動作切換時呼叫 reset() 或直接傳入新的 action，
    內部會自動重置狀態並保留各動作的累計次數。
    """

    def __init__(self, min_period_s=0.7, hysteresis=0.30,
                 history_s=8.0, min_range=None):
        self.min_period_s = float(min_period_s)
        self.hysteresis = float(hysteresis)
        self.history_s = float(history_s)
        # 各訊號型別的最小動作幅度門檻 (低於此視為沒在動)
        self.min_range = min_range or {
            "knee_angle": 25.0, "elbow_angle": 25.0,
            "ankle_spread": 0.25, "knee_height": 0.20, "wrist_height": 0.25,
        }

        self.counts = {}            # {動作名稱: 次數}
        self.hold_seconds = {}      # {靜態動作名稱: 持續秒數}
        self._buf = deque()         # [(t, 訊號值)]
        self._state = "high"
        self._last_rep_t = 0.0
        self._cur_action = None
        self._cur_kind = None
        self._hold_start = None
        self.last_signal = None
        self.last_norm = None

    # ------------------------------------------------------------------
    def reset(self):
        self._buf.clear()
        self._state = "high"
        self._last_rep_t = 0.0
        self._hold_start = None
        self.last_signal = self.last_norm = None

    @property
    def total(self):
        return int(sum(self.counts.values()))

    def count_of(self, action):
        return int(self.counts.get(action, 0))

    # ------------------------------------------------------------------
    def update(self, kps, action, t=None):
        """
        餵入一影格的關節點與當前動作名稱。
        回傳 dict: {action, kind, reps, hold_s, counted, norm, moving}
        """
        t = time.time() if t is None else float(t)
        kps = np.asarray(kps, dtype=np.float64)

        if action != self._cur_action:
            self._cur_action = action
            self._cur_kind = classify(action)
            self.reset()
            self.counts.setdefault(action, 0)

        kind = self._cur_kind
        out = {"action": action, "kind": kind, "counted": False,
               "reps": self.counts.get(action, 0),
               "hold_s": self.hold_seconds.get(action, 0.0),
               "norm": None, "moving": False}

        # --- 靜態支撐：計時不計次 ---
        if kind == "static":
            if self._hold_start is None:
                self._hold_start = t
            self.hold_seconds[action] = t - self._hold_start
            out["hold_s"] = self.hold_seconds[action]
            return out

        val, ok = signal_of(kps, kind)
        if not ok:
            return out
        self.last_signal = val

        self._buf.append((t, val))
        while self._buf and t - self._buf[0][0] > self.history_s:
            self._buf.popleft()
        if len(self._buf) < 8:
            return out

        vals = np.array([v for _, v in self._buf])
        lo, hi = float(vals.min()), float(vals.max())
        rng = hi - lo
        out["moving"] = rng >= self.min_range.get(kind, 0.2)
        if not out["moving"]:
            return out                       # 幅度不足，視為沒在做動作

        norm = (val - lo) / rng
        out["norm"] = self.last_norm = float(norm)

        lo_thr = 0.5 - self.hysteresis / 2
        hi_thr = 0.5 + self.hysteresis / 2

        if self._state == "high" and norm < lo_thr:
            self._state = "low"              # 進入收縮期 (蹲下 / 下推 / 開腿)
        elif self._state == "low" and norm > hi_thr:
            self._state = "high"             # 回到起始位 → 完成一次
            if t - self._last_rep_t >= self.min_period_s:
                self.counts[action] = self.counts.get(action, 0) + 1
                self._last_rep_t = t
                out["counted"] = True

        out["reps"] = self.counts.get(action, 0)
        return out
