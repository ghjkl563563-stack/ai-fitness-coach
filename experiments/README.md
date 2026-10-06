# TANET 論文實驗手冊 — 實驗 B / 實驗 C

本目錄是兩項實驗的完整程式與操作流程：

| 實驗 | 主題 | 適合議程軌 |
|---|---|---|
| **B** | 雙軌生理感測與優雅降級 | 智慧醫療／健康照護、IoT |
| **C** | 端-雲協同架構的頻寬與承載分析 | 網路技術與應用 |

---

## 0. 檔案總覽

```
專案根目錄/
├── rppg_algorithms.py              rPPG 演算法核心 (POS / CHROM / GREEN + SNR)
├── rPPG_service.py                 v2 視覺心率服務 (實際系統使用)
├── rep_counter.py                  骨架動作計次器
└── experiments/
    ├── README.md                   ← 本檔
    ├── rPPG_service_v1_baseline.py 原始 v1 實作備份 (論文對照組)
    │
    ├── test_rppg_synth.py          合成訊號驗證演算法正確性
    ├── test_rppg_service.py        v2 服務煙霧測試
    │
    ├── collect_hr.py               【實驗 B】資料收集
    ├── synth_session.py            【實驗 B】合成資料產生器 (測試管線用)
    ├── analyze_hr.py               【實驗 B】一致性分析
    ├── analyze_failover.py         【實驗 B】優雅降級分析
    │
    ├── aggregate_subjects.py       【實驗 B】跨受試者彙整 + 統計檢定
    ├── bench_transport.py          【實驗 C】傳輸效能量測 (本機)
    ├── bench_e2e_live.py           【實驗 C】真實佈署端到端量測 + 壓力測試
    ├── client_s2.py                【實驗 C】S2 骨架卸載參考客戶端 (可實跑)
    ├── test_s2_path.py             S2 伺服器端路徑驗證
    ├── test_rep_counter.py         計次器驗證
    ├── test_agent_hr_path.py       agent 心率路徑驗證
    └── results/                    所有輸出 (CSV / 報告 / 圖)
```

---

## 1. 環境需求

套件版本已固定在專案根目錄的 [requirements.txt](../requirements.txt)：

```bash
pip install -r requirements.txt
```

> torch 預設裝 CPU 版。要用 GPU 請先依 pytorch.org 的指示安裝對應 CUDA 版本。

**環境變數**：複製 [.env.example](../.env.example) 後依說明設定。
AI 教練講評需要 `GEMINI_API_KEY`，未設定不影響其他功能。

**硬體**：
- 網路攝影機（實驗 B、C 都需要）
- BLE 心率帶（實驗 B 的參考標準；可設定 `BLE_HR_ADDRESS` 環境變數，
  或用 `--ble-addr` 指定。未指定時會自動掃描）

**Windows 下執行請加編碼設定**，否則終端機中文會變亂碼：

```bash
set PYTHONIOENCODING=utf-8
```

---

## 2. 先跑自我檢驗

在收任何真實資料之前，先確認演算法與服務本身是正確的。這兩支測試是論文
數據可信度的前提，也可以直接寫進論文的「實作驗證」小節。

```bash
python experiments/test_rppg_synth.py
```

用已知頻率的合成脈波檢驗三種演算法。預期輸出包含帶內運動假影情境下的
MAE 對照（GREEN 會明顯劣化，POS/CHROM 應維持低誤差），以及視窗長度對
頻率解析度的取捨表。

```bash
python experiments/test_rppg_service.py
```

檢驗 v2 服務：心率估計正確、純雜訊與無臉輸入會被正確判為不可信、
每影格處理耗時。

```bash
python experiments/test_agent_hr_path.py
python experiments/test_rep_counter.py
python experiments/test_s2_path.py
```

前者確認兩支 agent 的心率路徑在 10 / 15 / 30 fps 下都正確（舊版寫死
fps，換了串流速率就會失準）；後者確認計次器準確且靜態動作改計時。

---

## 3. 實驗 B：雙軌生理感測

### 3.1 先用合成資料確認管線能跑

還沒找到受試者前，先確認分析流程正常：

```bash
python experiments/synth_session.py --subject SYN01
python experiments/analyze_hr.py experiments/results/hr_SYN01_synthetic.csv --step 2.0
python experiments/analyze_failover.py experiments/results/hr_SYN01_synthetic.csv
```

> ⚠️ **合成資料的數字不可當作論文結果**。它只驗證程式正確。
> 合成模型無法完整重現真實的 ROI 內容變化、光照非線性與追蹤抖動，
> 真實數據的誤差一定更大。

### 3.2 受試者實驗協定

每位受試者一輪約 **7.5 分鐘**（5 階段 × 90 秒）。建議 **8–10 位**
（統計檢定的硬性下限是 6 位，理由見 3.5 節）。

| 階段 | 動作 | 時長 | 目的 |
|---|---|---|---|
| `rest` | 靜止坐姿，放鬆不說話 | 90s | 取得基線、最佳情況誤差 |
| `light` | 原地踏步 | 90s | 輕度運動 |
| `moderate` | 深蹲 | 90s | 中度運動 |
| `vigorous` | 開合跳 | 90s | 高強度，rPPG 最嚴苛的情境 |
| `recovery` | 坐下恢復，保持不動 | 90s | 心率快速下降段，考驗追蹤能力 |

**環境控制（會直接影響結果，務必記錄下來寫進論文）：**

- 光源：固定的室內照明，避免日光燈閃爍與背光；不要在窗邊
- 距離：臉部距鏡頭約 60–100 cm，臉部在畫面中佔比不要太小
- 鏡頭：固定在腳架或桌面，全程不可移動
- 受試者：避免濃妝、瀏海遮額頭；全程不要說話（說話會造成強烈假影）
- 心率帶：實驗開始前先確認已穩定連線並有讀數

**執行：**

```bash
python experiments/collect_hr.py --subject S01
```

操作鍵：

| 鍵 | 作用 |
|---|---|
| `SPACE` | 手動提前進入下一階段 |
| `d` | 標記一次人為藍牙斷線（**拔除感測器的同時按下**） |
| `q` | 結束並存檔 |

**失效轉移測試：** 在 `moderate` 或 `vigorous` 階段中途，按 `d` 並同時
關閉／拔除心率帶，維持 30–60 秒後再開啟。每位受試者至少做 1 次，
這是實驗 B-2 的資料來源。

沒有藍牙裝置時可用 `--no-ble`，但這樣就沒有參考標準，只能拿來測 rPPG
本身的穩定度，無法算一致性指標。

### 3.3 分析

```bash
python experiments/analyze_hr.py experiments/results/hr_S01_<時間戳>.csv
python experiments/analyze_failover.py experiments/results/hr_S01_<時間戳>.csv
```

常用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--window-sec` | 10 | 分析視窗長度（秒） |
| `--step` | 1.0 | 視窗步進；調大可加速 |
| `--skip-sweeps` | 關 | 略過視窗長度掃描（較慢的那段） |
| `--min-conf` | 0.55 | 失效轉移分析的信心門檻（見下方說明） |

#### 信心門檻怎麼選

`rPPG_service.py` 的 `min_conf` 預設 **0.55**，依合成訊號的實測分布訂定：

| 門檻 | 純雜訊被誤判為可信 | 真實脈波保留率 |
|---|---|---|
| 0.35 | 19.5% | 96.7% |
| **0.55** | **1.0%** | **90.0%** |
| 0.65 | 0.0% | 81.3% |

正式實驗請用 `analyze_hr.py` 產出的「表 3　SNR 信心門檻」掃描表，
依自己受試者資料的分布重新選定，並在論文中說明選定依據。

### 3.4 產出對應到論文的哪張表

| 輸出檔 | 論文用途 |
|---|---|
| `*_report.md` 表 1 | 四種演算法的整體一致性（MAE/RMSE/r/Bland-Altman） |
| `*_report.md` 表 2 | **依運動強度分層** — 論文核心論證 |
| `*_report.md` 表 3 | SNR 門檻的準確度 vs 覆蓋率取捨 |
| `*_report.md` 表 4 | 視窗長度取捨 |
| `*_failover_report.md` | 常駐並行 vs 延遲啟動的接管延遲 |
| `*_fig_bland_altman.png` | 標準一致性分析圖，醫療類期刊必備 |
| `*_fig_stage_mae.png` | 誤差隨強度上升的長條圖 |
| `*_fig_failover.png` | 降級時序圖，最適合當論文首圖 |

### 3.5 多位受試者的彙整

每位受試者各自跑完 `analyze_hr.py` 與 `analyze_failover.py` 之後，用這支
彙整成論文表格：

```bash
python experiments/aggregate_subjects.py
```

預設會排除合成資料；要一併納入請加 `--include-synthetic`。

產出：

| 檔案 | 內容 |
|---|---|
| `aggregate_report.md` | 論文表格：mean ± SD、分層、統計檢定、降級彙整 |
| `aggregate_metrics.csv` | 各方法 × 各階段的 mean / SD / n |
| `aggregate_per_subject.csv` | 逐受試者明細，用來檢查離群值 |
| `aggregate_fig_stage_mae.png` | 含誤差棒的分層長條圖 |
| `aggregate_fig_method_box.png` | 各演算法的逐受試者盒鬚圖 |

#### ⚠️ 受試者數的硬性下限

統計檢定採雙尾 **Wilcoxon 符號等級檢定**（小樣本、不假設常態分布，
以受試者為配對單位）。這個檢定在樣本數 n 下**理論上可達到的最小 p 值
是 2/2ⁿ**：

| n | 可達最小 p | 能否宣稱 p < 0.05 |
|---|---|---|
| 4 | 0.125 | ✗ |
| 5 | 0.0625 | ✗ |
| **6** | **0.03125** | ✓ |
| 8 | 0.0078 | ✓ |
| 10 | 0.002 | ✓ |

也就是說，**只收 5 位受試者的話，即使所有人的結果都朝同一方向，
數學上也不可能達到統計顯著**。若論文需要顯著性結論，
**至少要 6 位，建議 8–10 位**以保留餘裕（有人資料品質不佳時可剔除）。

報告會自動依實際 n 顯示這項警告。

---

## 4. 實驗 C：端-雲協同傳輸分析

### 4.1 執行

```bash
python experiments/bench_transport.py --frames 60
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--source` | `webcam` | `synthetic` 為無攝影機時的替代方案 |
| `--frames` | 40 | 取樣影格數 |
| `--jpeg-up` | 40 | 上行 JPEG 品質，對應 `index.html` 的 `toDataURL(..., 0.4)` |
| `--jpeg-down` | 70 | 下行 JPEG 品質，對應 `app.py` 的 `IMWRITE_JPEG_QUALITY, 70` |
| `--no-ws` | 關 | 略過 WebSocket 迴路實測 |

> **務必用真實攝影機畫面跑**。JPEG 壓縮後的大小與畫面內容高度相關，
> 合成畫面的數字不能寫進論文。建議錄製時畫面中有人、背景是真實房間。

### 4.2 三種比較方案

| 方案 | 作法 | 實作狀態 |
|---|---|---|
| **S1 雲端全幀** | 瀏覽器送 JPEG，伺服器跑完把標註影格送回 | `app.py` 的 `/ws` |
| **S2 骨架卸載** | 端側跑姿態估計與生理感測，只上傳 17×3 關節點與心率 | **已實作**：`app.py` 的 `/ws_skeleton` + [client_s2.py](client_s2.py) |
| **S3 全端側** | 全部在裝置上運算，不用網路 | 對照組（不需另外實作） |

S2 不是紙上推估——伺服器端點與客戶端都已實作可跑：

```bash
# 伺服器
uvicorn app:app --host 0.0.0.0 --port 8000

# 客戶端（端側跑 YOLO 與心率，只上傳關節點）
python experiments/client_s2.py --url ws://localhost:8000/ws_skeleton
```

客戶端會即時顯示伺服器回傳的評分、次數、心率，以及實測的往返時間與上行位元組數。
結束時印出完整統計。

不啟動伺服器也能驗證伺服器端邏輯：

```bash
python experiments/test_s2_path.py
```

**S2 的隱私特性**：影像與臉部完全不離開使用者裝置，只有關節點座標與心率數值上傳。
這點值得在論文中獨立提出。

### 4.3 伺服器端評分成本（重要）

動作評分（ST-GCN backbone + 430 個範本比對）**S1 與 S2 都要付**，
實測單次約 **55 ms**，其中 ST-GCN 佔約 45 ms。

原本 `assess_quality()` 是每一影格都跑，等於每秒對同一個 50 幀滑動視窗
重算 30 次。現已加入節流（`FitnessAIAgent.ASSESS_INTERVAL_S = 0.2`，即 5 Hz），
30 fps 下攤提成本由 55 ms 降為 **9.2 ms/影格**。

`bench_transport.py` 會實際載入模型量測這個數字，不再使用假設值：

```bash
python experiments/bench_transport.py --frames 60          # 自動量測
python experiments/bench_transport.py --skip-scoring-measure  # 不載入模型，用預設值
python experiments/bench_transport.py --scoring-ms 55         # 直接指定
```

因此兩種方案的伺服器成本為：

| | 每影格伺服器成本 @30fps |
|---|---|
| S1 | YOLO 111 ms + JPEG 編碼 + 評分攤提 9.2 ms ≈ **121 ms** |
| S2 | 只有評分攤提 ≈ **9.2 ms** |

差距約 **13 倍**（不是早期版本估的 51 倍——那份估計把 S2 的伺服器成本
誤設為 3 ms，且漏算了 S1 也要付的評分成本）。

### 4.4 產出

| 輸出檔 | 論文用途 |
|---|---|
| `transport_report.md` 表 1 | 單影格傳輸量 |
| 表 2 | 不同 fps 下的頻寬需求 |
| 表 3 | 端到端延遲組成 |
| 表 5 | **伺服器承載能力** — 用實測評分成本計算 |
| `transport_fig_payload.png` | 傳輸量對數長條圖 |
| `transport_fig_bandwidth.png` | 頻寬 vs fps，含 4G 上限參考線 |

### 4.5 這個實驗的限制（論文要誠實寫出來）

1. **延遲的結論有前提。** 總延遲被姿態推論主導。本測試在 CPU 上跑
   YOLOv8n-pose；S2 把這段移到客戶端後，實務上瀏覽器會用
   TensorFlow.js 或 MediaPipe 的 WebGL 加速模型，耗時與本數據不同。
   要下延遲的結論，必須用真實客戶端重新量測。
2. **廣域網路時間是推估的。** 依頻寬與 RTT 模型計算，非實地量測。
   若要實測，可在 ngrok 佈署上用相同大小的 payload 測往返時間後代入。
3. **頻寬與伺服器承載兩項不受上述限制影響** — 它們只取決於傳輸量與
   伺服器端工作量，與客戶端硬體無關，可以直接當結論。

### 4.6 真實佈署量測（`bench_e2e_live.py`）

`bench_transport.py` 量的是本機處理成本加上依頻寬推估的網路時間。
這支則是對**真的跑起來的伺服器**量測，補上推估不來的兩塊數據。

#### (a) 端到端延遲與可達影格率

先啟動伺服器：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

本機量測：

```bash
python experiments/bench_e2e_live.py --url ws://localhost:8000/ws --frames 100
```

經 ngrok 的真實網路量測（含通道與 TLS 成本）：

```bash
ngrok http 8000
python experiments/bench_e2e_live.py --url wss://xxxx.ngrok-free.app/ws --frames 100
```

#### (b) 多連線壓力測試

驗證表 5 的伺服器承載推估。`--watch-cpu` 會找出本機的 uvicorn 行程並記錄 CPU：

```bash
python experiments/bench_e2e_live.py --url ws://localhost:8000/ws --clients 1,2,4,8 --frames 40 --watch-cpu
```

輸出 `e2e_live_fig_scaling.png`：並發數 vs 延遲與每連線影格率的劣化曲線。

#### (c) 純網路傳輸時間（隔離運算成本）

伺服器端只做回聲、不做任何運算，因此 S1 與 S2 的差異純粹來自傳輸量。
在伺服器那台機器上：

```bash
python experiments/bench_e2e_live.py --serve-echo --port 8899
```

另開通道：

```bash
ngrok http 8899
```

從客戶端量測：

```bash
python experiments/bench_e2e_live.py --echo-url wss://xxxx.ngrok-free.app --frames 100
```

這組數據最有價值——它是 S1 與 S2 在**同一條真實網路路徑**上的直接對照，
不受伺服器硬體與客戶端推論速度影響，可以直接當結論寫進論文。

> 注意：`--url` 模式量的是現況的 S1 架構。S2 骨架卸載要同法量測，
> 需先在伺服器實作對應端點；若只想比較兩者的網路傳輸成本，用上面的
> 回聲模式即可。

---

## 5. 已修正的問題

先前盤點出的五項問題已全部處理完畢：

| 問題 | 原本的狀況 | 現在 |
|---|---|---|
| Gemini API 金鑰硬寫在原始碼 | 金鑰明文出現在兩支 agent | 一律讀 `GEMINI_API_KEY` 環境變數，未設定則停用講評 |
| 心率沒進融合模型 | `hr_norm` 寫死 `-0.5`（等同假設心率恆為 75） | 以 `(HR-100)/50` 正規化後送入，與訓練時一致 |
| 次數計算是計時器 | 分數≥50 且距上次>1.5 秒就 +1 | [rep_counter.py](../rep_counter.py) 以骨架動作訊號計次 |
| 自動分流用假心率 | `app.py` 傳入寫死的 85，心率條件恆為真 | 決策引擎自備雙軌心率，無可信讀數時誠實改走純視覺 |
| 熱量固定 8 kcal/min | 與動作、心率都無關 | 有心率用 Keytel (2005) 迴歸式，無心率依動作查 MET 表 |

另外順手修掉的：

- **merged agent 網頁路徑的 fps 錯誤**：`process_frame()` 把取樣率寫死 30，
  但瀏覽器串流實際約 10 fps，算出的頻率是真值 3 倍、直接超出 0.8–3.0 Hz
  頻帶，顯示的心率等同亂數。四處 rPPG 區塊已全改用 v2 `update()`，
  由影格時間戳自行估計取樣率。
- **決策引擎交接時的資源洩漏**：`app.py` 在切換模式時只做 `del agent`，
  沒關掉 BLE 執行緒，會佔住心率帶讓接手的 agent 連不上。
- **`Mode_Selector.py` 的 `__main__`**：呼叫了不存在的參數與方法，
  直接執行必然拋 `TypeError`。

### ⚠️ 仍需你手動處理

**撤銷舊的 Gemini API 金鑰。** 它曾經明文存在於原始碼中，必須當作已外洩
處理——到 Google AI Studio 刪除該金鑰並重新產生一把，然後用環境變數提供：

```bash
setx GEMINI_API_KEY "你的新金鑰"
```

若專案曾經推上 GitHub，Git 歷史中仍留有舊金鑰，撤銷是唯一有效的補救。

### 計次器的效果

以合成骨架（已知次數）驗證：

| | 平均誤差 |
|---|---|
| 原本的計時器 | 4.25 次 |
| **rep_counter** | **0.25 次** |

且靜止不動不會計次、棒式等靜態支撐改為回報持續秒數。
驗證程式：`python experiments/test_rep_counter.py`

---

## 6. 建議的執行順序

1. 跑兩支自我檢驗 → 確認環境正常
2. 跑合成資料流程 → 確認分析管線正常
3. 自己先當受試者跑一輪完整協定 → 檢查真實資料的品質與臉部偵測率
4. 依第 3 步的結果調整環境（光源、距離），再正式收 8–10 位受試者
5. 實驗 C 用真實攝影機畫面跑一次
6. 每位受試者各跑一次 `analyze_hr.py` 與 `analyze_failover.py`
7. 執行 `aggregate_subjects.py` 產生論文表格
8. 開始寫論文
