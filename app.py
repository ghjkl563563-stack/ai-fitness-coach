# http://localhost:8000
# uvicorn app:app --host 0.0.0.0 --port 8000
# .\ngrok http 8000
import base64
import cv2
import numpy as np
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# 🌟 新增：匯入改寫好的網頁版決策引擎
from Mode_Selector import WebModeSelector
from AI_Agent_Strength_Training import StrengthTrainingAgent
from AI_Agent_crossatt_multimodal_merged import FitnessAIAgent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.get("/")
async def get_webpage():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

def decode_image(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 網頁端已連線！")

    agent = None

    while True:
        try:
            data = await websocket.receive_json()

            # ==========================================
            # 判斷網頁傳來的「指令」
            # ==========================================
            if "action" in data:

                # --- 邏輯 1：啟動模式 ---
                if data["action"] == "switch_mode":
                    mode = data["mode"]
                    if mode == "auto":
                        print("🚀 啟動【智能決策引擎】...")
                        # ✅ 修正 1：呼叫專為網頁設計的 WebModeSelector
                        agent = WebModeSelector()
                    elif mode == "1":
                        print("🚀 啟動【手動重訓模式】...")
                        agent = StrengthTrainingAgent(cam_id=0)
                    elif mode == "2":
                        print("🚀 啟動【手動居家模式】...")
                        agent = FitnessAIAgent(weight_path="best_model 1.pth", stgcn_path="models/gcn_weight.pth", exemplar_path="exemplar_bank.pt")

                    await websocket.send_json({"status": "ready"})

                # --- 邏輯 2：停止模式與釋放資源 ---
                elif data["action"] == "stop":
                    print("🛑 收到停止指令，正在釋放資源...")
                    if agent is not None:
                        if hasattr(agent, 'close'):
                            agent.close()
                        for attr in ('ble_sensor', 'hr_sensor', 'hr_sensor_ble'):
                            snr = getattr(agent, attr, None)
                            if snr is not None and hasattr(snr, '_stop_event'):
                                snr._stop_event.set()
                        rp = getattr(agent, 'hr_sensor_rppg', None)
                        if rp is not None and hasattr(rp, 'close'):
                            rp.close()

                        del agent
                        agent = None
                    await websocket.send_json({"status": "stopped"})

                # --- ✨ 邏輯 3：處理網頁傳來的語音聊天 ✨ ---
                elif data["action"] == "chat" and agent is not None:
                    user_msg = data["message"]
                    print(f"👤 玩家語音說: {user_msg}")
                    if hasattr(agent, 'chat_with_coach'):
                        reply = agent.chat_with_coach(user_msg)
                        await websocket.send_json({"advice": reply})

            # ==========================================
            # 判斷網頁傳來的「影像」
            # ==========================================
            elif "image" in data and agent is not None:
                frame = decode_image(data["image"])

                # ✅ 修正 2：針對「決策引擎」和「AI 教練」進行分流處理
                if isinstance(agent, WebModeSelector):
                    # 由決策引擎自己的雙軌感測器讀取真實心率。
                    # 先前這裡傳入寫死的 85，使心率條件恆為真，等同只用姿態決策。
                    processed_frame, decision = agent.process_frame(frame)

                    # 如果 AI 已經做好決定了 (decision 不再是 None)
                    if decision is not None:
                        print(f"🔄 偵測完畢！正在自動無縫切換至模式：{decision}...")
                        # 必須先關掉決策引擎的感測器，否則它的 BLE 執行緒會繼續
                        # 佔住心率帶，接手的 agent 將連不上同一台裝置。
                        agent.close()
                        del agent

                        # 自動交接給對應的主廚
                        if decision == "1":
                            agent = StrengthTrainingAgent(cam_id=0)
                        elif decision == "2":
                            agent = FitnessAIAgent(weight_path="best_model 1.pth", stgcn_path="models/gcn_weight.pth", exemplar_path="exemplar_bank.pt")

                    current_advice = "AI 正在掃描您的姿態與環境..."

                else:
                    # 這是一般的教練處理模式
                    processed_frame = agent.process_frame(frame)
                    current_advice = getattr(agent, 'gemini_advice', "")

                # 把圖片跟講評一起包裝傳給網頁
                out_b64 = encode_image(processed_frame)
                await websocket.send_json({
                    "image": out_b64,
                    "advice": current_advice
                })

        except Exception as e:
            print("🔴 連線中斷:", e)
            break


# ======================================================================
# S2 骨架卸載端點 (實驗 C 提出的架構)
# ======================================================================
# 與 /ws 的差別：
#   /ws          客戶端上傳 JPEG 影格，伺服器跑 YOLO + 評分，回傳標註後的影格
#   /ws_skeleton 客戶端自己跑姿態估計，只上傳 17 個關節點，
#                伺服器只做評分，回傳結構化結果（無影像）
#
# 效益：每影格傳輸量約由 31 KB 降至 0.4 KB，且伺服器不需做視覺推論，
#       單機可服務的並發連線數大幅提升。
# 隱私：影像與臉部完全不離開使用者裝置。
#
# 協定
#   → {"action": "start"}                              建立評分器
#   ← {"status": "ready"}
#   → {"kps": [[x, y, conf] × 17], "hr": 92}           每影格 (hr 可省略)
#   ← {"score":…, "action":…, "reps":…, "hr":…, "advice":…}
#   → {"action": "stop"}
# ======================================================================
@app.websocket("/ws_skeleton")
async def websocket_skeleton(websocket: WebSocket):
    await websocket.accept()
    print("🟢 [S2] 骨架卸載端已連線")

    agent = None
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start":
                print("🚀 [S2] 建立評分器 ...")
                # use_sensors=False：S2 架構下心率由客戶端量測後隨關節點一起
                # 上傳，伺服器沒有影像也就不需要 (也無法) 跑 rPPG。
                agent = FitnessAIAgent(
                    weight_path="best_model 1.pth",
                    stgcn_path="models/gcn_weight.pth",
                    exemplar_path="exemplar_bank.pt",
                    use_sensors=False,
                )
                await websocket.send_json({"status": "ready"})
                continue

            if data.get("action") == "stop":
                if agent is not None:
                    del agent
                    agent = None
                await websocket.send_json({"status": "stopped"})
                continue

            if "kps" in data:
                if agent is None:
                    await websocket.send_json({"error": "尚未啟動，請先送 action=start"})
                    continue
                result = agent.process_keypoints(data["kps"], data.get("hr"))
                await websocket.send_json(result)

    except Exception as e:
        print("🔴 [S2] 連線中斷:", e)
    finally:
        if agent is not None:
            del agent
