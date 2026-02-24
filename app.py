import streamlit as st
from ultralytics import YOLO
from PIL import Image
import collections
import numpy as np

# --- 0. 介面設定 ---
st.set_page_config(page_title="AI 麻將聽牌小幫手", layout="centered")

st.markdown("""
    <style>
        /* 按鈕樣式 (麻將牌按鈕) */
        .stButton > button {
            border: 2px solid #333 !important; background-color: white !important;
            height: 90px !important; width: 70px !important; margin: 1px !important;
            display: flex !important; align-items: center !important; justify-content: center !important;
            padding: 0 !important;
        }
        .stButton > button div p { font-size: 55px !important; color: #1B1B3A !important; font-family: "Segoe UI Emoji" !important; margin: 0 !important; }
        
        /* 區塊標題 */
        .section-header { font-size: 24px; font-weight: bold; color: #1B1B3A; margin: 20px 0 10px 0; border-bottom: 3px solid #CCCCFF; padding-bottom: 5px; }
        .count-badge { background-color: #1B1B3A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 18px; margin-left: 10px; vertical-align: middle; }
        
        /* 一般操作按鈕的容器 (用來覆寫麻將牌的強制大小，讓一般按鈕恢復正常) */
        .normal-btn .stButton > button {
            height: auto !important; 
            width: 100% !important;
            background-color: #f0f2f6 !important;
            border: 1px solid #ccc !important;
            padding: 10px !important;
            border-radius: 8px !important;
        }
        .normal-btn .stButton > button div p {
            font-size: 20px !important;
            color: #333 !important;
            font-family: inherit !important;
        }
        
        /* --- 結果顯示區樣式 (加大字體) --- */
        .result-box {
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
            opacity: 0.9;
        }
        .result-content {
            font-size: 40px; /* 加大文字 */
            font-weight: 800;
            line-height: 1.4;
        }
        
        /* 聽牌列表的圖示樣式 */
        .waiting-tiles-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
        }
        .waiting-tile {
            background-color: #fff;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 60px; /* 超大麻將圖示 */
            line-height: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .waiting-name {
            font-size: 20px;
            font-weight: normal;
            margin-top: 5px;
            color: #555;
        }
        
        /* 錯誤訊息樣式 */
        .error-msg {
            font-size: 28px;
            font-weight: bold;
            color: #721c24;
        }
        .hint-msg {
            font-size: 20px;
            color: #666;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🀄️ 麻將聽牌小幫手")

# --- 1. 設定與定義 ---
with st.sidebar:
    st.title("⚙️ 系統設定")
    model_choice = st.selectbox("辨識模型", ("yolov8s(2).pt", "yolov8n(2).pt", "YOLOv8s_obb.pt", "YOLOv8n_obb.pt"))
    st.info("本工具僅進行聽牌分析，不計算台數。")

@st.cache_resource
def load_yolo_model(name): return YOLO(name)
model = load_yolo_model(model_choice)

# 定義麻將牌資料
TILE_INFO = {
    '1w': {'name': '一萬', 'icon': '🀇', 'w': 1, 'type': 'w', 'val': 1}, '2w': {'name': '二萬', 'icon': '🀈', 'w': 2, 'type': 'w', 'val': 2},
    '3w': {'name': '三萬', 'icon': '🀉', 'w': 3, 'type': 'w', 'val': 3}, '4w': {'name': '四萬', 'icon': '🀊', 'w': 4, 'type': 'w', 'val': 4},
    '5w': {'name': '五萬', 'icon': '🀋', 'w': 5, 'type': 'w', 'val': 5}, '6w': {'name': '六萬', 'icon': '🀌', 'w': 6, 'type': 'w', 'val': 6},
    '7w': {'name': '七萬', 'icon': '🀍', 'w': 7, 'type': 'w', 'val': 7}, '8w': {'name': '八萬', 'icon': '🀎', 'w': 8, 'type': 'w', 'val': 8},
    '9w': {'name': '九萬', 'icon': '🀏', 'w': 9, 'type': 'w', 'val': 9},
    '1D': {'name': '一筒', 'icon': '🀙', 'w': 11, 'type': 'D', 'val': 1}, '2D': {'name': '二筒', 'icon': '🀚', 'w': 12, 'type': 'D', 'val': 2},
    '3D': {'name': '三筒', 'icon': '🀛', 'w': 13, 'type': 'D', 'val': 3}, '4D': {'name': '四筒', 'icon': '🀜', 'w': 14, 'type': 'D', 'val': 4},
    '5D': {'name': '五筒', 'icon': '🀝', 'w': 15, 'type': 'D', 'val': 5}, '6D': {'name': '六筒', 'icon': '🀞', 'w': 16, 'type': 'D', 'val': 6},
    '7D': {'name': '七筒', 'icon': '🀟', 'w': 17, 'type': 'D', 'val': 7}, '8D': {'name': '八筒', 'icon': '🀠', 'w': 18, 'type': 'D', 'val': 8},
    '9D': {'name': '九筒', 'icon': '🀡', 'w': 19, 'type': 'D', 'val': 9},
    '1s': {'name': '一條', 'icon': '🀐', 'w': 21, 'type': 's', 'val': 1}, '2s': {'name': '二條', 'icon': '🀑', 'w': 22, 'type': 's', 'val': 2},
    '3s': {'name': '三條', 'icon': '🀒', 'w': 23, 'type': 's', 'val': 3}, '4s': {'name': '四條', 'icon': '🀓', 'w': 24, 'type': 's', 'val': 4},
    '5s': {'name': '五條', 'icon': '🀔', 'w': 25, 'type': 's', 'val': 5}, '6s': {'name': '六條', 'icon': '🀕', 'w': 26, 'type': 's', 'val': 6},
    '7s': {'name': '七條', 'icon': '🀖', 'w': 27, 'type': 's', 'val': 7}, '8s': {'name': '八條', 'icon': '🀗', 'w':
