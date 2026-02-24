import streamlit as st
from ultralytics import YOLO
from PIL import Image
import collections
import numpy as np

# --- 0. 介面設定 ---
st.set_page_config(page_title="AI 麻將聽牌小幫手", layout="centered")

st.markdown("""
    <style>
        /* =========================================
           1. 全域按鈕設定 (預設給麻將牌使用)
           這是你指定的：大尺寸、粗框、大字體
           ========================================= */
        div[data-testid="column"] .stButton > button {
            border: 2px solid #333 !important; 
            background-color: white !important;
            height: 100px !important; 
            width: 80px !important; 
            margin: 2px auto !important;
            display: flex !important; 
            align-items: center !important; 
            justify-content: center !important;
            padding: 0 !important;
            border-radius: 6px !important;
        }
        
        /* 麻將牌的文字 (大 Emoji) */
        .stButton > button p { 
            font-size: 70px !important; 
            color: #1B1B3A !important; 
            font-family: "Segoe UI Emoji", sans-serif !important; 
            margin: 0 !important; 
            line-height: 1 !important;
        }

        /* =========================================
           2. 正常功能按鈕的「特效藥」 (重新分析、交換手牌)
           使用 .normal-button 包裹，強制覆蓋上面的設定
           ========================================= */
        .normal-button .stButton > button {
            width: 100% !important;        /* 寬度填滿 */
            height: auto !important;       /* 高度自動 */
            padding: 12px 20px !important; /* 舒服的內距 */
            background-color: #f0f2f6 !important; /* 淺灰背景 */
            border: 1px solid #ccc !important;    /* 淺灰邊框 */
            margin: 10px 0 !important;
        }
        
        /* 正常按鈕的文字 (正常大小、橫向) */
        .normal-button .stButton > button p {
            font-size: 20px !important;    
            color: #333 !important;
            font-family: sans-serif !important;
            writing-mode: horizontal-tb !important; /* 強制橫排 */
        }

        /* =========================================
           3. 其他介面樣式
           ========================================= */
        .section-header { font-size: 24px; font-weight: bold; color: #1B1B3A; margin: 20px 0 10px 0; border-bottom: 3px solid #CCCCFF; padding-bottom: 5px; }
        .count-badge { background-color: #1B1B3A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 18px; margin-left: 10px; vertical-align: middle; }
        
        /* 結果顯示區 */
        .result-box {
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-title { font-size: 32px; font-weight: bold; margin-bottom: 20px; opacity: 0.9; }
        .result-content { font-size: 40px; font-weight: 800; line-height: 1.4; }
        
        /* 聽牌列表圖示 */
        .waiting-tiles-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-top: 20px; }
        .waiting-tile {
            background-color: #fff; border: 2px solid #333; border-radius: 8px;
            padding: 10px 20px; font-size: 60px; line-height: 1;
            display: flex; flex-direction: column; align-items: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .waiting-name { font-size: 20px; font-weight: normal; margin-top: 5px; color: #555; }
        
        /* 錯誤訊息 */
        .error-msg { font-size: 28px; font-weight: bold; color: #721c24; }
        .hint-msg { font-size: 20px; color: #666; margin-top: 10px; }
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
    '7s': {'name': '七條', 'icon': '🀖', 'w': 27, 'type': 's', 'val': 7}, '8s': {'name': '八條', 'icon': '🀗', 'w': 28, 'type': 's', 'val': 8},
    '9s': {'name': '九條', 'icon': '🀘', 'w': 29, 'type': 's', 'val': 9},
    'ew': {'name': '東', 'icon': '🀀', 'w': 31, 'type': 'z'}, 'sw': {'name': '南', 'icon': '🀁', 'w': 32, 'type': 'z'},
    'ww': {'name': '西', 'icon': '🀂', 'w': 33, 'type': 'z'}, 'nw': {'name': '北', 'icon': '🀃', 'w': 34, 'type': 'z'},
    'zhong': {'name': '中', 'icon': '🀄︎', 'w': 35, 'type': 'z'}, 'fa': {'name': '發', 'icon': '🀅', 'w': 36, 'type': 'z'},
    'wd': {'name': '白', 'icon': '🀆', 'w': 37, 'type': 'z'},
    '1rf': {'name': '春', 'icon': '🀦', 'w': 51, 'type': 'h', 'suit': 'rf', 'v': 1}, '2rf': {'name': '夏', 'icon': '🀧', 'w': 52, 'type': 'h', 'suit': 'rf', 'v': 2},
    '3rf': {'name': '秋', 'icon': '🀨', 'w': 53, 'type': 'h', 'suit': 'rf', 'v': 3}, '4rf': {'name': '冬', 'icon': '🀩', 'w': 54, 'type': 'h', 'suit': 'rf', 'v': 4},
    '1bf': {'name': '梅', 'icon': '🀢', 'w': 55, 'type': 'h', 'suit': 'bf', 'v': 1}, '2bf': {'name': '蘭', 'icon': '🀣', 'w': 56, 'type': 'h', 'suit': 'bf', 'v': 2},
    '3bf': {'name': '竹', 'icon': '🀤', 'w': 57, 'type': 'h', 'suit': 'bf', 'v': 3}, '4bf': {'name': '菊', 'icon': '🀥', 'w': 58, 'type': 'h', 'suit': 'bf', 'v': 4}
}

# --- 2. 演算法邏輯 ---
def recursive_decompose(counts, sets_needed, win_tile, current_sets=[]):
    if sum(counts.values()) == 0:
        return (sets_needed == 0), current_sets
    if sets_needed <= 0: return False, []
    
    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    
    for take in [4, 3]:
        if counts[tile] >= take:
            temp = counts.copy(); temp[tile] -= take
            ok, res = recursive_decompose(temp, sets_needed - 1, win_tile, current_sets + [(f'set_{take}', tile)])
            if ok: return True, res
            
    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info.get('val', 0) <= 7:
        t2 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+1), None)
        t3 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+2), None)
        if t2 and t3 and counts.get(t2,0) > 0 and counts.get(t3,0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[t2]-=1; temp[t3]-=1
            seq = [tile, t2, t3]; pos = seq.index(win_tile) if win_tile in seq else -1
            ok, res = recursive_decompose(temp, sets_needed - 1, win_tile, current_sets + [('seq', seq, pos)])
            if ok: return True, res
    return False, []

def check_hu_for_waiting(counts):
    for eye in counts:
        if counts[eye] >= 2:
            temp = counts.copy()
            temp[eye] -= 2
            rem_tiles = sum(temp.values())
            if rem_tiles % 3 != 0: continue
            sets_needed = rem_tiles // 3
            ok, _ = recursive_decompose(temp, sets_needed, '1w')
            if ok: return True
    return False

def get_waiting_tiles(hand_codes):
    counts = collections.Counter(hand_codes)
    waiting = []
    all_tiles = [k for k,v in TILE_INFO.items() if v['type'] != 'h']
    
    for t in all_tiles:
        temp = counts.copy()
        temp[t] += 1
        if check_hu_for_waiting(temp):
            waiting.append(t)
            
    return waiting

# --- 3. 核心處理函數 ---
def analyze_waiting_status(con, exp):
    hand_only = [c for c in con if TILE_INFO[c]['type'] != 'h']
    
    total_counts = collections.Counter(con + exp)
    for code, count in total_counts.items():
        info = TILE_INFO[code]
        if info['type'] != 'h' and count > 4:
            return "error", f"牌數錯誤：**{info['name']}** 有 {count} 張 (上限 4)", []

    hand_len = len(hand_only)
    if hand_len % 3 == 0:
        return "error", f"手牌數量為 {hand_len} 張 (相公)。<br>若要聽牌，手牌應為 1, 4, 7, 10, 13, 16... 張 (少一張牌的狀態)。", []
    elif hand_len % 3 == 2:
        return "error", f"手牌數量為 {hand_len} 張。<br>這是已經胡牌或未打牌的數量，請移除一張牌以計算聽牌。", []
    
    waiting_list = get_waiting_tiles(hand_only)
    
    if waiting_list:
        return "waiting", "聽牌中！", waiting_list
    else:
        return "not_waiting", "尚未聽牌", []


# --- 4. 影像偵測 ---
def process_detection(image_obj, source_type, current_model_name):
    img_id_base = getattr(image_obj, 'name', 'camera') if source_type == 'upload' else 'camera_shot'
    cache_key = (img_id_base, current_model_name)
    
    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        st.session_state.current_cache_key = cache_key
        st.session_state.current_image = image_obj
        
        results = model(image_obj)
        st.session_state.current_plot = results[0].plot()
        
        tile_data = []
        for r in results:
            source = r.obb if hasattr(r, 'obb') and r.obb is not None else r.boxes
            if source is not None:
                for i, c in enumerate(source.cls):
                    try:
                        if hasattr(source, 'xywhr'): box = source.xywhr[i].cpu().numpy()
                        else: box = source.xywh[i].cpu().numpy()
                        
                        tile_data.append({
                            'code': model.names[int(c)], 
                            'x': float(box[0]), 
                            'y': float(box[1])
                        })
                    except: continue 

        if not tile_data:
            st.warning("未偵測到任何麻將牌")
            st.session_state.con_manual = []
            st.session_state.exp_manual = []
            return

        sorted_y = sorted(tile_data, key=lambda x: x['y'])
        gaps = np.diff([d['y'] for d in sorted_y])
        max_idx = np.argmax(gaps) if len(gaps) > 0 else -1
        
        threshold = (sorted_y[max_idx]['y'] + sorted_y[max_idx+1]['y'])/2 if (max_idx != -1 and gaps[max_idx] > 40) else 9999
        
        st.session_state.con_manual = [d['code'] for d in tile_data if d['y'] >= threshold]
        st.session_state.exp_manual = [d['code'] for d in tile_data if d['y'] < threshold]

# --- 5. UI 渲染主程式 ---
def render_main_ui():
    if 'current_plot' not in st.session_state:
        st.info("☝️ 請先從上方上傳照片或使用相機拍照，AI 將自動辨識手牌。")
        return
    
    st.image(st.session_state.current_plot, caption=f"AI 辨識結果", use_container_width=True)

    # --- 牌面管理區域 ---
    all_codes = st.session_state.con_manual + st.session_state.exp_manual
    st.markdown(f'<div class="section-header">🎴 牌面管理 <span class="count-badge">總張數：{len(all_codes)}</span></div>', unsafe_allow_html=True)
    
    # 1. 手牌區
    st.write(f"**🐹 手牌 (Concealed)：**")
    codes = st.session_state.con_manual
    s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
    cols = st.columns(11) # 放在 column 內的按鈕會被套用大麻將牌樣式
    for i, idx in enumerate(s_idx):
        with cols[i % 11]:
            if st.button(TILE_INFO[codes[idx]]['icon'], key=f"h_{i}"):
                st.session_state.con_manual.pop(idx); st.rerun()
    
    with st.popover(f"➕ 新增手牌"):
        st.write("點擊圖示加入：")
        all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
        cols_add = st.columns(8)
        for idx, (k, v) in enumerate(all_keys):
            with cols_add[idx % 8]:
                if st.button(v['icon'], key=f"add_h_{k}"):
                    st.session_state.con_manual.append(k); st.rerun()

    # ★ 這裡使用 .normal-button 來包裝，強制套用縮小樣式
    st.markdown('<div class="normal-button">', unsafe_allow_html=True)
    if st.button("🔃 交換手牌 與 門前牌", help="AI 分錯排時使用"):
        st.session_state.con_manual, st.session_state.exp_manual = st.session_state.exp_manual, st.session_state.con_manual
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. 門前牌區
    st.write(f"**🐥 門前牌 (Exposed)：**")
    codes = st.session_state.exp_manual
    s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
    cols = st.columns(11)
    for i, idx in enumerate(s_idx):
        with cols[i % 11]:
            if st.button(TILE_INFO[codes[idx]]['icon'], key=f"d_{i}"):
                st.session_state.exp_manual.pop(idx); st.rerun()
    
    with st.popover(f"➕ 新增門前"):
        st.write("點擊圖示加入：")
        all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
        cols_add = st.columns(8)
        for idx, (k, v) in enumerate(all_keys):
            with cols_add[idx % 8]:
                if st.button(v['icon'], key=f"add_d_{k}"):
                    st.session_state.exp_manual.append(k); st.rerun()

    # --- 分析結果區域 ---
    st.markdown("---")
    
    status, title, data = analyze_waiting_status(st.session_state.con_manual, st.session_state.exp_manual)
    
    if status == "waiting":
        bg_color, text_color = "#cce5ff", "#004085"
        icon_html = ""
        for tile_code in data:
            icon_html += f'<div class="waiting-tile"><div>{TILE_INFO[tile_code]["icon"]}</div><div class="waiting-name">{TILE_INFO[tile_code]["name"]}</div></div>'
            
        html_content = f"""
<div class="result-box" style="background-color: {bg_color}; color: {text_color};">
    <div class="result-title">👀 聽牌分析</div>
    <div class="result-content">🔥 {title}</div>
    <div style="margin-top: 10px; font-size: 20px;">這手牌聽以下這些牌：</div>
    <div class="waiting-tiles-container">
        {icon_html}
    </div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)
        
    elif status == "not_waiting":
        bg_color, text_color = "#fff3cd", "#856404"
        html_content = f"""
<div class="result-box" style="background-color: {bg_color}; color: {text_color};">
    <div class="result-title">👀 聽牌分析</div>
    <div class="result-content">{title} 🧩</div>
    <div class="hint-msg">再加把勁！調整手牌組合看看。</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)
        
    else: 
        bg_color, text_color = "#f8d7da", "#721c24"
        html_content = f"""
<div class="result-box" style="background-color: {bg_color}; color: {text_color};">
    <div class="result-title">⚠️ 牌型異常</div>
    <div class="error-msg">相公 👻</div>
    <div class="hint-msg">{title}</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)

    # ★ 這裡使用 .normal-button 來包裝，強制套用縮小樣式
    st.markdown('<div class="normal-button">', unsafe_allow_html=True)
    if st.button("🔄 重新分析", key="refresh_all"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 啟動入口 ---
t1, t2 = st.tabs(["📷︎ 即時拍照", "📁 上傳照片"])
with t1:
    cam = st.camera_input("拍照")
    if cam: process_detection(Image.open(cam), 'camera', model_choice)
with t2:
    up = st.file_uploader("選照片", type=['png', 'jpg', 'jpeg'])
    if up: process_detection(Image.open(up), 'upload', model_choice)

render_main_ui()

