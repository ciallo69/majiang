import streamlit as st
from ultralytics import YOLO
from PIL import Image
import collections
import numpy as np
import base64
import os


# --- 0. 介面與 CSS 樣式設定 (精確對齊您指定的紫色風格) ---
st.set_page_config(page_title="AI麻將算台平台", layout="centered")

st.markdown("""
    <style>
        /* 修改 multiselect 標籤背景與邊框 */
        div[data-baseweb="tag"], span[data-baseweb="tag"] {
            background-color: #ccccff !important;
            border: 1px solid #B2B2FF !important;
            border-radius: 4px !important;
        }
        /* 修改標籤文字顏色 */
        div[data-baseweb="tag"] span, span[data-baseweb="tag"] span {
            color: #2E2E66 !important;
            font-weight: 500 !important;
        }
        /* 修改刪除按鈕顏色 */
        div[data-baseweb="tag"] div[role="button"] svg, 
        span[data-baseweb="tag"] div[role="button"] svg {
            fill: #2E2E66 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🀄️🀅 🀆 麻將自動算台 ")

# --- 1. 系統設定 (側邊欄) ---
st.sidebar.title("⚙️ 核心設定")
model_choice = st.sidebar.selectbox("選擇辨識模型", ("yolov8s(2).pt", "yolov8n(2).pt", "YOLOv8s_obb.pt", "YOLOv8n_obb.pt"))

# 🌟 新增：底台設定框
base_tai_input = st.sidebar.number_input("設定底台", min_value=0, step=1, value=0, help="設定胡牌的基本底台數")

@st.cache_resource
def load_yolo_model(name):
    return YOLO(name)

model = load_yolo_model(model_choice)

# --- 2. 牌名對照表 ---
TILE_INFO = {
    '1w': {'name': '一萬', 'icon': '🀇', 'w': 1, 'val': 1, 'type': 'w'}, '2w': {'name': '二萬', 'icon': '🀈', 'w': 2, 'val': 2, 'type': 'w'},
    '3w': {'name': '三萬', 'icon': '🀉', 'w': 3, 'val': 3, 'type': 'w'}, '4w': {'name': '四萬', 'icon': '🀊', 'w': 4, 'val': 4, 'type': 'w'},
    '5w': {'name': '五萬', 'icon': '🀋', 'w': 5, 'val': 5, 'type': 'w'}, '6w': {'name': '六萬', 'icon': '🀌', 'w': 6, 'val': 6, 'type': 'w'},
    '7w': {'name': '七萬', 'icon': '🀍', 'w': 7, 'val': 7, 'type': 'w'}, '8w': {'name': '八萬', 'icon': '🀎', 'w': 8, 'val': 8, 'type': 'w'},
    '9w': {'name': '九萬', 'icon': '🀏', 'w': 9, 'val': 9, 'type': 'w'},
    '1D': {'name': '一筒', 'icon': '🀙', 'w': 11, 'val': 1, 'type': 'D'}, '2D': {'name': '二筒', 'icon': '🀚', 'w': 12, 'val': 2, 'type': 'D'},
    '3D': {'name': '三筒', 'icon': '🀛', 'w': 13, 'val': 3, 'type': 'D'}, '4D': {'name': '四筒', 'icon': '🀜', 'w': 14, 'val': 4, 'type': 'D'},
    '5D': {'name': '五筒', 'icon': '🀝', 'w': 15, 'val': 5, 'type': 'D'}, '6D': {'name': '六筒', 'icon': '🀞', 'w': 16, 'val': 6, 'type': 'D'},
    '7D': {'name': '七筒', 'icon': '🀟', 'w': 17, 'val': 7, 'type': 'D'}, '8D': {'name': '八筒', 'icon': '🀠', 'w': 18, 'val': 8, 'type': 'D'},
    '9D': {'name': '九筒', 'icon': '🀡', 'w': 19, 'val': 9, 'type': 'D'},
    '1s': {'name': '一條', 'icon': '🀐', 'w': 21, 'val': 1, 'type': 's'}, '2s': {'name': '二條', 'icon': '🀑', 'w': 22, 'val': 2, 'type': 's'},
    '3s': {'name': '三條', 'icon': '🀒', 'w': 23, 'val': 3, 'type': 's'}, '4s': {'name': '四條', 'icon': '🀓', 'w': 24, 'val': 4, 'type': 's'},
    '5s': {'name': '五條', 'icon': '🀔', 'w': 25, 'val': 5, 'type': 's'}, '6s': {'name': '六條', 'icon': '🀕', 'w': 26, 'val': 6, 'type': 's'},
    '7s': {'name': '七條', 'icon': '🀖', 'w': 27, 'val': 7, 'type': 's'}, '8s': {'name': '八條', 'icon': '🀗', 'w': 28, 'val': 8, 'type': 's'},
    '9s': {'name': '九條', 'icon': '🀘', 'w': 29, 'val': 9, 'type': 's'},
    'ew': {'name': '東', 'icon': '🀀', 'w': 31, 'type': 'z'}, 'sw': {'name': '南', 'icon': '🀁', 'w': 32, 'type': 'z'},
    'ww': {'name': '西', 'icon': '🀂', 'w': 33, 'type': 'z'}, 'nw': {'name': '北', 'icon': '🀃', 'w': 34, 'type': 'z'},
    'zhong': {'name': '中', 'icon': '🀄︎', 'w': 35, 'type': 'z'}, 'fa': {'name': '發', 'icon': '🀅', 'w': 36, 'type': 'z'},
    'wd': {'name': '白', 'icon': '🀆', 'w': 37, 'type': 'z'},
    '1rf': {'name': '春', 'icon': '🀦', 'w': 51, 'type': 'h'}, '2rf': {'name': '夏', 'icon': '🀧', 'w': 52, 'type': 'h'},
    '3rf': {'name': '秋', 'icon': '🀨', 'w': 53, 'type': 'h'}, '4rf': {'name': '冬', 'icon': '🀩', 'w': 54, 'type': 'h'},
    '1bf': {'name': '梅', 'icon': '🀢', 'w': 55, 'type': 'h'}, '2bf': {'name': '蘭', 'icon': '🀣', 'w': 56, 'type': 'h'},
    '3bf': {'name': '竹', 'icon': '🀤', 'w': 57, 'type': 'h'}, '4bf': {'name': '菊', 'icon': '🀥', 'w': 58, 'type': 'h'}
}

# --- 3. 核心胡牌判定演算法 ---
def is_decomposable(counts, sets_needed):
    if sum(counts.values()) == 0: return sets_needed == 0
    if sets_needed <= 0: return False
    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    for take in [4, 3]:
        if counts[tile] >= take:
            temp = counts.copy(); temp[tile] -= take
            if is_decomposable(temp, sets_needed - 1): return True
    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info['val'] <= 7:
        tile_b = next((k for k,v in TILE_INFO.items() if v['type']==info['type'] and v['val']==info['val']+1), None)
        tile_c = next((k for k,v in TILE_INFO.items() if v['type']==info['type'] and v['val']==info['val']+2), None)
        if tile_b and tile_c and counts.get(tile_b, 0) > 0 and counts.get(tile_c, 0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[tile_b]-=1; temp[tile_c]-=1
            if is_decomposable(temp, sets_needed - 1): return True
    return False

def can_hu(codes):
    hand_tiles = [c for c in codes if TILE_INFO[c]['type'] != 'h']
    num_hand = len(hand_tiles)
    if not (17 <= num_hand <= 22): return False, f"手牌張數非法 ({num_hand}張)"
    counts = collections.Counter(hand_tiles)
    for tile, count in counts.items():
        if count >= 2:
            temp = counts.copy(); temp[tile] -= 2
            if is_decomposable(temp, 5): return True, "胡牌成功"
    return False, "結構不符合胡牌型 (有孤牌)，判定為相公"

# 🌟 新增：傳入 base_tai 參數進行計算
def calculate_tai_system(codes, is_z, streak, m_list, base_tai):
    tai, reason = 0, []
    
    # 加入底台
    if base_tai > 0:
        tai += base_tai
        reason.append(f"底台 {base_tai}台")

    hand_codes = [c for c in codes if TILE_INFO[c]['type'] != 'h']
    hua_codes = [c for c in codes if TILE_INFO[c]['type'] == 'h']
    hand_names = [TILE_INFO[c]['name'] for c in hand_codes]
    n_counts = collections.Counter([TILE_INFO[c]['name'] for c in codes])
    
    if is_z:
        s_tai = 2 * streak + 1
        tai += s_tai; reason.append(f"莊家連 {streak} 拉 {streak} 共 {s_tai}台" if streak > 0 else "莊家 1台")

    m_map = {"自摸": 1, "中洞": 1, "邊張": 1, "單調": 1, "門清": 1,  "河底撈魚": 1, "搶槓": 1, "槓上開花+自摸": 2, "海底撈月+自摸": 2,
             "全求人(含單調)": 2, "平胡": 2, "三暗刻": 2, "門清一摸三": 3, "四暗刻": 5, "咪幾": 8, "哩咕": 8, "五暗刻": 8}
    for m in m_list:
        if m in m_map: tai += m_map[m]; reason.append(f"手動：{m} {m_map[m]}台")

    h_counts = collections.Counter(hand_codes)
    for t, c in h_counts.items():
        if c >= 2:
            temp = h_counts.copy(); temp[t] -= 2
            # 若剩餘牌皆為 3 或 4 的倍數，代表全由刻子/槓子組成
            if all(v in [0, 3, 4] for v in temp.values()):
                tai += 4; reason.append("碰碰胡 4台")
                break

    u_hua = len(set(hua_codes))
    if u_hua == 8: tai += 8; reason.append("八仙過海 8台")
    elif u_hua == 7: tai += 8; reason.append("七搶一(七張花) 8台")
    else:
        if all(c in hua_codes for c in ['1rf','2rf','3rf','4rf']): tai += 2; reason.append("花槓 (春夏秋冬) 2台")
        if all(c in hua_codes for c in ['1bf','2bf','3bf','4bf']): tai += 2; reason.append("花槓 (梅蘭竹菊) 2台")

    w_tri = sum(1 for w in ['東','南','西','北'] if n_counts[w] >= 3)
    if w_tri == 4: tai += 16; reason.append("大四喜 16台")
    elif w_tri == 3 and sum(1 for w in ['東','南','西','北'] if n_counts[w] == 2) == 1: tai += 8; reason.append("小四喜 8台")
    d_tri = sum(1 for d in ['中','發','白'] if n_counts[d] >= 3)
    if d_tri == 3: tai += 8; reason.append("大三元 8台")
    elif d_tri == 2 and sum(1 for d in ['中','發','白'] if n_counts[d] == 2) == 1: tai += 4; reason.append("小三元 4台")
    
    zi_c = sum(1 for n in hand_names if n in ['東','南','西','北','中','發','白'])
    wan_c, tong_c, tiao_c = sum(1 for n in hand_names if '萬' in n), sum(1 for n in hand_names if '筒' in n), sum(1 for n in hand_names if '條' in n)
    if zi_c == len(hand_names): tai += 16; reason.append("字一色 16台")
    elif any(c == len(hand_names) for c in [wan_c, tong_c, tiao_c]): tai += 8; reason.append("清一色 8台")
    elif any((c+zi_c == len(hand_names) and c>0 and zi_c>0) for c in [wan_c, tong_c, tiao_c]): tai += 4; reason.append("湊(混)一色 4台")
    
    return tai, reason

# --- 4. 辨識與結果呈現 ---
def run_detection(img):
    results = model(img)
    ai_raw = []
    for r in results:
        st.image(r.plot(), caption="AI 辨識畫面", use_container_width=True)
        source = r.obb if hasattr(r, 'obb') and r.obb is not None else r.boxes
        if source:
            for c in source.cls: ai_raw.append(model.names[int(c)])

    st.write("#### 玩家牌面")
    counter = collections.Counter(ai_raw)
    default_list = []
    for code, count in counter.items():
        limit = 1 if TILE_INFO[code]['type'] == 'h' else 4
        for i in range(1, min(count, limit) + 1): default_list.append(f"{TILE_INFO[code]['name']} ({code}) #{i}")

    all_opts = []
    for k, v in TILE_INFO.items():
        limit = 1 if v['type'] == 'h' else 4
        for i in range(1, limit + 1): all_opts.append(f"{v['name']} ({k}) #{i}")

    manual_tiles = st.multiselect("修改區：", options=all_opts, default=default_list)
    final_codes = [s.split('(')[1].split(')')[0] for s in manual_tiles]

    if final_codes:
        sorted_codes = sorted(final_codes, key=lambda x: TILE_INFO[x]['w'])
        hand_c = [c for c in sorted_codes if TILE_INFO[c]['type'] != 'h']
        num_h, num_all = len(hand_c), len(sorted_codes)

        # 📊 統計看板
        st.info(f"📊 **統計：共偵測到 {num_all} 張牌** (手牌: {num_h}, 花牌: {num_all - num_h})")
        h_counts = collections.Counter(hand_c)
        over = [TILE_INFO[k]['name'] for k, v in h_counts.items() if v > 4]
        if over: st.error(f"🚨 手牌超過上限：{', '.join(over)}")

        cols = st.columns(9)
        for i, c in enumerate(sorted_codes):
            info = TILE_INFO.get(c, {'name':'未知','icon':'❓'})
            with cols[i%9]: st.markdown(f"<div style='text-align:center;'><span style='font-size:30px;'>{info['icon']}</span><br><small>{info['name']}</small></div>", unsafe_allow_html=True)
        
        st.divider()

        c_left, c_right = st.columns(2)
        with c_left:
            st.write("**🏠 莊家設定**")
            is_z = st.checkbox("我是莊家", value=False)
            s_c = st.number_input("連莊 (N)", min_value=0, step=1, value=0) if is_z else 0
        with c_right:
            st.write("**🃏 可能台數(自選)**")
            m_list = st.multiselect("勾選額外台數：", ["自摸", "中洞", "邊張", "單調", "門清", "門清一摸三", "搶槓", "槓上開花+自摸", "海底撈月+自摸", "河底撈魚", "全求人(含單調)", "平胡", "三暗刻", "四暗刻", "五暗刻", "咪幾", "哩咕"])

        # --- 🛡️ 總台數看板邏輯 ---
        is_hu, msg = can_hu(sorted_codes) if not over else (False, "防呆未通過")
        
        if  is_hu:
            # 傳入側邊欄底台參數
            t_tai, details = calculate_tai_system(sorted_codes, is_z, s_c, m_list, base_tai_input) if 17 <= num_h <= 22 else (0, [f"未胡牌原因：{msg}"])
            
            # 🌟 若台數僅含底台，顯示無特殊牌型
            if is_hu and t_tai == base_tai_input:
                details.append("無特殊牌型")
                
            bg_card, txt_card = "#d4edda", "#155724" # 綠色
        else:
            t_tai, details = 0, ["目前張數不符胡牌規則(相公)"]
            bg_card, txt_card = "#f8d7da", "#721c24" # 紅色

        st.markdown(f"""<div style="background-color:{bg_card}; color:{txt_card}; padding:20px; border-radius:10px; text-align:center; border:1px solid {txt_card}; margin:20px 0;">
            <p style="margin:0; font-size:18px; font-weight:bold;">🏆️預估總台數</p>
            <h1 style="margin:0; font-size:48px;">{t_tai} <span style="font-size:24px;">台</span></h1></div>""", unsafe_allow_html=True)
        
        st.write("**台數明細：**")
        for d in details: st.write(f"📌 {d}")
        st.info("💡 提示：本系統以您的修正與手動更改為最終判定標準。")
    else: st.warning("⚠️ 請上傳照片或修正牌面。")

# --- UI 分頁 ---
tab1, tab2 = st.tabs(["📷︎ 即時拍照", "📁 上傳照片"])
with tab1:
    cam = st.camera_input("拍照算台")
    if cam: run_detection(Image.open(cam))
with tab2:
    up = st.file_uploader("選擇照片", type=['png', 'jpg', 'jpeg'])
    if up: run_detection(Image.open(up))

