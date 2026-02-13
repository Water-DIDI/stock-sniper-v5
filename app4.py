import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="美股 V9 智能輪動模型", layout="wide", page_icon="🧠")
st.title("🧠 美股 V9 智能輪動模型 (Regime + Top3 Sector)")

# --- 2. 策略參數定義 (源自文件) ---
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊' # IYR取代XLRE以獲取更長歷史
}

# --- 3. 核心數據處理函式 (修復 None 問題) ---
@st.cache_data(ttl=3600)
def get_data_v9():
    """
    抓取清洗後的數據，避免 MultiIndex 造成 None
    """
    tickers = list(SECTOR_MAP.keys()) + ['SPY', '^VIX']
    
    # 下載數據
    data = yf.download(tickers, period="2y", auto_adjust=True)
    
    # 處理 yfinance 新版 MultiIndex 問題
    if isinstance(data.columns, pd.MultiIndex):
        # 嘗試只取 'Close'，如果失敗則直接用 data
        try:
            df = data['Close'].copy()
        except KeyError:
            df = data.copy()
    else:
        df = data['Close'].copy()

    # 強制填補空值 (Forward Fill)，解決假日/數據延遲導致的 NaN
    df = df.ffill()
    return df

# --- 4. 策略邏輯實作 ---
def calculate_momentum_score(df):
    """
    依據文件公式計算動能分數:
    Score = 0.5*3M + 0.3*6M + 0.2*1M
    """
    # 計算各週期報酬率 (21, 63, 126 天)
    ret_1m = df.pct_change(21).iloc[-1]
    ret_3m = df.pct_change(63).iloc[-1]
    ret_6m = df.pct_change(126).iloc[-1]
    
    # 計算分數
    score = (0.5 * ret_3m) + (0.3 * ret_6m) + (0.2 * ret_1m)
    
    # 50MA 濾網判斷
    ma50 = df.rolling(50).mean().iloc[-1]
    price = df.iloc[-1]
    above_ma50 = price > ma50
    
    return score, ret_1m, ret_3m, ret_6m, above_ma50

def check_market_regime(df):
    """
    市場風控濾網:
    1. SPY > 200MA
    2. VIX 5MA < VIX 20MA
    兩者皆 True 才為 Risk ON
    """
    if 'SPY' not in df or '^VIX' not in df:
        return False, "數據不足"
        
    spy = df['SPY']
    vix = df['^VIX']
    
    # SPY 條件
    spy_price = spy.iloc[-1]
    spy_ma200 = spy.rolling(200).mean().iloc[-1]
    cond_spy = spy_price > spy_ma200
    
    # VIX 條件
    vix_ma5 = vix.rolling(5).mean().iloc[-1]
    vix_ma20 = vix.rolling(20).mean().iloc[-1]
    cond_vix = vix_ma5 < vix_ma20
    
    is_bull = cond_spy and cond_vix
    
    detail = f"""
    - SPY vs 200MA: {'✅ 多頭' if cond_spy else '❌ 空頭'} ({spy_price:.2f} / {spy_ma200:.2f})
    - VIX 結構: {'✅ 穩定' if cond_vix else '❌ 恐慌'} (5MA:{vix_ma5:.2f} / 20MA:{vix_ma20:.2f})
    """
    return is_bull, detail

# --- 5. 介面呈現 ---

try:
    with st.spinner('正在執行 V9 演算法運算...'):
        df = get_data_v9()

    # A. 市場風控儀表板 (Regime Filter)
    st.header("1️⃣ 市場風控濾網 (Market Regime)")
    is_risk_on, regime_detail = check_market_regime(df)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if is_risk_on:
            st.success("🟢 **RISK ON (積極進攻)**\n\n建議：買入 Top 3 板塊")
        else:
            st.error("🔴 **RISK OFF (防禦/現金)**\n\n建議：持有現金或美債，暫停輪動")
    with col2:
        with st.expander("查看風控細節"):
            st.text(regime_detail)

    # B. Top 3 推薦模型 (Ranking Model)
    st.header("2️⃣ 本月輪動冠軍 (Top 3 Sectors)")
    
    if is_risk_on:
        st.caption("根據模型：Score = 0.5*3M + 0.3*6M + 0.2*1M，且股價 > 50MA")
        
        scores = []
        for ticker in SECTOR_MAP.keys():
            if ticker in df:
                s, r1, r3, r6, flt = calculate_momentum_score(df[ticker])
                scores.append({
                    "代號": ticker,
                    "板塊": SECTOR_MAP[ticker],
                    "綜合評分": s * 100, # 轉百分比顯示
                    "1M": r1, "3M": r3, "6M": r6,
                    ">50MA": "✅" if flt else "❌ (剔除)"
                })
        
        df_score = pd.DataFrame(scores)
        
        # 1. 先篩選掉跌破 50MA 的
        df_valid = df_score[df_score[">50MA"] == "✅"].copy()
        
        # 2. 排序取前三
        df_valid = df_valid.sort_values(by="綜合評分", ascending=False)
        top3 = df_valid.head(3)
        
        # 顯示 Top 3 卡片
        c1, c2, c3 = st.columns(3)
        if len(top3) >= 1:
            row = top3.iloc[0]
            c1.metric(label=f"🥇 冠軍: {row['板塊']} ({row['代號']})", value=f"{row['綜合評分']:.1f}分", delta=f"1M: {row['1M']:.1%}")
        if len(top3) >= 2:
            row = top3.iloc[1]
            c2.metric(label=f"🥈 亞軍: {row['板塊']} ({row['代號']})", value=f"{row['綜合評分']:.1f}分", delta=f"1M: {row['1M']:.1%}")
        if len(top3) >= 3:
            row = top3.iloc[2]
            c3.metric(label=f"🥉 季軍: {row['板塊']} ({row['代號']})", value=f"{row['綜合評分']:.1f}分", delta=f"1M: {row['1M']:.1%}")
            
        st.markdown("---")
        st.subheader("📊 完整評分排行榜")
        # 格式化顯示
        st.dataframe(
            df_score.sort_values(by="綜合評分", ascending=False).style.format({
                "綜合評分": "{:.2f}", "1M": "{:.2%}", "3M": "{:.2%}", "6M": "{:.2%}"
            }).background_gradient(subset=["綜合評分"], cmap="Greens"),
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ 目前市場處於 **Risk Off** 狀態，模型建議 **不持有任何股票板塊**，請轉往現金 (USD) 或 短債 (SGOV/SHV)。")

    # C. 熱力圖與趨勢 (保留原功能作為輔助)
    st.markdown("---")
    with st.expander("查看原始數據圖表 (Heatmap & Charts)"):
        # 熱力圖
        periods = {'1M': 21, '3M': 63, '6M': 126}
        res_data = {}
        curr = df.iloc[-1]
        for t in SECTOR_MAP.keys():
            if t in df:
                row = {}
                for p_name, p_days in periods.items():
                    prev = df[t].iloc[-p_days]
                    row[p_name] = (curr[t] - prev) / prev
                res_data[t] = row
        st.dataframe(pd.DataFrame.from_dict(res_data, orient='index').style.format("{:.2%}"), use_container_width=True)

except Exception as e:
    st.error(f"系統錯誤: {e}")
    st.write("Debug info:", e)
