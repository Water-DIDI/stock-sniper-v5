import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- 1. 頁面設定 ---
st.set_page_config(page_title="美股 V7 戰情室", layout="wide", page_icon="🚀")
st.title("🚀 美股 V7 狙擊手戰情室")

# --- 2. 定義資料結構 ---
SECTOR_MAP = {
    'XLK': '科技 (Tech)', 'SMH': '半導體 (Chip)', 'XLE': '能源 (Energy)',
    'XLV': '醫療 (Health)', 'XLF': '金融 (Finance)', 'XLI': '工業 (Industry)',
    'XLP': '必需品 (Staples)', 'XLU': '公用事業 (Util)', 'XLY': '非必需 (Discret)',
    'XLB': '原物料 (Material)', 'XLC': '通訊 (Comm)', 'IYR': '房地產 (Real Est)',
    'QQQ': '那斯達克100', 'SPY': '標普500'
}

SECTOR_STOCKS = {
    "半導體": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU"],
    "科技": ["AAPL", "MSFT", "ORCL", "CRM", "ADBE", "CSCO", "IBM", "META", "GOOGL"],
    "軟體": ["PANW", "SNOW", "PLTR", "CRWD", "DDOG", "ZS", "NET"],
    "能源": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO"],
    "原物料": ["GLD", "SLV", "FCX", "SCCO", "AA", "NEM"],
    "工業": ["GE", "CAT", "DE", "HON", "LMT", "RTX"],
    "加密": ["IBIT", "COIN", "MSTR", "MARA", "CLSK"]
}

MEGA_CAPS = ["TSM", "NVDA", "AAPL", "MSFT", "GOOGL", "META", "XOM", "CVX", "JPM", "GLD"]
HIGH_BETA = ["MSTR", "COIN", "MARA", "CLSK", "PLTR", "SOFI", "AI"]

# --- 3. 核心函式庫 ---

def get_strategy_params(ticker):
    if ticker in MEGA_CAPS: return 1.1, 0.0, 20, "🐢權值穩健"
    elif ticker in HIGH_BETA: return 2.0, 2.0, 10, "🐇投機飆股"
    else: return 1.5, 1.0, 14, "🐆循環動能"

@st.cache_data(ttl=3600)
def get_sector_data():
    tickers = list(SECTOR_MAP.keys())
    data = yf.download(tickers, period="400d", auto_adjust=True)['Close']
    return data

def get_trend_emoji(price, ma20):
    """判斷紅綠燈: 價格 > 月線 = 🟢"""
    return "🟢" if price > ma20 else "🔴"

def check_stock(ticker, df, spy_close):
    if len(df) < 50: return None
    close, high, vol = df["Close"], df["High"], df["Volume"]
    
    rvol_th, rs_th, lookback, mode_name = get_strategy_params(ticker)

    # 1. 趨勢 & 2. 突破
    ma20 = close.rolling(20).mean()
    if not ((close.iloc[-1] > ma20.iloc[-1]) and (close.iloc[-2] > ma20.iloc[-2])): return None
    
    highest_high = high.shift(1).rolling(window=lookback).max()
    if not (close.iloc[-1] > highest_high.iloc[-1]): return None

    # 3. RS & 4. RVOL & 5. 紅K
    idx = close.index.intersection(spy_close.index)
    if len(idx) < 30: return None
    rs_ratio = close.loc[idx] / spy_close.loc[idx]
    rs_val = (rs_ratio.iloc[-1] / rs_ratio.iloc[-21] - 1) * 100
    
    vol_avg = vol.rolling(20).mean()
    avg_vol = vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
    rvol_val = vol.iloc[-1] / avg_vol
    
    is_red = close.iloc[-1] > df["Open"].iloc[-1]

    if rs_val > rs_th and rvol_val > rvol_th and is_red:
        return {
            "Mode": mode_name, "RS": rs_val, "RVOL": rvol_val, 
            "Breakout": lookback, "Price": close.iloc[-1],
            "Chg": (close.iloc[-1]/close.iloc[-2]-1)*100
        }
    return None

# --- 4. 介面佈局 ---
tab1, tab2 = st.tabs(["📊 板塊戰情室 (Macro)", "🚀 個股狙擊手 (Scanner)"])

# ==========================================
# Tab 1: 板塊戰情室
# ==========================================
with tab1:
    st.markdown("### 資金流向熱力圖 (Heatmap)")
    try:
        with st.spinner('載入板塊數據...'):
            df_close = get_sector_data()

        periods = {'1M (近1月)': 21, '3M (近1季)': 63}
        res_data = {}
        curr = df_close.iloc[-1]
        
        for t in SECTOR_MAP.keys():
            row = {}
            for p_name, p_days in periods.items():
                if len(df_close) > p_days:
                    prev = df_close[t].iloc[-p_days]
                    row[p_name] = (curr[t] - prev) / prev
                else: row[p_name] = 0.0
            res_data[f"{t} {SECTOR_MAP[t]}"] = row

        df_ret = pd.DataFrame.from_dict(res_data, orient='index').sort_values(by='1M (近1月)', ascending=False)
        st.dataframe(df_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1), use_container_width=True, height=500)

        # 趨勢圖
        st.markdown("---")
        default_sectors = ['XLE', 'XLK', 'SMH', 'XLU']
        selected = st.multiselect("對比板塊:", list(SECTOR_MAP.keys()), default=[k for k in default_sectors if k in SECTOR_MAP])
        if selected:
            lookback = st.slider("回測天數", 30, 365, 120)
            chart_data = df_close[selected].iloc[-lookback:].copy()
            chart_data = (chart_data / chart_data.iloc[0] - 1) * 100
            st.plotly_chart(px.line(chart_data, title=f"近 {lookback} 天趨勢 (%)"), use_container_width=True)

    except Exception as e: st.error(f"數據錯誤: {e}")

# ==========================================
# Tab 2: 個股狙擊手 (加回紅綠燈!)
# ==========================================
with tab2:
    st.markdown("### V7.6 智能個股掃描")
    
    # --- [新增] 板塊紅綠燈區塊 ---
    st.info("🚦 **掃描前確認：板塊紅綠燈 (月線趨勢)**")
    
    # 這裡我們快速計算幾個關鍵板塊的燈號
    if 'df_close' in locals() and not df_close.empty:
        cols = st.columns(4)
        
        # 定義要監控的關鍵板塊
        key_sectors = ['SMH', 'XLK', 'XLE', 'XLU']
        
        for i, ticker in enumerate(key_sectors):
            series = df_close[ticker]
            ma20 = series.rolling(20).mean().iloc[-1]
            price = series.iloc[-1]
            emoji = get_trend_emoji(price, ma20)
            name = SECTOR_MAP[ticker].split(' ')[0] # 只取中文名
            
            with cols[i]:
                st.metric(label=f"{name} ({ticker})", value=f"{price:.2f}", delta=emoji)
        
        st.markdown("---")
    # ----------------------------------

    if st.button("🚀 開始掃描火箭 (Start Scan)"):
        status_text = st.empty()
        status_text.text("⏳ 掃描中...")
        
        all_tickers = []
        for s in SECTOR_STOCKS.values(): all_tickers.extend(s)
        all_tickers.append("SPY")
        all_tickers = list(set(all_tickers))
        
        try:
            data = yf.download(all_tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
            if "SPY" in data:
                spy_close = data["SPY"]["Close"]
                results = []
                for sector, tickers in SECTOR_STOCKS.items():
                    for t in tickers:
                        try:
                            if t not in data.columns.levels[0]: continue
                            res = check_stock(t, data[t], spy_close)
                            if res:
                                results.append({
                                    "板塊": sector, "代號": t, "模式": res['Mode'],
                                    "價格": f"{res['Price']:.2f}", "漲幅": f"{res['Chg']:.2f}%",
                                    "RVOL": f"{res['RVOL']}x", "突破": f"{res['Breakout']}日", "RS": f"{res['RS']:.2f}"
                                })
                        except: continue
                
                status_text.empty()
                if results:
                    st.success(f"🎉 發現 {len(results)} 檔火箭")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else: st.warning("💤 今日無火箭")
        except Exception as e: st.error(f"錯誤: {e}")
