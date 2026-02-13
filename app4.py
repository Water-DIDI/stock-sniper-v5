import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- 1. 頁面設定 ---
st.set_page_config(page_title="美股 V8 全方位戰情室", layout="wide", page_icon="🛡️")
st.title("🛡️ 美股 V8 全方位戰情室")

# --- 2. 資料定義 ---
SECTOR_MAP = {
    'XLK': '科技', 'SMH': '半導體', 'XLE': '能源', 'XLV': '醫療',
    'XLF': '金融', 'XLI': '工業', 'XLP': '必需品', 'XLU': '公用',
    'XLY': '非必需', 'XLB': '原物料', 'XLC': '通訊', 'IYR': '房地產',
    'QQQ': '那斯達克', 'SPY': '標普'
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

# --- 3. 核心邏輯 ---

def get_strategy_params(ticker):
    if ticker in MEGA_CAPS: return 1.1, 0.0, 20, "🐢權值穩健"
    elif ticker in HIGH_BETA: return 2.0, 2.0, 10, "🐇投機飆股"
    else: return 1.5, 1.0, 14, "🐆循環動能"

@st.cache_data(ttl=1800) # 快取 30 分鐘
def get_market_data():
    """下載總經與板塊數據"""
    # 增加 ^VIX, ^TNX (10年債), DX-Y.NYB (美元)
    tickers = list(SECTOR_MAP.keys()) + ['^VIX', '^TNX', 'DX-Y.NYB']
    data = yf.download(tickers, period="400d", auto_adjust=True)['Close']
    return data

def get_trend_emoji(price, ma20):
    return "🟢" if price > ma20 else "🔴"

def check_stock(ticker, df, spy_close):
    if len(df) < 50: return None
    close, high, vol = df["Close"], df["High"], df["Volume"]
    rvol_th, rs_th, lookback, mode_name = get_strategy_params(ticker)

    # 1. 趨勢
    ma20 = close.rolling(20).mean()
    if not (close.iloc[-1] > ma20.iloc[-1]): return None
    
    # 2. 突破
    highest_high = high.shift(1).rolling(window=lookback).max()
    if not (close.iloc[-1] > highest_high.iloc[-1]): return None

    # 3. RS 強度
    idx = close.index.intersection(spy_close.index)
    if len(idx) < 30: return None
    rs_ratio = close.loc[idx] / spy_close.loc[idx]
    rs_val = (rs_ratio.iloc[-1] / rs_ratio.iloc[-21] - 1) * 100
    
    # 4. 乖離率 (Bias) - 新增濾網
    bias = (close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100
    
    # 5. 量能 & 紅K
    vol_avg = vol.rolling(20).mean()
    avg_vol = vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
    rvol_val = vol.iloc[-1] / avg_vol
    is_red = close.iloc[-1] > df["Open"].iloc[-1]

    if rs_val > rs_th and rvol_val > rvol_th and is_red:
        return {
            "Mode": mode_name, "RS": rs_val, "RVOL": rvol_val, 
            "Breakout": lookback, "Price": close.iloc[-1],
            "Bias": bias, # 回傳乖離率
            "Chg": (close.iloc[-1]/close.iloc[-2]-1)*100
        }
    return None

# --- 4. 介面開始 ---

# [優化 1] 總經儀表板 (Macro Dashboard)
with st.container():
    st.markdown("### 🌍 市場天候監測 (Market Regime)")
    try:
        with st.spinner('連線華爾街數據庫...'):
            df_all = get_market_data()
        
        m_cols = st.columns(4)
        
        # VIX
        if '^VIX' in df_all:
            vix = df_all['^VIX'].iloc[-1]
            vix_chg = df_all['^VIX'].iloc[-1] - df_all['^VIX'].iloc[-2]
            vix_color = "inverse" if vix > 20 else "normal" # VIX 高代表危險
            m_cols[0].metric("恐慌指數 (VIX)", f"{vix:.2f}", f"{vix_chg:.2f}", delta_color=vix_color)
            
        # 10年債 (TNX)
        if '^TNX' in df_all:
            tnx = df_all['^TNX'].iloc[-1]
            tnx_chg = df_all['^TNX'].iloc[-1] - df_all['^TNX'].iloc[-2]
            m_cols[1].metric("10年美債殖利率", f"{tnx:.2f}%", f"{tnx_chg:.2f}", delta_color="inverse")

        # 美元 (DXY)
        if 'DX-Y.NYB' in df_all:
            dxy = df_all['DX-Y.NYB'].iloc[-1]
            dxy_chg = df_all['DX-Y.NYB'].iloc[-1] - df_all['DX-Y.NYB'].iloc[-2]
            m_cols[2].metric("美元指數 (DXY)", f"{dxy:.2f}", f"{dxy_chg:.2f}")

        # SPY 狀態
        if 'SPY' in df_all:
            spy_p = df_all['SPY'].iloc[-1]
            spy_ma20 = df_all['SPY'].rolling(20).mean().iloc[-1]
            trend = "🐂 多頭" if spy_p > spy_ma20 else "🐻 空頭"
            m_cols[3].metric("大盤趨勢 (SPY)", trend, f"{(spy_p/df_all['SPY'].iloc[-2]-1)*100:.2f}%")
            
        st.markdown("---")
        
    except Exception as e:
        st.error(f"總經數據載入失敗: {e}")

# 分頁區
tab1, tab2 = st.tabs(["📊 板塊戰情室 (Sector)", "🚀 個股狙擊手 (Scanner)"])

# ==========================================
# Tab 1: 板塊 (優化：相對強度圖)
# ==========================================
with tab1:
    st.markdown("### 資金流向熱力榜")
    periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '12M': 252}
    res_data = {}
    curr = df_all.iloc[-1]
    
    for t in SECTOR_MAP.keys():
        if t not in df_all: continue
        row = {}
        for p_name, p_days in periods.items():
            if len(df_all) > p_days:
                prev = df_all[t].iloc[-p_days]
                row[p_name] = (curr[t] - prev) / prev
            else: row[p_name] = 0.0
        res_data[f"{t} {SECTOR_MAP[t]}"] = row

    df_ret = pd.DataFrame.from_dict(res_data, orient='index').sort_values(by='1M', ascending=False)
    st.dataframe(df_ret.style.format("{:.2%}").background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1), use_container_width=True)

    # [優化 2] 相對強度趨勢圖 (Relative Strength)
    st.markdown("---")
    st.subheader("📈 真實強度分析 (Relative Strength vs SPY)")
    st.caption("💡 曲線向上代表「跑贏大盤」，向下代表「跑輸大盤」。請尋找往右上角噴出的板塊。")
    
    default_sectors = ['XLE', 'XLK', 'SMH', 'XLU']
    selected = st.multiselect("選擇板塊:", list(SECTOR_MAP.keys()), default=[k for k in default_sectors if k in df_all])
    
    if selected and 'SPY' in df_all:
        lookback = st.slider("回測天數", 30, 365, 120)
        # 計算 RS Ratio: 板塊股價 / SPY股價
        rs_data = pd.DataFrame()
        for s in selected:
            rs_data[s] = df_all[s] / df_all['SPY']
        
        chart_data = rs_data.iloc[-lookback:].copy()
        # 歸一化
        chart_data = (chart_data / chart_data.iloc[0] - 1) * 100
        
        fig = px.line(chart_data, title=f"相對 SPY 強度表現 (%)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tab 2: 個股 (優化：乖離率 Bias)
# ==========================================
with tab2:
    st.markdown("### V8.0 智能個股掃描")
    
    # 紅綠燈
    st.info("🚦 **板塊紅綠燈**")
    cols = st.columns(4)
    key_sectors = ['SMH', 'XLK', 'XLE', 'XLU']
    for i, ticker in enumerate(key_sectors):
        if ticker in df_all:
            p = df_all[ticker].iloc[-1]
            ma20 = df_all[ticker].rolling(20).mean().iloc[-1]
            cols[i].metric(f"{ticker}", f"{p:.2f}", get_trend_emoji(p, ma20))
    st.markdown("---")

    if st.button("🚀 開始掃描 (Start Scan)"):
        st.text("⏳ 掃描數據中...")
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
                                # [優化 3] 乖離率過大警示
                                bias_str = f"{res['Bias']:.1f}%"
                                if res['Bias'] > 10: bias_str += " ⚠️過熱"
                                
                                results.append({
                                    "板塊": sector, "代號": t, 
                                    "價格": f"{res['Price']:.2f}", 
                                    "乖離率(Bias)": bias_str, # 新欄位
                                    "RS值": f"{res['RS']:.2f}",
                                    "模式": res['Mode']
                                })
                        except: continue
                
                if results:
                    st.success(f"🎉 發現 {len(results)} 檔火箭")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                    st.caption("⚠️ **乖離率 > 10%** 代表短線過熱，建議等待回測 MA5/MA10 再佈局，不要追高。")
                else: st.warning("💤 無訊號")
        except Exception as e: st.error(f"錯誤: {e}")
