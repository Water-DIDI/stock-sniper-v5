import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- 0. 全局配置 ---
st.set_page_config(page_title="V6 美股全域戰情室", layout="wide")

# --- 1. 數據定義 ---
SECTOR_CONFIG = {
    "半導體 (SMH)": {"benchmark": "SMH", "components": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU"]},
    "科技巨頭 (XLK)": {"benchmark": "XLK", "components": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "ADBE", "CSCO"]},
    "軟體雲端 (IGV)": {"benchmark": "IGV", "components": ["MSFT", "CRM", "ADBE", "ORCL", "PANW", "SNOW", "PLTR", "CRWD", "DDOG"]},
    "通訊服務 (XLC)": {"benchmark": "XLC", "components": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "TMUS", "VZ"]},
    "金融銀行 (XLF)": {"benchmark": "XLF", "components": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "V", "MA"]},
    "生技醫療 (XBI)": {"benchmark": "XBI", "components": ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BNTX", "ILMN"]},
    "能源油氣 (XLE)": {"benchmark": "XLE", "components": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO"]},
    "貴金屬原物料 (XLB)": {"benchmark": "XLB", "components": ["GLD", "SLV", "GDX", "NEM", "FCX", "SCCO", "AA"]},
    "加密貨幣概念 (IBIT)": {"benchmark": "IBIT", "components": ["IBIT", "COIN", "MSTR", "MARA", "CLSK", "RIOT"]},
    "工業製造 (XLI)": {"benchmark": "XLI", "components": ["GE", "CAT", "DE", "HON", "UNP", "UPS", "LMT", "RTX"]}
}

# --- 2. 核心運算函數 ---

@st.cache_data(ttl=1800)
def fetch_data(tickers, period):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, threads=True)
        return data
    except Exception:
        return pd.DataFrame()

def calculate_trend_history(df_close, ma_window=20, lookback_days=3):
    """
    計算過去 N 天的趨勢狀態
    回傳: 燈號字串 (例如: 🟢🟢🟢)
    """
    if len(df_close) < ma_window + lookback_days:
        return "⚪⚪⚪" # 數據不足

    ma_series = df_close.rolling(ma_window).mean()
    
    # 取得最後 N 天的數據 (倒序: T-2, T-1, Today)
    status_icons = []
    
    # 我們要檢查 Today, Yesterday, Day before yesterday
    # Python index: -1 (今天), -2 (昨天), -3 (前天)
    for i in range(lookback_days, 0, -1): 
        idx = -1 * i # -3, -2, -1
        price = df_close.iloc[idx]
        ma = ma_series.iloc[idx]
        
        if price > ma:
            status_icons.append("🟢")
        else:
            status_icons.append("🔴")
            
    return "".join(status_icons)

def calculate_metrics(df_close, df_vol, spy_close):
    # 1. RS 動能
    idx = df_close.index.intersection(spy_close.index)
    if len(idx) < 30: return 0, 0, 0
    
    aligned_close = df_close.loc[idx]
    aligned_spy = spy_close.loc[idx]
    
    rs_ratio = aligned_close / aligned_spy
    rs_mom = (rs_ratio.iloc[-1] / rs_ratio.iloc[-21] - 1) * 100
    
    # 2. RVOL
    vol_avg = df_vol.rolling(20).mean()
    curr_vol = df_vol.iloc[-1]
    avg_val = vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
    rvol = curr_vol / avg_val
    
    # 3. 漲幅
    chg = (aligned_close.iloc[-1] / aligned_close.iloc[-6] - 1) * 100
    
    return round(rs_mom, 2), round(rvol, 2), round(chg, 2)

# --- 3. 主程式 ---
def main():
    st.title("🚀 V6 美股全域戰情室 (3-Day Confirmation)")
    st.markdown("---")
    
    # 下載全市場數據
    benchmarks = {k: v["benchmark"] for k, v in SECTOR_CONFIG.items()}
    all_tickers = list(benchmarks.values()) + ["SPY"]
    
    with st.spinner("正在進行時光回測 (Backtesting)..."):
        market_data = fetch_data(all_tickers, "6mo")
        
    if market_data.empty:
        st.error("數據下載失敗")
        return

    try:
        spy_close = market_data["SPY"]["Close"]
    except:
        spy_close = market_data["Close"] # 單一股票兼容

    # --- Step 1: 全板塊趨勢掃描 ---
    st.header("1️⃣ 全板塊趨勢確認 (Sector Trend Logic)")
    st.info("💡 **3日法則說明**：\n- 🟢🟢🟢 (全綠)：趨勢確認，資金穩定流入 -> **可積極操作**\n- 🔴🔴🟢 (紅紅綠)：首日轉強，可能是假突破 -> **建議觀察，不要重倉**\n- 🟢🟢🔴 (綠綠紅)：漲多拉回或轉弱 -> **暫停買入**")
    
    sector_list = []
    
    for name, ticker in benchmarks.items():
        if ticker not in market_data.columns.levels[0]: continue
        
        df = market_data[ticker]
        close = df["Close"]
        vol = df["Volume"]
        
        rs, rvol, chg = calculate_metrics(close, vol, spy_close)
        
        # [關鍵功能] 計算過去3天歷史
        trend_history = calculate_trend_history(close, 20, 3)
        
        # 判定是否為 "確認趨勢"
        is_confirmed = (trend_history == "🟢🟢🟢")
        
        sector_list.append({
            "板塊": name.split(" ")[0],
            "代號": ticker,
            "RS動能": rs,
            "RVOL": rvol,
            "週漲幅%": chg,
            "3日趨勢 (前天➜今天)": trend_history,
            "確認訊號": "✅ YES" if is_confirmed else "⚠️ Wait"
        })
        
    df_sec = pd.DataFrame(sector_list)
    
    # 顯示排行榜
    st.dataframe(
        df_sec.sort_values("RS動能", ascending=False),
        column_order=["板塊", "3日趨勢 (前天➜今天)", "確認訊號", "RS動能", "RVOL", "週漲幅%"],
        column_config={
            "RS動能": st.column_config.NumberColumn(format="%.2f"),
            "RVOL": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=3),
            "確認訊號": st.column_config.TextColumn(help="只有連續3天站上月線，才視為真趨勢")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 熱力圖
    st.subheader("🗺️ 資金流向分佈")
    fig = px.treemap(
        df_sec, path=['板塊'], values='RVOL', color='RS動能',
        color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        hover_data=['3日趨勢 (前天➜今天)'],
        title="面積=RVOL資金量 | 顏色=RS強度"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # --- Step 2: 個股鑽取 ---
    st.header("2️⃣ 強勢股篩選 (Drill Down)")
    
    # 自動選出最強且確認的板塊
    confirmed_sectors = df_sec[df_sec["確認訊號"] == "✅ YES"].sort_values("RS動能", ascending=False)
    
    if not confirmed_sectors.empty:
        top_name = confirmed_sectors.iloc[0]["板塊"]
        # 找回原始 key
        default_idx = 0
        keys = list(SECTOR_CONFIG.keys())
        for i, k in enumerate(keys):
            if top_name in k:
                default_idx = i
                break
    else:
        default_idx = 0 # 如果沒有確認的，就選第一個

    c1, c2 = st.columns([3, 7])
    with c1:
        target = st.selectbox("選擇板塊", list(SECTOR_CONFIG.keys()), index=default_idx)
    with c2:
        rs_th = st.slider("RS 門檻", -5.0, 5.0, 0.0, step=0.5)
        rvol_th = st.slider("RVOL 門檻", 0.5, 3.0, 1.2, step=0.1)

    # 下載個股
    comps = SECTOR_CONFIG[target]["components"]
    with st.spinner(f"掃描 {target} 成分股..."):
        comp_data = fetch_data(comps, "6mo")
        
    if not comp_data.empty:
        stock_list = []
        for t in comps:
            try:
                if len(comps)>1:
                    if t not in comp_data.columns.levels[0]: continue
                    df = comp_data[t]
                else:
                    df = comp_data
                
                close = df["Close"]
                vol = df["Volume"]
                
                rs, rvol, chg = calculate_metrics(close, vol, spy_close)
                
                # 個股也要看3日趨勢
                history = calculate_trend_history(close, 20, 3)
                
                # 嚴格篩選: 必須趨勢確認 + RS強 + 爆量
                if rs > rs_th and rvol > rvol_th and history == "🟢🟢🟢":
                    stock_list.append({
                        "代號": t,
                        "3日趨勢": history,
                        "RS值": rs,
                        "RVOL": rvol,
                        "TV連結": f"https://www.tradingview.com/chart/?symbol={t}"
                    })
            except:
                continue
        
        df_st = pd.DataFrame(stock_list)
        
        c3, c4 = st.columns([5, 5])
        with c3:
            st.subheader(f"🚀 {target} 火箭清單 (僅列出 3日強勢股)")
            if not df_st.empty:
                st.data_editor(
                    df_st.sort_values("RVOL", ascending=False),
                    column_config={
                        "TV連結": st.column_config.LinkColumn("圖表", display_text="Open TV"),
                        "RVOL": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=5)
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("⚠️ 該板塊雖強，但無個股同時滿足「3日全紅」且「爆量」條件。")
                
        with c4:
            st.subheader("動能分佈")
            if not df_st.empty:
                fig_s = px.scatter(df_st, x="RS值", y="RVOL", size="RVOL", text="代號", title="尋找右上角領頭羊", template="plotly_dark")
                st.plotly_chart(fig_s, use_container_width=True)

if __name__ == "__main__":
    main()
