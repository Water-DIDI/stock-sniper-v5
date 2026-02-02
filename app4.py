import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- 0. 全局配置 ---
st.set_page_config(page_title="V5 美股全域戰情室", layout="wide")

# --- 1. 數據定義 (板塊與成分股) ---
SECTOR_CONFIG = {
    "半導體 (SMH)": {
        "benchmark": "SMH",
        "components": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU", "ADI", "KLAC", "MRVL", "ARM"]
    },
    "科技巨頭 (XLK)": {
        "benchmark": "XLK",
        "components": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "IBM", "NOW"]
    },
    "軟體雲端 (IGV)": {
        "benchmark": "IGV",
        "components": ["MSFT", "CRM", "ADBE", "ORCL", "PANW", "SNOW", "PLTR", "CRWD", "DDOG", "ZS", "NET", "MDB"]
    },
    "通訊服務 (XLC)": {
        "benchmark": "XLC",
        "components": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "WBD"]
    },
    "金融銀行 (XLF)": {
        "benchmark": "XLF",
        "components": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "V", "MA", "C", "BRK-B"]
    },
    "生技醫療 (XBI)": {
        "benchmark": "XBI",
        "components": ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BNTX", "ILMN", "ISRG", "LLY", "PFE"]
    },
    "能源油氣 (XLE)": {
        "benchmark": "XLE",
        "components": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "HAL"]
    },
    "貴金屬原物料 (XLB)": {
        "benchmark": "XLB",
        "components": ["GLD", "SLV", "GDX", "NEM", "FCX", "SCCO", "AA", "CLF", "RIO", "BHP"]
    },
    "加密貨幣概念 (IBIT)": {
        "benchmark": "IBIT",
        "components": ["IBIT", "COIN", "MSTR", "MARA", "CLSK", "RIOT", "HUT", "HOOD", "SI"]
    },
    "工業製造 (XLI)": {
        "benchmark": "XLI",
        "components": ["GE", "CAT", "DE", "HON", "UNP", "UPS", "LMT", "RTX", "BA"]
    }
}

# --- 2. 核心運算函數 ---

@st.cache_data(ttl=1800) # 30分鐘快取
def fetch_data(tickers, period):
    if not tickers: return pd.DataFrame()
    try:
        # 下載數據，強制 group_by='ticker'
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, threads=True)
        return data
    except Exception:
        return pd.DataFrame()

def calculate_rs_rvol(df_close, df_vol, spy_close, window=20):
    """通用計算函數：計算 RS動能 與 RVOL"""
    # 1. RS 動能 (相對於 SPY)
    # 確保索引對齊
    idx = df_close.index.intersection(spy_close.index)
    if len(idx) < window + 5: return 0, 0, 0 # 數據不足

    aligned_close = df_close.loc[idx]
    aligned_spy = spy_close.loc[idx]
    
    rs_ratio = aligned_close / aligned_spy
    # 動能公式: (現在RS / N天前RS - 1) * 100
    rs_mom = (rs_ratio.iloc[-1] / rs_ratio.iloc[-window] - 1) * 100
    
    # 2. RVOL
    vol_avg = df_vol.rolling(window).mean()
    curr_vol = df_vol.iloc[-1]
    # 防呆除以0
    avg_vol_val = vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
    rvol = curr_vol / avg_vol_val
    
    # 3. 漲跌幅
    chg = (aligned_close.iloc[-1] / aligned_close.iloc[-5] - 1) * 100
    
    return round(rs_mom, 2), round(rvol, 2), round(chg, 2)

def analyze_trend_light(price, ma20, ma50):
    """紅綠燈判斷"""
    if price > ma20 and ma20 > ma50:
        return "🟢 強勢多頭"
    elif price > ma50:
        return "🟡 震盪偏多"
    else:
        return "🔴 空頭修正"

# --- 3. 主程式 ---
def main():
    st.title("🚀 V5 美股全域戰情室 (Top-Down Strategy)")
    
    # --- Step 0: 準備全板塊 ETF 數據 ---
    sector_benchmarks = {k: v["benchmark"] for k, v in SECTOR_CONFIG.items()}
    all_etfs = list(sector_benchmarks.values()) + ["SPY"]
    
    with st.spinner("正在掃描全市場板塊資金流向..."):
        # 抓取 ETF 數據
        market_data = fetch_data(all_etfs, "6mo")
    
    if market_data.empty:
        st.error("無法連線至數據庫，請檢查網路。")
        return

    # 提取 SPY (修復 AttributeError 的關鍵寫法)
    try:
        spy_df = market_data["SPY"]
        spy_close = spy_df["Close"]
    except KeyError:
        # 如果只有單一股票，結構不同，但這裡我們下載了多檔，通常不會進這
        st.error("SPY 數據缺失，無法計算相對強度。")
        return

    # --- Step 1: 全板塊熱力總覽 (The General's Map) ---
    st.header("1️⃣ 全板塊氣象站 (Sector Overview)")
    
    sector_metrics = []
    
    for name, ticker in sector_benchmarks.items():
        if ticker not in market_data.columns.levels[0]: continue
        
        df = market_data[ticker]
        close = df["Close"]
        vol = df["Volume"]
        
        # 計算指標
        rs, rvol, chg = calculate_rs_rvol(close, vol, spy_close)
        
        # 趨勢紅綠燈
        curr_price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        status = analyze_trend_light(curr_price, ma20, ma50)
        
        sector_metrics.append({
            "板塊": name.split(" ")[0], # 簡化名稱
            "代號": ticker,
            "RS動能": rs,
            "RVOL": rvol,
            "週漲幅%": chg,
            "狀態": status
        })
    
    df_sectors = pd.DataFrame(sector_metrics)
    
    # 顯示全板塊指標
    col_map, col_stat = st.columns([6, 4])
    
    with col_map:
        # 板塊熱力圖
        fig_sec = px.treemap(
            df_sectors,
            path=['板塊'],
            values='RVOL', # 大小 = 資金熱度
            color='RS動能', # 顏色 = 強度 (越綠越強)
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="板塊資金流向 (面積=熱度, 顏色=RS強度)",
            hover_data=['狀態', '週漲幅%']
        )
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with col_stat:
        # 排行榜
        st.markdown("#### 🏆 強勢板塊排行 (依 RS 強度)")
        st.dataframe(
            df_sectors.sort_values("RS動能", ascending=False)[["板塊", "狀態", "RS動能", "RVOL"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "RS動能": st.column_config.NumberColumn(format="%.2f"),
                "RVOL": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=3)
            }
        )

    st.markdown("---")

    # --- Step 2: 深入單一板塊 (Drill-Down) ---
    st.header("2️⃣ 戰術打擊 (Sector Drill-Down)")
    
    # 預設選擇 RS 最強的板塊
    top_sector = df_sectors.sort_values("RS動能", ascending=False).iloc[0]["板塊"]
    # 找出完整的 key name
    default_idx = 0
    keys_list = list(SECTOR_CONFIG.keys())
    for i, k in enumerate(keys_list):
        if top_sector in k:
            default_idx = i
            break
            
    col_sel, col_param = st.columns([3, 7])
    with col_sel:
        target_sector = st.selectbox("選擇進攻板塊", keys_list, index=default_idx)
        st.info(f"當前關注：{target_sector}")
        
    with col_param:
        # 篩選參數
        rs_th = st.slider("個股 RS 強度門檻", -5.0, 5.0, 0.0, step=0.5)
        rvol_th = st.slider("個股 RVOL 爆量門檻", 0.5, 5.0, 1.2, step=0.1)

    # --- Step 3: 獲取成分股數據 ---
    components = SECTOR_CONFIG[target_sector]["components"]
    
    with st.spinner(f"正在掃描 {target_sector} 成分股..."):
        comp_data = fetch_data(components, "6mo")
        
    if comp_data.empty:
        st.warning("無數據")
        return

    comp_metrics = []
    for ticker in components:
        # 處理單一/多重索引
        try:
            if len(components) > 1:
                if ticker not in comp_data.columns.levels[0]: continue
                df = comp_data[ticker]
            else:
                df = comp_data
            
            close = df["Close"]
            vol = df["Volume"]
            open_p = df["Open"]
            
            if len(close) < 30: continue
            
            # 計算個股指標
            rs, rvol, chg = calculate_rs_rvol(close, vol, spy_close)
            
            # 嚴格過濾條件
            curr_price = close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            is_uptrend = curr_price > ma20
            is_red_k = curr_price > open_p.iloc[-1]
            
            comp_metrics.append({
                "代號": ticker,
                "現價": round(curr_price, 2),
                "RS值": rs,
                "RVOL": rvol,
                "週漲幅%": chg,
                "多頭": is_uptrend,
                "紅K": is_red_k,
                "TV連結": f"https://www.tradingview.com/chart/?symbol={ticker}"
            })
        except Exception:
            continue

    df_comp = pd.DataFrame(comp_metrics)
    
    if df_comp.empty:
        st.warning("無法計算成分股指標")
        return

    # --- Step 4: 呈現細節 (火箭清單 + 散佈圖) ---
    
    # 篩選火箭
    mask = (df_comp["RS值"] > rs_th) & (df_comp["RVOL"] > rvol_th) & (df_comp["多頭"]==True) & (df_comp["紅K"]==True)
    rockets = df_comp[mask].sort_values("RVOL", ascending=False)
    
    c1, c2 = st.columns([4, 6])
    
    with c1:
        st.subheader(f"🚀 火箭清單 ({len(rockets)})")
        if not rockets.empty:
            st.data_editor(
                rockets[["代號", "RS值", "RVOL", "TV連結"]],
                column_config={
                    "TV連結": st.column_config.LinkColumn("圖表", display_text="Open TV"),
                    "RVOL": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=5),
                    "RS值": st.column_config.NumberColumn(format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("⚠️ 無符合「嚴格」條件個股。請嘗試降低門檻或更換板塊。")
            
    with c2:
        st.subheader("🎯 動能分佈 (個股)")
        # 散佈圖
        fig_scat = px.scatter(
            df_comp,
            x="RS值", y="RVOL", size="RVOL", color="多頭",
            text="代號", hover_data=["現價", "週漲幅%"],
            title=f"{target_sector} 成分股動能分佈",
            template="plotly_dark",
            height=450
        )
        # 畫過濾線
        fig_scat.add_vline(x=rs_th, line_dash="dash", line_color="yellow")
        fig_scat.add_hline(y=rvol_th, line_dash="dash", line_color="yellow")
        
        st.plotly_chart(fig_scat, use_container_width=True)

if __name__ == "__main__":
    main()
