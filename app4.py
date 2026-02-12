import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="美股 V7 戰情室", layout="wide", page_icon="🚀")

st.title("🚀 美股 V7 狙擊手戰情室")
st.markdown("### 資金流向與板塊輪動監控")

# --- 2. 定義板塊 ---
SECTOR_MAP = {
    'XLK': '科技 (Tech)',
    'SMH': '半導體 (Chip)',
    'XLE': '能源 (Energy)',
    'XLV': '醫療 (Health)',
    'XLF': '金融 (Finance)',
    'XLI': '工業 (Industry)',
    'XLP': '必需品 (Staples)',
    'XLU': '公用事業 (Util)',
    'XLY': '非必需 (Discret)',
    'XLB': '原物料 (Material)',
    'XLC': '通訊 (Comm)',
    'IYR': '房地產 (Real Est)',
    'QQQ': '那斯達克100',
    'SPY': '標普500'
}

# --- 3. 數據抓取函式 ---
@st.cache_data(ttl=3600) # 設定快取 1 小時，避免重複抓取
def get_sector_data():
    tickers = list(SECTOR_MAP.keys())
    # 抓取 400 天數據以計算年報酬
    data = yf.download(tickers, period="400d", auto_adjust=True)['Close']
    return data

try:
    with st.spinner('正在從華爾街下載最新數據...'):
        df_close = get_sector_data()

    # --- 4. 計算報酬率表 (Heatmap Table) ---
    st.subheader("📊 各板塊績效熱力榜 (由強至弱)")
    
    periods = {
        '1M (近1月)': 21,
        '3M (近1季)': 63,
        '6M (半年)': 126,
        '9M (三季)': 189,
        '12M (一年)': 252
    }
    
    res_data = {}
    current_prices = df_close.iloc[-1]
    
    for ticker in SECTOR_MAP.keys():
        row = {}
        # 顯示名稱
        name = SECTOR_MAP[ticker]
        
        for p_name, p_days in periods.items():
            if len(df_close) > p_days:
                prev_price = df_close[ticker].iloc[-p_days]
                ret = (current_prices[ticker] - prev_price) / prev_price
                row[p_name] = ret
            else:
                row[p_name] = 0.0
        res_data[f"{ticker} - {name}"] = row

    df_ret = pd.DataFrame.from_dict(res_data, orient='index')
    
    # 依照「近1月」強度排序
    df_ret = df_ret.sort_values(by='1M (近1月)', ascending=False)

    # 格式化顯示 (百分比 + 顏色條)
    st.dataframe(
        df_ret.style.format("{:.2%}")
        .background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1),
        use_container_width=True,
        height=500
    )
    
    st.markdown("💡 **解讀**：**綠色**越深代表資金流入越強，**紅色**越深代表拋售越重。請關注 **1M** 與 **3M** 皆強的板塊。")

    # --- 5. 互動式趨勢圖 (Trend Chart) ---
    st.markdown("---")
    st.subheader("📈 板塊資金流向趨勢圖 (Normalize)")
    
    # 讓用戶選擇要比較的板塊
    default_sectors = ['XLE', 'XLK', 'SMH', 'XLU'] # 預設顯示這幾個
    selected_tickers = st.multiselect(
        "選擇要對比的板塊/指數 (可多選):", 
        options=list(SECTOR_MAP.keys()),
        default=[k for k in default_sectors if k in SECTOR_MAP],
        format_func=lambda x: f"{x} - {SECTOR_MAP[x]}"
    )

    if selected_tickers:
        # 選擇時間範圍
        lookback_days = st.slider("回測天數 (Lookback)", min_value=30, max_value=365, value=120)
        
        # 截取數據並歸一化 (以起始日為 0%)
        chart_data = df_close[selected_tickers].iloc[-lookback_days:].copy()
        chart_data = (chart_data / chart_data.iloc[0] - 1) * 100
        
        # 使用 Plotly 畫圖
        fig = px.line(chart_data, x=chart_data.index, y=chart_data.columns, 
                      labels={"value": "報酬率 (%)", "variable": "板塊"},
                      title=f"近 {lookback_days} 天資金流向對比")
        
        # 優化圖表樣式
        fig.update_layout(hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **工程師提示**：試著把 **XLK (科技)** 和 **XLE (能源)** 同時選起來，觀察最近是否出現「剪刀差」背離現象。")

except Exception as e:
    st.error(f"系統發生錯誤，請檢查網路或數據源: {e}")
    # 印出詳細錯誤給開發者看
    st.write(e)
