import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V11.1 時空旅人 (2010-Now)", layout="wide", page_icon="⏳")
st.title("⏳ V11.1 時空旅人 (2010-Present)")

# --- 2. 策略參數 ---
# 這些 ETF 在 2010 年都已經存在，確保回測真實性
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊' # 注意: XLC 成立較晚，程式會自動處理缺值
}
BENCHMARK = 'SPY'
RIVAL = 'QQQ'
SAFE_ASSET = 'SHV' # 短債 (現金替代品)
RISK_FREE_RATE = 0.03 # 長期平均無風險利率

# --- 3. 數據引擎 (從 2010 開始) ---
@st.cache_data(ttl=3600)
def get_long_history_data():
    tickers = list(SECTOR_MAP.keys()) + [BENCHMARK, RIVAL, SAFE_ASSET, '^VIX']
    
    # [關鍵修正] 設定 start="2010-01-01" 抓取長週期數據
    data = yf.download(tickers, start="2010-01-01", auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df = data['Close'].copy()
        except KeyError:
            df = data.copy()
    else:
        df = data['Close'].copy()
    
    # 數值清洗
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 處理時區問題 (移除 timezone 避免對齊錯誤)
    df.index = df.index.tz_localize(None)
    
    # 填補空值 (XLC 在 2018 前是空的，這沒關係，會自動被排除在排名外)
    df = df.ffill().dropna(how='all')
    
    return df

# --- 4. 演算法核心 (Smart Rebalance) ---

def run_strategy_2010(df_in, lookback_1m=21, lookback_3m=63, lookback_6m=126):
    df = df_in.copy()
    
    # 1. 計算動能分數 (Weighted Momentum)
    # 演算法：重視近期爆發力 (1M) 但也要有中期續航力 (3M/6M)
    ret_1m = df.pct_change(lookback_1m)
    ret_3m = df.pct_change(lookback_3m)
    ret_6m = df.pct_change(lookback_6m)
    score = (0.5 * ret_3m) + (0.3 * ret_6m) + (0.2 * ret_1m)
    
    # 2. 市場風控濾網 (Regime Filter)
    spy = df[BENCHMARK]
    vix = df['^VIX']
    
    # 長期趨勢線 (200MA)
    sma200 = spy.rolling(200).mean()
    # 只有當 "跌破年線" 且 "VIX > 20" 時才視為真正熊市，避免被假跌破洗出場
    is_bear = (spy < sma200) & (vix > 20) 
    
    # 3. 換倉邏輯 (月底結算)
    # 找出每個月的最後一個交易日
    unique_months = df.index.to_period('M').unique()
    rebalance_dates = []
    for m in unique_months:
        mask = (df.index.to_period('M') == m)
        if mask.any():
            rebalance_dates.append(df.index[mask][-1])
            
    strategy_returns = pd.Series(0.0, index=df.index)
    positions_history = {} 
    
    # 逐月回測
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 確保當天有數據
        if curr_date not in score.index: continue
        
        # 取得當下環境
        bear_market = is_bear.loc[curr_date]
        
        # 計算區間遮罩
        mask = (df.index > curr_date) & (df.index <= next_date)
        
        if not bear_market:
            # === 牛市邏輯 (Bull Market) ===
            # 挑選分數最高的 1 檔 (Top 1 Winner)
            current_scores = score.loc[curr_date]
            # 排除非板塊代號
            valid_scores = current_scores.drop([BENCHMARK, RIVAL, SAFE_ASSET, '^VIX'], errors='ignore')
            # 排除沒有數據的 (例如早期的 XLC)
            valid_scores = valid_scores.dropna()
            
            # [進階濾網] 只有當該板塊股價 > 自己的 50MA 才買 (確保不是接刀)
            curr_prices = df.loc[curr_date]
            curr_ma50 = df.rolling(50).mean().loc[curr_date]
            valid_scores = valid_scores[curr_prices > curr_ma50]
            
            top_list = valid_scores.nlargest(1).index.tolist()
            
            if top_list:
                target = top_list[0]
                # 全倉買入最強板塊
                strategy_returns.loc[mask] = df.loc[mask, target].pct_change()
                positions_history[next_date] = [f"{target} (Top 1)"]
            else:
                # 沒強勢股，暫泊短債
                strategy_returns.loc[mask] = df.loc[mask, SAFE_ASSET].pct_change()
                positions_history[next_date] = [SAFE_ASSET]
        else:
            # === 熊市邏輯 (Bear Market) ===
            # 全倉買短債避險 (Cash is King)
            strategy_returns.loc[mask] = df.loc[mask, SAFE_ASSET].pct_change()
            positions_history[next_date] = [f"{SAFE_ASSET} (避險)"]

    # 計算淨值 (從 1.0 開始)
    strategy_equity = (1 + strategy_returns).cumprod()
    
    # 計算 QQQ 淨值 (對齊起點)
    rival_ret = df[RIVAL].pct_change()
    rival_equity = (1 + rival_ret).cumprod()
    
    # 確保兩者從同一個時間點開始比較 (移除回測前的 NaN)
    valid_start = strategy_equity.first_valid_index()
    strategy_equity = strategy_equity.loc[valid_start:]
    rival_equity = rival_equity.loc[valid_start:]
    
    # 重新歸一化 (Base = 1.0)
    strategy_equity = strategy_equity / strategy_equity.iloc[0]
    rival_equity = rival_equity / rival_equity.iloc[0]
    
    return strategy_equity, rival_equity, positions_history, strategy_returns

# --- 5. 介面呈現 ---

try:
    with st.spinner('正在穿越時空 (下載 2010-2026 數據)...'):
        df = get_long_history_data()

    if df.empty:
        st.error("數據下載失敗")
        st.stop()
    
    # 參數設定
    with st.expander("⚙️ 調整回測參數", expanded=False):
        lookback_1m = st.slider("動能週期 1", 10, 40, 21)
        lookback_3m = st.slider("動能週期 2", 40, 80, 63)
        lookback_6m = st.slider("動能週期 3", 100, 150, 126)

    # 執行策略
    strat_eq, rival_eq, positions, strat_rets = run_strategy_2010(df, lookback_1m, lookback_3m, lookback_6m)

    # --- 績效計算 ---
    def get_kpi(equity):
        if equity.empty: return 0, 0, 0
        total_ret = equity.iloc[-1] - 1
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        cagr = (equity.iloc[-1])**(1/years) - 1
        max_dd = ((equity / equity.cummax()) - 1).min()
        return total_ret, cagr, max_dd

    v11_tot, v11_cagr, v11_dd = get_kpi(strat_eq)
    qqq_tot, qqq_cagr, qqq_dd = get_kpi(rival_eq)
    
    # 計算 Sharpe (使用日報酬)
    v11_vol = strat_rets.std() * np.sqrt(252)
    v11_sharpe = (v11_cagr - RISK_FREE_RATE) / v11_vol if v11_vol > 0 else 0
    
    rival_rets = df[RIVAL].pct_change().loc[strat_eq.index]
    qqq_vol = rival_rets.std() * np.sqrt(252)
    qqq_sharpe = (qqq_cagr - RISK_FREE_RATE) / qqq_vol if qqq_vol > 0 else 0

    # --- 顯示區 ---
    
    # 1. 冠軍賽比分
    st.markdown("### 🥊 15年回測總結 (2010 - Present)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("總報酬 (Total Return)", f"{v11_tot:.2%}", f"vs QQQ: {v11_tot-qqq_tot:.2%}")
    col2.metric("年化報酬 (CAGR)", f"{v11_cagr:.2%}", f"QQQ: {qqq_cagr:.2%}")
    # Sharpe 亮燈邏輯
    delta_sharpe = v11_sharpe - qqq_sharpe
    col3.metric("夏普比率 (Sharpe)", f"{v11_sharpe:.2f}", f"{delta_sharpe:.2f}", delta_color="normal" if delta_sharpe > 0 else "inverse")
    col4.metric("最大回檔 (MaxDD)", f"{v11_dd:.2%}", f"QQQ: {qqq_dd:.2%}", delta_color="inverse")

    # 2. 互動式圖表 (Plotly)
    st.subheader("📈 資產淨值走勢 (可縮放)")
    st.caption("💡 提示：使用滑鼠滾輪可縮放圖表，移到線條上可查看詳細數值。")
    
    chart_data = pd.DataFrame({
        "V11 策略": strat_eq,
        "QQQ (那斯達克)": rival_eq
    })
    
    fig = px.line(chart_data, color_discrete_map={"V11 策略": "#00FF00", "QQQ (那斯達克)": "#FF3333"})
    
    # 優化圖表設定 (淺顯易懂的 Tooltip)
    fig.update_layout(
        hovermode="x unified", # 統一顯示 x 軸資訊
        xaxis_title="年份",
        yaxis_title="資產淨值 (起始=1.0)",
        legend_title="策略比較",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. 換倉歷史分析
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📜 歷史換倉紀錄 (History)")
        st.caption("演算法每月月底會進行一次決策。以下是最近 10 次的換倉動作：")
        
        if positions:
            rec_pos = pd.DataFrame.from_dict(positions, orient='index', columns=['持有標的'])
            rec_pos.index.name = '換倉日期'
            # 格式化日期
            rec_pos.index = pd.to_datetime(rec_pos.index).strftime('%Y-%m-%d')
            st.table(rec_pos.tail(10))
            
    with c2:
        st.subheader("🤖 演算法運作說明")
        with st.expander("多久換一次倉？", expanded=True):
            st.write("""
            **頻率：每月一次 (Monthly Rebalance)**
            
            程式會在**每個月的最後一個交易日**收盤後進行運算：
            1. 計算所有板塊的動能分數。
            2. 檢查大盤是否崩盤 (SPY < 200MA + VIX > 20)。
            3. **下個月的第一個交易日** 開盤執行買賣。
            """)
            
        with st.expander("如何決定買什麼？", expanded=True):
            st.write("""
            **邏輯：強者恆強 (Winner Takes All)**
            
            1. **牛市時**：買進**分數最高**的那一檔板塊 (Top 1)。
               *(例如：科技股最強就全買 XLK，能源強就全買 XLE)*
            2. **熊市時**：全數賣出股票，買進 **SHV (短債)** 領利息避險。
            """)

except Exception as e:
    st.error(f"系統錯誤: {e}")
