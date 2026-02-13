import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V12.0 量子槓桿 (Beat QQQ)", layout="wide", page_icon="⚡")
st.title("⚡ V12.0 量子槓桿：終極 Alpha (2010-Present)")

# --- 2. 策略資產池 (引入槓桿 ETF) ---
# 注意：為了回測 2010 年，若 3x ETF 尚未成立，程式會自動降級為 1x
ASSETS = {
    'RiskOn_1x': ['XLK', 'SMH', 'XLE', 'XLV', 'XLY'], # 基礎攻擊
    'RiskOn_3x': ['TECL', 'SOXL', 'ERX', 'CURE', 'TECL'], # 槓桿攻擊 (對應上面)
    'Hedge': ['TLT', 'SHV'], # 長債(避險) + 短債(現金)
    'Benchmark': ['QQQ', 'SPY'],
    'Macro': ['^VIX']
}

# 對應關係映射
LEVERAGE_MAP = {
    'XLK': 'TECL', 'SMH': 'SOXL', 'XLE': 'ERX', 'XLV': 'CURE', 'XLY': 'TECL' # XLY用TECL代替
}

RISK_FREE_RATE = 0.03

# --- 3. 數據引擎 ---
@st.cache_data(ttl=3600)
def get_v12_data():
    # 收集所有 tickers
    all_tickers = []
    for k, v in ASSETS.items(): all_tickers += v
    all_tickers = list(set(all_tickers))
    
    # 下載數據 (2010-Now)
    data = yf.download(all_tickers, start="2010-01-01", auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try: df = data['Close'].copy()
        except: df = data.copy()
    else:
        df = data['Close'].copy()
        
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().dropna(how='all')
    
    return df

# --- 4. 核心演算法 (Vol-Targeted Leverage) ---

def run_v12_strategy(df_in, lookback_3m=63, vol_threshold=20):
    df = df_in.copy()
    
    # 1. 計算動能 (只看 3M 強度，反應最快)
    candidates = ASSETS['RiskOn_1x']
    momentum = df[candidates].pct_change(lookback_3m)
    
    # 2. 環境濾網
    vix = df['^VIX']
    qqq = df['QQQ']
    # 簡單均線濾網 (200MA)
    ma200 = qqq.rolling(200).mean()
    
    # 換倉日 (月底)
    rebalance_dates = df.resample('M').last().index
    
    strategy_returns = pd.Series(0.0, index=df.index)
    positions_history = {}
    
    # 為了避免 lookahead bias，我們用 shift 1 (看到訊號，次日執行)
    # 但為了簡化回測代碼，我們在迴圈內取當日訊號，填入次月報酬
    
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 找最後一個有效交易日
        if curr_date not in df.index:
            # 嘗試往前找最近的交易日
            try:
                curr_date = df.index[df.index <= curr_date][-1]
            except: continue
            
        if curr_date not in momentum.index: continue
        
        # --- 決策邏輯 ---
        
        # A. 選股：誰最強？
        curr_mom = momentum.loc[curr_date]
        if curr_mom.isnull().all(): continue
        
        best_sector = curr_mom.idxmax() # 找出 1x 代码 (e.g., 'XLK')
        
        # B. 決定槓桿倍數 (Risk Management)
        curr_vix = vix.loc[curr_date]
        is_bull = qqq.loc[curr_date] > ma200.loc[curr_date]
        
        target_ticker = ""
        
        if not is_bull:
            # 熊市 (QQQ < 200MA)：全倉 TLT (長債避險)
            # 歷史證明 2008, 2020 熊市 TLT 表現極佳
            target_ticker = 'TLT' 
            
            # [2022年 特例處理] 股債雙殺時，TLT 也會死，改 SHV
            # 簡單判定：如果 TLT 也在跌 (動能<0)，就去 SHV
            if df['TLT'].pct_change(63).loc[curr_date] < -0.05:
                target_ticker = 'SHV'
                
        else:
            # 牛市 (QQQ > 200MA)
            if curr_vix < vol_threshold:
                # 波動低：開 3 倍槓桿攻擊！
                # 檢查 3x ETF 是否存在 (2010年有些還沒出)
                lev_ticker = LEVERAGE_MAP.get(best_sector, best_sector)
                if lev_ticker in df.columns and not pd.isna(df.loc[curr_date, lev_ticker]):
                    target_ticker = lev_ticker
                else:
                    target_ticker = best_sector # 降級回 1x
            else:
                # 波動高 (VIX > 20)：降回 1x
                target_ticker = best_sector

        # --- 執行回測 ---
        mask = (df.index > curr_date) & (df.index <= next_date)
        if target_ticker in df.columns:
            # 記錄持倉
            positions_history[next_date] = target_ticker
            # 計算報酬
            strategy_returns.loc[mask] = df.loc[mask, target_ticker].pct_change()
        else:
            strategy_returns.loc[mask] = 0.0

    # 計算淨值
    strategy_equity = (1 + strategy_returns).cumprod()
    qqq_equity = (1 + df['QQQ'].pct_change()).cumprod()
    
    # 對齊
    valid_start = strategy_equity[strategy_equity != 1.0].index[0]
    strategy_equity = strategy_equity.loc[valid_start:]
    qqq_equity = qqq_equity.loc[valid_start:]
    
    # 歸一
    strategy_equity = strategy_equity / strategy_equity.iloc[0]
    qqq_equity = qqq_equity / qqq_equity.iloc[0]
    
    return strategy_equity, qqq_equity, positions_history, strategy_returns

# --- 5. 介面呈現 ---

try:
    with st.spinner('正在載入 V12 量子槓桿數據 (包含 3x ETF)...'):
        df = get_v12_data()

    if df.empty:
        st.error("數據下載失敗")
        st.stop()
        
    # 參數
    with st.sidebar:
        st.header("⚡ 策略參數")
        vol_threshold = st.slider("VIX 警戒線 (降槓桿)", 15, 30, 20)
        st.info("💡 當 VIX 低於此值且為牛市時，策略會買入 3x 槓桿 ETF (TECL/SOXL)。")

    # 運行
    strat_eq, qqq_eq, positions, strat_rets = run_v12_strategy(df, vol_threshold=vol_threshold)

    # --- KPI 計算 ---
    def get_kpi(equity, rets, rf=0.0):
        if equity.empty: return 0,0,0,0
        total = equity.iloc[-1] - 1
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        cagr = (equity.iloc[-1])**(1/years) - 1 if years > 0 else 0
        vol = rets.std() * np.sqrt(252)
        sharpe = (cagr - rf) / vol if vol > 0 else 0
        dd = ((equity / equity.cummax()) - 1).min()
        return total, cagr, sharpe, dd

    s_tot, s_cagr, s_sharpe, s_dd = get_kpi(strat_eq, strat_rets, RISK_FREE_RATE)
    q_tot, q_cagr, q_sharpe, q_dd = get_kpi(qqq_eq, df['QQQ'].pct_change().loc[strat_eq.index], RISK_FREE_RATE)

    # --- 顯示結果 ---
    st.markdown(f"### ⚡ V12 終極對決 (2010 - Present)")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("總報酬 (Total)", f"{s_tot:.2%}", f"vs QQQ: {s_tot-q_tot:.2%}")
    k2.metric("年化報酬 (CAGR)", f"{s_cagr:.2%}", f"QQQ: {q_cagr:.2%}")
    
    # Sharpe 顏色判定
    sharpe_delta = s_sharpe - q_sharpe
    k3.metric("夏普比率 (Sharpe)", f"{s_sharpe:.2f}", f"vs QQQ: {sharpe_delta:.2f}", 
              delta_color="normal" if sharpe_delta > 0 else "inverse")
    
    k4.metric("最大回檔 (MaxDD)", f"{s_dd:.2%}", f"QQQ: {q_dd:.2%}", delta_color="inverse")

    # 繪圖
    st.subheader("📈 資產淨值 (Log Scale)")
    chart_data = pd.DataFrame({"V12 策略 (3x Leveraged)": strat_eq, "QQQ (Benchmark)": qqq_eq})
    fig = px.line(chart_data, log_y=True) # 使用對數坐標，因為複利效應巨大
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 歷史
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📜 關鍵換倉歷史 (Top 10)")
        if positions:
            hist_df = pd.DataFrame(list(positions.items()), columns=['日期', '持倉標的'])
            hist_df['日期'] = hist_df['日期'].dt.strftime('%Y-%m-%d')
            st.table(hist_df.tail(10).set_index('日期'))
            
    with c2:
        st.subheader("🧠 策略邏輯解密")
        st.markdown("""
        1. **攻擊 (Bull Market)**
           - 若 VIX < 20: 開 **3倍槓桿** (TECL/SOXL)。
           - 若 VIX > 20: 降 **1倍槓桿** (XLK/SMH)。
        2. **防守 (Bear Market)**
           - 若 QQQ < 200MA: 全倉 **TLT (長債)**。
           - 若債券也跌: 轉 **SHV (現金)**。
        """)

except Exception as e:
    st.error(f"執行錯誤: {e}")
