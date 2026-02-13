import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V11 Alpha 掠奪者", layout="wide", page_icon="🦅")
st.title("🦅 V11.0 Alpha 掠奪者 (The Predator)")

# --- 2. 策略參數 ---
# 我們加入 SHV (短債) 作為現金替代品，QQQ 作為對手
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊'
}
BENCHMARK = 'SPY'
RIVAL = 'QQQ'
SAFE_ASSET = 'SHV' # iShares Short Treasury Bond ETF (類現金但有息)
RISK_FREE_RATE = 0.04 

# --- 3. 數據引擎 ---
@st.cache_data(ttl=3600)
def get_predator_data():
    tickers = list(SECTOR_MAP.keys()) + [BENCHMARK, RIVAL, SAFE_ASSET, '^VIX']
    data = yf.download(tickers, period="5y", auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df = data['Close'].copy()
        except KeyError:
            df = data.copy()
    else:
        df = data['Close'].copy()
    
    # 清洗數據
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().dropna(how='all')
    df.index = pd.to_datetime(df.index)
    
    return df

# --- 4. Alpha 核心演算法 ---

def run_alpha_strategy(df_in, lookback_1m=21, lookback_3m=63, lookback_6m=126):
    """
    V11 獨贏策略：
    1. 只選最強的 1 檔 (Top 1)
    2. 空頭時持有 SHV (短債)
    3. VIX 過高時強制減半倉位 (Vol Control)
    """
    df = df_in.copy()
    
    # 計算動能分數
    ret_1m = df.pct_change(lookback_1m)
    ret_3m = df.pct_change(lookback_3m)
    ret_6m = df.pct_change(lookback_6m)
    score = (0.5 * ret_3m) + (0.3 * ret_6m) + (0.2 * ret_1m)
    
    # 定義環境
    spy = df[BENCHMARK]
    vix = df['^VIX']
    
    # 黃金交叉濾網 (Golden Cross): 50MA > 200MA
    sma50 = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    is_bull = sma50 > sma200
    
    # 換倉日計算
    unique_months = df.index.to_period('M').unique()
    rebalance_dates = []
    for m in unique_months:
        mask = (df.index.to_period('M') == m)
        if mask.any():
            rebalance_dates.append(df.index[mask][-1])
            
    # 回測容器
    strategy_returns = pd.Series(0.0, index=df.index)
    positions_history = {} 
    
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        if curr_date not in score.index: continue
        
        # 1. 判斷多空
        bull_market = is_bull.loc[curr_date]
        
        # 2. 判斷波動率 (恐慌濾網)
        current_vix = vix.loc[curr_date]
        is_panic = current_vix > 25 # VIX 高於 25 代表恐慌
        
        mask = (df.index > curr_date) & (df.index <= next_date)
        
        if bull_market:
            # 牛市：選最強的一檔 (Top 1)
            current_scores = score.loc[curr_date]
            # 排除非板塊
            valid_scores = current_scores.drop([BENCHMARK, RIVAL, SAFE_ASSET, '^VIX'], errors='ignore')
            
            # 50MA 濾網 (個股也要強)
            curr_prices = df.loc[curr_date]
            curr_ma50 = df.rolling(50).mean().loc[curr_date]
            valid_scores = valid_scores[curr_prices > curr_ma50]
            
            top_sector = valid_scores.nlargest(1).index.tolist()
            
            if top_sector:
                target = top_sector[0]
                sector_ret = df.loc[mask, target].pct_change()
                
                # [Vol Control] 如果恐慌，只持倉 50%，剩下 50% 買短債
                if is_panic:
                    safe_ret = df.loc[mask, SAFE_ASSET].pct_change()
                    strategy_returns.loc[mask] = 0.5 * sector_ret + 0.5 * safe_ret
                    positions_history[next_date] = [f"{target} (50%)", f"{SAFE_ASSET} (50%)"]
                else:
                    strategy_returns.loc[mask] = sector_ret
                    positions_history[next_date] = [f"{target} (100%)"]
            else:
                # 沒股票選，買短債
                strategy_returns.loc[mask] = df.loc[mask, SAFE_ASSET].pct_change()
                positions_history[next_date] = [SAFE_ASSET]
        else:
            # 熊市：全倉短債 (Active Hedge)
            strategy_returns.loc[mask] = df.loc[mask, SAFE_ASSET].pct_change()
            positions_history[next_date] = [f"{SAFE_ASSET} (Bear Hedge)"]

    # 計算累積淨值
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + df[BENCHMARK].pct_change()).cumprod()
    rival_equity = (1 + df[RIVAL].pct_change()).cumprod() # QQQ
    
    # 正規化
    if not strategy_equity.empty:
        base = strategy_equity.iloc[0]
        strategy_equity /= base
        benchmark_equity /= benchmark_equity.iloc[0]
        rival_equity /= rival_equity.iloc[0]
    
    return strategy_equity, benchmark_equity, rival_equity, positions_history, strategy_returns

# --- 5. 介面呈現 ---

try:
    with st.spinner('啟動 V11 Alpha 引擎...'):
        df = get_predator_data()

    if df.empty:
        st.error("數據下載失敗")
        st.stop()

    # 側邊欄
    st.sidebar.header("🦅 掠奪者參數")
    lookback_1m = st.sidebar.slider("動能週期 1", 10, 40, 21)
    lookback_3m = st.sidebar.slider("動能週期 2", 40, 80, 63)
    lookback_6m = st.sidebar.slider("動能週期 3", 100, 150, 126)

    # 執行策略
    strat_eq, spy_eq, qqq_eq, positions, strat_rets = run_alpha_strategy(df, lookback_1m, lookback_3m, lookback_6m)

    # --- 計算績效指標 ---
    def calc_metrics(equity, rets):
        if equity.empty: return 0, 0, 0, 0
        total_ret = equity.iloc[-1] - 1
        days = len(equity)
        cagr = (equity.iloc[-1])**(252/days) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        max_drawdown = ((equity / equity.cummax()) - 1).min()
        return total_ret, cagr, sharpe, max_drawdown

    v11_m = calc_metrics(strat_eq, strat_rets)
    qqq_m = calc_metrics(qqq_eq, df[RIVAL].pct_change())

    # --- 顯示 ---
    
    # 1. 頂部 KPI 對決
    st.markdown(f"### 🥊 冠軍賽：V11 vs {RIVAL}")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # 顏色判斷
    sharpe_delta = v11_m[2] - qqq_m[2]
    sharpe_color = "normal" if sharpe_delta >= 0 else "inverse"
    
    kpi1.metric("總報酬 (Total Return)", f"{v11_m[0]:.2%}", f"vs QQQ: {v11_m[0]-qqq_m[0]:.2%}")
    kpi2.metric("夏普比率 (Sharpe)", f"{v11_m[2]:.2f}", f"vs QQQ: {sharpe_delta:.2f}", delta_color=sharpe_color)
    kpi3.metric("年化報酬 (CAGR)", f"{v11_m[1]:.2%}", f"vs QQQ: {v11_m[1]-qqq_m[1]:.2%}")
    kpi4.metric("最大回檔 (MaxDD)", f"{v11_m[3]:.2%}", f"QQQ: {qqq_m[3]:.2%}", delta_color="inverse")

    # 2. 淨值曲線
    st.subheader("📈 淨值走勢 (Equity Curve)")
    chart_df = pd.DataFrame({
        "V11 掠奪者": strat_eq,
        "QQQ (對手)": qqq_eq,
        "SPY (大盤)": spy_eq
    })
    fig = px.line(chart_df, color_discrete_map={"V11 掠奪者": "#00FF00", "QQQ (對手)": "#FF0000", "SPY (大盤)": "#888888"})
    st.plotly_chart(fig, use_container_width=True)

    # 3. 戰術面板
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📡 當前持倉訊號 (Live)")
        if positions:
            last_date_obj = pd.to_datetime(max(positions.keys()))
            latest_pos = positions[max(positions.keys())]
            
            # 樣式優化
            st.info(f"""
            **決策日期**: {last_date_obj.strftime('%Y-%m-%d')}
            
            **🎯 攻擊目標**: {latest_pos}
            """)
            
            # 顯示目前環境
            last_spy = df[BENCHMARK].iloc[-1]
            last_ma200 = df[BENCHMARK].rolling(200).mean().iloc[-1]
            last_vix = df['^VIX'].iloc[-1]
            
            status_md = ""
            status_md += f"- **大盤趨勢**: {'🐂 牛市' if last_spy > last_ma200 else '🐻 熊市'}\n"
            status_md += f"- **恐慌指數**: {'🔥 恐慌' if last_vix > 25 else '😌 平靜'} ({last_vix:.2f})"
            st.markdown(status_md)
            
    with c2:
        st.subheader("🔥 板塊動能排行 (Heatmap Fix)")
        # 修正：確保顯示代號
        curr = df.iloc[-1]
        prev_1m = df.iloc[-21]
        chg = (curr - prev_1m) / prev_1m
        
        # 只取板塊
        target_cols = [c for c in SECTOR_MAP.keys() if c in chg.index]
        sec_chg = chg[target_cols].sort_values(ascending=False)
        
        # 建立一個有中文名稱的 DataFrame
        display_df = pd.DataFrame({
            "代號": sec_chg.index,
            "板塊名稱": [SECTOR_MAP[t] for t in sec_chg.index],
            "近1月漲幅": sec_chg.values
        })
        display_df = display_df.set_index("代號")
        
        st.dataframe(
            display_df.style.format({"近1月漲幅": "{:.2%}"}).background_gradient(subset=["近1月漲幅"], cmap='RdYlGn'),
            use_container_width=True
        )

    # 4. 歷史持倉
    with st.expander("📜 查看詳細換倉歷史"):
        if positions:
            rec_pos = pd.DataFrame.from_dict(positions, orient='index')
            rec_pos.columns = ['持倉內容'] + [f'資產{i}' for i in range(1, len(rec_pos.columns))]
            rec_pos.index = pd.to_datetime(rec_pos.index).strftime('%Y-%m-%d')
            st.table(rec_pos.tail(12))

except Exception as e:
    st.error(f"系統錯誤: {e}")
