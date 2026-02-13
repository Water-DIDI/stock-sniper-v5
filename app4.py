import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. 頁面設定 (量子黑風格) ---
st.set_page_config(page_title="V10 量子對沖基金戰情室", layout="wide", page_icon="⚛️")
st.title("⚛️ V10.0 量子對沖基金戰情室 (Quant Lab)")

# --- 2. 策略參數 (可調整區) ---
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊'
}
BENCHMARK = 'SPY'
RISK_FREE_RATE = 0.04 # 4% 無風險利率

# --- 3. 高速數據引擎 ---
@st.cache_data(ttl=3600)
def get_quant_data():
    """下載所有相關數據 (包含總經與板塊)"""
    tickers = list(SECTOR_MAP.keys()) + [BENCHMARK, '^VIX']
    # 抓 5 年數據以進行有效回測
    data = yf.download(tickers, period="5y", auto_adjust=True)
    
    # 處理 MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df = data['Close'].copy()
        except KeyError:
            df = data.copy()
    else:
        df = data['Close'].copy()
        
    return df.ffill()

# --- 4. 量化邏輯核心 (Vectorized Logic) ---

def run_backtest(df, lookback_1m=21, lookback_3m=63, lookback_6m=126, top_n=3):
    """
    高速向量化回測引擎
    回傳: 策略淨值曲線, 基準淨值曲線, 買賣訊號
    """
    # 1. 計算動能分數 (Momentum Score)
    # Score = 0.5*3M + 0.3*6M + 0.2*1M
    ret_1m = df.pct_change(lookback_1m)
    ret_3m = df.pct_change(lookback_3m)
    ret_6m = df.pct_change(lookback_6m)
    
    score = (0.5 * ret_3m) + (0.3 * ret_6m) + (0.2 * ret_1m)
    
    # 2. 市場風控濾網 (Regime Filter)
    spy = df[BENCHMARK]
    vix = df['^VIX']
    
    # SPY > 200MA
    regime_bull = spy > spy.rolling(200).mean()
    # VIX 5MA < 20MA
    vix_calm = vix.rolling(5).mean() < vix.rolling(20).mean()
    
    # 總體 Risk ON 信號 (True = 進場, False = 空手/現金)
    risk_on = regime_bull & vix_calm
    
    # 3. 模擬逐月換倉 (Monthly Rebalance)
    # 我們取每個月最後一個交易日進行判定
    monthly_data = df.resample('M').last()
    monthly_score = score.loc[monthly_data.index]
    monthly_risk_on = risk_on.loc[monthly_data.index]
    
    # 建立策略報酬率容器
    strategy_returns = pd.Series(0.0, index=df.index)
    
    # 為了模擬真實操作，我們使用 "Shift 1" (這個月的訊號，下個月初執行)
    positions_history = {} # 紀錄持倉
    
    current_date_idx = 0
    rebalance_dates = monthly_data.index
    
    # 這邊因為要模擬換倉，稍微用迴圈處理每個月，但內部運算還是向量化的
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 取得這段期間的日資料遮罩
        mask = (df.index > curr_date) & (df.index <= next_date)
        
        # 判斷當下是否 Risk On
        if monthly_risk_on.loc[curr_date]:
            # 選出前 N 名
            current_scores = monthly_score.loc[curr_date]
            # 排除 SPY, VIX
            valid_scores = current_scores.drop([BENCHMARK, '^VIX'], errors='ignore')
            
            # 50MA 濾網: 價格需 > 50MA (這裡用當月最後一天的價格判斷)
            current_prices = df.loc[curr_date]
            ma50 = df.rolling(50).mean().loc[curr_date]
            valid_scores = valid_scores[current_prices > ma50]
            
            # 取 Top N
            top_sectors = valid_scores.nlargest(top_n).index.tolist()
            
            # 紀錄持倉
            positions_history[next_date] = top_sectors
            
            # 計算下個月的平均報酬 (等權重)
            if top_sectors:
                # 獲取這些板塊下個月的日報酬
                sector_returns = df.loc[mask, top_sectors].pct_change()
                # 策略日報酬 = 板塊平均
                strategy_returns.loc[mask] = sector_returns.mean(axis=1)
            else:
                # 沒標的，空手 (0報酬 或 無風險利率)
                strategy_returns.loc[mask] = 0.0
        else:
            # Risk Off: 空手 (或持有 SHV/IEF，這裡簡化為 0 報酬現金)
            positions_history[next_date] = ['CASH']
            strategy_returns.loc[mask] = 0.0

    # 計算累積淨值
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + df[BENCHMARK].pct_change()).cumprod()
    
    # 對齊起點為 1
    strategy_equity = strategy_equity / strategy_equity.iloc[0]
    benchmark_equity = benchmark_equity / benchmark_equity.iloc[0]
    
    return strategy_equity, benchmark_equity, positions_history, strategy_returns

def monte_carlo_sim(returns, n_sims=1000, days=126):
    """蒙地卡羅模擬未來走勢"""
    mu = returns.mean()
    sigma = returns.std()
    
    simulations = np.zeros((days, n_sims))
    
    # 使用幾何布朗運動 (GBM) 或 簡單常態分佈模擬日報酬
    # P_t = P_t-1 * (1 + r)
    for i in range(n_sims):
        rand_rets = np.random.normal(mu, sigma, days)
        price_path = (1 + rand_rets).cumprod()
        simulations[:, i] = price_path
        
    return simulations

# --- 5. 介面佈局 ---

try:
    with st.spinner('啟動量子運算核心...'):
        df = get_quant_data()

    # 側邊欄：進階參數
    st.sidebar.header("⚙️ 實驗室參數")
    lookback_1m = st.sidebar.slider("動能週期 1 (短)", 10, 40, 21)
    lookback_3m = st.sidebar.slider("動能週期 2 (中)", 40, 80, 63)
    lookback_6m = st.sidebar.slider("動能週期 3 (長)", 100, 150, 126)
    sim_days = st.sidebar.slider("蒙地卡羅預測天數", 30, 252, 126)

    # 執行回測
    strat_eq, bench_eq, positions, strat_rets = run_backtest(df, lookback_1m, lookback_3m, lookback_6m)

    # 分頁設計
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 即時戰情 (Live)", 
        "🔙 歷史回測 (Backtest)", 
        "🎲 蒙地卡羅 (Monte Carlo)", 
        "🧮 參數矩陣 (Optimization)"
    ])

    # ==========================================
    # Tab 1: 即時戰情 (原本的 V9 功能)
    # ==========================================
    with tab1:
        st.subheader("📡 市場即時訊號")
        
        # 取得最新一天的 Regime
        spy = df[BENCHMARK]
        vix = df['^VIX']
        is_bull = (spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]) and \
                  (vix.rolling(5).mean().iloc[-1] < vix.rolling(20).mean().iloc[-1])
        
        c1, c2 = st.columns([1, 3])
        with c1:
            if is_bull:
                st.success("🟢 **RISK ON**\n\n建議：全力進攻")
            else:
                st.error("🔴 **RISK OFF**\n\n建議：現金/防禦")
                
        with c2:
            # 顯示本月持倉建議 (基於最新數據)
            latest_pos = positions[max(positions.keys())]
            st.info(f"📋 **本月模型建議持倉**: {', '.join(latest_pos)}")

        # 簡單熱力圖 (保留視覺化)
        st.markdown("---")
        st.caption("板塊動能掃描")
        curr = df.iloc[-1]
        prev_1m = df.iloc[-21]
        chg = (curr - prev_1m) / prev_1m
        # 只取板塊
        sec_chg = chg[list(SECTOR_MAP.keys())].sort_values(ascending=False)
        st.dataframe(sec_chg.to_frame(name="近1月漲幅").style.format("{:.2%}"), use_container_width=True)

    # ==========================================
    # Tab 2: 歷史回測 (新功能)
    # ==========================================
    with tab2:
        st.subheader("📈 策略 vs 大盤 (5年回測)")
        
        # 計算績效指標
        total_ret = strat_eq.iloc[-1] - 1
        bench_ret = bench_eq.iloc[-1] - 1
        cagr = (strat_eq.iloc[-1])**(252/len(strat_eq)) - 1
        vol = strat_rets.std() * np.sqrt(252)
        sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總報酬率", f"{total_ret:.2%}", f"{total_ret-bench_ret:.2%}")
        m2.metric("年化報酬 (CAGR)", f"{cagr:.2%}")
        m3.metric("夏普比率 (Sharpe)", f"{sharpe:.2f}")
        m4.metric("波動率 (Vol)", f"{vol:.2%}")

        # 畫淨值曲線
        chart_df = pd.DataFrame({
            "Strategy (V10)": strat_eq,
            "SPY (Benchmark)": bench_eq
        })
        fig = px.line(chart_df, title="策略淨值曲線 (Equity Curve)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📜 持倉歷史記錄 (最近 6 個月)")
        # 顯示最近幾次的換倉紀錄
        rec_pos = pd.DataFrame.from_dict(positions, orient='index').tail(6)
        st.table(rec_pos)

    # ==========================================
    # Tab 3: 蒙地卡羅 (新功能)
    # ==========================================
    with tab3:
        st.subheader("🎲 未來風險模擬 (Monte Carlo Simulation)")
        st.caption(f"基於策略過去表現，模擬未來 {sim_days} 天的 1,000 種可能路徑")
        
        sims = monte_carlo_sim(strat_rets, days=sim_days)
        
        # 畫出模擬圖 (只畫前 50 條以免太亂)
        fig_mc = go.Figure()
        for i in range(50):
            fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
        # 加上平均線
        avg_path = sims.mean(axis=1)
        fig_mc.add_trace(go.Scatter(y=avg_path, mode='lines', line=dict(color='yellow', width=3), name='平均路徑'))
        
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # 統計分佈
        final_values = sims[-1, :]
        p95 = np.percentile(final_values, 95)
        p50 = np.percentile(final_values, 50)
        p05 = np.percentile(final_values, 5)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("樂觀情境 (95%)", f"x {p95:.2f}")
        c2.metric("中性情境 (50%)", f"x {p50:.2f}")
        c3.metric("悲觀情境 (5%)", f"x {p05:.2f} (VaR)")

    # ==========================================
    # Tab 4: 參數優化矩陣 (新功能)
    # ==========================================
    with tab4:
        st.subheader("🧮 參數敏感度分析 (避免 Overfitting)")
        st.write("測試不同 [短週期 vs 長週期] 組合下的年化報酬率")
        
        if st.button("🚀 開始矩陣運算 (可能需要幾秒鐘)"):
            results = {}
            # 簡化測試範圍
            short_range = [10, 21, 42]
            long_range = [63, 126, 200]
            
            for s in short_range:
                row = {}
                for l in long_range:
                    if s >= l: 
                        row[l] = 0
                        continue
                    # 快速跑回測
                    s_eq, _, _, _ = run_backtest(df, lookback_1m=s, lookback_3m=(s+l)//2, lookback_6m=l)
                    ann_ret = (s_eq.iloc[-1])**(252/len(s_eq)) - 1
                    row[f"長週期 {l}"] = ann_ret
                results[f"短週期 {s}"] = row
            
            res_df = pd.DataFrame(results).T
            st.dataframe(res_df.style.format("{:.2%}").background_gradient(cmap='RdYlGn'), use_container_width=True)
            st.caption("💡 顏色越綠越好。如果整個矩陣都是綠的，代表策略邏輯強健 (Robust)；如果只有一格綠，代表過度擬合 (Overfitting)。")

except Exception as e:
    st.error(f"系統崩潰 (Margin Call): {e}")
    st.write("Debug info:", e)
