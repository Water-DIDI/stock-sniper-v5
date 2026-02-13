import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V10.4 量子戰情室 (Final)", layout="wide", page_icon="⚛️")
st.title("⚛️ V10.4 量子對沖基金戰情室 (Final Stable)")

# --- 2. 策略參數 ---
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊'
}
BENCHMARK = 'SPY'
RISK_FREE_RATE = 0.04 

# --- 3. 高速數據引擎 (防彈版) ---
@st.cache_data(ttl=3600)
def get_quant_data():
    tickers = list(SECTOR_MAP.keys()) + [BENCHMARK, '^VIX']
    # 下載數據
    data = yf.download(tickers, period="5y", auto_adjust=True)
    
    # 處理 yfinance 多層索引問題
    if isinstance(data.columns, pd.MultiIndex):
        try:
            df = data['Close'].copy()
        except KeyError:
            df = data.copy()
    else:
        df = data['Close'].copy()
    
    # [Fix] 強制轉為數值，遇到非數值轉為 NaN，確保沒有 Object 混入
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 填補空值並刪除全空的列
    df = df.ffill().dropna(how='all')
    
    # 確保 Index 是 DatetimeIndex
    df.index = pd.to_datetime(df.index)
    
    return df

# --- 4. 量化邏輯核心 (零污染版) ---

def run_backtest(df_in, lookback_1m=21, lookback_3m=63, lookback_6m=126, top_n=3):
    """
    高速向量化回測引擎 (V10.4 零副作用版)
    """
    # 建立副本，確保不汙染快取
    df = df_in.copy()
    
    # 1. 計算動能分數
    ret_1m = df.pct_change(lookback_1m)
    ret_3m = df.pct_change(lookback_3m)
    ret_6m = df.pct_change(lookback_6m)
    
    score = (0.5 * ret_3m) + (0.3 * ret_6m) + (0.2 * ret_1m)
    
    # 2. 市場風控濾網
    spy = df[BENCHMARK]
    vix = df['^VIX']
    
    regime_bull = spy > spy.rolling(200).mean()
    vix_calm = vix.rolling(5).mean() < vix.rolling(20).mean()
    risk_on = regime_bull & vix_calm
    
    # 3. 抓取換倉日 (使用最安全的迴圈法，避開 Period 型態錯誤)
    # 我們只看 index，不把週期寫入 dataframe，避免 Cannot aggregate 錯誤
    unique_months = df.index.to_period('M').unique()
    rebalance_dates = []
    
    for m in unique_months:
        # 找出該月份在 df 中的最後一個交易日
        # 使用 boolean masking 絕對安全
        mask = (df.index.to_period('M') == m)
        if mask.any():
            last_date = df.index[mask][-1]
            rebalance_dates.append(last_date)
            
    # 4. 逐月回測
    strategy_returns = pd.Series(0.0, index=df.index)
    positions_history = {} 
    
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 確保日期有效 (因為 pct_change 前期會有 NaN)
        if curr_date not in score.index: continue
        
        # 取得該月份的分數 row
        # 這裡需注意：如果 score 在該日全是 NaN (例如數據剛開始)，跳過
        current_scores = score.loc[curr_date]
        if current_scores.isna().all(): continue

        mask = (df.index > curr_date) & (df.index <= next_date)
        
        if risk_on.loc[curr_date]:
            # 排除非板塊
            valid_scores = current_scores.drop([BENCHMARK, '^VIX'], errors='ignore')
            
            # 50MA 濾網
            current_prices = df.loc[curr_date]
            ma50 = df.rolling(50).mean().loc[curr_date]
            
            # 確保欄位對齊
            common_cols = valid_scores.index.intersection(current_prices.index)
            valid_scores = valid_scores.loc[common_cols]
            
            # 價格 > 50MA
            cond_ma = current_prices[common_cols] > ma50[common_cols]
            valid_scores = valid_scores[cond_ma]
            
            # 取 Top N
            top_sectors = valid_scores.nlargest(top_n).index.tolist()
            positions_history[next_date] = top_sectors
            
            if top_sectors:
                # 計算平均報酬
                sector_rets = df.loc[mask, top_sectors].pct_change().mean(axis=1)
                strategy_returns.loc[mask] = sector_rets.fillna(0)
            else:
                strategy_returns.loc[mask] = 0.0
        else:
            positions_history[next_date] = ['CASH (Risk Off)']
            strategy_returns.loc[mask] = 0.0

    # 計算淨值
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + df[BENCHMARK].pct_change()).cumprod()
    
    # 正規化起點
    if not strategy_equity.empty:
        strategy_equity = strategy_equity / strategy_equity.iloc[0]
        benchmark_equity = benchmark_equity / benchmark_equity.iloc[0]
    
    return strategy_equity, benchmark_equity, positions_history, strategy_returns

def monte_carlo_sim(returns, n_sims=1000, days=126):
    returns = returns.dropna()
    if len(returns) < 10: return np.zeros((days, n_sims))

    mu = returns.mean()
    sigma = returns.std()
    
    simulations = np.zeros((days, n_sims))
    for i in range(n_sims):
        rand_rets = np.random.normal(mu, sigma, days)
        price_path = (1 + rand_rets).cumprod()
        simulations[:, i] = price_path
        
    return simulations

# --- 5. 介面佈局 ---

try:
    with st.spinner('啟動 V10.4 量子核心 (數據清洗與下載)...'):
        df = get_quant_data()
    
    if df.empty:
        st.error("❌ 無法下載數據，請檢查網路。")
        st.stop()

    st.sidebar.header("⚙️ 實驗室參數")
    lookback_1m = st.sidebar.slider("動能週期 1 (短)", 10, 40, 21)
    lookback_3m = st.sidebar.slider("動能週期 2 (中)", 40, 80, 63)
    lookback_6m = st.sidebar.slider("動能週期 3 (長)", 100, 150, 126)
    sim_days = st.sidebar.slider("蒙地卡羅預測天數", 30, 252, 126)

    strat_eq, bench_eq, positions, strat_rets = run_backtest(df, lookback_1m, lookback_3m, lookback_6m)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 即時戰情 (Live)", 
        "🔙 歷史回測 (Backtest)", 
        "🎲 蒙地卡羅 (Monte Carlo)", 
        "🧮 參數矩陣 (Optimization)"
    ])

    # Tab 1: Live
    with tab1:
        st.subheader("📡 市場即時訊號")
        spy = df[BENCHMARK]
        vix = df['^VIX']
        
        # 確保有足夠數據計算 MA
        if len(df) > 200:
            is_bull = (spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]) and \
                      (vix.rolling(5).mean().iloc[-1] < vix.rolling(20).mean().iloc[-1])
        else:
            is_bull = False # 數據不足預設保守
            
        c1, c2 = st.columns([1, 3])
        with c1:
            if is_bull:
                st.success("🟢 **RISK ON**\n\n建議：全力進攻")
            else:
                st.error("🔴 **RISK OFF**\n\n建議：現金/防禦")
        with c2:
            if positions:
                # [Fix] 強制轉型 datetime 防止 numpy 報錯
                last_date_obj = pd.to_datetime(max(positions.keys()))
                latest_pos = positions[max(positions.keys())]
                st.info(f"📋 **本月模型建議持倉 ({last_date_obj.strftime('%Y-%m-%d')})**: {', '.join(latest_pos)}")
            else:
                st.warning("數據不足以產生持倉建議")

        st.markdown("---")
        st.caption("板塊動能掃描 (由強至弱)")
        
        # 安全取得最新數據
        curr = df.iloc[-1]
        prev_1m = df.iloc[-21]
        chg = (curr - prev_1m) / prev_1m
        
        # 只顯示板塊
        target_cols = [c for c in SECTOR_MAP.keys() if c in chg.index]
        sec_chg = chg[target_cols].sort_values(ascending=False)
        
        st.dataframe(
            sec_chg.to_frame(name="近1月漲幅").style.format("{:.2%}").background_gradient(cmap='RdYlGn'),
            use_container_width=True
        )

    # Tab 2: Backtest
    with tab2:
        st.subheader("📈 策略 vs 大盤")
        
        total_ret = strat_eq.iloc[-1] - 1
        bench_ret = bench_eq.iloc[-1] - 1
        days_len = len(strat_eq)
        cagr = (strat_eq.iloc[-1])**(252/days_len) - 1 if days_len > 0 else 0
        vol = strat_rets.std() * np.sqrt(252)
        sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("總報酬率", f"{total_ret:.2%}", f"{total_ret-bench_ret:.2%} (Alpha)")
        m2.metric("年化報酬 (CAGR)", f"{cagr:.2%}")
        m3.metric("夏普比率", f"{sharpe:.2f}")
        m4.metric("波動率", f"{vol:.2%}")

        chart_df = pd.DataFrame({"V10 策略": strat_eq, "SPY 大盤": bench_eq})
        st.plotly_chart(px.line(chart_df, title="淨值曲線"), use_container_width=True)
        
        st.markdown("#### 📜 換倉歷史記錄")
        if positions:
            rec_pos = pd.DataFrame.from_dict(positions, orient='index').tail(6)
            # [Fix] 索引轉字串防止格式錯誤
            rec_pos.index = pd.to_datetime(rec_pos.index).strftime('%Y-%m-%d')
            st.table(rec_pos)

    # Tab 3: Monte Carlo
    with tab3:
        st.subheader("🎲 未來風險模擬")
        sims = monte_carlo_sim(strat_rets, days=sim_days)
        
        fig_mc = go.Figure()
        # 畫前 50 條
        limit_n = min(50, sims.shape[1])
        for i in range(limit_n):
            fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
        avg_path = sims.mean(axis=1)
        fig_mc.add_trace(go.Scatter(y=avg_path, mode='lines', line=dict(color='yellow', width=3), name='平均預期'))
        st.plotly_chart(fig_mc, use_container_width=True)
        
        final_vals = sims[-1, :]
        c1, c2, c3 = st.columns(3)
        c1.metric("樂觀 (95%)", f"x {np.percentile(final_vals, 95):.2f}")
        c2.metric("中性 (50%)", f"x {np.percentile(final_vals, 50):.2f}")
        c3.metric("悲觀 (VaR 5%)", f"x {np.percentile(final_vals, 5):.2f}", delta_color="inverse")

    # Tab 4: Optimization
    with tab4:
        st.subheader("🧮 參數敏感度")
        if st.button("🚀 開始運算"):
            results = {}
            with st.spinner("量子矩陣運算中..."):
                for s in [10, 21, 42]:
                    row = {}
                    for l in [63, 126, 200]:
                        if s >= l: 
                            row[l] = 0
                            continue
                        s_eq, _, _, _ = run_backtest(df, lookback_1m=s, lookback_3m=(s+l)//2, lookback_6m=l)
                        # 安全計算 CAGR
                        if len(s_eq) > 0 and s_eq.iloc[-1] > 0:
                            ann = (s_eq.iloc[-1])**(252/len(s_eq)) - 1
                        else:
                            ann = 0
                        row[f"長週期 {l}"] = ann
                    results[f"短週期 {s}"] = row
            st.dataframe(pd.DataFrame(results).T.style.format("{:.2%}").background_gradient(cmap='RdYlGn'), use_container_width=True)

except Exception as e:
    st.error(f"系統發生錯誤: {e}")
    # 印出少量 Debug 資訊供參考
    st.text(str(e))
