import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V14 真·板塊輪動戰情室", layout="wide", page_icon="🛡️")
st.title("🛡️ V14.0 真·板塊輪動戰情室 (Dual Momentum)")

# --- 2. 策略參數 ---
# 這些是 11 大板塊 ETF + 避險資產
SECTOR_MAP = {
    'XLE': '能源', 'XLK': '科技', 'XLV': '醫療', 'XLF': '金融', 
    'XLY': '非必需', 'XLP': '必需品', 'XLI': '工業', 'XLB': '原物料', 
    'XLU': '公用', 'IYR': '房地產', 'XLC': '通訊'
}
SECTORS = list(SECTOR_MAP.keys())
ASSETS = SECTORS + ['SPY', 'QQQ', 'IEF', 'SHV']
RISK_FREE_RATE = 0.03

# --- 3. 數據引擎 ---
@st.cache_data(ttl=3600)
def get_data_v14():
    # 抓取長週期數據 (2010 ~ Now)
    data = yf.download(ASSETS, start="2010-01-01", auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try: df = data['Close'].copy()
        except: df = data.copy()
    else:
        df = data['Close'].copy()
    
    # 轉數值並移除全空行
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    
    # 填補空值 (ffill) 針對某些 ETF 早期數據缺失
    df = df.ffill()
    # 再次移除無法填補的早期數據，確保對齊
    df = df.dropna()
    
    return df

# --- 4. 核心演算法：雙重動能板塊輪動 ---
def run_sector_rotation(df_in, lookback_1m=21, lookback_3m=63, lookback_6m=126, top_n=2):
    df = df_in.copy()
    
    # 1. 計算動能分數 (Momentum Score)
    # 權重設計：重視中期趨勢 (3M/6M) 避免短期雜訊
    ret_1m = df[SECTORS].pct_change(lookback_1m)
    ret_3m = df[SECTORS].pct_change(lookback_3m)
    ret_6m = df[SECTORS].pct_change(lookback_6m)
    
    # 綜合評分公式
    score = (0.4 * ret_3m) + (0.3 * ret_6m) + (0.3 * ret_1m)
    
    # 2. 市場濾網 (Market Filter / Regime)
    # SPY 站上 200 日線才做多，否則避險
    spy = df['SPY']
    ma200 = spy.rolling(200).mean()
    is_bull = (spy > ma200).shift(1) # shift(1) 避免未來函數
    
    # 3. 回測變數初始化
    rebalance_dates = df.resample('M').last().index # 每月底換倉
    strategy_returns = pd.Series(0.0, index=df.index)
    positions_history = {}
    
    # 4. 逐月回測迴圈
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # 找最接近的有效交易日
        if curr_date not in df.index:
            # 簡單處理：若月底非交易日，找前一天
            try: curr_date = df.index[df.index <= curr_date][-1]
            except: continue
        
        if curr_date not in score.index: continue
        
        # --- 決策邏輯 ---
        
        # 判斷大盤環境
        bull_market = is_bull.loc[curr_date]
        
        mask = (df.index > curr_date) & (df.index <= next_date)
        
        if bull_market:
            # 牛市：選分數最高的 Top N 板塊
            current_scores = score.loc[curr_date]
            top_sectors = current_scores.nlargest(top_n).index.tolist()
            
            # 紀錄持倉
            positions_history[next_date] = top_sectors
            
            # 計算報酬 (等權重)
            if top_sectors:
                # 取得這幾個板塊下個月的日報酬平均
                daily_rets = df.loc[mask, top_sectors].pct_change().mean(axis=1)
                strategy_returns.loc[mask] = daily_rets
        else:
            # 熊市：全倉轉入 IEF (公債)
            # 進階優化：如果連公債都在跌 (例如 2022)，轉入 SHV (現金)
            ief_mom = df['IEF'].pct_change(63).loc[curr_date]
            
            if ief_mom < -0.02: # 公債動能也是負的
                target = 'SHV' # 現金
            else:
                target = 'IEF' # 公債
                
            positions_history[next_date] = [f"{target} (避險)"]
            strategy_returns.loc[mask] = df.loc[mask, target].pct_change()

    # 計算淨值
    strategy_equity = (1 + strategy_returns).cumprod()
    qqq_equity = (1 + df['QQQ'].pct_change()).cumprod()
    
    # 對齊起點
    valid_start = strategy_equity[strategy_equity != 1.0].index[0]
    strategy_equity = strategy_equity.loc[valid_start:]
    qqq_equity = qqq_equity.loc[valid_start:]
    
    # 歸一化
    strategy_equity /= strategy_equity.iloc[0]
    qqq_equity /= qqq_equity.iloc[0]
    
    return strategy_equity, qqq_equity, positions_history

# --- 5. 介面呈現 ---

try:
    with st.spinner('正在分析板塊輪動數據 (2010-Now)...'):
        df = get_data_v14()

    if df.empty:
        st.error("數據下載失敗")
        st.stop()

    # 參數側邊欄
    with st.sidebar:
        st.header("⚙️ 策略參數")
        top_n = st.selectbox("持有板塊數量", [1, 2, 3], index=1, help="持有前幾強的板塊")
        st.info("💡 建議持有 2 檔以分散單一板塊風險。")

    # 執行策略
    strat_eq, qqq_eq, positions = run_sector_rotation(df, top_n=top_n)

    # --- KPI 計算 ---
    def calc_kpi(equity):
        if equity.empty: return 0,0,0,0
        total_ret = equity.iloc[-1] - 1
        days = (equity.index[-1] - equity.index[0]).days
        cagr = (equity.iloc[-1])**(365.25/days) - 1 if days > 0 else 0
        
        daily_ret = equity.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        
        cummax = equity.cummax()
        dd = (equity / cummax - 1).min()
        return total_ret, cagr, sharpe, dd

    s_tot, s_cagr, s_sharpe, s_dd = calc_kpi(strat_eq)
    q_tot, q_cagr, q_sharpe, q_dd = calc_kpi(qqq_eq)

    # --- 顯示結果 ---
    st.markdown("### 🛡️ V14 真·板塊輪動 vs QQQ")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("總報酬 (Total)", f"{s_tot:.2%}", f"vs QQQ: {s_tot - q_tot:.2%}")
    k2.metric("年化報酬 (CAGR)", f"{s_cagr:.2%}", f"QQQ: {q_cagr:.2%}")
    
    # Sharpe 顏色
    s_color = "normal" if s_sharpe >= q_sharpe else "inverse"
    k3.metric("夏普比率 (Sharpe)", f"{s_sharpe:.2f}", f"vs QQQ: {s_sharpe - q_sharpe:.2f}", delta_color=s_color)
    k4.metric("最大回檔 (MaxDD)", f"{s_dd:.2%}", f"QQQ: {q_dd:.2%}", delta_color="inverse")

    # 圖表
    st.subheader("📈 資產淨值 (Log Scale)")
    chart_data = pd.DataFrame({
        "板塊輪動策略": strat_eq,
        "QQQ (大盤)": qqq_eq
    })
    fig = px.line(chart_data, log_y=True, color_discrete_map={"板塊輪動策略": "#00FF00", "QQQ (大盤)": "#FF3333"})
    fig.update_layout(hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 操作面板
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📡 本月持倉建議")
        if positions:
            last_date = max(positions.keys())
            latest_pos = positions[last_date]
            st.info(f"**日期**: {last_date.strftime('%Y-%m-%d')}")
            
            # 判斷是避險還是進攻
            if "避險" in latest_pos[0]:
                st.error(f"### 🛑 風險規避模式")
                st.write(f"持有標的：**{latest_pos[0]}**")
                st.caption("原因：SPY 跌破年線 或 債券動能轉弱")
            else:
                st.success(f"### 🚀 動能攻擊模式")
                st.write(f"持有前 {top_n} 強勢板塊：")
                for p in latest_pos:
                    st.write(f"- **{p} ({SECTOR_MAP.get(p, p)})**")

    with c2:
        st.subheader("🔥 即時熱力掃描 (Momentum)")
        # 顯示最新一日的動能排名
        latest_score = (0.4 * df[SECTORS].pct_change(63).iloc[-1] + 
                        0.3 * df[SECTORS].pct_change(126).iloc[-1] + 
                        0.3 * df[SECTORS].pct_change(21).iloc[-1]).sort_values(ascending=False)
        
        disp_df = pd.DataFrame({
            "板塊": [SECTOR_MAP[t] for t in latest_score.index],
            "綜合動能分": latest_score.values
        }, index=latest_score.index)
        
        st.dataframe(disp_df.style.background_gradient(cmap='Greens'), use_container_width=True)

    # 歷史紀錄
    with st.expander("📜 查看換倉歷史紀錄"):
        if positions:
            hist_list = []
            for d, p in positions.items():
                hist_list.append({"日期": d.strftime('%Y-%m-%d'), "持倉": ", ".join(p)})
            st.dataframe(pd.DataFrame(hist_list).set_index("日期").tail(20), use_container_width=True)

except Exception as e:
    st.error(f"系統錯誤: {e}")
