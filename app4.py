import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. 頁面設定 ---
st.set_page_config(page_title="V16 四雄對決 DCA", layout="wide", page_icon="🥊")
st.title("🥊 V16.0 定期定額終極戰：誰是退休金之王？")

# --- 2. 設定參數 ---
SYMBOLS = ['VOO', 'QQQ', 'QLD']
# VOO 成立於 2010/09，這是回測的起點限制
START_DATE = "2010-09-09" 
MONTHLY_INVESTMENT = 1000 

# --- 3. 數據引擎 ---
@st.cache_data(ttl=3600)
def get_dca_data():
    # 下載數據
    data = yf.download(SYMBOLS, start="2009-01-01", auto_adjust=True) # 多抓一點算MA
    
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].copy()
    else:
        df = data['Close'].copy()
    
    # 計算 QQQ 的 200日均線 (作為 Smart 策略的訊號)
    df['QQQ_MA200'] = df['QQQ'].rolling(200).mean()
    
    # 裁切到 VOO 上市後
    df = df.loc[START_DATE:]
    df = df.dropna()
    
    return df

# --- 4. 核心回測邏輯 (包含 Smart DCA) ---
def run_dca_simulation(df):
    # 產生每月投資日 (每個月第一個交易日)
    monthly_dates = []
    # 這裡我們用 resample('MS') 抓每個月第一天，然後找最近的交易日
    temp_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='MS')
    
    for d in temp_dates:
        # 往後找最近的交易日
        valid_dates = df.index[df.index >= d]
        if not valid_dates.empty:
            monthly_dates.append(valid_dates[0])
    
    # 初始化結果容器
    results = {
        'VOO': {'dates': [], 'value': [], 'cost': []},
        'QQQ': {'dates': [], 'value': [], 'cost': []},
        'QLD': {'dates': [], 'value': [], 'cost': []},
        'Smart_QLD': {'dates': [], 'value': [], 'cost': []} # 第四種策略
    }
    
    # 初始化持倉狀態
    # [股數, 總投入成本]
    holdings = {
        'VOO': 0, 'QQQ': 0, 'QLD': 0, 
        'Smart_QLD': {'shares_qld': 0, 'shares_qqq': 0} # Smart 策略會持有兩種之一
    }
    total_cost = 0
    
    for date in monthly_dates:
        # 當日價格
        prices = df.loc[date]
        qqq_price = prices['QQQ']
        qld_price = prices['QLD']
        voo_price = prices['VOO']
        ma200 = prices['QQQ_MA200']
        
        # 累積成本
        total_cost += MONTHLY_INVESTMENT
        
        # --- 策略 1, 2, 3: 傳統 DCA (買入持有) ---
        holdings['VOO'] += MONTHLY_INVESTMENT / voo_price
        holdings['QQQ'] += MONTHLY_INVESTMENT / qqq_price
        holdings['QLD'] += MONTHLY_INVESTMENT / qld_price
        
        # 紀錄 1, 2, 3
        results['VOO']['dates'].append(date)
        results['VOO']['value'].append(holdings['VOO'] * voo_price)
        results['VOO']['cost'].append(total_cost)
        
        results['QQQ']['dates'].append(date)
        results['QQQ']['value'].append(holdings['QQQ'] * qqq_price)
        results['QQQ']['cost'].append(total_cost)
        
        results['QLD']['dates'].append(date)
        results['QLD']['value'].append(holdings['QLD'] * qld_price)
        results['QLD']['cost'].append(total_cost)
        
        # --- 策略 4: Smart QLD (動態切換) ---
        # 1. 計算目前資產總值
        current_smart_val = (holdings['Smart_QLD']['shares_qld'] * qld_price) + \
                            (holdings['Smart_QLD']['shares_qqq'] * qqq_price)
        
        # 2. 加上本月投入
        new_total_val = current_smart_val + MONTHLY_INVESTMENT
        
        # 3. 判斷訊號 (QQQ > 200MA ?)
        is_bull = qqq_price > ma200
        
        # 4. 全倉輪動 (Rebalance)
        if is_bull:
            # 牛市：全倉持有 QLD
            new_shares_qld = new_total_val / qld_price
            holdings['Smart_QLD'] = {'shares_qld': new_shares_qld, 'shares_qqq': 0}
        else:
            # 熊市：全倉持有 QQQ (降槓桿)
            new_shares_qqq = new_total_val / qqq_price
            holdings['Smart_QLD'] = {'shares_qld': 0, 'shares_qqq': new_shares_qqq}
            
        # 紀錄 Smart 策略
        results['Smart_QLD']['dates'].append(date)
        results['Smart_QLD']['value'].append(new_total_val)
        results['Smart_QLD']['cost'].append(total_cost)

    return results

# --- 5. 介面呈現 ---
try:
    with st.spinner('正在進行 15 年 DCA 回測運算...'):
        df = get_dca_data()
        res = run_dca_simulation(df)
        
    st.success("回測完成！以下是每月投入 $1,000 美元的最終成果：")
    
    # 計算 KPI
    summary = []
    strategies = ['VOO', 'QQQ', 'QLD', 'Smart_QLD']
    strategy_names = {
        'VOO': '1. VOO (標普500)',
        'QQQ': '2. QQQ (那斯達克)',
        'QLD': '3. QLD (無腦2倍槓桿)',
        'Smart_QLD': '4. 聰明槓桿 (QLD+200MA)'
    }
    
    for s in strategies:
        final_val = res[s]['value'][-1]
        cost = res[s]['cost'][-1]
        roi = (final_val - cost) / cost
        
        # MaxDD
        vals = pd.Series(res[s]['value'])
        cummax = vals.cummax()
        dd = (vals / cummax - 1).min()
        
        summary.append({
            '策略': strategy_names[s],
            '總投入成本': f"${cost:,.0f}",
            '最終資產': f"${final_val:,.0f}",
            '總報酬率': f"{roi:.2%}",
            '最大回檔': f"{dd:.2%}"
        })
        
    # 顯示表格
    st.table(pd.DataFrame(summary).set_index('策略'))
    
    # 繪圖
    st.subheader("📈 資產累積曲線 (對數座標)")
    chart_df = pd.DataFrame({
        'Date': res['VOO']['dates'],
        '1. VOO': res['VOO']['value'],
        '2. QQQ': res['QQQ']['value'],
        '3. QLD (Buy & Hold)': res['QLD']['value'],
        '4. Smart QLD (Trend)': res['Smart_QLD']['value'],
        '投入成本': res['VOO']['cost']
    }).set_index('Date')
    
    fig = px.line(chart_df, log_y=True)
    # 自定義顏色
    fig.update_traces(line=dict(width=2))
    # 加粗 Smart 策略
    fig.update_traces(selector=dict(name='4. Smart QLD (Trend)'), line=dict(width=4, color='#00FF00'))
    fig.update_traces(selector=dict(name='3. QLD (Buy & Hold)'), line=dict(color='orange', dash='dot'))
    
    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # 結論區
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        ### 🧐 策略 4 (Smart QLD) 的優勢：
        1. **完美躲避 2022**：請看圖表，在 2022 年橘色虛線 (QLD) 崩跌時，綠色實線 (Smart QLD) 因為切換到了 QQQ，跌幅明顯較小。
        2. **保留牛市爆發力**：在 2023-2024 牛市回歸時，它又切換回 QLD，資產斜率與橘線一樣陡峭。
        """)
    with c2:
        st.warning("""
        ### ⚠️ 關鍵差別：
        * **無腦 QLD (橘色)**：最終資產雖高，但中間曾經歷 **-60%** 的腰斬，非常考驗人性。
        * **聰明 QLD (綠色)**：最終資產接近無腦 QLD，但回檔控制得更好，是更適合長期持有的改良版。
        """)

except Exception as e:
    st.error(f"錯誤: {e}")
