import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. 設定參數 ---
SYMBOLS = ['VOO', 'QQQ', 'QLD']
START_DATE = "2010-09-09" # VOO 成立於 2010/09，統一從這天開始
END_DATE = "2026-02-01"
MONTHLY_INVESTMENT = 1000 # 每月投入

# --- 2. 下載數據 ---
data = yf.download(SYMBOLS, start=START_DATE, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    df = data['Close']
else:
    df = data['Close']

df = df.dropna() # 確保對齊

# --- 3. 執行 DCA 回測 ---
results = {}

for symbol in SYMBOLS:
    dates = []
    portfolio_value = []
    total_invested = []
    
    shares = 0
    cash_in = 0
    
    # 產生每月 1 號
    monthly_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    for date in monthly_dates:
        # 找交易日
        if date not in df.index:
            future_dates = df.index[df.index >= date]
            if future_dates.empty: continue
            trade_date = future_dates[0]
        else:
            trade_date = date
            
        # 買入
        price = df.loc[trade_date, symbol]
        shares += MONTHLY_INVESTMENT / price
        cash_in += MONTHLY_INVESTMENT
        
        # 紀錄
        dates.append(trade_date)
        total_invested.append(cash_in)
        portfolio_value.append(shares * price)
        
    results[symbol] = pd.DataFrame({
        'Date': dates,
        'Value': portfolio_value,
        'Cost': total_invested
    }).set_index('Date')

# --- 4. 績效計算與顯示 ---
st.markdown("### 🥊 三雄對決：定期定額 (2010/09 - Present)")

cols = st.columns(3)
metrics = []

for i, symbol in enumerate(SYMBOLS):
    final_val = results[symbol]['Value'].iloc[-1]
    cost = results[symbol]['Cost'].iloc[-1]
    profit = final_val - cost
    roi = profit / cost
    
    # 計算最大回檔 (Max Drawdown)
    daily_val = results[symbol]['Value']
    cummax = daily_val.cummax()
    dd = (daily_val / cummax - 1).min()
    
    metrics.append((symbol, final_val, roi, dd))
    
    with cols[i]:
        st.subheader(f"{symbol}")
        st.metric("最終資產", f"${final_val:,.0f}")
        st.metric("總報酬率", f"{roi:.2%}")
        st.metric("最大回檔", f"{dd:.2%}", delta_color="inverse")

# --- 5. 繪製走勢圖 ---
chart_data = pd.DataFrame({
    'VOO (標普500)': results['VOO']['Value'],
    'QQQ (那斯達克)': results['QQQ']['Value'],
    'QLD (2倍槓桿)': results['QLD']['Value'],
    '投入成本 (Principal)': results['VOO']['Cost'] # 成本都一樣
})

fig = px.line(chart_data, log_y=True) # 使用對數座標
fig.update_layout(
    title="定期定額資產累積 (對數座標)",
    xaxis_title="年份",
    yaxis_title="資產價值 (USD)",
    height=600,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.info("""
💡 **數據解讀：**
1. **QLD** 是絕對的獲利王者，但請看 **2022 年的跌幅**。它從高點腰斬再腰斬，如果您那時急需用錢，會非常痛苦。
2. **QQQ** 取得了最佳的平衡，報酬遠勝 VOO，風險卻比 QLD 低得多。
3. **VOO** 是穩健的底層資產，適合當作安全氣囊。
""")
