import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import optimize

# --- 1. 設定參數 ---
SYMBOL = "QQQ"
START_DATE = "2010-01-01"
END_DATE = "2026-02-01"  # 回測截止日
MONTHLY_INVESTMENT = 1000 # 每月投入金額 (美元)

# --- 2. 下載數據 ---
print(f"正在下載 {SYMBOL} 數據 ({START_DATE} ~ {END_DATE})...")
data = yf.download(SYMBOL, start=START_DATE, end=pd.to_datetime(END_DATE) + pd.Timedelta(days=5), auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    df = data['Close']
else:
    df = data['Close']

# --- 3. 執行定期定額 (DCA) ---
cash_flow = []       # 用於計算 XIRR 的現金流
dates = []           # 日期紀錄
portfolio_value = [] # 資產價值紀錄
total_invested = []  # 總投入成本紀錄

shares_owned = 0
invested_capital = 0

# 產生每個月 1 號的日期序列
monthly_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

print("開始執行回測...")
for date in monthly_dates:
    # 尋找該月 1 號 (或之後最近的交易日)
    # 使用 asof 或 searchsorted 確保找到有效交易日
    if date not in df.index:
        # 往後找最近的交易日
        future_dates = df.index[df.index >= date]
        if future_dates.empty: continue
        trade_date = future_dates[0]
    else:
        trade_date = date
    
    # 執行買入
    price = float(df.loc[trade_date])
    shares_bought = MONTHLY_INVESTMENT / price
    shares_owned += shares_bought
    invested_capital += MONTHLY_INVESTMENT
    
    # 紀錄數據
    current_value = shares_owned * price
    
    dates.append(trade_date)
    total_invested.append(invested_capital)
    portfolio_value.append(current_value)
    
    # 紀錄現金流 (負值代表流出/投資)
    cash_flow.append((trade_date, -MONTHLY_INVESTMENT))

# --- 4. 計算最終結果 ---
final_date = dates[-1]
final_price = float(df.loc[final_date])
final_balance = shares_owned * final_price

# 加入最後一筆正現金流 (假設期末全部賣出，用於計算 XIRR)
cash_flow.append((final_date, final_balance))

# 計算 XIRR (內部報酬率)
def xirr(cashflows):
    years = [(cf[0] - cashflows[0][0]).days / 365.0 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    try:
        return optimize.newton(lambda r: sum([a / ((1 + r) ** y) for a, y in zip(amounts, years)]), 0.1)
    except:
        return 0.0

final_xirr = xirr(cash_flow)
total_return_pct = (final_balance - invested_capital) / invested_capital

# --- 5. 顯示結果報告 ---
print("-" * 40)
print(f"🚀 定期定額回測報告: {SYMBOL}")
print("-" * 40)
print(f"回測期間: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
print(f"扣款次數: {len(dates)} 次")
print(f"總投入成本: ${invested_capital:,.0f}")
print(f"最終資產價值: ${final_balance:,.0f}")
print(f"資產淨獲利: ${final_balance - invested_capital:,.0f}")
print("-" * 40)
print(f"總報酬率 (ROI): {total_return_pct:.2%}")
print(f"年化報酬率 (XIRR): {final_xirr:.2%}")
print("-" * 40)

# --- 6. 繪製互動圖表 ---
fig = go.Figure()

# 總投入成本線 (紅色虛線)
fig.add_trace(go.Scatter(
    x=dates, y=total_invested,
    mode='lines', name='總投入成本 (Principal)',
    line=dict(color='red', width=2, dash='dash')
))

# 資產市值線 (綠色實線)
fig.add_trace(go.Scatter(
    x=dates, y=portfolio_value,
    mode='lines', name='資產市值 (Market Value)',
    line=dict(color='#00FF00', width=3)
))

# 加上 QQQ 價格 (右軸，參考用)
fig.add_trace(go.Scatter(
    x=dates, y=[df.loc[d] for d in dates],
    mode='lines', name='QQQ 股價',
    line=dict(color='gray', width=1),
    yaxis='y2', opacity=0.3
))

fig.update_layout(
    title=f"定期定額 {SYMBOL} 資產累積圖 (Monthly $1000)",
    xaxis_title="年份",
    yaxis_title="資產價值 (USD)",
    yaxis2=dict(title="QQQ 股價", overlaying='y', side='right', showgrid=False),
    hovermode="x unified",
    height=600,
    legend=dict(x=0.01, y=0.99)
)

fig.show()
