import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

# --- A. 環境變數 ---
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# --- B. 定義板塊 ---
SECTOR_CONFIG = {
    "半導體": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU"],
    "科技": ["AAPL", "MSFT", "ORCL", "CRM", "ADBE", "CSCO", "IBM", "META", "GOOGL"],
    "軟體": ["PANW", "SNOW", "PLTR", "CRWD", "DDOG", "ZS", "NET"],
    "能源": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO"],
    "原物料": ["GLD", "SLV", "FCX", "SCCO", "AA", "NEM"],
    "工業": ["GE", "CAT", "DE", "HON", "LMT", "RTX"],
    "生技": ["AMGN", "GILD", "VRTX", "REGN", "MRNA"],
    "加密": ["IBIT", "COIN", "MSTR", "MARA", "CLSK"]
}

# --- C. 屬性名單 ---
MEGA_CAPS = ["TSM", "NVDA", "AAPL", "MSFT", "GOOGL", "META", "XOM", "CVX", "JPM", "GLD"]
HIGH_BETA = ["MSTR", "COIN", "MARA", "CLSK", "PLTR", "SOFI", "AI"]

def send_telegram_notify(msg):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def get_strategy_params(ticker):
    """
    對齊 TradingView V5 的參數邏輯
    回傳: (rvol_th, rs_th, lookback_days, mode_name)
    """
    if ticker in MEGA_CAPS:
        # 權值穩健: 突破20日高點
        return 1.1, 0.0, 20, "🐢穩健"
    elif ticker in HIGH_BETA:
        # 投機飆股: 突破10日高點
        return 2.0, 2.0, 10, "🐇飆股"
    else:
        # 循環動能: 突破14日高點
        return 1.5, 1.0, 14, "🐆動能"

def fetch_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
        return data
    except: return pd.DataFrame()

def check_stock(ticker, df, spy_close):
    # 確保數據夠長 (至少要能算 20日均線 + 突破)
    if len(df) < 50: return None
    
    close = df["Close"]
    high = df["High"]
    vol = df["Volume"]
    
    # 1. 取得參數 (含 lookback)
    rvol_th, rs_th, lookback, mode_name = get_strategy_params(ticker)

    # 2. [V6] 3日趨勢確認
    ma20 = close.rolling(20).mean()
    is_confirmed = (close.iloc[-1] > ma20.iloc[-1]) and \
                   (close.iloc[-2] > ma20.iloc[-2]) and \
                   (close.iloc[-3] > ma20.iloc[-3])
    if not is_confirmed: return None 

    # 3. [V7.1 NEW] 突破前高邏輯 (Donchian Breakout)
    # 取得「昨天以前」的過去 N 天最高價
    # shift(1) 代表不包含今天 (因為我們要看是不是今天突破過去)
    highest_high = high.shift(1).rolling(window=lookback).max()
    
    # 判定：今天的收盤價 > 過去 N 天的最高價
    is_breakout = close.iloc[-1] > highest_high.iloc[-1]
    
    if not is_breakout: return None # 沒突破就過濾掉

    # 4. RS 動能
    idx = close.index.intersection(spy_close.index)
    if len(idx) < 30: return None
    rs_ratio = close.loc[idx] / spy_close.loc[idx]
    rs_val = (rs_ratio.iloc[-1] / rs_ratio.iloc[-21] - 1) * 100
    
    # 5. RVOL
    vol_avg = vol.rolling(20).mean()
    avg_vol = vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
    rvol_val = vol.iloc[-1] / avg_vol
    
    # 6. 紅K
    is_red = close.iloc[-1] > df["Open"].iloc[-1]

    # 最終篩選
    if rs_val > rs_th and rvol_val > rvol_th and is_red:
        return {
            "Mode": mode_name,
            "RS": round(rs_val, 2),
            "RVOL": round(rvol_val, 2),
            "Breakout": lookback,
            "Chg": round((close.iloc[-1]/close.iloc[-2]-1)*100, 2)
        }
    return None

def main():
    print("🚀 開始掃描美股 (V7.1 Breakout Edition)...")
    all_stocks = []
    for s in SECTOR_CONFIG.values(): all_stocks.extend(s)
    all_stocks.append("SPY")
    all_stocks = list(set(all_stocks))
    
    data = fetch_data(all_stocks)
    if data.empty:
        send_telegram_notify("⚠️ 數據下載失敗")
        return

    try: spy_close = data["SPY"]["Close"]
    except: return
    
    results = {}
    for sector, tickers in SECTOR_CONFIG.items():
        hit_list = []
        for t in tickers:
            try:
                if t not in data.columns.levels[0]: continue
                res = check_stock(t, data[t], spy_close)
                if res:
                    # 顯示資訊增加「突破天數」
                    hit_list.append(f"*{t}* {res['Mode']} 破{res['Breakout']}日高 (+{res['Chg']}%)")
            except: continue
        if hit_list: results[sector] = hit_list

    today = datetime.now().strftime("%Y-%m-%d")
    if results:
        msg = f"🚀 *美股狙擊手 V7.1* [{today}]\n"
        msg += "🔥 *突破火箭名單 (Align with TV)*：\n"
        msg += "━━━━━━━━━━━━━━\n"
        for sec, stocks in results.items():
            msg += f"📂 *{sec}*\n" + "\n".join(stocks) + "\n\n"
        msg += "━━━━━━━━━━━━━━\n請打開 TradingView 確認！"
    else:
        msg = f"💤 *美股狙擊手 V7.1* [{today}]\n今日無「突破前高 + 爆量」標的。\n市場盤整中，建議觀望。"
    
    print(msg)
    send_telegram_notify(msg)

if __name__ == "__main__":
    main()
