import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import io
from datetime import datetime, timedelta

# --- 環境變數 ---
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# --- 定義 11 大板塊 ETF ---
SECTOR_MAP = {
    'XLK': '科技 (Tech)',
    'SMH': '半導體 (Chip)', # 特別加入，因為它是領先指標
    'XLE': '能源 (Energy)',
    'XLV': '醫療 (Health)',
    'XLF': '金融 (Finance)',
    'XLI': '工業 (Industry)',
    'XLP': '必需品 (Staples)',
    'XLU': '公用事業 (Util)',
    'XLY': '非必需 (Discret)',
    'XLB': '原物料 (Material)',
    'XLC': '通訊 (Comm)',
    'IYR': '房地產 (Real Est)'
}

def send_telegram_photo(caption, image_buffer):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    files = {'photo': ('heatmap.png', image_buffer, 'image/png')}
    data = {'chat_id': TG_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=data, files=files)
    except Exception as e:
        print(f"❌ 發送失敗: {e}")

def get_sector_performance():
    tickers = list(SECTOR_MAP.keys())
    # 抓取過去 400 天數據 (確保能算到 12 個月)
    data = yf.download(tickers, period="2y", auto_adjust=True)['Close']
    
    # 計算各週期報酬率
    periods = {
        '1M': 21,
        '3M': 63,
        '6M': 126,
        '9M': 189,
        '12M': 252
    }
    
    results = {}
    for ticker in tickers:
        ticker_res = []
        current_price = data[ticker].iloc[-1]
        
        for p_name, p_days in periods.items():
            if len(data) > p_days:
                past_price = data[ticker].iloc[-p_days]
                ret = (current_price - past_price) / past_price * 100
                ticker_res.append(ret)
            else:
                ticker_res.append(0.0)
        results[SECTOR_MAP[ticker]] = ticker_res

    # 轉成 DataFrame
    df = pd.DataFrame.from_dict(results, orient='index', columns=periods.keys())
    # 依照「最近 1 個月 (1M)」的強弱排序
    df = df.sort_values(by='1M', ascending=False)
    return df

def generate_heatmap(df):
    # 設定繪圖風格 (深色模式)
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    
    # 繪製熱力圖
    # cmap='RdYlGn': 紅色跌，綠色漲 (美股慣例通常綠漲紅跌，但在熱力圖中我們用 綠=強, 紅=弱)
    # 若習慣台股 (紅漲綠跌)，可以把 cmap 改成 'RdYlGn_r'
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                linewidths=.5, cbar_kws={'label': '報酬率 (%)'})
    
    plt.title(f"🔥 美股板塊資金流向熱力圖 ({datetime.now().strftime('%Y-%m-%d')})", fontsize=14, pad=20)
    plt.tight_layout()
    
    # 存到記憶體
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def main():
    print("📊 開始分析板塊資金流向...")
    try:
        df = get_sector_performance()
        if df.empty: return
        
        # 產生圖片
        img_buf = generate_heatmap(df)
        
        # 產生文字摘要 (前三強)
        top3 = df.index[:3].tolist()
        msg = f"📊 *V7.3 板塊資金報告*\n"
        msg += f"🔥 *近期最強*: {top3[0]}, {top3[1]}, {top3[2]}\n"
        msg += f"🧊 *近期最弱*: {df.index[-1]}\n"
        msg += "💡 顏色越綠代表資金流入越多，越紅代表流出。"

        send_telegram_photo(msg, img_buf)
        print("✅ 報告已發送")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    main()
