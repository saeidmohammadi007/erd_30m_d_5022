import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dtw import dtw
import ccxt
from collections import defaultdict
import requests

# ═══════════════════════════════════════════════════════════
# ۰. تنظیمات ثابت
# ═══════════════════════════════════════════════════════════
n_candles = 120          # تعداد کندل‌های الگوی بلند
n_candles_short = 25     # تعداد کندل‌های الگوی کوتاه (جدید)
tf_input = "30m"
search_interval = "1d"
macd_fast = 12          # دورهٔ میانگین سریع برای MACD
macd_slow = 26          # دورهٔ میانگین کند برای MACD

# تنظیمات تلگرام
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    def send_telegram_message(text):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"خطا در ارسال پیام تلگرام: {e}")

    def send_telegram_long_message(text):
        max_len = 4096
        if len(text) <= max_len:
            send_telegram_message(text)
            return
        lines = text.split('\n')
        chunk = ''
        for line in lines:
            if len(chunk) + len(line) + 1 > max_len:
                if chunk:
                    send_telegram_message(chunk.strip())
                    chunk = line + '\n'
                else:
                    send_telegram_message(line[:max_len])
                    chunk = ''
            else:
                chunk += line + '\n'
        if chunk.strip():
            send_telegram_message(chunk.strip())
else:
    def send_telegram_message(text): pass
    def send_telegram_long_message(text): pass

crypto_symbols = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD",
    "VET-USD", "LINK-USD", "SHIB-USD", "DOGE-USD", "ADA-USD",
    "SAND-USD", "AR-USD", "HBAR-USD", "IOTA-USD", "TRX-USD",
    "AVAX-USD", "NEAR-USD", "ONDO-USD", "HYPE-USD", "FLOKI-USD",
    "BONK-USD", "YFI-USD", "TST-USD", "PONS-USD", "NIL-USD",
    "SQD-USD", "BMT-USD", "BOME-USD", "CYS-USD", "GUN-USD",
    "JTO-USD", "COAI-USD", "WLD-USD", "H-USD", "AIO-USD",
    "RE-USD", "DODO-USD", "DELL-USD", "MYX-USD", "BEAT-USD",
    "BLESS-USD", "PROM-USD", "VELVET-USD", "EVAA-USD", "BTW-USD",
    "KAITO-USD", "TUT-USD", "MVLL-USD", "SKYAI-USD", "USELESS-USD"
]

symbols_to_search = [
    "AAPL", "GC=F",
    "ETH-USD", "XRP-USD", "LTC-USD", "BTC-USD", "DOGE-USD",
    "BNB-USD", "ADA-USD", "LINK-USD", "VET-USD",
    "TRX-USD", "ATOM-USD", "XTZ-USD", "HBAR-USD",
    "XLM-USD", "IOTA-USD", "SOL-USD", "FIL-USD",
    "AVAX-USD", "DOT-USD", "SHIB-USD"
]

symbol_colors = {
    "AAPL": "#1f77b4", "GC=F": "#ff7f0e",
    "XRP-USD": "#2ca02c", "LTC-USD": "#d62728",
    "BTC-USD": "#9467bd", "DOGE-USD": "#8c564b",
    "BNB-USD": "#e377c2", "ADA-USD": "#7f7f7f",
    "LINK-USD": "#bcbd22", "VET-USD": "#17becf",
    "TRX-USD": "#aec7e8", "ATOM-USD": "#ffbb78",
    "XTZ-USD": "#98df8a", "HBAR-USD": "#ff9896",
    "XLM-USD": "#c5b0d5", "IOTA-USD": "#c49c94",
    "SOL-USD": "#f7b6d2", "FIL-USD": "#dbdb8d",
    "AVAX-USD": "#9edae5", "DOT-USD": "#393b79",
    "SHIB-USD": "#e7ba52", "ETH-USD": "#ff6347",
    "SAND-USD": "#f39c12", "AR-USD": "#2ecc71"
}

def fetch_lbank_data(yahoo_symbol, tf):
    print(f"Yahoo داده‌ای برای {yahoo_symbol} ندارد. دریافت از LBank...")
    base = yahoo_symbol.upper().replace("-USD", "").replace("/USD", "")
    lbank_pair = f"{base}/USDT"
    tf_map = {"5m": "5m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1w"}
    lbank_tf = tf_map.get(tf, "1d")
    exchange = ccxt.lbank({'enableRateLimit': True})
    since = exchange.parse8601('2017-01-01T00:00:00Z')
    all_candles = []
    limit = 1000
    while True:
        try:
            candles = exchange.fetch_ohlcv(lbank_pair, timeframe=lbank_tf, since=since, limit=limit)
        except Exception as e:
            print(f"خطا در دریافت از LBank: {e}")
            break
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
    if not all_candles:
        raise ValueError(f"LBank نیز داده‌ای برای {lbank_pair} ندارد.")
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"داده‌های LBank دریافت شد: {len(df)} کندل برای {lbank_pair}")
    return df

def z_norm(seq):
    std = np.std(seq)
    if std == 0:
        return seq - np.mean(seq)
    return (seq - np.mean(seq)) / std

def compute_macd_line(close_series, fast=12, slow=26):
    """محاسبه خط MACD (تفاضل دو میانگین نمایی)"""
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

all_telegram_parts = []

for crypto_symbol in crypto_symbols:
    print(f"\n{'='*80}")
    print(f"شروع تحلیل برای نماد الگو: {crypto_symbol}")
    print(f"{'='*80}")

    if tf_input == "4h":
        download_interval = "1h"
        resample_rule = "4h"
    elif tf_input == "5m":
        download_interval = "5m"
        resample_rule = None
    elif tf_input == "30m":
        download_interval = "30m"
        resample_rule = None
    elif tf_input == "1wk":
        download_interval = "1wk"
        resample_rule = None
    else:
        download_interval = "1d"
        resample_rule = None

    crypto_df = yf.download(crypto_symbol, interval=download_interval, period="max")
    if crypto_df.empty:
        try:
            crypto_df = fetch_lbank_data(crypto_symbol, download_interval)
        except:
            print(f"  داده‌ای برای {crypto_symbol} یافت نشد. رد می‌شود.")
            continue
    if crypto_df.empty:
        print(f"  داده‌ای برای {crypto_symbol} خالی است. رد می‌شود.")
        continue

    close_series = crypto_df['Close'].dropna()
    if resample_rule:
        close_series = close_series.resample(resample_rule).last().dropna()

    # محاسبه خط MACD خام
    macd_series = compute_macd_line(close_series, macd_fast, macd_slow)
    macd_series = macd_series.dropna()

    if len(macd_series) < n_candles:
        print(f"  تعداد کندل‌های معتبر MACD ({len(macd_series)}) کمتر از {n_candles} است. رد می‌شود.")
        continue

    # الگوها: بلند (120) و کوتاه (25)
    pattern_macd_long = macd_series.iloc[-n_candles:].values
    pattern_dates = macd_series.index[-n_candles:]
    pattern_start_date = pattern_dates[0].normalize().tz_localize(None)

    pattern_macd_short = pattern_macd_long[-n_candles_short:]

    pattern_norm_long = z_norm(pattern_macd_long)
    pattern_norm_short = z_norm(pattern_macd_short)

    print(f"الگوی {crypto_symbol} ({tf_input}) با {n_candles} کندل بلند و {n_candles_short} کندل کوتاه از MACD استخراج شد.")
    print(f"بازه الگو: {pattern_dates[0]} تا {pattern_dates[-1]}")

    def search_raw_matches(symbol, interval="1d", window_fraction=0.5):
        df = yf.download(symbol, interval=interval, period="max")
        if df.empty:
            if symbol.endswith("-USD"):
                try:
                    df = fetch_lbank_data(symbol, interval)
                except:
                    return None, []
            else:
                return None, []
        if df.empty:
            return None, []

        close = df['Close'].dropna()
        if len(close) < n_candles:
            return None, []

        macd_line = compute_macd_line(close, macd_fast, macd_slow).dropna()

        if len(macd_line) < n_candles:
            return None, []

        N = len(macd_line)
        macd_vals = macd_line.values
        window_size = max(2, int(window_fraction * n_candles))
        window_size_short = max(2, int(window_fraction * n_candles_short))
        raw_matches = []

        for i in range(N - n_candles + 1):
            window_end_date = macd_line.index[i + n_candles - 1]
            window_end_date_naive = window_end_date.normalize().tz_localize(None)
            if window_end_date_naive >= pattern_start_date:
                continue

            win_macd_long = macd_vals[i:i + n_candles]
            win_macd_short = win_macd_long[-n_candles_short:]

            win_norm_long = z_norm(win_macd_long)
            win_norm_short = z_norm(win_macd_short)

            # فاصله DTW برای الگوی بلند
            dist_long = dtw(pattern_norm_long, win_norm_long,
                            keep_internals=False,
                            window_type='sakoechiba',
                            window_args={'window_size': window_size}).distance

            # فاصله DTW برای الگوی کوتاه
            dist_short = dtw(pattern_norm_short, win_norm_short,
                             keep_internals=False,
                             window_type='sakoechiba',
                             window_args={'window_size': window_size_short}).distance

            # امتیاز ترکیبی با وزن برابر
            combined_score = 0.5 * (dist_long / n_candles) + 0.5 * (dist_short / n_candles_short)

            raw_matches.append((combined_score, dist_long, dist_short,
                                macd_line.index[i], win_macd_long))

        return macd_line, raw_matches

    all_macd_series = {}
    global_raw_matches = []

    for sym in symbols_to_search:
        macd_series_sym, raw_list = search_raw_matches(sym, interval=search_interval, window_fraction=0.5)
        all_macd_series[sym] = macd_series_sym
        if raw_list:
            for item in raw_list:
                global_raw_matches.append((sym,) + item)

    if not global_raw_matches:
        print("هیچ تطابقی برای این الگو یافت نشد. به نماد بعدی می‌رویم.")
        continue

    # ساخت لیست نهایی: (sym, combined_score, dist_long, dist_short, start_date, win_macd)
    scored_matches = []
    for match in global_raw_matches:
        sym, combined_score, dist_long, dist_short, start_date, win_macd = match
        scored_matches.append((sym, combined_score, dist_long, dist_short, start_date, win_macd))

    # گروه‌بندی بر اساس نماد
    symbol_matches = defaultdict(list)
    for m in scored_matches:
        sym = m[0]
        symbol_matches[sym].append(m)

    # انتخاب بهترین‌ها با جلوگیری از هم‌پوشانی
    selected_per_symbol = []
    for sym, matches in symbol_matches.items():
        matches.sort(key=lambda x: x[1])  # مرتب‌سازی بر اساس combined_score
        selected = []
        for match in matches:
            start = match[4]
            overlap = False
            for sel in selected:
                if abs((start - sel[4]).days) < n_candles:
                    overlap = True
                    break
            if not overlap:
                selected.append(match)
                if len(selected) == 2:
                    break
        selected_per_symbol.extend(selected)

    selected_per_symbol.sort(key=lambda x: x[1])

    # حذف آستانه و انتخاب بهترین تطابق کلی
    if not selected_per_symbol:
        print(f"هیچ تطابقی برای {crypto_symbol} یافت نشد.")
        continue

    best_match = selected_per_symbol[0]

    print(f"\nبهترین تطابق برای الگوی {crypto_symbol}:")
    print("─" * 80)
    sym, combined_score, dist_long, dist_short, start_date, win_macd = best_match
    print(f"⭐ {sym} | امتیاز ترکیبی: {combined_score:.3f} | DTW بلند: {dist_long:.2f} | DTW کوتاه: {dist_short:.2f} | شروع: {start_date.strftime('%Y-%m-%d')}")

    msg = (
        f"📊 <b>الگو:</b> {tf_input} {crypto_symbol}\n"
        f"🪙 <b>نماد:</b> {sym}\n"
        f"📅 <b>تاریخ شروع:</b> {start_date.strftime('%Y-%m-%d')}\n"
        f"⭐ <b>امتیاز ترکیبی:</b> {combined_score:.3f}\n"
        f"<i>DTW بلند:</i> {dist_long:.2f}\n"
        f"<i>DTW کوتاه:</i> {dist_short:.2f}"
    )
    all_telegram_parts.append(msg)

    # ═══════════════════════════════════════════════════════════
    # رسم نمودار ساده: الگو و بهترین تطابق
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # نمودار الگو (بلند)
    axes[0].plot(pattern_dates, pattern_macd_long, label='MACD', color='purple', linewidth=2)
    axes[0].set_title(f'Pattern: {crypto_symbol} ({tf_input})')
    axes[0].legend()
    axes[0].grid(True)

    # نمودار بهترین تطابق
    color = symbol_colors.get(sym, 'gray')
    if search_interval == "1d":
        end_date = start_date + pd.DateOffset(days=n_candles - 1)
    else:
        end_date = start_date + pd.DateOffset(weeks=n_candles - 1)
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_candles)
    axes[1].plot(date_range, win_macd, color=color, linewidth=2.5, label='MACD')
    axes[1].set_title(f'Best Match: {sym}\nScore: {combined_score:.3f} | {start_date.strftime("%Y-%m-%d")}')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f'Best match for {crypto_symbol} pattern (MACD)')
    plt.tight_layout()
    plt.savefig(f"pattern_plot_{crypto_symbol}.png")
    plt.close(fig)

if all_telegram_parts:
    print("\nارسال همهٔ نتایج به تلگرام...")
    full_message = "\n\n".join(all_telegram_parts)
    send_telegram_long_message(full_message)
    print("ارسال انجام شد.")
else:
    print("\nهیچ پیامی برای ارسال وجود ندارد.")

print("\n\nپایان تحلیل تمام نمادها.")
