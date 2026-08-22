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
n_candles = 150
tf_input = "30m"
search_interval = "1d"

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

def norm(val, mn, rng):
    return (val - mn) / rng if rng != 0 else 0.0

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

    bb_period = 20
    bb_std = 2
    bb_mid = close_series.rolling(bb_period).mean()
    bb_std_val = close_series.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    bb_mid = bb_mid.dropna()
    bb_upper = bb_upper.reindex(bb_mid.index)
    bb_lower = bb_lower.reindex(bb_mid.index)

    if len(bb_mid) < n_candles:
        print(f"  تعداد کندل‌های معتبر باند بولینگر ({len(bb_mid)}) کمتر از {n_candles} است. رد می‌شود.")
        continue

    pattern_upper = bb_upper.iloc[-n_candles:].values
    pattern_lower = bb_lower.iloc[-n_candles:].values
    pattern_dates = bb_upper.index[-n_candles:]
    pattern_start_date = pattern_dates[0].normalize().tz_localize(None)

    pattern_norm_upper = z_norm(pattern_upper)
    pattern_norm_lower = z_norm(pattern_lower)

    print(f"الگوی {crypto_symbol} ({tf_input}) با {n_candles} کندل استخراج شد (فقط دو خط).")
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
        if len(close) < bb_period + n_candles:
            return None, []

        mid = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std

        mid = mid.dropna()
        upper = upper.reindex(mid.index)
        lower = lower.reindex(mid.index)

        N = len(mid)
        if N < n_candles:
            return mid, []

        mid_vals = mid.values
        upper_vals = upper.values
        lower_vals = lower.values

        window_size = max(2, int(window_fraction * n_candles))
        raw_matches = []

        for i in range(N - n_candles + 1):
            window_end_date = mid.index[i + n_candles - 1]
            window_end_date_naive = window_end_date.normalize().tz_localize(None)
            if window_end_date_naive >= pattern_start_date:
                continue

            win_upper = upper_vals[i:i + n_candles]
            win_lower = lower_vals[i:i + n_candles]

            win_upper_norm = z_norm(win_upper)
            win_lower_norm = z_norm(win_lower)

            dtw_up = dtw(pattern_norm_upper, win_upper_norm,
                         keep_internals=False,
                         window_type='sakoechiba',
                         window_args={'window_size': window_size}).distance

            dtw_low = dtw(pattern_norm_lower, win_lower_norm,
                          keep_internals=False,
                          window_type='sakoechiba',
                          window_args={'window_size': window_size}).distance

            # فاصله اقلیدسی حذف شد
            raw_matches.append((dtw_up, dtw_low,
                                mid.index[i], win_upper, win_lower))

        return mid, raw_matches

    all_mid_series = {}
    global_raw_matches = []

    for sym in symbols_to_search:
        mid_series, raw_list = search_raw_matches(sym, interval=search_interval, window_fraction=0.5)
        all_mid_series[sym] = mid_series
        if raw_list:
            for item in raw_list:
                global_raw_matches.append((sym,) + item)

    if not global_raw_matches:
        print("هیچ تطابقی برای این الگو یافت نشد. به نماد بعدی می‌رویم.")
        continue

    # استخراج آرایه‌های DTW
    dtw_up_all  = np.array([m[1] for m in global_raw_matches])
    dtw_low_all = np.array([m[2] for m in global_raw_matches])

    ranges = {}
    for name, arr in [("dtw_up", dtw_up_all), ("dtw_low", dtw_low_all)]:
        mn, mx = arr.min(), arr.max()
        rng = mx - mn if mx != mn else 1.0
        ranges[name] = (mn, rng)

    scored_matches = []
    for match in global_raw_matches:
        sym, dtw_up, dtw_low, start_date, w_up, w_low = match

        n_dtw_up = norm(dtw_up, ranges["dtw_up"][0], ranges["dtw_up"][1])
        n_dtw_low = norm(dtw_low, ranges["dtw_low"][0], ranges["dtw_low"][1])

        # امتیاز نهایی میانگین دو DTW
        final_score = (n_dtw_up + n_dtw_low) / 2

        scored_matches.append((final_score, dtw_up, dtw_low,
                               sym, start_date, w_up, w_low))

    symbol_matches = defaultdict(list)
    for m in scored_matches:
        sym = m[3]   # جایگاه جدید نماد
        symbol_matches[sym].append(m)

    selected_per_symbol = []
    for sym, matches in symbol_matches.items():
        matches.sort(key=lambda x: x[0])
        selected = []
        for match in matches:
            start = match[4]   # جایگاه جدید تاریخ شروع
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

    selected_per_symbol.sort(key=lambda x: x[0])
    filtered_matches = [m for m in selected_per_symbol if m[0] < 0.02]

    if not filtered_matches:
        print(f"هیچ تطابقی با امتیاز زیر 0.02 برای {crypto_symbol} یافت نشد.")
        continue

    print(f"\nنتایج با امتیاز < 0.02 برای الگوی {crypto_symbol}:")
    print("─" * 80)
    for idx, (score, dtw_up, dtw_low, sym, start_date, _, _) in enumerate(filtered_matches):
        star = "⭐" if idx == 0 else "  "
        print(f"{star} رتبه {idx+1}: {sym} | امتیاز: {score:.3f} | "
              f"DTW_UP:{dtw_up:.2f} | DTW_LOW:{dtw_low:.2f} | شروع: {start_date.strftime('%Y-%m-%d')}")

        msg = (
            f"📊 <b>الگو:</b> {tf_input} {crypto_symbol}\n"
            f"🪙 <b>نماد:</b> {sym}\n"
            f"📅 <b>تاریخ شروع:</b> {start_date.strftime('%Y-%m-%d')}\n"
            f"⭐ <b>امتیاز:</b> {score:.3f}\n"
            f"<i>DTW_UP:</i> {dtw_up:.2f} | <i>DTW_LOW:</i> {dtw_low:.2f}"
        )
        all_telegram_parts.append(msg)

    n_matches = len(filtered_matches)
    fig, axes = plt.subplots(10, 5, figsize=(25, 50))
    axes = axes.flatten()

    pattern_indices = [0, 24, 49]
    plot_data = [None] * 50
    for idx in pattern_indices:
        plot_data[idx] = ("pattern", None)

    match_idx = 0
    for i in range(50):
        if plot_data[i] is None and match_idx < n_matches:
            plot_data[i] = ("match", match_idx)
            match_idx += 1

    for i in range(50):
        content = plot_data[i]
        ax = axes[i]
        if content is None:
            ax.set_visible(False)
            continue

        kind, data = content
        if kind == "pattern":
            ax.plot(pattern_dates, pattern_upper, label='Upper', color='red', alpha=0.7)
            ax.plot(pattern_dates, pattern_lower, label='Lower', color='green', alpha=0.7)
            title = f'Pattern: {crypto_symbol} ({tf_input})'
            if i == 0: title += ' (start)'
            elif i == 49: title += ' (end)'
            else: title += ' (middle)'
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
        else:
            global_idx = data
            (score, dtw_up, dtw_low, sym, start_date, w_up, w_low) = filtered_matches[global_idx]

            color = symbol_colors.get(sym, 'gray')
            is_best = (global_idx == 0)

            if search_interval == "1d":
                end_date = start_date + pd.DateOffset(days=n_candles - 1)
            else:
                end_date = start_date + pd.DateOffset(weeks=n_candles - 1)
            date_range = pd.date_range(start=start_date, end=end_date, periods=n_candles)

            lw = 2.5 if is_best else 1.5
            ax.plot(date_range, w_up, color=color, linewidth=lw, linestyle='--', label='Upper')
            ax.plot(date_range, w_low, color=color, linewidth=lw, linestyle=':', label='Lower')
            title = f'{sym} #{global_idx+1}'
            if is_best: title += ' ⭐'
            title += f'\nScore:{score:.3f} | {start_date.strftime("%Y-%m-%d")}'
            ax.set_title(title, fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True)

    plt.suptitle(f'تحلیل الگوی {crypto_symbol} - باند بولینگر دوخطی (امتیاز < 0.02)', fontsize=16)
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
