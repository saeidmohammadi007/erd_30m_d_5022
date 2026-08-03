import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # برای محیط‌های بدون سرور (Actions/Codespaces headless)
import matplotlib.pyplot as plt
from dtw import dtw
import ccxt
from collections import defaultdict
import requests

# ═══════════════════════════════════════════════════════════
# ۰. تنظیمات ثابت
# ═══════════════════════════════════════════════════════════
n_candles = 85
tf_input = "30m"
search_interval = "1d"

W_DTW_MACD = 0.5
W_EUC_MACD = 0.5
W_DTW_SMA  = 0.5
W_EUC_SMA  = 0.5
W_MACD     = 0.5
W_SMA      = 0.5

# ═══════════ تنظیمات تلگرام (خواندن از متغیر محیطی) ═══════════
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8782544489:AAGE52tiaHi8IOmf0n9xxH0oZO0OoNyFJv8")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "6146445006")

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
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

crypto_symbols = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD",
    "VET-USD", "LINK-USD", "SHIB-USD", "DOGE-USD", "ADA-USD",
    "SAND-USD", "AR-USD", "HBAR-USD", "IOTA-USD", "TRX-USD"
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

# ═══════════════════════════════════════════════════════════
# ۱. توابع کمکی
# ═══════════════════════════════════════════════════════════
def fetch_lbank_data(yahoo_symbol, tf):
    print(f"Yahoo داده‌ای برای {yahoo_symbol} ندارد. دریافت از LBank...")
    base = yahoo_symbol.upper().replace("-USD", "").replace("/USD", "")
    lbank_pair = f"{base}/USDT"
    tf_map = {
        "5m": "5m", "30m": "30m", "1h": "1h",
        "4h": "4h", "1d": "1d", "1wk": "1w"
    }
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

# ═══════════════════════════════════════════════════════════
# ۲. جمع‌آوری همهٔ پیام‌ها برای ارسال یکجا
# ═══════════════════════════════════════════════════════════
all_telegram_parts = []

# ═══════════════════════════════════════════════════════════
# ۳. حلقه اصلی برای هر نماد الگو
# ═══════════════════════════════════════════════════════════
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

    if len(close_series) < n_candles:
        print(f"  تعداد کندل‌های {tf_input} ({len(close_series)}) کمتر از {n_candles} است. رد می‌شود.")
        continue

    ema_fast = close_series.ewm(span=12, adjust=False).mean()
    ema_slow = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sma9 = close_series.rolling(9).mean()

    pattern_macd = macd_line.iloc[-n_candles:].values
    pattern_sma  = sma9.iloc[-n_candles:].values
    pattern_dates = macd_line.index[-n_candles:]
    pattern_start_date = pattern_dates[0].normalize().tz_localize(None)

    pattern_norm_macd = z_norm(pattern_macd)
    pattern_norm_sma  = z_norm(pattern_sma)

    print(f"الگوی {crypto_symbol} ({tf_input}) با {n_candles} کندل استخراج شد.")
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

        ema_f = close.ewm(span=12, adjust=False).mean()
        ema_s = close.ewm(span=26, adjust=False).mean()
        macd = ema_f - ema_s
        sma  = close.rolling(9).mean()

        macd = macd.dropna()
        N = len(macd)
        if N < n_candles:
            return macd, []

        macd_vals = macd.values
        sma_vals  = sma.values

        window_size = max(2, int(window_fraction * n_candles))
        raw_matches = []

        start_i = 8
        for i in range(start_i, N - n_candles + 1):
            window_end_date = macd.index[i + n_candles - 1]
            if window_end_date >= pattern_start_date:
                continue

            win_macd = macd_vals[i:i + n_candles]
            win_sma  = sma_vals[i:i + n_candles]
            if np.isnan(win_sma).any():
                continue

            win_macd_norm = z_norm(win_macd)
            win_sma_norm  = z_norm(win_sma)

            alignment_macd = dtw(pattern_norm_macd, win_macd_norm,
                                 keep_internals=False,
                                 window_type='sakoechiba',
                                 window_args={'window_size': window_size})
            dtw_macd_dist = alignment_macd.distance
            euc_macd_dist  = np.sqrt(np.sum((pattern_norm_macd - win_macd_norm) ** 2))

            alignment_sma = dtw(pattern_norm_sma, win_sma_norm,
                                keep_internals=False,
                                window_type='sakoechiba',
                                window_args={'window_size': window_size})
            dtw_sma_dist = alignment_sma.distance
            euc_sma_dist  = np.sqrt(np.sum((pattern_norm_sma - win_sma_norm) ** 2))

            raw_matches.append((dtw_macd_dist, euc_macd_dist,
                                dtw_sma_dist, euc_sma_dist,
                                macd.index[i], win_macd))

        return macd, raw_matches

    all_macd_series = {}
    global_raw_matches = []

    for sym in symbols_to_search:
        macd_series, raw_list = search_raw_matches(sym, interval=search_interval, window_fraction=0.5)
        all_macd_series[sym] = macd_series
        if raw_list:
            for item in raw_list:
                dtw_m, euc_m, dtw_s, euc_s, start_date, win_macd = item
                global_raw_matches.append((sym, dtw_m, euc_m, dtw_s, euc_s, start_date, win_macd))

    if not global_raw_matches:
        print("هیچ تطابقی برای این الگو یافت نشد. به نماد بعدی می‌رویم.")
        continue

    dtw_macd_all = np.array([m[1] for m in global_raw_matches])
    euc_macd_all = np.array([m[2] for m in global_raw_matches])
    dtw_sma_all  = np.array([m[3] for m in global_raw_matches])
    euc_sma_all  = np.array([m[4] for m in global_raw_matches])

    min_dtw_macd, max_dtw_macd = dtw_macd_all.min(), dtw_macd_all.max()
    min_euc_macd, max_euc_macd = euc_macd_all.min(), euc_macd_all.max()
    min_dtw_sma,  max_dtw_sma  = dtw_sma_all.min(),  dtw_sma_all.max()
    min_euc_sma,  max_euc_sma  = euc_sma_all.min(),  euc_sma_all.max()

    range_dtw_macd = max_dtw_macd - min_dtw_macd if max_dtw_macd != min_dtw_macd else 1
    range_euc_macd = max_euc_macd - min_euc_macd if max_euc_macd != min_euc_macd else 1
    range_dtw_sma  = max_dtw_sma  - min_dtw_sma  if max_dtw_sma  != min_dtw_sma  else 1
    range_euc_sma  = max_euc_sma  - min_euc_sma  if max_euc_sma  != min_euc_sma  else 1

    scored_matches = []
    for (sym, dtw_m, euc_m, dtw_s, euc_s, start_date, win_macd) in global_raw_matches:
        norm_dtw_macd = (dtw_m - min_dtw_macd) / range_dtw_macd
        norm_euc_macd = (euc_m - min_euc_macd) / range_euc_macd
        norm_dtw_sma  = (dtw_s - min_dtw_sma)  / range_dtw_sma
        norm_euc_sma  = (euc_s - min_euc_sma)  / range_euc_sma

        macd_score = (W_DTW_MACD * norm_dtw_macd + W_EUC_MACD * norm_euc_macd) / (W_DTW_MACD + W_EUC_MACD)
        sma_score  = (W_DTW_SMA  * norm_dtw_sma  + W_EUC_SMA  * norm_euc_sma ) / (W_DTW_SMA  + W_EUC_SMA)
        final_score = (W_MACD * macd_score + W_SMA * sma_score) / (W_MACD + W_SMA)

        scored_matches.append((final_score, dtw_m, euc_m, dtw_s, euc_s, sym, start_date, win_macd))

    symbol_matches = defaultdict(list)
    for m in scored_matches:
        sym = m[5]
        symbol_matches[sym].append(m)

    selected_per_symbol = []
    for sym, matches in symbol_matches.items():
        matches.sort(key=lambda x: x[0])
        selected = []
        for match in matches:
            score, dtw_m, euc_m, dtw_s, euc_s, _, start_date, win_macd = match
            overlap = False
            for sel in selected:
                sel_start = sel[6]
                if abs((start_date - sel_start).days) < n_candles:
                    overlap = True
                    break
            if not overlap:
                selected.append(match)
                if len(selected) == 2:
                    break
        selected_per_symbol.extend(selected)

    selected_per_symbol.sort(key=lambda x: x[0])

    all_matches_flat = []
    for rank, (score, dtw_m, euc_m, dtw_s, euc_s, sym, start_date, win_macd) in enumerate(selected_per_symbol, 1):
        all_matches_flat.append((sym, rank, score, dtw_m, euc_m, dtw_s, euc_s, start_date, win_macd))

    filtered_matches = [m for m in all_matches_flat if m[2] < 0.02]

    if not filtered_matches:
        print(f"هیچ تطابقی با امتیاز زیر 0.02 برای {crypto_symbol} یافت نشد.")
        continue

    print(f"\nنتایج با امتیاز < 0.02 برای الگوی {crypto_symbol}:")
    print("─" * 80)

    for idx, (sym, rank, score, dtw_m, euc_m, dtw_s, euc_s, start_date, _) in enumerate(filtered_matches):
        star = "⭐" if idx == 0 else "  "
        print(f"{star} رتبه {idx+1}: {sym} | امتیاز: {score:.3f} | "
              f"DTW_MACD:{dtw_m:.2f} EUC_MACD:{euc_m:.2f} | "
              f"DTW_SMA:{dtw_s:.2f} EUC_SMA:{euc_s:.2f} | شروع: {start_date.strftime('%Y-%m-%d')}")

        msg = (
            f"📊 <b>الگو:</b> {tf_input} {crypto_symbol}\n"
            f"🪙 <b>نماد:</b> {sym}\n"
            f"📅 <b>تاریخ شروع:</b> {start_date.strftime('%Y-%m-%d')}\n"
            f"⭐ <b>امتیاز:</b> {score:.3f}\n"
            f"<i>DTW_MACD:</i> {dtw_m:.2f}  |  <i>EUC_MACD:</i> {euc_m:.2f}\n"
            f"<i>DTW_SMA:</i>  {dtw_s:.2f}  |  <i>EUC_SMA:</i>  {euc_s:.2f}"
        )
        all_telegram_parts.append(msg)

    print("─" * 80)

    n_matches = len(filtered_matches)
    total_plots = n_matches + 3
    print(f"تعداد کل نمودارهای فعال: {total_plots}")

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
            ax.plot(pattern_dates, pattern_macd, label=f'MACD ({tf_input})', color='blue')
            title = f'Pattern: {crypto_symbol} ({tf_input})'
            if i == 0:
                title += ' (start)'
            elif i == 49:
                title += ' (end)'
            else:
                title += ' (middle)'
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True)
        else:
            global_idx = data
            (sym, rank, score,
             dtw_m, euc_m, dtw_s, euc_s,
             start_date, win_macd) = filtered_matches[global_idx]

            macd_series = all_macd_series.get(sym, None)
            color = symbol_colors.get(sym, 'gray')
            is_best = (global_idx == 0)

            if macd_series is not None:
                if search_interval == "1d":
                    end_date = start_date + pd.DateOffset(days=n_candles - 1)
                else:
                    end_date = start_date + pd.DateOffset(weeks=n_candles - 1)

                date_range = pd.date_range(start=start_date, end=end_date, periods=n_candles)

                lw = 2.5 if is_best else 1.5
                ax.plot(date_range, win_macd, color=color, linewidth=lw,
                        label=f'{sym} MACD #{rank}')
                title = f'{sym} #{rank}'
                if is_best:
                    title += ' ⭐'
                title += f'\nScore:{score:.3f} DTW:{dtw_m:.1f} EUC:{euc_m:.1f}'
                title += f'\n{start_date.strftime("%Y-%m-%d")}'
                ax.set_title(title, fontsize=8)
                ax.legend(fontsize=6)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'{sym} data missing', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'{sym} #{rank}', fontsize=8)

    plt.suptitle(f'تحلیل الگوی {crypto_symbol} - فقط امتیاز < 0.02', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"pattern_plot_{crypto_symbol}.png")   # ذخیره به‌جای نمایش
    plt.close(fig)

if all_telegram_parts:
    print("\nارسال همهٔ نتایج به تلگرام...")
    full_message = "\n\n".join(all_telegram_parts)
    send_telegram_long_message(full_message)
    print("ارسال انجام شد.")
else:
    print("\nهیچ پیامی برای ارسال وجود ندارد.")

print("\n\nپایان تحلیل تمام نمادها.")
