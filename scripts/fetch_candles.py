#!/usr/bin/env python3
"""Reproduce the raw candle data behind this dataset.

Downloads 5-minute OHLCV for Binance USDT-M perpetual pairs via the public API
(no key needed) using ccxt, exactly as the dataset was built. Raw candles are
NOT redistributed in this repo by design: they are ~1 GB, freely available
from the source, and bulk re-publishing exchange feeds is not ours to do —
this script is the reproducibility recipe instead.

Usage:
    pip install ccxt pandas
    python3 scripts/fetch_candles.py BTC/USDT 1460    # one pair, days back
    python3 scripts/fetch_candles.py --universe 50 1460
        # top-50 by 24h quote volume (index/stablecoin pairs excluded),
        # saved into cache/ for scripts/measure.py
"""
import os
import sys
import time

import ccxt
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
EXCLUDE = {'BTCDOMUSDT', 'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT'}


def exchange():
    return ccxt.binance({'enableRateLimit': True,
                         'options': {'defaultType': 'future'}})


def fetch(ex, symbol: str, days: int) -> pd.DataFrame:
    since = ex.milliseconds() - days * 86_400_000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, '5m', since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low',
                                     'close', 'volume'])
    return df.drop_duplicates('timestamp').reset_index(drop=True)


def top_universe(ex, n=50):
    """Top-n TRADING USDT-margined perpetuals by 24h quote volume."""
    info = ex.fapiPublicGetExchangeInfo()
    perps = {s['symbol'] for s in info['symbols']
             if s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL'
             and s['quoteAsset'] == 'USDT' and s['symbol'] not in EXCLUDE}
    tickers = ex.fapiPublicGetTicker24hr()
    rows = [(t['symbol'], float(t['quoteVolume'])) for t in tickers if t['symbol'] in perps]
    rows.sort(key=lambda r: -r[1])
    return [sym[:-4] + '/USDT' for sym, _ in rows[:n]]


def main():
    args = [a for a in sys.argv[1:]]
    ex = exchange()
    if args and args[0] == '--universe':
        n = int(args[1]) if len(args) > 1 else 50
        days = int(args[2]) if len(args) > 2 else 1460
        os.makedirs(CACHE, exist_ok=True)
        syms = top_universe(ex, n)
        for i, sym in enumerate(syms, 1):
            out = os.path.join(CACHE, sym.replace('/', '-') + '_5m.csv')
            print(f"[{i}/{len(syms)}] {sym} ...", flush=True)
            try:
                fetch(ex, sym, days).to_csv(out, index=False)
            except Exception as e:
                print(f"   FAILED {sym}: {e} — waiting 30s, moving on")
                time.sleep(30)
        print(f"Done -> {CACHE}")
        return
    sym = args[0] if args else 'BTC/USDT'
    days = int(args[1]) if len(args) > 1 else 1460
    df = fetch(ex, sym, days)
    out = sym.replace('/', '-') + '_5m.csv'
    df.to_csv(out, index=False)
    print(f'{sym}: {len(df):,} candles -> {out}')


if __name__ == '__main__':
    main()
