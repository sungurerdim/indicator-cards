#!/usr/bin/env python3
"""Build data/dataset.json from the measurement store (build/skill.csv) and
the cache inventory. Then run embed_data.py and gen_readme_inventory.py to
push the numbers into the page and the README.

Full chain:
    python3 scripts/fetch_candles.py --universe 50 1460
    python3 scripts/measure.py
    python3 scripts/build_dataset.py
    python3 scripts/embed_data.py
    python3 scripts/gen_readme_inventory.py
"""
import glob
import json
import os
from datetime import datetime, timezone

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache')
STORE = os.path.join(BASE, 'build', 'skill.csv')
OUT = os.path.join(BASE, 'data', 'dataset.json')


def main():
    universe = []
    for f in sorted(glob.glob(os.path.join(CACHE, '*_5m.csv'))):
        d = pd.read_csv(f, usecols=['timestamp'])
        universe.append({
            's': os.path.basename(f)[:-len('_5m.csv')].replace('-USDT', ''),
            'from': datetime.fromtimestamp(d['timestamp'].iloc[0] / 1000).strftime('%Y-%m-%d'),
            'to': datetime.fromtimestamp(d['timestamp'].iloc[-1] / 1000).strftime('%Y-%m-%d'),
            'bars': int(len(d))})

    sk = pd.read_csv(STORE)
    sk['key_extra'] = sk['key_extra'].fillna('').astype(str)
    sk['sym'] = sk['symbol'].str.replace('/USDT', '', regex=False)
    cards = []
    for (ind, tf, h), g in sk.groupby(['indicator', 'tf', 'key_extra']):
        rows = [[r['sym'], round(float(r['edge']), 2), round(float(r['ic']), 4),
                 round(float(r['hit']), 2), int(r['n'])]
                for _, r in g.sort_values('edge', ascending=False).iterrows()]
        cards.append({'i': ind, 'tf': tf, 'h': int(h.lstrip('h') or 6), 'rows': rows})

    out = {
        'meta': {
            'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'coins': int(sk['symbol'].nunique()),
            'indicators': int(sk['indicator'].nunique()),
            'measurements': int(len(sk)),
            'source': 'Binance USDT-M perpetual futures, 5m OHLCV',
            'tf_note': 'All timeframes above 5m are exact resamples of the raw 5m candles.',
            'row_schema': ['symbol', 'edge_pp', 'ic_spearman', 'hit_pct', 'n_bars'],
        },
        'universe': universe,
        'cards': cards,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print(f"Dataset: {OUT} ({os.path.getsize(OUT) >> 10} KB, {len(cards)} cards)")


if __name__ == '__main__':
    main()
