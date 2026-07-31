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

    cards, total, n_ind, n_coin = [], 0, 0, 0
    variants = []
    for variant, path in (('raw', STORE),
                          ('ha', os.path.join(BASE, 'build', 'skill_ha.csv'))):
        if not os.path.exists(path):
            continue
        variants.append(variant)
        sk = pd.read_csv(path)
        sk['key_extra'] = sk['key_extra'].fillna('').astype(str)
        # Artımlı store eski konfigürasyonların satırlarını da saklar; yayına
        # her mantıksal ölçümün YALNIZ en yeni hali girer.
        sk = sk.drop_duplicates(subset=['symbol', 'tf', 'indicator', 'key_extra'],
                                keep='last')
        sk['sym'] = sk['symbol'].str.replace('/USDT', '', regex=False)
        total += len(sk)
        if variant == 'raw':
            n_ind = int(sk['indicator'].nunique())
            n_coin = int(sk['symbol'].nunique())
        for (ind, tf, h), g in sk.groupby(['indicator', 'tf', 'key_extra']):
            rows = [[r['sym'], round(float(r['edge']), 2), round(float(r['ic']), 4),
                     round(float(r['hit']), 2), int(r['n'])]
                    for _, r in g.sort_values('edge', ascending=False).iterrows()]
            cards.append({'i': ind, 'tf': tf, 'h': int(h.lstrip('h') or 6),
                          'v': variant, 'rows': rows})

    out = {
        'meta': {
            'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'coins': n_coin,
            'indicators': n_ind,
            'measurements': int(total),
            'variants': variants,
            'ha_note': 'ha cards: states from Heikin Ashi-smoothed candles; '
                       'forward returns always from real prices.',
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
