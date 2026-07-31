#!/usr/bin/env python3
"""Measure the directional accuracy of every indicator state — the raw numbers
behind the dataset.

For each symbol × timeframe × horizon: at every candle close the state
(+1/-1) from `indicators.py` is compared with the sign of the return over the
next N bars, and the state's continuous value is rank-correlated (Spearman IC)
with that forward return. No look-ahead, no costs, no strategy.

Incremental by design: results accumulate in build/skill.csv keyed by
(symbol, tf, config hash, data fingerprint); unchanged work is skipped, so
re-running after a cache refresh only recomputes what changed.

Usage:
    python3 scripts/measure.py                 # full run: 6 TFs × horizons 6,24
    python3 scripts/measure.py --tfs 4h,1d --horizons 24 --symbols BTC/USDT
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from indicators import state_series, to_heikin_ashi

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE, 'cache')
BUILD_DIR = os.path.join(BASE, 'build')

TF_MIN = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}


def cfg_hash(d):
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def data_fingerprint(symbol):
    """Size+mtime fingerprint: if the cached data changed, work is redone."""
    p = os.path.join(CACHE_DIR, symbol.replace('/', '-') + '_5m.csv')
    try:
        st = os.stat(p)
        return f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return "missing"


def cached_symbols():
    return [os.path.basename(f)[:-len('_5m.csv')].replace('-', '/')
            for f in sorted(glob.glob(os.path.join(CACHE_DIR, '*_5m.csv')))]


def load_tf(symbol, tf):
    path = os.path.join(CACHE_DIR, symbol.replace('/', '-') + '_5m.csv')
    if not os.path.exists(path):
        return None
    pq = path[:-4] + '.parquet'
    try:  # parquet sidecar: ~10x faster than CSV; refreshed when CSV is newer
        if os.path.exists(pq) and os.path.getmtime(pq) >= os.path.getmtime(path):
            df = pd.read_parquet(pq)
        else:
            df = pd.read_csv(path)
            df.to_parquet(pq, index=False)
    except Exception:
        df = pd.read_csv(path)
    if tf != '5m':
        step = TF_MIN[tf] * 60_000
        df['bucket'] = (df['timestamp'] // step) * step
        need = TF_MIN[tf] // 5
        df = df.groupby('bucket').agg(
            open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
            close=('close', 'last'), volume=('volume', 'sum'), n=('close', 'size'))
        df = df[df['n'] == need].drop(columns='n').reset_index().rename(columns={'bucket': 'timestamp'})
    return df.reset_index(drop=True)


_PMAP_FN = None


def _pmap_worker(x):
    return _PMAP_FN(x)


def pmap(fn, items):
    """Symbol-level parallelism (fork + module-global pattern: closures don't
    pickle, but fork children inherit the global)."""
    global _PMAP_FN
    import multiprocessing as mp
    jobs = max(1, mp.cpu_count() - 1)
    if jobs == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    _PMAP_FN = fn
    with mp.get_context('fork').Pool(jobs) as pool:
        return pool.map(_pmap_worker, items)


def store_incremental(cfg, tasks, worker, store_name='skill'):
    """Central incremental store build/<store_name>.csv. Key: (symbol, tf,
    cfg_hash, data_fp); rows present for unchanged data are never recomputed."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    path = os.path.join(BUILD_DIR, f'{store_name}.csv')
    ch = cfg_hash(cfg)
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(old):
        old['key_extra'] = old['key_extra'].fillna('').astype(str)
        old = old.drop_duplicates(
            subset=['symbol', 'tf', 'cfg_hash', 'key_extra', 'indicator'], keep='last')
    have = set()
    if len(old):
        for _, r in old.iterrows():
            have.add((r['symbol'], r['tf'], r['cfg_hash'], r['data_fp'],
                      str(r.get('key_extra', '') or '')))
    key_extra = f"h{cfg['horizon']}"
    todo, skipped = [], 0
    for t in tasks:
        k = (t[0], t[1], ch, data_fingerprint(t[0]), key_extra)
        if k in have:
            skipped += 1
        else:
            todo.append(t)
    print(f"[store:h{cfg['horizon']}] {len(tasks)} tasks -> {skipped} up to date, {len(todo)} to compute")
    if todo:
        results = pmap(worker, todo)
        new_rows = []
        for t, lst in zip(todo, results):
            fp = data_fingerprint(t[0])
            for r in (lst or []):
                r.update({'cfg_hash': ch, 'data_fp': fp, 'key_extra': key_extra})
                new_rows.append(r)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if len(old):
                keycols = ['symbol', 'tf', 'cfg_hash', 'key_extra', 'indicator']
                keep = ~old.set_index(keycols).index.isin(new_df.set_index(keycols).index)
                old = old[np.asarray(keep)]
            merged = pd.concat([old, new_df], ignore_index=True) if len(old) else new_df
            merged.to_csv(path, index=False)


def run(symbols, tfs, horizon, use_ha=False):
    from scipy.stats import spearmanr
    states = state_series()

    def one(sym_tf):
        sym, tf = sym_tf
        d = load_tf(sym, tf)
        if d is None or len(d) < 400:
            return []
        c = d['close'].values
        fwd = np.full(len(c), np.nan)
        fwd[:-horizon] = c[horizon:] / c[:-horizon] - 1
        d_sig = to_heikin_ashi(d) if use_ha else d
        out = []
        for name, fn in states:
            try:
                st, val = fn(d_sig)
            except Exception:
                continue
            m = (st != 0) & ~np.isnan(fwd) & ~np.isnan(val)
            if m.sum() < 200:
                continue
            hit = float((np.sign(st[m]) == np.sign(fwd[m])).mean())
            ic = float(spearmanr(val[m], fwd[m]).statistic)
            out.append({'symbol': sym, 'tf': tf, 'indicator': name, 'n': int(m.sum()),
                        'hit': round(hit * 100, 2), 'edge': round((hit - 0.5) * 100, 2),
                        'ic': round(ic, 4)})
        return out

    tasks = [(s, tf) for s in symbols for tf in tfs]
    store_incremental({'horizon': horizon, 'n_ind': len(states)}, tasks, one,
                      store_name='skill_ha' if use_ha else 'skill')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='', help='comma list; default: all cached')
    ap.add_argument('--tfs', default='5m,15m,30m,1h,4h,1d')
    ap.add_argument('--horizons', default='6,24')
    ap.add_argument('--ha', action='store_true',
                    help='Heikin Ashi signal variant (build/skill_ha.csv)')
    args = ap.parse_args()
    symbols = args.symbols.split(',') if args.symbols else cached_symbols()
    if not symbols:
        raise SystemExit("cache/ is empty — run scripts/fetch_candles.py --universe first")
    for h in (int(x) for x in args.horizons.split(',')):
        run(symbols, args.tfs.split(','), h, use_ha=args.ha)
    print(f"Store: {os.path.join(BUILD_DIR, 'skill_ha.csv' if args.ha else 'skill.csv')}")


if __name__ == '__main__':
    main()
