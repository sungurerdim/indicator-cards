#!/usr/bin/env python3
"""The 33 indicator state series behind the dataset.

Each entry produces, per candle, a direction state (+1 up / -1 down / 0 warmup)
and a continuous value (used for the rank-correlation IC). States use only
closed candles — no look-ahead. The math is the exact code the published
dataset was built with; changing anything here changes the dataset.

IDs follow one convention: name_param_variant (ema_20_pos, ema_cross_10_50,
supertrend_10_2p5, rsi_14_midline…); classics with fixed defaults stay bare
(macd_hist, obv_slope, psar).
"""
import numpy as np
import pandas as pd


def atr(df, n=14):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values


def tema(x, n):
    e1 = ema(x, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def dema(x, n):
    e1 = ema(x, n); e2 = ema(e1, n)
    return 2 * e1 - e2


def hma(x, n):
    wma = lambda s, k: pd.Series(s).rolling(k).apply(
        lambda w: np.dot(w, np.arange(1, len(w) + 1)) / np.arange(1, len(w) + 1).sum(), raw=True).values
    return wma(2 * wma(x, n // 2) - wma(x, n), max(2, int(np.sqrt(n))))


def zlema(x, n):
    lag = (n - 1) // 2
    adj = 2 * np.asarray(x) - np.roll(x, lag)
    adj[:lag] = x[:lag]
    return ema(adj, n)


def vwma(df, n):
    pv = pd.Series(df['close'].values * df['volume'].values).rolling(n).sum()
    vv = pd.Series(df['volume'].values).rolling(n).sum()
    return (pv / vv.replace(0, np.nan)).values


MA_KINDS = {'tema': lambda d, n: tema(d['close'].values, n),
            'dema': lambda d, n: dema(d['close'].values, n),
            'hma': lambda d, n: hma(d['close'].values, n),
            'zlema': lambda d, n: zlema(d['close'].values, n),
            'vwma': lambda d, n: vwma(d, n)}


def supertrend_dir(df, n=10, mult=3.0):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    a = atr(df, n)
    mid = (h + l) / 2
    ub, lb = mid + mult * a, mid - mult * a
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
    d = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i] > fub[i-1]: d[i] = 1
        elif c[i] < flb[i-1]: d[i] = -1
        else: d[i] = d[i-1]
    return d


def _ma_state(d, m):
    c = d['close'].values
    val = (c - m) / np.where(m > 0, m, np.nan)
    st = np.sign(np.nan_to_num(val))
    return st, val


def _cross_state(d, f, s):
    c = d['close'].values
    diff = ema(c, f) - ema(c, s)
    val = diff / c
    st = np.sign(np.nan_to_num(val)); st[:s] = 0
    return st, val


def _st_state(d, n, m):
    st = supertrend_dir(d, n, m); st[:n + 2] = 0
    return st, st.astype(float)


def _rsi_state(d, n):
    r = rsi(d['close'].values, n)
    return np.sign(r - 50), r - 50


def _roc_state(d, n):
    c = d['close'].values
    v = np.zeros(len(c)); v[n:] = c[n:] / c[:-n] - 1
    st = np.sign(v); st[:n] = 0
    return st, v


def _macd_state(d):
    c = d['close'].values
    h = ema(c, 12) - ema(c, 26); h = h - ema(h, 9)
    st = np.sign(h); st[:35] = 0
    return st, h / c


def _obv_state(d):
    c, v = d['close'].values, d['volume'].values
    dirn = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(dirn * v)
    sl = obv - pd.Series(obv).rolling(20).mean().values
    st = np.sign(np.nan_to_num(sl)); st[:20] = 0
    return st, sl


def _cci_state(d, n=20):
    tp = (d['high'].values + d['low'].values + d['close'].values) / 3
    ma = pd.Series(tp).rolling(n).mean().values
    md = pd.Series(np.abs(tp - ma)).rolling(n).mean().values
    v = (tp - ma) / np.where(md > 0, 0.015 * md, np.nan)
    st = np.sign(np.nan_to_num(v))
    return st, v


def _willr_state(d, n=14):
    hh = pd.Series(d['high']).rolling(n).max().values
    ll = pd.Series(d['low']).rolling(n).min().values
    v = -100 * (hh - d['close'].values) / np.where((hh - ll) > 0, hh - ll, np.nan)
    st = np.sign(np.nan_to_num(v + 50))   # -50 midline
    return st, v


def _aroon_state(d, n=25):
    h = pd.Series(d['high']); l = pd.Series(d['low'])
    up = 100 * h.rolling(n + 1).apply(lambda w: float(np.argmax(w)) / n, raw=True).values
    dn = 100 * l.rolling(n + 1).apply(lambda w: float(np.argmin(w)) / n, raw=True).values
    v = up - dn
    st = np.sign(np.nan_to_num(v))
    return st, v


def _dc_mid_state(d, n=20):
    hh = pd.Series(d['high']).rolling(n).max().values
    ll = pd.Series(d['low']).rolling(n).min().values
    mid = (hh + ll) / 2
    v = (d['close'].values - mid) / d['close'].values
    st = np.sign(np.nan_to_num(v)); st[:n] = 0
    return st, v


def _psar_state(d, af=0.02, af_max=0.2):
    h, l = d['high'].values, d['low'].values
    n = len(h); st = np.zeros(n)
    if n < 3: return st, st
    up, sar, ep, a = True, l[0], h[0], af
    for i in range(1, n):
        sar = sar + a * (ep - sar)
        if up:
            if l[i] < sar: up, sar, ep, a = False, ep, l[i], af
            elif h[i] > ep: ep, a = h[i], min(a + af, af_max)
        else:
            if h[i] > sar: up, sar, ep, a = True, ep, h[i], af
            elif l[i] < ep: ep, a = l[i], min(a + af, af_max)
        st[i] = 1 if up else -1
    return st, st.astype(float)


def state_series():
    """(id, fn(df) -> (state ±1/0, continuous value)) for all 33 states."""
    S = []
    for n in (20, 50, 100, 200):
        S.append((f"ema_{n}_pos", lambda d, n=n: _ma_state(d, ema(d['close'].values, n))))
    for kind in ('tema', 'dema', 'hma', 'zlema', 'vwma'):
        for n in (20, 50):
            S.append((f"{kind}_{n}_pos", lambda d, k=kind, n=n: _ma_state(d, MA_KINDS[k](d, n))))
    for f, s in ((10, 50), (20, 100), (50, 200)):
        S.append((f"ema_cross_{f}_{s}", lambda d, a=f, b=s: _cross_state(d, a, b)))
    for n, m in ((7, 2.0), (10, 3.0), (14, 4.0)):
        mtag = str(m).rstrip('0').rstrip('.').replace('.', 'p')
        S.append((f"supertrend_{n}_{mtag}", lambda d, a=n, b=m: _st_state(d, a, b)))
    for n in (7, 14, 21):
        S.append((f"rsi_{n}_midline", lambda d, n=n: _rsi_state(d, n)))
    for n in (24, 48, 96):
        S.append((f"roc_{n}", lambda d, n=n: _roc_state(d, n)))
    S.append(("macd_hist", _macd_state))
    S.append(("obv_slope", _obv_state))
    S.append(("cci_20", _cci_state))
    S.append(("willr_14", _willr_state))
    S.append(("aroon_25", _aroon_state))
    S.append(("donchian_mid_20", _dc_mid_state))
    S.append(("psar", _psar_state))
    return S
