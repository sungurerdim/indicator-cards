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
    # Deliberate variant: Wilder RMA (ewm alpha=1/n), not SMA-ATR (audit 2026-08-02)
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
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
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def dema(x, n):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 2 * e1 - e2


def hma(x, n):
    # Deliberate variant: final smoothing length = int(sqrt(n)) (floor), the
    # common open implementation; not round(sqrt(n))
    wma = lambda s, k: (
        pd.Series(s)
        .rolling(k)
        .apply(
            lambda w: (
                np.dot(w, np.arange(1, len(w) + 1)) / np.arange(1, len(w) + 1).sum()
            ),
            raw=True,
        )
        .values
    )
    return wma(2 * wma(x, n // 2) - wma(x, n), max(2, int(np.sqrt(n))))


def zlema(x, n):
    lag = (n - 1) // 2
    adj = 2 * np.asarray(x) - np.roll(x, lag)
    adj[:lag] = x[:lag]
    return ema(adj, n)


def vwma(df, n):
    pv = pd.Series(df["close"].values * df["volume"].values).rolling(n).sum()
    vv = pd.Series(df["volume"].values).rolling(n).sum()
    return (pv / vv.replace(0, np.nan)).values


MA_KINDS = {
    "tema": lambda d, n: tema(d["close"].values, n),
    "dema": lambda d, n: dema(d["close"].values, n),
    "hma": lambda d, n: hma(d["close"].values, n),
    "zlema": lambda d, n: zlema(d["close"].values, n),
    "vwma": lambda d, n: vwma(d, n),
}


def supertrend_dir(df, n=10, mult=3.0):
    # Seed note: direction starts at +1 (d[0]=1) by convention; state users
    # zero the first n+2 bars (_st_state), so the seed never leaks into states
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = atr(df, n)
    mid = (h + l) / 2
    ub, lb = mid + mult * a, mid - mult * a
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
    d = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i] > fub[i - 1]:
            d[i] = 1
        elif c[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return d


def _warm_state(pair, n):
    """v5 fix: EMA-derived states get an n-bar warmup zero (seed convergence);
    rolling-window MAs already produce NaN warmups and need no zeroing."""
    st, val = pair
    st[:n] = 0
    return st, val


def _ma_state(d, m):
    c = d["close"].values
    val = (c - m) / np.where(m > 0, m, np.nan)
    st = np.sign(np.nan_to_num(val))
    return st, val


def _cross_state(d, f, s):
    c = d["close"].values
    diff = ema(c, f) - ema(c, s)
    val = diff / c
    st = np.sign(np.nan_to_num(val))
    st[:s] = 0
    return st, val


def _st_state(d, n, m):
    st = supertrend_dir(d, n, m)
    st[: n + 2] = 0
    return st, st.astype(float)


def _rsi_state(d, n):
    r = rsi(d["close"].values, n)
    return np.sign(r - 50), r - 50


def _roc_state(d, n):
    c = d["close"].values
    v = np.zeros(len(c))
    v[n:] = c[n:] / c[:-n] - 1
    st = np.sign(v)
    st[:n] = 0
    return st, v


def _macd_state(d):
    c = d["close"].values
    h = ema(c, 12) - ema(c, 26)
    h = h - ema(h, 9)
    st = np.sign(h)
    st[:35] = 0
    return st, h / c


def _obv_state(d):
    c, v = d["close"].values, d["volume"].values
    dirn = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(dirn * v)
    sl = obv - pd.Series(obv).rolling(20).mean().values
    st = np.sign(np.nan_to_num(sl))
    st[:20] = 0
    return st, sl


def _cci_state(d, n=20):
    # Deliberate variant: mean deviation approximated with a rolling mean of
    # |tp - ma| (ma re-used across the window), not the textbook per-window
    # mean absolute deviation — cheaper, direction-equivalent in practice
    tp = (d["high"].values + d["low"].values + d["close"].values) / 3
    ma = pd.Series(tp).rolling(n).mean().values
    md = pd.Series(np.abs(tp - ma)).rolling(n).mean().values
    v = (tp - ma) / np.where(md > 0, 0.015 * md, np.nan)
    st = np.sign(np.nan_to_num(v))
    return st, v


def _willr_state(d, n=14):
    hh = pd.Series(d["high"]).rolling(n).max().values
    ll = pd.Series(d["low"]).rolling(n).min().values
    v = -100 * (hh - d["close"].values) / np.where((hh - ll) > 0, hh - ll, np.nan)
    st = np.sign(np.nan_to_num(v + 50))  # -50 midline
    return st, v


def _aroon_state(d, n=25):
    h = pd.Series(d["high"])
    l = pd.Series(d["low"])
    up = (
        100 * h.rolling(n + 1).apply(lambda w: float(np.argmax(w)) / n, raw=True).values
    )
    dn = (
        100 * l.rolling(n + 1).apply(lambda w: float(np.argmin(w)) / n, raw=True).values
    )
    v = up - dn
    st = np.sign(np.nan_to_num(v))
    return st, v


def _dc_mid_state(d, n=20):
    hh = pd.Series(d["high"]).rolling(n).max().values
    ll = pd.Series(d["low"]).rolling(n).min().values
    mid = (hh + ll) / 2
    v = (d["close"].values - mid) / d["close"].values
    st = np.sign(np.nan_to_num(v))
    st[:n] = 0
    return st, v


def _psar_state(d, af=0.02, af_max=0.2):
    # v5 fix: standard Wilder two-bar clamp — SAR may not enter the range of
    # the previous two bars (identical code in trader_ai_v2 psar_levels)
    h, l = d["high"].values, d["low"].values
    n = len(h)
    st = np.zeros(n)
    if n < 3:
        return st, st
    up, sar, ep, a = True, l[0], h[0], af
    for i in range(1, n):
        sar = sar + a * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, a = False, ep, l[i], af
            elif h[i] > ep:
                ep, a = h[i], min(a + af, af_max)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, a = True, ep, h[i], af
            elif l[i] < ep:
                ep, a = l[i], min(a + af, af_max)
        st[i] = 1 if up else -1
    return st, st.astype(float)


# ---------------------------------------------------------------- extensions
# Second measurement wave: the remaining candidates from the public inventory.
def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def _conv_ma(x, w):
    out = np.full(len(x), np.nan)
    if len(x) >= len(w):
        out[len(w) - 1 :] = np.convolve(x, w[::-1], "valid")
    return out


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    return _conv_ma(np.asarray(x, float), w / w.sum())


def alma(x, n=20, offset=0.85, sigma=6.0):
    m = offset * (n - 1)
    s = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    return _conv_ma(np.asarray(x, float), w / w.sum())


def kama(x, n=10, fast=2, slow=30):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    change = np.abs(x - np.roll(x, n))
    change[:n] = np.nan
    vol = pd.Series(np.abs(np.diff(x, prepend=x[0]))).rolling(n).sum().values
    er = change / np.where(vol > 0, vol, np.nan)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    sc_slow = (2 / (slow + 1)) ** 2
    out[n] = x[n]
    for i in range(n + 1, len(x)):
        s = sc[i] if np.isfinite(sc[i]) else sc_slow
        out[i] = out[i - 1] + s * (x[i] - out[i - 1])
    return out


def t3(x, n=20, a=0.7):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    c1 = -(a**3)
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = 1 + 3 * a + a**3 + 3 * a**2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def mcginley(x, n=20, k=0.6):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if not len(x):
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        ratio = max(x[i] / out[i - 1], 1e-9)
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / (k * n * ratio**4)
    out[:n] = np.nan
    return out


def _tr(df):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def _di(df, n=14):
    h, l = df["high"].values, df["low"].values
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = pd.Series(_tr(df)).ewm(alpha=1 / n, adjust=False).mean().values
    pdi = (
        100
        * pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean().values
        / np.maximum(atr_s, 1e-9)
    )
    ndi = (
        100
        * pd.Series(ndm).ewm(alpha=1 / n, adjust=False).mean().values
        / np.maximum(atr_s, 1e-9)
    )
    return pdi, ndi


def _vortex_state(d, n=14):
    h, l = d["high"].values, d["low"].values
    pl = np.roll(l, 1)
    pl[0] = l[0]
    ph = np.roll(h, 1)
    ph[0] = h[0]
    vmp = pd.Series(np.abs(h - pl)).rolling(n).sum().values
    vmm = pd.Series(np.abs(l - ph)).rolling(n).sum().values
    trs = pd.Series(_tr(d)).rolling(n).sum().values
    v = (vmp - vmm) / np.where(trs > 0, trs, np.nan)
    st = np.sign(np.nan_to_num(v))
    st[:n] = 0
    return st, v


def _ichimoku_tk_state(d, t=9, k=26):
    h = pd.Series(d["high"])
    l = pd.Series(d["low"])
    ten = (h.rolling(t).max() + l.rolling(t).min()).values / 2
    kij = (h.rolling(k).max() + l.rolling(k).min()).values / 2
    v = (ten - kij) / d["close"].values
    st = np.sign(np.nan_to_num(v))
    st[:k] = 0
    return st, v


def _ichimoku_kumo_state(d, t=9, k=26, b=52):
    h = pd.Series(d["high"])
    l = pd.Series(d["low"])
    ten = (h.rolling(t).max() + l.rolling(t).min()).values / 2
    kij = (h.rolling(k).max() + l.rolling(k).min()).values / 2
    spa = np.roll((ten + kij) / 2, k)
    spa[:k] = np.nan
    spb = np.roll((h.rolling(b).max() + l.rolling(b).min()).values / 2, k)
    spb[:k] = np.nan
    c = d["close"].values
    hi = np.maximum(spa, spb)
    lo = np.minimum(spa, spb)
    st = np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0))
    st[np.isnan(hi)] = 0
    v = (c - (spa + spb) / 2) / c
    return st, v


def _adx_di_state(d, n=14):
    pdi, ndi = _di(d, n)
    v = (pdi - ndi) / 100
    st = np.sign(v)
    st[: 2 * n] = 0
    return st, v


def _linreg_state(d, n=20):
    t = np.arange(n, dtype=float)
    w = t - t.mean()
    w /= (w**2).sum()
    c = np.asarray(d["close"].values, float)
    sl = _conv_ma(c, w / 1)  # convolve zaten w'yi ters uygular
    v = sl * n / c
    st = np.sign(np.nan_to_num(v))
    st[:n] = 0
    return st, v


def _stoch_state(d, n=14, sm=3):
    c = d["close"].values
    hh = pd.Series(d["high"]).rolling(n).max().values
    ll = pd.Series(d["low"]).rolling(n).min().values
    kraw = 100 * (c - ll) / np.where((hh - ll) > 0, hh - ll, np.nan)
    kk = pd.Series(kraw).rolling(sm).mean().values
    v = kk - 50
    st = np.sign(np.nan_to_num(v))
    st[: n + sm] = 0
    return st, v


def _stochrsi_state(d, n=14, sm=3):
    r = pd.Series(rsi(d["close"].values, n))
    mn = r.rolling(n).min()
    mx = r.rolling(n).max()
    sr = ((r - mn) / (mx - mn).replace(0, np.nan)).rolling(sm).mean().values
    v = sr - 0.5
    st = np.sign(np.nan_to_num(v))
    st[: 2 * n] = 0
    return st, v


def _mfi_state(d, n=14):
    tp = (d["high"].values + d["low"].values + d["close"].values) / 3
    mf = tp * d["volume"].values
    dtp = np.diff(tp, prepend=tp[0])
    pos = pd.Series(np.where(dtp > 0, mf, 0.0)).rolling(n).sum()
    neg = pd.Series(np.where(dtp < 0, mf, 0.0)).rolling(n).sum()
    mfi = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).values
    v = mfi - 50
    st = np.sign(np.nan_to_num(v))
    st[:n] = 0
    return st, v


def _tsi_state(d, r=25, s=13):
    m = np.diff(d["close"].values, prepend=d["close"].values[0])
    num = ema(ema(m, r), s)
    den = ema(ema(np.abs(m), r), s)
    v = 100 * num / np.where(den > 0, den, np.nan)
    st = np.sign(np.nan_to_num(v))
    st[: r + s] = 0
    return st, v


def _trix_state(d, n=15):
    c = d["close"].values
    e3 = ema(ema(ema(c, n), n), n)
    v = np.zeros(len(c))
    v[1:] = e3[1:] / e3[:-1] - 1
    st = np.sign(v)
    st[: 3 * n] = 0
    return st, v


def _kst_state(d):
    c = d["close"].values

    def roc(n):
        r = np.zeros(len(c))
        r[n:] = c[n:] / c[:-n] - 1
        return r

    k = (
        sma(roc(10), 10)
        + 2 * sma(roc(15), 10)
        + 3 * sma(roc(20), 10)
        + 4 * sma(roc(30), 15)
    )
    st = np.sign(np.nan_to_num(k))
    st[:45] = 0
    return st, k


def _ultosc_state(d, n1=7, n2=14, n3=28):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    bp = c - np.minimum(l, pc)
    tr = np.maximum(h, pc) - np.minimum(l, pc)

    def avg(n):
        return (
            pd.Series(bp).rolling(n).sum()
            / pd.Series(tr).rolling(n).sum().replace(0, np.nan)
        ).values

    uo = 100 * (4 * avg(n1) + 2 * avg(n2) + avg(n3)) / 7
    v = uo - 50
    st = np.sign(np.nan_to_num(v))
    st[:n3] = 0
    return st, v


def _ao_state(d):
    mp = (d["high"].values + d["low"].values) / 2
    v = sma(mp, 5) - sma(mp, 34)
    st = np.sign(np.nan_to_num(v))
    st[:34] = 0
    return st, np.nan_to_num(v) / d["close"].values


def _fisher_state(d, n=9):
    mp = (d["high"].values + d["low"].values) / 2
    mn = pd.Series(mp).rolling(n).min().values
    mx = pd.Series(mp).rolling(n).max().values
    raw = 2 * ((mp - mn) / np.where((mx - mn) > 0, mx - mn, np.nan) - 0.5)
    raw = np.clip(np.nan_to_num(raw), -0.999, 0.999)
    val = np.zeros(len(mp))
    fish = np.zeros(len(mp))
    for i in range(1, len(mp)):
        val[i] = 0.66 * raw[i] + 0.34 * val[i - 1]
        x = np.clip(val[i], -0.999, 0.999)
        fish[i] = 0.5 * np.log((1 + x) / (1 - x)) + 0.5 * fish[i - 1]
    st = np.sign(fish)
    st[:n] = 0
    return st, fish


def _cmf_state(d, n=20):
    h, l, c, v = (
        d["high"].values,
        d["low"].values,
        d["close"].values,
        d["volume"].values,
    )
    mfm = ((c - l) - (h - c)) / np.where((h - l) > 0, h - l, np.nan)
    mfv = np.nan_to_num(mfm) * v
    cmf = (
        pd.Series(mfv).rolling(n).sum()
        / pd.Series(v).rolling(n).sum().replace(0, np.nan)
    ).values
    st = np.sign(np.nan_to_num(cmf))
    st[:n] = 0
    return st, cmf


def _adl_state(d, n=20):
    h, l, c, v = (
        d["high"].values,
        d["low"].values,
        d["close"].values,
        d["volume"].values,
    )
    mfm = ((c - l) - (h - c)) / np.where((h - l) > 0, h - l, np.nan)
    adl = np.cumsum(np.nan_to_num(mfm) * v)
    sl = adl - pd.Series(adl).rolling(n).mean().values
    st = np.sign(np.nan_to_num(sl))
    st[:n] = 0
    return st, sl


def _force_state(d, n=13):
    c, v = d["close"].values, d["volume"].values
    fi = np.diff(c, prepend=c[0]) * v
    e = ema(fi, n)
    st = np.sign(e)
    st[:n] = 0
    return st, e


def _eom_state(d, n=14):
    h, l, v = d["high"].values, d["low"].values, d["volume"].values
    mid = (h + l) / 2
    dm = np.diff(mid, prepend=mid[0])
    box = (v / 1e8) / np.maximum(h - l, 1e-9)
    # v5 fix: zero-volume bars have no meaningful box ratio — the old
    # max(box, 1e-12) floor turned them into astronomic spikes that dominated
    # the rolling mean; NaN propagates through the window → state 0 there
    ratio = np.where(v > 0, dm / np.maximum(box, 1e-12), np.nan)
    eom = pd.Series(ratio).rolling(n).mean().values
    st = np.sign(np.nan_to_num(eom))
    st[:n] = 0
    return st, eom


def _vwap_pos_state(d):
    """Weekly-anchored VWAP relative position. Deliberate variant: week
    boundary = epoch//7d → resets Thursday 00:00 UTC (not ISO Monday).
    v5 fix: bars with zero cumulative volume have no VWAP — state 0 there
    (the old code collapsed VWAP to ~0 and emitted a spurious +1)."""
    ts = d["timestamp"].values
    week = ts // (7 * 86400_000)
    tp = (d["high"].values + d["low"].values + d["close"].values) / 3
    v = d["volume"].values
    vwap = np.zeros(len(v))
    v_run = np.zeros(len(v))
    run_pv = run_v = 0.0
    for i in range(len(v)):
        if i and week[i] != week[i - 1]:
            run_pv = run_v = 0.0
        run_pv += tp[i] * v[i]
        run_v += v[i]
        vwap[i] = run_pv / max(run_v, 1e-9)
        v_run[i] = run_v
    c = d["close"].values
    val = np.where(v_run > 0, (c - vwap) / np.where(vwap > 0, vwap, np.nan), np.nan)
    st = np.sign(np.nan_to_num(val))
    return st, val


def _ext_states():
    S = []
    for n in (50, 200):
        S.append(
            (f"sma_{n}_pos", lambda d, n=n: _ma_state(d, sma(d["close"].values, n)))
        )
    for n in (20, 50):
        S.append(
            (f"wma_{n}_pos", lambda d, n=n: _ma_state(d, wma(d["close"].values, n)))
        )
    S.append(("kama_10_pos", lambda d: _ma_state(d, kama(d["close"].values))))
    # v5 fix: t3 is a 6×EMA chain — n-bar warmup zero like other EMA-derived MAs
    S.append(
        ("t3_20_pos", lambda d: _warm_state(_ma_state(d, t3(d["close"].values)), 20))
    )
    S.append(("alma_20_pos", lambda d: _ma_state(d, alma(d["close"].values))))
    S.append(("mcginley_20_pos", lambda d: _ma_state(d, mcginley(d["close"].values))))
    S.append(("vortex_14", _vortex_state))
    S.append(("ichimoku_tk", _ichimoku_tk_state))
    S.append(("ichimoku_kumo", _ichimoku_kumo_state))
    S.append(("adx_di_14", _adx_di_state))
    S.append(("linreg_slope_20", _linreg_state))
    S.append(("stoch_14", _stoch_state))
    S.append(("stochrsi_14", _stochrsi_state))
    S.append(("mfi_14", _mfi_state))
    S.append(("tsi_25_13", _tsi_state))
    S.append(("trix_15", _trix_state))
    S.append(("kst", _kst_state))
    S.append(("ultosc", _ultosc_state))
    S.append(("ao", _ao_state))
    S.append(("fisher_9", _fisher_state))
    S.append(("cmf_20", _cmf_state))
    S.append(("adl_slope", _adl_state))
    S.append(("force_13", _force_state))
    S.append(("eom_14", _eom_state))
    S.append(("vwap_pos", _vwap_pos_state))
    return S


def to_heikin_ashi(df):
    """Heikin Ashi transform, for the HA signal variant: states are computed
    on the smoothed candles, forward returns ALWAYS come from real prices."""
    o, h, l, c = (
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
    )
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    out = df.copy()
    out["open"] = ho
    out["close"] = hc
    out["high"] = np.maximum(h, np.maximum(ho, hc))
    out["low"] = np.minimum(l, np.minimum(ho, hc))
    return out


def state_series():
    """(id, fn(df) -> (state ±1/0, continuous value)) for all 33 states."""
    S = []
    for n in (20, 50, 100, 200):
        S.append(
            (
                f"ema_{n}_pos",
                lambda d, n=n: _warm_state(_ma_state(d, ema(d["close"].values, n)), n),
            )
        )
    for kind in ("tema", "dema", "hma", "zlema", "vwma"):
        for n in (20, 50):
            # v5 fix: EMA-derived MAs (tema/dema/zlema) get an n-bar warmup
            # zero; hma/vwma already have rolling-NaN warmups
            if kind in ("tema", "dema", "zlema"):
                S.append(
                    (
                        f"{kind}_{n}_pos",
                        lambda d, k=kind, n=n: _warm_state(
                            _ma_state(d, MA_KINDS[k](d, n)), n
                        ),
                    )
                )
            else:
                S.append(
                    (
                        f"{kind}_{n}_pos",
                        lambda d, k=kind, n=n: _ma_state(d, MA_KINDS[k](d, n)),
                    )
                )
    for f, s in ((10, 50), (20, 100), (50, 200)):
        S.append((f"ema_cross_{f}_{s}", lambda d, a=f, b=s: _cross_state(d, a, b)))
    for n, m in ((7, 2.0), (10, 3.0), (14, 4.0)):
        mtag = str(m).rstrip("0").rstrip(".").replace(".", "p")
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
    S += _ext_states()
    return S
