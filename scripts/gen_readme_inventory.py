#!/usr/bin/env python3
"""README'nin veriye bağlı bölümlerini data/dataset.json'dan yeniden üretir:
AUTO-SCOPE (başlıktaki kapsam cümlesi) + AUTO-INVENTORY (envanter bloğu) +
AUTO-CAVEAT (hücre-sayısı uyarısı). Kapsam sayıları elle yazılmaz."""
import json
import re

TF_MIN = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}


def humanize(minutes):
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def group(i):
    if re.match(r'^(ema|tema|dema|hma|zlema|vwma)_\d+_pos$', i):
        return 'MA position (price vs MA)'
    if i.startswith('ema_cross_'):
        return 'MA cross'
    if re.match(r'^(supertrend|psar|aroon|donchian)', i):
        return 'Trend system'
    if re.match(r'^(rsi|roc|macd|cci|willr)', i):
        return 'Oscillator'
    if i.startswith('obv'):
        return 'Volume-based'
    return 'Other'


d = json.load(open('data/dataset.json', encoding='utf-8'))
inds = sorted({c['i'] for c in d['cards']})
tfs = sorted({c['tf'] for c in d['cards']}, key=list(TF_MIN).index)
hs = sorted({c['h'] for c in d['cards']})
u = d['universe']
meas = d['meta']['measurements']
yrs = max(x['bars'] for x in u) * 5 / (60 * 24 * 365)

scope = (f"This dataset scores the raw directional accuracy of "
         f"**{len(inds)} popular technical indicator states** on "
         f"**{len(u)} of the highest-volume Binance USDT-M perpetual futures pairs**, "
         f"across **{len(tfs)} timeframes** ({tfs[0]} → {tfs[-1]}) and "
         f"**{len(hs)} forecast horizons** — up to **{yrs:.0f} years** of history, "
         f"**{meas:,} individual measurements**.")

groups = {}
for i in inds:
    groups.setdefault(group(i), []).append(i)
ind_lines = [f"  - *{g}* ({len(v)}): " + ', '.join(f'`{i}`' for i in v)
             for g, v in sorted(groups.items())]

hor = ' · '.join(
    f"{'/'.join(humanize(h * TF_MIN[tf]) for h in hs)} @{tf}" for tf in tfs)

inventory = '\n'.join([
    f"### Tested inventory (auto-generated from `data/dataset.json`)",
    f"- **Indicator states ({len(inds)}):**",
    *ind_lines,
    f"- **Timeframes ({len(tfs)}):** {', '.join(tfs)} — 5m is the raw feed; "
    f"everything above is an exact resample of it",
    f"- **Horizons ({len(hs)}):** {' and '.join(map(str, hs))} bars ahead "
    f"(in wall-clock time: {hor})",
    f"- **Universe ({len(u)} pairs):** " + ', '.join(x['s'] for x in u),
    f"- **History:** {min(x['from'] for x in u)} → {max(x['to'] for x in u)} · "
    f"{sum(x['bars'] for x in u):,} five-minute bars · {meas:,} measurements",
])

caveat = (f"Results are period-specific — edges decay. A few cells out of "
          f"{len(inds)} × {len(tfs)} × {len(hs)} will look good by pure chance; "
          f"judge patterns, not single cells. **This is not financial advice "
          f"and not a trading strategy.**")

r = open('README.md', encoding='utf-8').read()
for tag, body in (('SCOPE', scope), ('INVENTORY', inventory), ('CAVEAT', caveat)):
    pat = rf'(<!-- AUTO-{tag}:START[^>]*-->\n).*?(<!-- AUTO-{tag}:END -->)'
    r, n = re.subn(pat, lambda m: m.group(1) + body + '\n' + m.group(2), r, flags=re.S)
    assert n == 1, f"AUTO-{tag} bloğu README'de yok"
open('README.md', 'w', encoding='utf-8').write(r)
print(f"README güncellendi: {len(inds)} gösterge · {len(tfs)} TF · {len(hs)} ufuk · {meas:,} ölçüm")
