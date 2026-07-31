#!/usr/bin/env python3
"""data/dataset.json'u index.html'e işler (tek dosya/offline): gömülü DATASET
bloğu + veriden türetilen <meta name="description"> (kapsam sayıları elle
yazılmaz — kaynak her zaman veri seti)."""
import json
import re

raw = open('data/dataset.json', encoding='utf-8').read()
d = json.loads(raw)
tfs = {c['tf'] for c in d['cards']}
yrs = max(u['bars'] for u in d['universe']) * 5 / (60 * 24 * 365)
nvar = len(d['meta'].get('variants', ['raw']))
desc = (f"P&L-free accuracy report cards for {d['meta']['indicators']} popular "
        f"technical indicators, measured on {d['meta']['coins']} top-volume "
        f"Binance USDT-M futures pairs across {len(tfs)} timeframes and "
        f"{nvar} signal variants (raw & Heikin Ashi) over up to "
        f"{yrs:.0f} years of 5-minute data. Open data, open method.")

s = open('index.html', encoding='utf-8').read()
s = re.sub(r'(<script id="DATASET" type="application/json">).*?(</script>)',
           lambda m: m.group(1) + raw.replace('</', '<\\/') + m.group(2), s, flags=re.S)
s, n = re.subn(r'(<meta name="description" content=").*?(">)',
               lambda m: m.group(1) + desc + m.group(2), s)
assert n == 1, "description meta bulunamadı"
s, n = re.subn(r'(<meta property="og:description" content=").*?(">)',
               lambda m: m.group(1) + desc + m.group(2), s)
assert n == 1, "og:description meta bulunamadı"

# Statik (JS'siz) gövde kopyaları, sayfanın kendi i18n EN metinlerinden (tek
# doğruluk kaynağı) token'ları veriden çözerek yeniden yazılır — idempotent:
# i18n şablonlarındaki token'lar kaynakta kalır, t() çalışma anında doldurur.
TF_ORDER = ('5m', '15m', '30m', '1h', '4h', '1d')
tf_sorted = sorted(tfs, key=TF_ORDER.index)
tokens = {'{TF_HI}': ', '.join(t for t in tf_sorted if t != '5m'),
          '{NIND}': str(d['meta']['indicators']), '{NTF}': str(len(tfs)),
          '{NH}': str(len({c['h'] for c in d['cards']})),
          '{NCOIN}': str(d['meta']['coins']), '{YRS}': f"{yrs:.0f}"}
for key in ('tagline', 'p_tf', 'm1', 'm2', 'm4'):
    m = re.search(rf"\b{key}:'(.*?)',\n", s)
    assert m, f"i18n EN '{key}' bulunamadı"
    text = m.group(1)
    for k, v in tokens.items():
        text = text.replace(k, v)
    cut = s.index('<script')
    pat = rf'(<(\w+)[^>]*data-i18n="{key}"[^>]*>).*?(</\2>)'
    head, n = re.subn(pat, lambda mm: mm.group(1) + text + mm.group(3),
                      s[:cut], flags=re.S)
    assert n == 1, f"statik '{key}' ögesi bulunamadı ({n})"
    s = head + s[cut:]
open('index.html', 'w', encoding='utf-8').write(s)
print(f"DATASET gömüldü + description güncellendi ({d['meta']['measurements']:,} ölçüm)")
