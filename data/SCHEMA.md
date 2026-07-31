# dataset.json — schema

Single JSON object with three top-level keys.

## `meta`
| Field | Meaning |
|---|---|
| `generated` | build timestamp (UTC) |
| `coins` | number of pairs in the universe |
| `indicators` | distinct indicator states tested |
| `measurements` | total rows across all cards |
| `source` | raw data source (Binance USDT-M perpetuals, 5m OHLCV) |
| `tf_note` | all higher timeframes are exact resamples of the raw 5m candles |
| `variants` | signal variants present in `cards[]` (`raw`, `ha`) |
| `ha_note` | how the Heikin Ashi variant is computed |
| `row_schema` | column order of every `cards[].rows` entry |

## `universe[]`
One entry per pair: `s` (symbol, quote USDT implied), `from` / `to`
(first/last candle date), `bars` (number of 5-minute candles).

## `cards[]`
One entry per indicator × timeframe × horizon × signal variant:

| Field | Meaning |
|---|---|
| `i` | indicator state id — standardized `name_param[_param][_variant]` (e.g. `ema_20_pos`, `ema_cross_10_50`, `supertrend_10_2p5`, `rsi_14_midline`; `p` = decimal point) |
| `tf` | timeframe (`5m`…`1d`) |
| `h` | forecast horizon in bars (6 or 24) |
| `v` | signal variant: `raw` (states from raw candles) or `ha` (states from Heikin Ashi-smoothed candles; forward returns always from real prices) |
| `rows` | per-coin results, ordered by edge desc; each row follows `meta.row_schema`: `[symbol, edge_pp, ic_spearman, hit_pct, n_bars]` |

### Row fields
- `edge_pp` — direction hit-rate minus 50, percentage points (0 ≈ coin flip)
- `ic_spearman` — Spearman rank correlation of indicator value vs forward return
- `hit_pct` — raw direction hit-rate
- `n_bars` — sample count (cells under 200 samples are dropped at build)

No look-ahead: states are computed on closed candles; forward returns start on
the next bar. License: CC BY 4.0.
