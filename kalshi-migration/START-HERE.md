# Kalshi Migration — Start Here

**Read this file first.** It is the entry point for the session that does the
actual migration. Written 2026-08-03.

Status: **research complete, decision made, no production code written yet.**
The grouper still reads DraftKings. Nothing in this folder is wired into
`../group.py`.

## The three documents, in reading order

| File | What it is | Trust level |
|---|---|---|
| `START-HERE.md` | This file. Decisions, remaining work, traps. | Current |
| `SPORTSBOOK-OPTIONS.md` | Why Kalshi over ESPN / Polymarket / The Odds API / DataGolf. Evidence and measurements. | Current (2026-08-03) |
| `ORIENTATION.md` | First exploration pass. Still the best API reference. **Carries a SUPERSEDED banner — two of its conclusions were reversed.** | Read the banner before the body |

Code: `kalshi_odds.py` (working client, needs two changes — see §3).
Raw findings: `api_exploration.json`.

---

## 1. What was decided — do not re-litigate

**Use Kalshi. Use the outright Winner market. Use the ask price.**

- **Kalshi over DraftKings** — documented, stable, unauthenticated, free. Kills
  the annual endpoint-hunt, which is the entire point of this exercise.
- **Start with Winner (`KXPGATOUR-*`). Keep the market type swappable.**
  This reverses `ORIENTATION.md`, which recommended Top 5 as the better pool
  metric. Winner ticks at $0.001 below 10¢
  (`price_level_structure: tapered_deci_cent`); Top 5 / Top 10 / MakeCut are
  `linear_cent`, flat $0.01 — 10× coarser exactly where a golf field lives.

  But the market type is **just a series prefix on the event ticker**
  (`KXPGATOUR-WYC26` vs `KXPGATOP5-WYC26`), and every one of them returns the
  identical market shape. So this is a default, not an architectural commitment
  — the direct analogue of the existing `ODDS_TYPE` knob in `../README.md`.
  Build against Winner, leave the prefix configurable, and swapping later costs
  one string.

  Do update the `../README.md` "Hidden Option" section, which currently
  recommends Top 5 as "another good option" — on Kalshi it is a strictly worse
  signal, even though it still works.
- **Ask, not mid, not bid.** Measured on a live 143-golfer field:

  | | Distinct levels | Sum | Golfers at 0.0000 |
  |---|---|---|---|
  | Ask | 28 | 1.294 | **0** |
  | Bid | 30 | 0.877 | **47** |

  Bid is disqualified — a third of the field would enter the grouper at zero
  probability, and the book sums under 1.0 so it isn't a distribution at all.
  Mid is worse than it looks: with 47 missing bids, "mid" is often just half the
  ask, a number the code invented. Ask is a real quoted price on the tick grid
  and is **always present** (see §2).

- **Liquidity is not a problem.** `ORIENTATION.md` §7 flagged this as the
  blocking unknown. Answer: the thinness was an artifact of capturing a finished
  tournament. Pre-tournament Wyndham had 97/143 two-sided, median spread 0.2¢,
  ~$3.2M volume.

- **ESPN is not an odds source.** Verified against a working NFL control: the
  golf odds route returns `count: 0`. Keep ESPN for field lists and live scoring
  only. (The probe script lives outside this repo, in `espntest/`.)

## 2. Why the ask never has gaps

This matters because it removes a whole error-handling branch.

Measured across 143 active Wyndham markets: `noAsk: 0`, `noBid: 47`,
`noLast: 16`, **`noAskAndNoLast: 0`**.

The reason is structural, not luck. Take Zecheng Dou — $0 volume, never traded.
His orderbook:

```
yes_dollars: []                              <- YES side genuinely empty
no_dollars:  [0.9940 x10060, 0.9970 x21300]  <- NO side deep
```

...yet `yes_ask_dollars` reads `0.0030`. On a binary contract a NO bid at 0.9970
*is* a YES ask at 0.0030. Kalshi synthesizes the YES ask off the NO book. Someone
is always willing to sell NO on a 1000/1 golfer, so the ask side never empties.

**Assert `ask > 0` for every active market and raise if it fails.** Do not add a
silent fallback — per `ORIENTATION.md` §5.1, a quiet zero is exactly how this
project previously produced a confident wrong answer.

## 3. The work remaining

### 3.1 `kalshi_odds.py` — two changes

1. Change the `--price` default from `mid` to `ask` (line ~240), and update the
   `to_golfers()` docstring, which currently calls mid "the fairest read." It
   isn't, for the reason in §1.
2. Carry `custom_strike.golf_competitor` through into the emitted golfer dict.
   It's a stable per-golfer UUID — a far better join key to ESPN than display
   names, and it kills the "Matt / Matthew Fitzpatrick" class of bug.

Note `_f()` already exists and handles the string coercion described in §4.

**No change needed for market-type switching** — `--event` already takes a full
event ticker, so `--event KXPGATOUR-WYC26` and `--event KXPGATOP5-WYC26` both
work today and return the same shape. When you wire this into `group.py`, keep
that as a config value rather than a constant. Two things to watch if you do
switch:

- **The book sums to the market, not to 1.0.** Winner sums ~1.29; a Top 5 book
  sums toward 5, Top 10 toward 10. Any de-vig must normalize by the observed
  field sum, never assume ~1.0. See §3.3.
- **Top N events post ~21h after Winner.** Winner appeared Sunday 23:00Z for
  both tournaments observed; Top 5 / Top 10 appeared Monday ~20:00Z. Confirmed
  again 2026-08-03 16:35Z — Wyndham Winner was live, Top N still absent. If you
  build against Top N you inherit a later, narrower pull window.

### 3.2 `../group.py` — the actual blocker

`list_dk_golf_odds()` expects the DraftKings envelope: top-level `markets[]`
matched on `name`, `selections[]` filtered on `marketId`, reading
`participants[0].name` and `displayOdds.fractional`.

`kalshi_odds.py` emits the *output* of that function — a flat
`[{"golfer_name", "odds"}, ...]` list. So `--write-dk-data` produces a file
`group.py` cannot read.

Add `list_kalshi_golf_odds()` alongside the existing parser and branch on input
shape. Keep the DK path working — you may still want a DK payload for the
comparison in §5.

### 3.3 `odds_to_conditional()` — needs a look

It strips vig assuming a winner book summing just above 1.0. The ask book sums to
**1.294**. Normalizing by the field sum is the right de-vig and is probably all
that's needed, but confirm the exclusion threshold (`implied probability >
1/participants`) still behaves sensibly at that scale. This is a scaling
question, not a correctness bug.

## 4. Traps — every one of these cost time

**Money fields are strings.** `"yes_bid_dollars": "0.0110"`. Summing without
coercion silently concatenates. Not mentioned in `ORIENTATION.md`.

**`web_fetch` serves stale Kalshi cache.** A fetch of
`/events?series_ticker=KXPGATOUR` returned 65 events and omitted `WYC26`,
`ROC26`, `3MO26`; a live client returned 70 including all three. `ORIENTATION.md`
§5.5 says web_fetch "yields empty output for JSON" — that is now wrong, it
returns JSON fine, but *old* JSON. That is the more dangerous failure. Use
`kalshi_odds.py`.

**`/markets/trades` silently ignores `event_ticker`.** Passing
`event_ticker=KXPGATOUR-WYC26` returns HTTP 200 with crypto, tennis and esports
trades — zero golf. It only honours `ticker={single market}`. If you ever wire
this in, assert the returned tickers match what you asked for.

**`/markets/orderbooks` needs repeated params, not a comma list.**
`?tickers=A,B,C` returns HTTP 200 with one result whose `ticker` is the literal
comma-joined string and an empty book — a convincing wrong answer. Correct form
is `?tickers=A&tickers=B`. Cap is 100 per call.

**Do not pass `status=` to `/events`.** Every value returns zero results. The
unfiltered call already includes settled events.

**Settled markets quote `bid=0.0000, ask=1.0000`.** Filter to `status ==
"active"` or any sum is nonsense.

**Rate limiting is silent.** Kalshi returns HTTP 429 on bursts and a 429 is not
an exception. `kalshi_odds.py::get()` checks `response.ok`, backs off, and raises
rather than returning empty. Keep it that way.

## 5. Still open

- **No side-by-side against a live DraftKings field.** DK is unreachable from
  agent tooling (extension blocklist), so the head-to-head in
  `SPORTSBOOK-OPTIONS.md` was run against Polymarket instead — which Kalshi beat
  decisively, but that establishes Kalshi > Polymarket, not Kalshi > DK. If you
  can save one DK payload to `dk_data.json` during a tournament week,
  `../find_dk_endpoint.py` plus a direct comparison settles it. The parity claim
  is well-reasoned, not measured.
- **Favorite-longshot bias is a hypothesis, not a finding.** Bookmakers are known
  to load margin onto longshots; exchanges show this less. If true, DK was
  over-weighting the bottom of the field all along and your historical groups
  were subtly skewed. Testable with the DK payload above.

## 6. Reference — the pull

One call gets everything. 143 golfers, ~121ms, ~341KB:

```
GET https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker=KXPGATOUR-WYC26&limit=500
```

`limit=500` covers any golf field in one page — no cursor loop needed. Returns
`yes_sub_title` (golfer), `yes_bid_dollars`, `yes_ask_dollars`,
`last_price_dollars`, `volume_fp`, `open_interest_fp`, `custom_strike`, `status`.

**Event codes are not predictable.** 2026 Wyndham is `WYC26`; 2025 was `WC25`.
Always look them up:

```
GET /events?series_ticker=KXPGATOUR&limit=200
```

**Timing — pull Wednesday night.** Observed on two consecutive tournaments:

| | Winner markets created | Top 5 / Top 10 |
|---|---|---|
| 3M Open (tees Thu 07-23) | Sun 07-19 23:00Z | Mon 07-20 20:00Z |
| Wyndham (tees Thu 08-06) | Sun 08-02 23:00Z | Mon 08-03, later |

Markets keep being *added* through Wednesday as the field firms up — 3M's last
market was created Wed 07-22 16:01Z. A Sunday pull gets an incomplete field.

**Historical prices are recoverable** via
`/series/{series}/markets/{ticker}/candlesticks?start_ts=&end_ts=&period_interval=60`
(OHLC bid/ask on settled markets). Two uses: backtest the grouper against prior
seasons, and recover a snapshot if you miss the pull window.

## 7. Scope

Read-only market data. Nothing here places, executes, or simulates trades, and it
should stay that way.
