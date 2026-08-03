# Kalshi Migration — Orientation

**Status: exploratory research only. Nothing here is wired into the grouper yet.**

Written 2026-08-02. If you are an agent or a human picking this up cold, read this
file before touching anything else in `kalshi-migration/`.

> **START AT `START-HERE.md`, NOT HERE.** This file is the first exploration
> pass. It remains the best API reference in the folder, but two of its
> conclusions were reversed by later evidence.
>
> **SUPERSEDED IN PART — see `SPORTSBOOK-OPTIONS.md` (2026-08-03).**
> A pre-tournament capture of the 2026 Wyndham Championship answered the §7 open
> questions and reversed two of them. Short version:
> - Liquidity is fine (97/143 two-sided pre-tournament); the earlier thinness was
>   an artifact of capturing a finished tournament.
> - **Winner is the right market; Top 5 / Top 10 are worse, not better.** Winner
>   ticks at $0.001 below 10¢ (`tapered_deci_cent`); Top 5/10 are flat $0.01.
> - §5.5's claim that `web_fetch` "yields empty output for JSON" is now wrong —
>   it returns JSON, but **stale cached** JSON. Worse failure mode. Don't use it.
> - Money fields are **strings** (`"0.0110"`), not numbers. Coerce them.

---

## 1. Why this folder exists

The parent project (`golf-odds-grouper`) reads golf odds and partitions the field
into equal-weighted groups for a pool. Historically the odds came from DraftKings
Sportsbook. **DraftKings moves its odds endpoint every year**, so every season
opens with the same chore: sit in Chrome DevTools, watch the network tab, and work
out where the JSON lives now. `../find_dk_endpoint.py` automates the analysis half
of that ritual (feed it a HAR export).

Kalshi is being evaluated as a replacement because it publishes a **documented,
stable, unauthenticated REST API**. No annual archaeology.

### Verdict from the exploration

Kalshi has **full parity** with the DraftKings markets the grouper uses: outright
Winner, Top 5, and Top 10, each covering the complete tournament field. Migration
looks viable. The open questions are in §7.

---

## 2. The API in sixty seconds

Base URL:

```
https://api.elections.kalshi.com/trade-api/v2
```

No API key needed for market-data reads. Three-level hierarchy:

| Level | Endpoint | Notes |
|---|---|---|
| Series | `/series?category=Sports` | 3077 series; 112 tagged `Golf`. Single page, cursor is null. |
| Series (one) | `/series/KXPGATOUR` | |
| Events | `/events?series_ticker=KXPGATOUR&limit=200` | A series is a market *type*; an event is one tournament. |
| Markets | `/markets?event_ticker=KXPGATOUR-ROC26&limit=500` | One market per golfer. Paginate via `cursor`. |

**Do not pass `status=` to `/events`.** Every value tried (`open`, `closed`,
`settled`, `unopened`) returned zero results. The unfiltered call already includes
settled events, so filtering is unnecessary anyway.

---

## 3. Ticker map

Event tickers are `<SERIES>-<EVENTCODE>`. The 2026 Rocket Classic is `ROC26`.

| Series | Market | Use |
|---|---|---|
| `KXPGATOUR` | **Outright winner** | The DK `"Winner"` analogue. Primary target. |
| `KXPGATOP5` | Top 5 Finishers | DK `"Top 5 (Including Ties)"` analogue |
| `KXPGATOP10` | Top 10 Finisher | |
| `KXPGAMAKECUT` | To Make the Cut | |
| `KXPGAR3LEAD`, `KXPGAR3TOP10`, `KXPGAR1TOP5` | Single-round | Not useful for tournament-long pools |
| `KXPGAH2H` | Head-to-head pairs | Two golfers per market; not field-wide |
| `KXDPWORLDTOUR` | DP World Tour winner | Different tour |

Event codes are **not** predictable abbreviations. The 2025 Wyndham Championship is
`KXPGATOUR-WC25`, not `WYN25`. Always look the code up rather than constructing it.

Events appear to be created close to tournament week. As of 2026-08-02 the Wyndham
Championship had no 2026 event yet — only `WC25` existed.

---

## 4. Data shape, and how it maps to the grouper

A market object carries these fields (43–45 keys total):

```
ticker, event_ticker, yes_sub_title (golfer name),
yes_bid_dollars, yes_ask_dollars, last_price_dollars,
no_bid_dollars, no_ask_dollars,
volume_fp, open_interest_fp, liquidity_dollars,
status ("active" | "finalized"), result ("yes" | "no" | "")
```

**Prices are already implied probabilities.** A YES contract settles at $1.00, so
`yes_bid_dollars` / `yes_ask_dollars` are probabilities directly. The grouper's
`fractional_odds_to_implied_probability()` becomes unnecessary — that function only
exists to decode DK's `displayOdds.fractional`.

Observed on `KXPGATOUR-ROC26`: 151 markets (full field, matching DK), 73 active,
sum of last prices **1.161** — roughly 16% overround, tighter than a typical
sportsbook golf outright at 20–40%. Prices carry sub-cent granularity (0.099,
0.065), finer than the 1¢ grid older Kalshi docs describe.

### Integration gap

`../group.py::list_dk_golf_odds()` expects the DK envelope: top-level `markets[]`
(match on `name`, take `id`) and `selections[]` (filter on `marketId`, read
`participants[0].name` and `displayOdds.fractional`).

`kalshi_odds.py` emits the *output* of that function — a flat
`[{"golfer_name", "odds"}, ...]` list — not the envelope. So `--write-dk-data`
produces a file `group.py` cannot currently read. **A small shim in `group.py` is
the remaining work**: branch on input shape, or add a `list_kalshi_golf_odds()`
alongside the existing parser.

---

## 5. Traps — all of these cost time already

### 5.1 Rate limiting is silent and will lie to you

This is the big one. Kalshi returns **HTTP 429** on bursts. Measured: 8-wide
concurrency across 112 series produced **99 × HTTP 429**.

A 429 is *not* an exception. Code shaped like this:

```python
try:
    events = requests.get(url).json().get("events") or []
except Exception:
    events = []
```

reads a rate-limited response as **"this market does not exist."**

That exact bug produced a confident, wrong conclusion during this exploration:
"Kalshi has no PGA Tour outright winner market." It was then "independently
verified" by re-running the *same broken sweep*, which reproduced the same false
negative and made the error look confirmed. The market — `KXPGATOUR-ROC26`,
"Rocket Classic Winner" — existed the whole time, in a series that was already in
the list being scanned.

**Rules:** check `response.ok` explicitly, never swallow non-200s, keep concurrency
low, back off on 429. `kalshi_odds.py::get()` does all four and raises rather than
returning empty. Keep it that way.

The general lesson, which cost more than the specific bug: *absence of data in a
sweep is not evidence of absence in the world.* When a query returns nothing,
confirm the query worked before believing the result. Cross-check against a
different method — reading the actual website is what finally caught this.

### 5.2 Settled markets poison probability sums

Finalized markets quote `bid=0.0000, ask=1.0000`. Filter to `status == "active"`
or any overround calculation will be nonsense.

### 5.3 Field naming changed

This API version suffixes money fields `_dollars` and fixed-point counters `_fp`.
Older Kalshi docs and examples show integer-cent fields named `yes_bid`, `yes_ask`,
`volume`. Code written against the old names reads `undefined` / `None` and fails
**silently**.

### 5.4 Browser-driven exploration is a trap

If you explore via Claude in Chrome rather than a server-side client:

- **CORS**: `api.elections.kalshi.com` accepts fetches from a `kalshi.com` origin
  but not from arbitrary origins (an `example.com` fetch failed outright). Run JS
  from a kalshi.com page — `https://kalshi.com/robots.txt` is a light one that
  avoids freezing the renderer the way the heavy market pages do.
- **Timers are throttled** to ~1/min in a background tab, so `setTimeout`-based
  pacing stalls forever. Pace with chunked `await Promise.all` instead, or just
  use a server-side client.
- Long `await` chains hit the 45s CDP timeout. Kick work off into
  `window.__result` and poll in a separate call.

None of this applies to `kalshi_odds.py`, which is a normal server-side client.

### 5.5 DraftKings is unreachable from agent tooling

The Claude in Chrome extension blocks sportsbook domains outright
("This site is not allowed due to safety restrictions") — both
`sportsbook.draftkings.com` and `sportsbook-nash.draftkings.com`. It's an extension
blocklist, not a Chrome or network issue: `web_fetch` reaches the page fine, and
navigation to unrelated domains works. `web_fetch` is no help either, since it
returns HTML only and yields empty output for JSON content types.

Kalshi is **not** blocked. This asymmetry is itself a point in Kalshi's favour for
any automated workflow.

---

## 6. Files here

| File | What it is |
|---|---|
| `kalshi_odds.py` | Working client: discovery, pagination, 429 backoff, mid/last/bid/ask pricing, liquidity report. Emits the grouper's `{golfer_name, odds}` shape. |
| `api_exploration.json` | Raw findings: verified endpoints, ticker inventory, captured prices, and a `CORRECTION_NOTICE` documenting the false-negative bug. |
| `ORIENTATION.md` | This file. |

### Usage

```bash
pip install requests

python kalshi_odds.py --list-series          # every Golf-tagged series
python kalshi_odds.py --list-events          # tournaments with full-field markets
python kalshi_odds.py --event KXPGATOUR-ROC26            # winner odds
python kalshi_odds.py --event KXPGATOP5-ROC26 --price mid
```

Writes raw payloads and a grouper-ready odds file to `captures/`.

The pure functions (`to_golfers`, `liquidity_report`, `get` backoff) have been
unit-tested offline. **The live network path has not been run end-to-end** — the
sandbox where this was written had no outbound access to Kalshi. Run it once
against a real event before trusting it.

### Quick manual check

```bash
# Has a tournament been posted yet?
curl -s 'https://api.elections.kalshi.com/trade-api/v2/events?series_ticker=KXPGATOUR&limit=200' \
  | jq -r '.events[] | "\(.event_ticker)\t\(.title)"'

# Winner odds for one event
curl -s 'https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker=KXPGATOUR-ROC26&limit=500' \
  | jq -r '.markets[] | select(.status=="active")
           | "\(.yes_bid_dollars)  \(.yes_ask_dollars)  \(.yes_sub_title)"' \
  | sort -rn | head -20
```

If `jq` prints nothing, **check the raw response for a 429 before concluding the
market is absent.**

---

## 7. Open questions

1. **Liquidity.** On `KXPGATOP10-ROC26`, only 40 of 73 active markets had
   two-sided quotes; `liquidity_dollars` read 0 on every market sampled. On the
   winner market, 15 of 73 were two-sided. **But every capture was taken with the
   tournament nearly finished**, so both the price collapse and possibly the
   thinness are artifacts. A pre-tournament capture is needed before drawing any
   conclusion. Do this first — it decides whether migration is viable at all.

2. **Which market to pool on.** Winner is the direct DK replacement. Top 5 may
   actually be the better pool metric (the parent README already flags it as a good
   option) and may carry more liquidity.

3. **`odds_to_conditional()` needs rethinking.** It strips vig assuming a winner
   market summing slightly above 1.0. That holds for `KXPGATOUR`. It does **not**
   hold for a Top 5 book, which sums toward 5.0. And excluding Scheffler/McIlroy
   means something different when the metric is "top 5" rather than "wins."

4. **The `group.py` shim** (§4) — the concrete code change blocking migration.

5. **Bid/ask vs. last.** DK gives one price; Kalshi gives a spread. `to_golfers()`
   defaults to mid, which is probably right, but on thin one-sided markets mid is
   doing real work off a single quote. Worth sanity-checking against DK's implied
   probabilities on the same field.

---

## 8. Scope note

This is read-only market research. Nothing here places, executes, or simulates
trades, and it should stay that way.
