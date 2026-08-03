# Odds Source Evaluation — Directional Findings

Written 2026-08-03. Test case: **2026 Wyndham Championship** (`KXPGATOUR-WYC26`),
captured pre-tournament — winner markets went live 2026-08-02 23:00Z, tournament
tees off Thursday 08-06. This is the pre-tournament capture ORIENTATION.md §7
flagged as the blocking unknown.

---

## Bottom line

**Kalshi's outright Winner market is granular enough. Migrate to it.**

The concern that prompted this research — that Kalshi's golf winner odds are too
coarse — does not survive contact with the data. The opposite is closer to true:
Kalshi prices the longshot tail on a **0.1¢ tick**, finer than DraftKings can
express in fractional odds.

Two corrections to prior assumptions come out of this, both of which reverse
ORIENTATION.md's open questions:

1. **Top 5 / Top 10 are the *wrong* market, not the better one.** They trade on a
   flat 1¢ grid — 10× coarser than Winner exactly where the field sits.
2. **Liquidity is not a problem.** The thinness recorded earlier was an artifact of
   capturing a nearly-finished tournament, as suspected.

---

## 1. Granularity — the actual answer

Kalshi publishes its tick structure on every market object. This is the whole
question, settled in one field:

| Market | `price_level_structure` | Tick |
|---|---|---|
| `KXPGATOUR-*` (Winner) | `tapered_deci_cent` | **$0.001** from 0–10¢, $0.01 from 10–90¢, $0.001 from 90¢–$1 |
| `KXPGATOP5-*` | `linear_cent` | $0.01 flat |
| `KXPGATOP10-*` | `linear_cent` | $0.01 flat |
| `KXPGAMAKECUT-*` | `linear_cent` | $0.01 flat |

A golf outright field lives almost entirely under 10% — at Wyndham the favourite
is 9%. The Winner market's taper puts **100 distinct price levels** in exactly
that band. The Top 5 market puts 10 there.

### Measured on `KXPGATOUR-WYC26`, pre-tournament

| Metric | Value |
|---|---|
| Markets | 143, **all `active`** |
| Two-sided quotes | 97 / 143 |
| Spread — p25 / median / p75 / max | 0.1¢ / **0.2¢** / 0.4¢ / 1.0¢ |
| Sum of mids (overround) | **1.217** (~22%) |
| Distinct price levels used | 44 |
| Volume / open interest | ~$3.21M / ~$3.06M |
| Favourite | Cameron Young, 9.0¢ |
| Median golfer | 0.55¢ |

Compare the earlier post-tournament ROC26 capture: 15/73 two-sided. The
pre-tournament number is 97/143. **The liquidity worry was a measurement artifact.**

### The one real limitation

The field compresses at the bottom. Distribution of mid prices:

```
 9.0¢ x1    4.1¢ x1   3.7¢ x1   3.5¢ x1   3.2¢ x1   3.0¢ x2   2.8¢ x1
 ...
 0.60¢ x9   0.55¢ x6  0.45¢ x6  0.40¢ x10 0.35¢ x4
 0.30¢ x12  0.25¢ x9  0.20¢ x10 0.15¢ x3  0.10¢ x12
```

12 golfers sit on the 0.10¢ floor; ~25 are at or below 0.20¢. The bottom third of
the field is functionally tied.

**This does not matter for the grouper.** Those golfers contribute under 1% of
group weight each; whether one is "really" 0.12¢ and another 0.08¢ changes no
partition. Ties at the tail are a property of the *sport*, not of Kalshi — a
150-player field genuinely has 50 players who are indistinguishable longshots.
DraftKings expresses that same tail as a wall of 500/1s.

---

## 2. Alternatives evaluated

### ESPN — no odds. Rules it out as an odds source.

Verified against a working control. `sports.core.api.espn.com/.../golf/leagues/pga/events/{id}/competitions/{id}/odds`
returns HTTP 200 with `count: 0` — including on pre-event tournaments, where odds
would exist if they existed at all. `/seasons/2026/futures` likewise `count: 0`.
The site leaderboard payload contains no `odds` key.

The identical route shape for NFL returns populated ESPN BET and DraftKings
markets, so the method is sound — this is a real absence, not a broken probe.
Confidence high; would be falsified by odds appearing on a major rather than a
regular-season stop.

Also worth knowing: the core API's **default `dates` parameter is pinned to
`20260622`**. Omit it and you get U.S. Open week, silently. Always pass `?dates=`
explicitly. (This is what made a probe look "date-shifted".)

**Keep ESPN for what it is good at** — field lists, stable athlete ids, live
scoring. Your `espntest/espn_golf_probe.py` is the right use of it. It is a
scoring feed, not an odds feed.

### Polymarket — materially worse than Kalshi on the same tournament

`2026-wyndham-championship-winner` exists, so weekly non-majors are covered. But
measured head-to-head on the same event, same day:

| | Kalshi | Polymarket |
|---|---|---|
| Golfers priced | 143 | 100 (field truncated) |
| Sum of mids | **1.217** | **2.350** |
| Median spread | 0.2¢ | 0.8¢ |
| Widest spread | 1.0¢ | 73.1¢ |
| Most-crowded price | 12 golfers | **22 golfers** at one price |
| Volume | ~$3.21M | ~$34.4k |

The 2.35 sum is the tell: four separate golfers are quoted 35–37¢ to win the same
tournament. Those are untraded quotes, not prices. Top 5 is worse — 95 of 100
markets sit at exactly 49.5¢ (i.e. a 1¢/98¢ default quote), summing to **49.0**
where it should sum to ~5.

Slug naming is also unstable: 2025 was `pga-tour-wyndham-championship-winner`,
2026 is `2026-wyndham-championship-winner`. That is the DraftKings problem in a
new costume.

### The Odds API — majors only. Disqualified.

Their own golf page states coverage is the four majors. The live `/sports` list
returns exactly four golf keys. Solves 4 weeks of ~40. Pricing (free 500
credits/mo, $30/mo for 20k) was never the obstacle — coverage is.

### DataGolf — the credible paid fallback, $270/yr

`feeds.datagolf.com/betting-tools/outrights?tour=pga&market=win` returns win odds
from 11 sportsbooks *plus* DataGolf's own model probability, which is already
de-vigged — arguably a cleaner input to a balancing algorithm than any single
book. `field-updates` gives the field list in the same subscription. Requires the
Scratch PLUS tier ($30/mo or $270/yr); the $20/mo tier does **not** include API.

Worth it only if you later want model probabilities rather than market prices.
For the current use case Kalshi is free and sufficient.

### Ruled out

SportsDataIO, OddsJam, OpticOdds, OddsBlaze — enterprise sales-gated, no
self-serve pricing. Betfair Exchange — has the markets, but US access is
geo-blocked and the live data key carries a reported £299 one-off fee.

---

## 3. Operational notes for the migration

**Money fields are strings, not numbers.** `"yes_bid_dollars": "0.0110"`. Adding
them without coercion silently concatenates. Coerce everything through `Number()`.
Not documented in ORIENTATION.md — it cost a debugging cycle here.

**Do not use `web_fetch` against the Kalshi API — it serves stale cache.** A fetch
of `/events?series_ticker=KXPGATOUR` returned 65 events and omitted `WYC26`,
`ROC26` and `3MO26`; a live browser-origin fetch of the same URL returned 70
including all three. ORIENTATION.md §5.5 says web_fetch "yields empty output for
JSON" — that is now wrong, it returns JSON fine. But it returns *old* JSON, which
is more dangerous. Use `kalshi_odds.py`.

**`custom_strike.golf_competitor` is a stable UUID per golfer.** Use it as the
join key against ESPN athlete ids instead of matching display names. This kills an
entire class of "Matt Fitzpatrick" / "Matthew Fitzpatrick" bugs.

**Timing — pull Wednesday night.** Observed on both 3M Open and Wyndham:

| | Winner markets created | Top 5 / Top 10 created |
|---|---|---|
| 3M Open (tees Thu 07-23) | Sun 07-19 23:00Z | Mon 07-20 20:00Z |
| Wyndham (tees Thu 08-06) | Sun 08-02 23:00Z | not yet as of 08-03 |

Markets keep being *added* through Wednesday as the field firms up (3M's last
market was created Wed 07-22 16:01Z). Pulling Sunday gets you an incomplete field.

**Historical prices are recoverable.** `/series/{series}/markets/{ticker}/candlesticks?start_ts=&end_ts=&period_interval=60`
returns OHLC bid/ask history on settled markets. Two uses: backtest the grouper
against prior seasons, and recover a snapshot if you miss the pull window.

---

## 4. Recommendation

1. **Use `KXPGATOUR-<EVENT>` (Winner) as the odds source.** Mid price where
   two-sided, last price otherwise. It is the direct DraftKings `"Winner"`
   replacement and the tick structure is better suited to the tail.
2. **Drop the Top 5 idea** from the parent README's "Hidden Option". On Kalshi it
   is a strictly worse signal. If you want a Top-5-flavoured pool, derive it from
   winner probabilities rather than reading the Top 5 book.
3. **Build the `group.py` shim** (ORIENTATION.md §7 item 4) — still the only code
   blocking migration.
4. **Keep ESPN** for field membership and live scoring, joined on
   `custom_strike.golf_competitor` → ESPN athlete id.
5. Revisit **DataGolf** only if you decide you want model-derived probabilities
   instead of market prices.

### Still open

- `odds_to_conditional()` de-vigging still assumes a winner book summing just
  above 1.0. At 1.217 that assumption is looser than the code expects — worth a
  look, though it is a scaling question rather than a correctness one.
- No side-by-side against a live DraftKings field. DK remains unreachable from
  agent tooling, so this comparison was made against Polymarket instead. If you
  can export a DK payload manually one week, the direct check is worth doing.
