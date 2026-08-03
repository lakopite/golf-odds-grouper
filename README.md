# Golf Odds Grouper

Reads a tournament's odds, partitions the field into equal-weighted groups, and deals
those groups out to the pool's participants. The partition is provably optimal on a
normal field, and the run says so.

Odds come from the **Kalshi** prediction market. They used to come from DraftKings,
which moved its odds endpoint every season and started each year with an hour in
DevTools hunting for the new URL. Kalshi publishes a documented, stable,
unauthenticated REST API instead, so there is nothing to hunt for.

## Quick start

```bash
pip install -r requirements.txt

# Who is playing this week? (participants.json is a list of name strings)
echo '["Mo", "Diogo", "Luis", "Cody", "Darwin"]' > participants.json

# Pull this week's odds and build the groups in one step.
python group.py --event KXPGATOUR-WYC26
```

Groups are written to `output/BESTGROUPS.json`, with `output/GROUPING.json` alongside it
recording the delta, the proven floor, and whether the partition is optimal.

## Pre-requisites

- `participants.json` in the repo root: a JSON list of participant name strings, e.g.
  `["Mo", "Diogo", "Luis", "Cody", "Darwin"]`. One group is built per participant.
- Network access to `api.elections.kalshi.com`. No API key, no account, no auth.

## Getting the odds

`group.py` resolves its odds in this order:

1. `--data-file PATH`, if given.
2. `--event TICKER`, if given — a live pull from Kalshi.
3. A local `kalshi_data.json`, then a local `dk_data.json`.
4. Otherwise, a live pull of the newest `KXPGATOUR` event that has active markets.

So `python group.py` on its own works during tournament week. Because a local file
outranks a live pull, every run prints the event and capture time recorded in that file,
so last week's odds cannot win silently. A file written with `--include-closed` holds
settled markets quoting `ask=1.0000`; the grouper refuses it rather than grouping a
tournament that has already finished.

To capture the odds first and group later — useful if you want the raw payload kept:

```bash
python kalshi_odds.py --list-events            # what golf is on right now
python kalshi_odds.py --latest                 # newest live PGA Tour winner event
python kalshi_odds.py --event KXPGATOUR-WYC26 --write-odds-file
python group.py                                # reads the kalshi_data.json just written
```

**Event codes are not predictable.** The 2026 Wyndham Championship is `WYC26`; 2025 was
`WC25`. Always look the code up with `--list-events` rather than constructing it.

**Pull Wednesday night.** Winner markets post Sunday ~23:00Z of tournament week, but
markets keep being *added* through Wednesday as the field firms up. A Sunday pull gets
an incomplete field.

## Hidden option — which market to price off

The market type is nothing more than a series prefix on the event ticker, and every
series returns the identical shape. This is the direct replacement for the old
DraftKings `ODDS_TYPE` label. Change it with `--series`, or edit `MARKET_SERIES` in
`group.py`:

| Series | Market |
|---|---|
| `KXPGATOUR` | Outright Winner — **the default** |
| `KXPGATOP5` | Top 5 Finishers |
| `KXPGATOP10` | Top 10 Finishers |
| `KXPGAMAKECUT` | To Make the Cut |

**Prefer Winner.** The old README recommended `Top 5 (Including Ties)` as "another good
option" on DraftKings. On Kalshi that advice inverts, for two reasons.

*Granularity.* Winner ticks at $0.001 below 10¢ (`price_level_structure:
tapered_deci_cent`) while Top 5, Top 10 and MakeCut are flat $0.01 — 10× coarser exactly
where a golf field lives. Top 5 / Top 10 markets also post about 21 hours later than
Winner, so they carry a narrower pull window.

*Meaning.* Winner outcomes are mutually exclusive — exactly one golfer wins — which is
what makes the de-vig a probability and makes "exclude anyone over `1/participants`" mean
something. Top 5 outcomes are **not** mutually exclusive: five golfers finish top 5, so
the book sums toward 5 and dividing by that sum gives share-of-five-slots, not a
probability. Grouping still balances sensibly on those numbers, but read them as weights.
The grouper prints a warning whenever a loaded book sums above 1.6 for this reason.

`--price` picks which side of the book to read. It defaults to `ask`, and should stay
there. Measured on the live 143-golfer Wyndham field on 2026-08-03: the ask gives every
golfer a real quoted price and sums to 1.308; the bid leaves 46 of 143 at zero and sums
to 0.906, which is not a distribution at all. `mid` inherits the bid's hole — with a
third of the field unbid, "mid" is usually just half the ask, a number the code invented
rather than read. (Prices move, so those figures are a shape rather than a constant.)

## De-vig and exclusions

The raw book is not a probability distribution. A Winner ask book sums to about 1.30 —
that spread over 1.0 is the overround. The grouper always divides by the **observed**
field sum, so the same code is correct whether the book sums to 1.30, to 5 (Top 5), or
to 10 (Top 10). Group totals therefore always add to 1.0.

Exclusions run on top of that. `EXCLUDE_LIST` in `group.py` (or `--exclude NAME`, or
`--no-exclude`) drops named golfers and re-scales the rest, giving the probability of
winning *given that none of the excluded golfers wins*.

`--auto-exclude` applies the rule the exclusion list exists to serve: drop any golfer
whose de-vigged probability exceeds `1/participants`. Such a golfer is worth more than a
whole group's fair share on their own, so no partition can balance around them. It
iterates — removing one golfer redistributes their weight and can push the next over the
line. Every run prints who is above the threshold whether or not you act on it.

## Command reference

```
python group.py [--event TICKER] [--series KXPGATOUR] [--price ask|mid|bid|last]
                [--data-file PATH] [--dk-odds-type "Winner"]
                [--participants participants.json] [--output-dir output]
                [--exclude NAME ...] [--no-exclude] [--auto-exclude]
                [--time-limit SECONDS] [--seed N]
```

An archived DraftKings `dk_data.json` still parses — `list_dk_golf_odds()` and
`--dk-odds-type` are unchanged — so old captures still run and a DK payload can be
compared against Kalshi side by side if one is ever exported by hand.

### Runtime

A run finishes in well under a second on a 143-golfer field, at any group count. The
partitioner stops the moment it can prove its answer is optimal, which on a real field
is immediately. `--time-limit` (default 2s) caps the search in the case where the proof
is out of reach — see the fair-share note below, because that case is a property of the
field rather than of the search.

The same odds always produce the same groups. The field is sorted into one canonical
order on load, and the partitioner has no randomness in it. Group *assignment* to
participants is random by design — pass `--seed N` to make a whole run reproducible.

## Tests

```bash
python -m pytest tests/ -q          # offline, uses checked-in fixtures
KALSHI_LIVE=1 python -m pytest tests/test_live.py -v   # hits the real Kalshi API
```

The offline suite proves the code is self-consistent. The live suite proves the endpoint
still answers, still sends money fields as strings, and still quotes an ask on every
active market. Run the live suite before trusting a season's first pull.

## How the grouping works

The field is partitioned into groups of equal total implied probability. The measure is
the **delta**: the highest group total minus the lowest. Smaller is better.

### It is an integer problem, and that is the whole trick

Kalshi quotes on a price grid — the Winner series ticks at $0.001, the others at a flat
$0.01. Every price in a book is an exact multiple of that tick (measured on the live
143-golfer Wyndham field: zero prices off the grid), and the de-vig divides the whole
book by one constant, so the grid survives it.

So this is not a real-valued partition. It is a partition of **whole numbers of ticks**,
which hands us something a float formulation cannot have: a provable floor.

- **Divisibility.** 1151 ticks split 5 ways cannot beat a delta of 1 tick, because 1151
  is not divisible by 5. Reach 1 tick and you are done.
- **Concentration.** If one golfer is worth more than a group's fair share, their group
  is heavy no matter what happens elsewhere. Taking the *j* heaviest golfers together
  sharpens this into a bound that is tight in practice.

Every run reports the delta, the floor, and whether it reached it. `PROVEN OPTIMAL`
means exactly that: no partition of this field does better.

### The method

1. Recover the tick grid and convert the field to whole numbers.
2. Compute the floor.
3. Seed with **Karmarkar–Karp differencing** — hold the group totals as a tuple and
   repeatedly combine the two most-imbalanced, largest against smallest, so imbalances
   cancel instead of compounding.
4. Improve with **local search** over single moves and single swaps. Both are needed: a
   move alone cannot rebalance two groups that are already the right size, and a swap
   alone cannot change a group's size at all.
5. Stop at the floor and report the answer as optimal.

Group sizes are not constrained beyond "at least one golfer each". Forcing equal sizes
costs real accuracy — measured on the live field with a 30-second search, 12 groups
reaches 1 tick unconstrained and is stuck at 8 ticks with equal sizes. A group of one is
a legitimate answer; a group of none is not, and is prevented.

### When the delta cannot be small

At 13 groups and above on the live field, Cameron Young at 92 ticks exceeds the fair
share of 1151/13 = 88.5, and the floor rises with the group count:

| Groups | Proven floor | What binds |
|---|---|---|
| 2–12 | 1 tick | nothing |
| 13 | 4 ticks | Cameron Young, 1.04 fair shares |
| 16 | 22 ticks | Cameron Young, 1.28 fair shares |
| 25 | 48 ticks | Cameron Young, 2.00 fair shares |

**No algorithm beats those numbers.** Searching harder is the wrong response; excluding
the golfer is the right one. The run names whoever is responsible and says so. With
`--auto-exclude` dropping Cameron Young, every group count from 2 to 27 lands back at
1 tick or better.

### What this replaced

Five methods — backtracking, dynamic programming, simulated annealing, a genetic
algorithm, and greedy redistribution — were run on every field and the smallest delta
won. They agreed on the live field because a full golf field is the easy case: a long
tail of one-tick golfers gives almost any method the small change it needs to close the
last gap. The comparison was therefore measuring very little, and off that field the
methods separated badly. Measured deltas in ticks, lower is better:

| Test field | Floor | backtrack | dp | sa | ga | greedy | now |
|---|---|---|---|---|---|---|---|
| Live, 5 groups | 1 | 1 | 1 | 1 | 1 | 94 | **1** |
| Live, 12 groups | 1 | 1 | 1 | 8 | 1 | 92 | **1** |
| Live, 25 groups | 48 | 49 | 48 | 53 | 49 | 89 | **48** |
| Limited field, 70 golfers | 1 | 3 | 1 | 1 | 1 | 4 | **1** |
| Small field, 24 golfers | 3 | 14 | 12 | 21 | 3 | 92 | **3** |
| 40 items, large weights | — | 136210 | 136210 | 3130 | 3717 | 683185 | **401** |

Four of the five had defects that made them unfixable rather than untuned. The
backtracking recurrence used `dp(i, 0) = inf`, which blocks the exclude path and forces
the top golfer into the seed set every run; its objective had no relation to balance, and
its last five lines did all the work. The DP scaled prices with `int(odds * 100)`, so
after the de-vig almost every golfer scaled to zero, and its cost grew with the *sum* of
the weights — 609 seconds on a 40-item field. The annealer started at temperature 1000
against a delta of 0.001–0.08, so it accepted nearly every bad move for its first
thousand steps and refused every bad move after about 1,400 of its 100,000 — it never
annealed. The genetic algorithm drew its crossover point from the field size (143) but
sliced groups of about 12, so at 12 groups the crossover did nothing in 92% of draws, and
it needed 5–10 minutes to reach what the current code reaches in about 3 milliseconds.
Greedy redistribution was the worst method on all ten test fields; its swap was fixed
rather than chosen, so it overshot and oscillated instead of converging.

The current code is deterministic, which the annealer and the genetic algorithm were not.
Both returned a different answer to the same question on consecutive runs, which makes
"why did my group change?" unanswerable.

## Research record

`kalshi-migration/` holds the evaluation that led here: why Kalshi over ESPN, Polymarket,
The Odds API and DataGolf; the API traps that cost time; and the measurements behind the
Winner-and-ask decision. It is history, not live documentation — the code it describes as
"remaining work" now lives in `kalshi_odds.py` and `group.py`.
