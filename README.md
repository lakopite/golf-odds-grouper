# Golf Odds Grouper

Reads a tournament's odds, partitions the field into equal-weighted groups, and deals
those groups out to the pool's participants.

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

Groups are written to `output/BESTGROUPS.json`, one file per method alongside it.

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
                [--methods backtracking,dp,sa,ga,greedy]
                [--sa-iter N] [--ga-pop N] [--ga-generations N] [--ga-mutation F]
                [--seed N]
```

An archived DraftKings `dk_data.json` still parses — `list_dk_golf_odds()` and
`--dk-odds-type` are unchanged — so old captures still run and a DK payload can be
compared against Kalshi side by side if one is ever exported by hand.

### Runtime

A default run on a 143-golfer field takes about **5 minutes**, and around **10 minutes**
with 25 participants. Essentially all of it is the genetic algorithm at its default
`--ga-pop 500 --ga-generations 10000`; every other method finishes in under a second.
These defaults pre-date the Kalshi migration and are left alone deliberately.

The GA does not appear to be earning that time — on a measured 143-golfer run its best
delta and backtracking's agreed to twelve significant figures, and with 25 participants
the DP won outright. For a fast run that is unlikely to lose anything:

```bash
python group.py --ga-pop 40 --ga-generations 100        # seconds, not minutes
python group.py --methods backtracking,dp,sa,greedy     # skip the GA entirely
```

The same odds always produce the same groups regardless of how the odds file was
ordered: the field is sorted into one canonical order on load. Group *assignment* to
participants is random by design — pass `--seed N` to make a whole run reproducible.

## Tests

```bash
python -m pytest tests/ -q          # offline, uses checked-in fixtures
KALSHI_LIVE=1 python -m pytest tests/test_live.py -v   # hits the real Kalshi API
```

The offline suite proves the code is self-consistent. The live suite proves the endpoint
still answers, still sends money fields as strings, and still quotes an ask on every
active market. Run the live suite before trusting a season's first pull.

## Golfers Grouping Algorithms

This project provides five different methods to partition a list of golfers into a
specified number of balanced groups based on their odds of winning. The methods aim to
minimize the difference between the group with the highest cumulative odds and the group
with the lowest cumulative odds. Each is run, each is validated, and the one with the
smallest delta percentage wins.

## Methods

1. **Backtracking**
2. **Dynamic Programming**
3. **Simulated Annealing**
4. **Genetic Algorithm**
5. **Greedy Algorithm with Redistribution**

### 1. Backtracking

**Function:** `backtracking_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a backtracking approach to generate balanced groups of golfers. It recursively tries to assign golfers to groups to minimize the difference in total odds.

### 2. Dynamic Programming

**Function:** `dp_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a dynamic programming approach to generate balanced groups of golfers. It iteratively calculates the minimum difference in total odds between groups and assigns golfers accordingly.

**Known limitation (pre-dates the Kalshi migration; measured, not fixed).** The DP's
objective is degenerate: `dp[n][n_groups][k] == k` for every feasible `k`, so the
minimal difference it finds is always 0 and its reconstruction emits `n_groups`
singletons. All the actual balancing is then done by the "assign each remaining golfer
to the lightest group" fallback at the end of the function. Two consequences worth
knowing: raising the table's fixed-point resolution changes which singletons come out
but does not improve the partition, and the method must never be given fewer golfers
than groups — the reconstruction loop does not terminate on that input. `group.py`
checks the field size after exclusions to keep that state unreachable.

### 3. Simulated Annealing

**Function:** `sa_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.
- `max_iter`: Integer (default=1000), the maximum number of iterations to perform.
- `temp`: Float (default=1000), the initial temperature for the simulated annealing process. Think of temp as the "energy" or "excitement" level at the start of the process. A high temperature means the algorithm is more willing to accept changes, even if they make things worse temporarily, to explore more potential solutions.
- `cooling_rate`: Float (default=0.99), the rate at which the temperature decreases. The cooling_rate controls how quickly the temperature drops over time. A high cooling rate means the temperature drops quickly, and the algorithm becomes more selective faster. A low cooling rate means the temperature drops slowly, allowing the algorithm to explore a broader range of solutions for a longer time.

**Description:**
Uses a simulated annealing approach to generate balanced groups of golfers. It starts with a random initial solution and iteratively tries to improve it by making random swaps and accepting changes based on the temperature.

### 4. Genetic Algorithm

**Function:** `ga_generate_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.
- `pop_size`: Integer (default=100), the size of the population. This is the number of possible solutions (groups of golfers) the algorithm starts with. A larger population size means more potential solutions to choose from and evolve, but it also requires more computational resources.
- `generations`: Integer (default=1000), the number of generations to evolve. This is how many times the algorithm will improve the population. In each generation, the algorithm selects the best solutions, combines them to create new solutions, and possibly mutates them to explore new possibilities. More generations mean more chances to find a good solution.
- `mutation_rate`: Float (default=0.1), the probability of mutation. This is the chance that a small random change will be made to a solution. Mutations help the algorithm explore new solutions that might not be found through selection and crossover alone. A higher mutation rate means more randomness and diversity in the solutions, while a lower rate means more stability and refinement of existing solutions.

**Description:**
Uses a genetic algorithm to generate balanced groups of golfers. It initializes a population of random solutions, evaluates their fitness, and iteratively applies crossover and mutation operations to evolve better solutions.

### 5. Greedy Algorithm with Redistribution

**Function:** `greedy_redistribute_groups`

**Parameters:**
- `golfers`: List of dictionaries, where each dictionary contains `golfer_name` and `odds` of each golfer.
- `n_groups`: Integer, the number of groups to divide the golfers into.

**Description:**
Uses a greedy algorithm to initially distribute golfers into groups and then redistributes them to balance the total odds. It sorts golfers by their odds and assigns them to groups in a round-robin fashion, followed by a redistribution step to minimize the difference in total odds between groups.

## Research record

`kalshi-migration/` holds the evaluation that led here: why Kalshi over ESPN, Polymarket,
The Odds API and DataGolf; the API traps that cost time; and the measurements behind the
Winner-and-ask decision. It is history, not live documentation — the code it describes as
"remaining work" now lives in `kalshi_odds.py` and `group.py`.
