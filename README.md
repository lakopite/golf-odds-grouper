# Golf Odds Grouper

Reads a tournament's odds, partitions the field into equal-weighted groups, and deals
those groups out to the pool's participants. The partition is provably optimal on a
normal field, and the run says so.

Odds come from the **Kalshi** prediction market. They used to come from DraftKings,
which moved its odds endpoint every season and started each year with an hour in
DevTools hunting for the new URL. Kalshi publishes a documented, stable,
unauthenticated REST API instead, so there is nothing to hunt for.

There are two ways in. The **competition pipeline** takes a named league and a
tournament and hands back a scoreboard; the **grouper CLI** does the partition alone
and is what the pipeline is built on. Start with the pipeline.

## Quick start — a whole competition

```bash
pip install -r requirements.txt

# One file per league: team name, player name, team logo. Team ids are derived.
cp leagues/example-league.json leagues/my-league.json   # then edit it

python build_competition.py --league leagues/my-league.json --tournament "Wyndham"
python bundle_frontend.py --result build/result.json --out dist/
```

`build/result.json` is the source of truth for that competition: the teams and their
groups, the odds at the moment they were drawn, both API endpoints, who was excluded
and why, and the partition's optimality certificate. `dist/*.html` is that file baked
into a single static page — it opens from disk and needs no server. The page is
`frontend/scoreboard/`, the designed one; `--template frontend/template` bundles the
plain reference instead, which is the thing to reach for when the question is whether
the page is wrong or the data is (`frontend/README.md`). Whether it polls
ESPN for scores is not a setting anybody chooses; it is decided below. The `.zip`
beside it carries the page, the result JSON and a manifest.

Neither is written back into the repository. `build/` and `dist/` are gitignored.

### Two builds, and the ESPN leaderboard decides which

ESPN publishes **zero competitors** until the first tee time. That one fact splits this
program in half, and the split is read off the leaderboard payload rather than off a
flag — the result file records which half it is in `build_mode`:

| `build_mode` | When | What the file says, and what the page does |
|---|---|---|
| `groups` | ESPN has published no field — Wednesday night | no name join is attempted at all: `live` is null and every `golfers[].espn` is null. The exported page fetches nothing, from anywhere, and ranks nothing. It shows the rosters, the odds the groups were drawn on, and the optimality certificate |
| `live` | ESPN has published a field | the Kalshi names are joined against **this week's** actual competitors, every golfer that resolves carries an ESPN athlete id, and the page polls ESPN and ranks on that id |

(Three unrelated things in here are called "live": that build mode, the `live` block
inside the result file, and the test suites that hit the real APIs. This README always
says which one it means.)

A `groups` build is not a degraded scoreboard waiting for a fetch to succeed. It is the
groups sheet — the thing that actually exists on Wednesday — and it is complete,
deployable and honest on its own terms. So a normal week is two runs against one
competition, and the second one is a rebuild:

```bash
# Wednesday night, no field published: the groups sheet.
python build_competition.py --league leagues/my-league.json --tournament "Wyndham"

# Thursday, play under way: the same competition, now with scoring.
python build_competition.py --from-result build/result.json --output build/thursday.json
```

Because the result file describes the whole competition, it is the input to the next
build of it. The league, the tournament on both APIs, the market, the price mode, the
hand-picked exclusions, the seed and any reviewed golfer-name decisions all come back
out of the file. The groups and the odds they were drawn on are carried forward
untouched — **a rebuild never re-deals**, because people have already been told which
golfers they own — and the parts that have a "now" are redone: above all the ESPN side,
which on Wednesday had no field to work with and on Thursday has one. It does not go
back to Kalshi: the odds were read once, when the groups were drawn, and that reading
is the competition. `--regroup` pulls fresh odds and partitions again, which is the one
that deals everybody new golfers and says so.

A rebuild that finds no field where the last one found 150 competitors is refused
rather than written: ESPN does not unpublish a field, so that is a failed read, and
writing it would turn a working scoreboard back into a groups sheet with no sign that
anything went wrong.

A league's own art rides along. The crest and the banner can be handed in when the
competition is created, beside the roster:

```bash
python build_competition.py --league leagues/my-league.json --tournament "Wyndham" \
    --crest art/crest.png --banner art/banner.png
```

Both are inlined into the exported page, so it stays one portable file. They can also
be named in the league file, and a build offered neither uses the art the tool ships —
so a page always looks finished. Those two images are the only thing that changes
between leagues: everything else about how the page looks belongs to the template.
`leagues/README.md` has the precedence rules and how to ask for no art at all.

Inside a Claude Code session, `.claude/skills/golf-pool/` drives all of this from a
sentence: *"build this week's pool for my-league at the Wyndham"*.

| | |
|---|---|
| `leagues/README.md` | the league file format |
| `frontend/README.md` | the two templates, and what a third one has to honour |
| `docs/FRONTEND-SPEC.md` | the scoreboard's design brief and the full result-JSON schema |

## Quick start — the grouper alone

```bash
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

## The scoring side

Grouping is half the job; the other half is saying who is winning. That runs on ESPN's
unauthenticated leaderboard, and the rule is:

> Rank each team by the best leaderboard position it holds. Teams tied on that are
> separated by their next-best golfer, and so on. Groups are uneven, so a team can run
> out of golfers partway down — when it does, the team that still has one wins.

`standings.py` implements it and `frontend/lib.js` implements it again in the
browser, because the browser is where it actually runs. `tests/test_frontend_parity.py`
feeds both the same payloads and fails if they disagree — two implementations of a rule
are one implementation and one rumour unless something checks them against each other.
The standings rule is the *only* thing that is implemented twice. `lib.js` used to carry
a second copy of the golfer-name matcher as well, mirrored character for character
against Python; it joins on a baked ESPN athlete id now, so that copy is gone and the
parity test has one less subtle string algorithm to keep in step.

Three things about the ESPN payload are measured rather than assumed, and all three
have bitten:

- `competitors[]` is not in rank order.
- `score.displayValue` counts completed rounds only, so mid-round it was wrong for 42
  of 147 players. The running total is the sum of the linescores.
- `sortOrder` *is* the in-play rank — zero inversions — and it already puts every cut
  player below every player who made the cut.

### Joining the two fields, once, at build time

Kalshi's golfer UUID is stable across events and market series but is not an ESPN id, so
the two fields have to be joined by name. That happens once, in a `live`-mode build,
against the competitors ESPN published for **this** tournament — there is no runtime
join any more, and no browser-side copy of the matcher. Three tiers, tried in order, all
three exact or explicit:

| Tier | What it is |
|---|---|
| `decision` | a binding somebody reviewed and recorded for this competition — or a note that the golfer is genuinely not in the field |
| `alias` | an explicit Kalshi-name → ESPN-display-name entry from `data/espn_aliases.json`, reusable every week |
| `exact` | the two normalised display names are equal |

**There is deliberately no fuzzy tier.** Measured on the Rocket Classic — 151 Kalshi
markets against the 147 competitors ESPN listed — `exact` alone resolves 139. Eight of
the twelve it leaves are formal-vs-familiar first names (Zachary/Zach Bauchou,
Cameron/Cam Davis, Nicolas/Nico Echavarria, Matthew/Matt McCarty and four more), and a
first-initial-plus-last-name rule binds all eight correctly. That rule was also measured
collision-free inside that 147-player field, which is exactly the kind of measurement
that does not generalise: the week a field holds two J. Smiths, or a Cameron and a
Carson Young, it binds one of them to the wrong person and nothing downstream can tell
the two cases apart. So it survives as a **suggestion** and never as a match —
`suggest_matches` ranks the right athlete first and says why, and a person settles it.

The other four — Daniel Brown, Taylor Moore, Brooks Koepka, Jason Day — were genuinely
absent, withdrawn before play. "Confirmed absent" and "we could not find him" both score
nothing and are completely different facts, and only the first is knowable by looking,
so the report keeps them in separate piles: `absent` and `unresolved`. Only the second
is a reason to do anything.

Two more things make a build report fewer matches than it had names, and both are
refusals rather than failures:

- **Two Kalshi names that resolve to one athlete.** One ESPN athlete cannot be on two
  teams, so the second name is left unresolved and the report says who already holds
  them. Reviewed decisions are applied in a first pass, so which name counts as
  "second" never depends on how Kalshi happened to sort its markets.
- **One normalised name shared by two athletes in the same field.** The name is dropped
  from the index and *both* golfers come back unresolved. It has not happened in a
  measured field; the day it does, a coin flip is the wrong way to decide which of two
  people is on somebody's team.

Run the join on its own against a published field — `--match` takes a result file, a
Kalshi odds file, a JSON list of names or one name per line:

```bash
python espn_leaderboard.py --season 2026 --find "wyndham"     # the ESPN event id
python espn_leaderboard.py --event 401811961 --match build/result.json
```

`--match` requires `--event`, because the join is against one published field and before
the first tee time there is no field to join against.

### Settling the leftovers, and why it is load-bearing

A `live` build writes every name it would not guess at to `match-review.json` beside the
result file: each unmatched Kalshi name, what it is worth, whose team it is in, the ESPN
athletes nobody claimed, and up to three ranked suggestions with the reason for each.
Heaviest golfer first, because a reviewer who only gets through half the list should get
through the half that moves the standings. A golfer who is simply not in the field comes
back with an empty suggestion list, which is itself the answer — an ESPN field and a
Kalshi field for the same tournament are very nearly the same people.

Somebody — in practice Claude, driving `.claude/skills/golf-pool/` — fills in
`decisions`, binding an athlete id or recording an absence, and the next build reads it
back and records what it applied. Nothing is written at all when every name resolved and
there was nothing to record; a file whose only content is "nothing to do" is a file
somebody has to open to find that out.

`--match-review PATH` points at a different file, and a review file recorded against
another ESPN event is refused outright rather than applied — its athlete ids are ids in
somebody else's field. `--update-aliases` promotes the name bindings (never the
absences: a withdrawal is true of exactly one week) into `data/espn_aliases.json`, where
next month's build resolves them with nobody looking.

**A golfer left unresolved at build time is unscoreable for the life of that page.**
They carry no athlete id, so the page has nothing to look up, and no amount of polling
will change that — every refresh re-reads the same leaderboard and finds the same
nothing. The fix is a rebuild, not a refresh. That is the price of deleting the runtime
name match, and it is worth paying: the join is now checkable, once, by a person, before
it takes effect, instead of being re-guessed in every browser on every poll. See
`espn_leaderboard.py`, `match_review.py` and `docs/FRONTEND-SPEC.md` §8.

### Kalshi will not answer a browser, so the page does not ask it

Its API allowlists request origins — `kalshi.com` gets a 200, every other origin
including `localhost` and `file://` gets a 403 with no CORS headers. The exported page
therefore fetches at most one thing, the ESPN leaderboard, and in `groups` mode not even
that; its odds are baked in and time-stamped as of the draw. There are no live odds and
no relay to configure, in either mode.

There is no way to show prices moving either, and that is deliberate. A rebuild used to
be able to re-read Kalshi server-side and bake a second price in beside the drawn one,
which the page rendered as an arrow. It was accurate and it was a mistake: a golfer
showing two prices — the one his group was dealt on, and one from three days later —
reads as a draw being quietly adjusted after the fact, and a fairness claim that has to
be explained is not doing its job. One reading per competition, taken when the groups
are drawn. `--regroup` is the only thing that ever prices a field again, and it deals a
whole new draw when it does.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # offline, uses checked-in fixtures
KALSHI_LIVE=1 python -m pytest tests/test_live.py -v        # hits the real Kalshi API
ESPN_LIVE=1 python -m pytest tests/test_live_espn.py -v     # hits the real ESPN API
```

The offline suite proves the code is self-consistent. The two real-API suites — the ones
gated on `KALSHI_LIVE` and `ESPN_LIVE`, which have nothing to do with `live` build mode
— prove the endpoints still behave the way the code was measured against: that Kalshi
still answers, still sends money fields as strings and still quotes an ask on every
active market; and that ESPN still returns a whole season's calendar for a one-day
request (12 KB, against 35 MB for the year) and still publishes no field for a
tournament that has not started. That last one is the fact `build_mode` is read off, so
it is worth knowing the day it stops being true. Run them before trusting a season's
first pull.

Three suites need a runtime beyond Python and skip cleanly without it:
`test_frontend_parity.py` needs `node`, and `test_scoreboard_render.py` and
`test_frontend_render.py` drive the two bundled pages in a real browser through
Playwright.

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
