---
name: golf-pool
description: Run a golf pool competition end to end for a league in this repo - pull a tournament's odds from Kalshi, partition the field into equal-value groups, deal them to the league's teams, and export a self-contained scoreboard. Use when the user names a league and a tournament, asks to "build/run/draw the groups", "make this week's pool", "set up a league", "export the scoreboard", or mentions a Kalshi event ticker, an odds type (winner/top5/top10/makecut), or a league JSON. Also use to create or validate a new league file, to rebuild or refresh an existing competition from its result.json ("update the scoreboard", "refresh the odds", "the tournament has started", "redraw the groups"), or to match Kalshi golfer names to ESPN athletes.
---

# Golf pool

Turn **one league + one tournament + one market** into a result file and a
self-contained scoreboard the user can download.

```
league JSON ─┐
tournament  ─┼─> build_competition.py ─> result.json ─> bundle_frontend.py ─> .html + .zip
odds type   ─┘
```

The result file is the source of truth for that competition: teams, groups, the odds
at creation, both endpoints, and the grouping's optimality certificate. The bundle is
the same file baked into a static page.

**Neither is written back into the repository.** Build into `build/` and `dist/`
(both gitignored), hand the files over with SendUserFile, and do not commit them.

---

## Which job is this?

| The user wants | Go to |
|---|---|
| A competition built and exported | §1 |
| A new league defined, or an existing one checked | §2 |
| To know what tournaments are available | §3 |
| An existing competition updated, refreshed or re-drawn | §4 |
| To understand a number in an existing result file | §5 |

**If they have a `result.json` already, §4 is almost always the answer, not §1.**
Rebuilding from it keeps the groups people have been told about; building again from
scratch does not.

---

## 1. Build a competition

### 1.1 Settle three inputs before running anything

**League** — a path under `leagues/`. `ls leagues/*.json`. One file, one league; the
team count is the group count. If they have not made one yet, go to §2.

**Tournament** — a name ("Wyndham", "the Rocket Classic") or a Kalshi event code
(`WYC26`, `KXPGATOUR-WYC26`). Names are resolved live against both APIs, so a name is
fine. **Event codes are not derivable** — 2026 Wyndham is `WYC26`, 2025 was `WC25` —
so never construct one; look it up (§3).

**Odds type** — `winner` (default), `top5`, `top10`, `makecut`.

> **Prefer `winner`, and say so if they ask for another.** Winner ticks at $0.001
> below 10¢; the others are flat $0.01, ten times coarser exactly where a golf field
> lives. Winner outcomes are also mutually exclusive, so the de-vig is a probability.
> Five golfers finish top 5, so a Top 5 book sums toward 5 and its numbers are
> share-of-five-slots — the grouping still balances, but the numbers are weights, not
> odds. Mention it once; it is their pool.

Ask only for what is genuinely missing. A message naming a league and a tournament is
a complete instruction — build it.

### 1.2 Check the timing

Winner markets post **Sunday ~23:00Z of tournament week**, and markets keep being
*added* through Wednesday as the field firms up. **Pull Wednesday night.** A Sunday
pull groups an incomplete field, and it will not look incomplete.

If the field looks short (well under ~140 for a full-field event), say so before
building rather than after.

### 1.3 Run it

```bash
python build_competition.py \
  --league leagues/<league>.json \
  --tournament "<name or ticker>" \
  --odds winner \
  --output build/result.json
```

Useful flags:

| Flag | When |
|---|---|
| `--kalshi-event KXPGATOUR-WYC26` | pin the event; skips name resolution |
| `--espn-event 401811961` | pin the ESPN event; skips name resolution |
| `--seed 42` | reproducible deal of groups to teams |
| `--exclude "Scottie Scheffler"` | drop a named golfer; repeatable |
| `--no-auto-exclude` | keep golfers over the fair-share threshold |
| `--update-aliases` | save newly learned golfer name aliases to the repo |
| `--espn-history-events N` | how far back to look for golfer identities (default 4) |

**Read the output before moving on.** It reports:

- which Kalshi and ESPN events it matched, and the runners-up if the match was fuzzy —
  **if the tournament looks wrong, stop and re-run with `--kalshi-event`**, because a
  wrong tournament produces a completely valid-looking grouping;
- the book sum (a Winner ask book runs ~1.1–1.3; far outside that is a thin or stale
  field);
- who was excluded and why;
- the partition line. `PROVEN OPTIMAL` means no partition of this field does better.
  If it is not optimal, the run names the golfer responsible — someone worth more than
  a whole group's fair share, whom no partition can balance around. The fix is to
  exclude them, not to search harder;
- the ESPN join. Before Thursday it reports "no field published" and then matches the
  names against the season's earlier tournaments instead — expect ~146 of 150 on a full
  field. The handful left over are golfers with no start this season; the page retries
  them by name once the field posts. A join that resolves almost nothing means something
  is wrong (wrong season, ESPN unreachable), not that the golfers are unknown.

### 1.4 Bundle and hand over

```bash
python bundle_frontend.py --result build/result.json --out dist/
```

Writes `dist/<league>-<tournament>.html` (self-contained — opens from disk, no server)
and a matching `.zip` holding the page, `result.json`, and a manifest.

Then send both:

```
SendUserFile(files=["dist/<name>.html", "dist/<name>.zip"], status="normal",
             caption="…")
```

Send the `.html` with `display: "render"` if they want to look at it now, `"attach"`
if they are saving it. Both files, always — the zip carries the `result.json` needed
to rebuild.

### 1.5 Report

Short. The groups are in the file; do not paste 150 golfers into chat. Say:

- which tournament, which market, how many golfers, book sum;
- the partition quality in one line, and whether it is provably optimal;
- who was excluded and why, if anyone;
- each team's total odds, so they can see the draw was even;
- anything odd — a fuzzy tournament match, a short field, unresolved golfers.

If odds come up, say once that the page's odds are the snapshot from the draw and never
change on their own (§6). Do not offer live odds; there are none.

---

## 2. Define or check a league

A league file lives in `leagues/`. Two shapes work; prefer the first.

```json
{
  "league_name": "Sunday Fivesome",
  "teams": [
    { "team_name": "Bogey Boys", "player_name": "Mo", "team_logo": "logos/bogey-boys.svg" },
    { "team_name": "Mulligan Mafia", "player_name": "Luis", "team_logo": null }
  ]
}
```

A bare list of team objects also works and takes its league name from the filename.

- `team_name` and `player_name` are required and `team_name` must be unique — the team
  id is derived from it.
- `team_logo` is optional: a path relative to the league file (inlined into the export
  as a data URI, so it stays portable), an `https://` URL, or null.
- **Do not write `team_id`.** It is a UUIDv5 over (league, team name): stable across
  runs and machines, nothing to persist. `python league.py <file> --write-ids` pins
  them into the file if they ever need to survive a rename.
- One team = one group. Five teams, five groups.

Validate before building:

```bash
python league.py leagues/<name>.json
```

`leagues/example-league.json` is a working five-team example with SVG logos.

---

## 3. Find a tournament

```bash
python kalshi_odds.py --list-events          # golf events with full-field markets
python espn_leaderboard.py --season 2026     # ESPN's calendar with event ids
python espn_leaderboard.py --season 2026 --find "wyndham"
```

Both are unauthenticated. Kalshi rate-limits bursts, so do not loop these.

---

## 4. Update an existing competition

The result file describes the whole competition, so it is the input to the next build of
it. **Rebuild from it rather than building again from scratch** — a fresh build re-pulls
odds and re-partitions, and everyone gets different golfers from the ones they were told
about.

```bash
python build_competition.py --from-result build/result.json --output build/thursday.json
```

Everything comes back out of the file: league, both events, market, price mode, named
exclusions, seed. The groups and the odds at creation are carried forward untouched, and
anything typed on the command line overrides what was recorded. `--from-result` also
takes an exported `.zip` directly — that is usually the copy the user still has.

| They want | Add |
|---|---|
| The golfers matched to ESPN now the field is up | *(nothing — this is the default)* |
| Today's prices shown against the drawn ones | `--refresh-odds` |
| A genuinely new draw | `--regroup` |
| The same groups, a new frontend | nothing — re-bundle (below) |

### 4.1 Which one they actually want

**Default (`--from-result` alone).** Reads no odds at all. Redoes the ESPN join, updates
the tournament state, and rewrites the file. This is the Thursday-morning run: the field
is posted, so every golfer gets an athlete id and a headshot, and withdrawals are named.
It also works after the tournament has finished, which a fresh build does not — settled
markets quote nothing, so a `--regroup` at that point fails by design.

**`--refresh-odds`.** Also re-reads Kalshi and records what the market says now, in
`odds_snapshot.refreshed` and `golfers[].odds.current`. The odds at creation do not
move. This is the **only** way the exported page ever shows prices changing (§6) — the
page presents them as movement since the draw, frozen until the next rebuild. It also
names golfers priced after the draw who are in nobody's group, and drawn golfers whose
market has gone. Re-send the page afterwards; a copy somebody already has will not
update itself.

**`--regroup`.** Pulls fresh odds and partitions again. **Every team gets a different
group.** Only do this if they have asked for a redraw and nobody is holding the old
groups. It refuses to overwrite the file it read unless given `--overwrite`.

### 4.2 After any of them

Re-bundle and re-send, exactly as §1.4. To re-bundle with no rebuild at all — a new
template against the same competition:

```bash
python bundle_frontend.py --result <existing result.json> --template <dir> --out dist/
```

### 4.3 Matching golfer names to ESPN on its own

The join runs inside every build, but it is also a step you can run and read:

```bash
python espn_leaderboard.py --season 2026 --event 401811961 --match build/result.json
```

It prints each Kalshi name, the ESPN athlete it resolved to, the tier that found it, and
— when the field is not posted — which earlier tournament answered. It also reads a
Kalshi odds file, a JSON list of names, or one name per line. Use it when a build
reports unresolved golfers and the question is why.

---

## 5. Explain a result file

Everything is in there; read it rather than recomputing.

| Question | Where |
|---|---|
| Why did I get this group? | `grouping.summary`, `grouping.optimal`, `generator.seed` |
| Was the draw fair? | `teams[].total_odds` — all ≈ 1/n |
| Why is X missing? | `odds_snapshot.excluded[]`, with a reason each |
| What was X worth? | `golfers[].odds` — `raw`, `devigged`, `grouping_weight`, `current` |
| Is X actually playing? | `golfers[].espn.in_field`, and `sources.espn.match_report.not_in_field` |
| Who is X on ESPN, and how do we know? | `golfers[].espn.source` / `.from_event` — `history` means the identity came from an earlier tournament, with no scores attached |
| Where did the numbers come from? | `sources.kalshi` / `sources.espn`, plus `odds_snapshot.captured_at` |
| Has this file been rebuilt? | `rebuilt_from` — null on a first build, else the mode and the count |
| How is the winner decided? | `standings_rules`, and docs/FRONTEND-SPEC.md §6 |

`raw` is the quoted price, `devigged` divides the whole field by the observed book
sum, and `grouping_weight` is what the partitioner actually saw after exclusions —
those sum to 1.0 across every grouped golfer.

---

## 6. Two facts that shape every answer

**Kalshi will not answer a browser, so the page has no live odds — by design.** Its API
allowlists request origins: `kalshi.com` gets a 200, and every other origin — localhost,
GitHub Pages, `file://` — gets a 403 with no CORS headers, preflight included. The
exported page therefore fetches **one** thing, the ESPN leaderboard, and carries its
odds baked in: real, time-stamped, and fixed as of the draw. There is no relay flag and
no live-odds setting to look for. **Never offer or promise live odds.**

There is one honest way to show prices moving: rebuild with `--refresh-odds` (§4) and
re-send the page. That bakes a second reading in beside the first, and the page shows
the movement since the draw. It is a new page each time, not a feed — say that rather
than calling it live.

**ESPN publishes no field before the tournament starts.** A `pre` event returns zero
competitors, so there is no leaderboard to join against on Wednesday night. The build
matches the names against the season's finished tournaments instead and takes identity
from them — athlete id, display name, headshot — and never scores, because those
tournaments are over. Measured on a real 150-golfer field with nothing published: 146
identified. The rest come out `deferred`, which is not a problem: the page redoes the
join by name at runtime once the field exists.

So a page built on Wednesday already knows who everybody is, and shows every roster with
its odds at creation. What it does not show is a ranking, because there is nothing to
rank on yet.

---

## 7. Repository map

| File | What |
|---|---|
| `build_competition.py` | the pipeline → result JSON, and the rebuild of one (§4) |
| `bundle_frontend.py` | result JSON + template → self-contained page + zip |
| `league.py` | league file loading, validation, team ids |
| `espn_leaderboard.py` | ESPN event resolution, leaderboard parsing, the name join — including the fallback to earlier tournaments when this week's field is empty |
| `standings.py` | the standings rule, in Python, for testing |
| `frontend/template/` | the reference scoreboard — `lib.js` holds the rules |
| `docs/FRONTEND-SPEC.md` | the design brief and the full result-JSON schema |
| `groupers.py`, `group.py`, `kalshi_odds.py` | the engine; see README.md |

Do not edit the engine to change a competition. Every knob is a flag.

Tests: `python -m pytest tests/ -q`. The parity test proves `lib.js` and
`standings.py` agree; the render test drives the bundled page in a real browser.
Run both after touching the frontend or the standings rule.
