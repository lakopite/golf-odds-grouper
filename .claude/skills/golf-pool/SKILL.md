---
name: golf-pool
description: Run a golf pool competition end to end for a league in this repo - pull a tournament's odds from Kalshi, partition the field into equal-value groups, deal them to the league's teams, and export a self-contained scoreboard. Use when the user names a league and a tournament, asks to "build/run/draw the groups", "make this week's pool", "set up a league", "export the scoreboard", or mentions a Kalshi event ticker, an odds type (winner/top5/top10/makecut), or a league JSON. Also use to create or validate a new league file.
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
| Something rebuilt with a change | §4 |
| To understand a number in an existing result file | §5 |

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
| `--kalshi-proxy "https://relay/?u={url}"` | enable live odds in the page (§6) |
| `--update-aliases` | save newly learned golfer name aliases to the repo |

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
- the ESPN join, or `deferred` if the field is not posted yet, which is normal before
  Thursday and is handled at runtime by the page.

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

Mention that live odds are snapshot-only unless a relay is configured (§6) — once, not
every time.

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

## 4. Rebuild with a change

Re-run §1.3 with the change and §1.4 again. Two things to keep in mind:

- **The deal is random unless seeded.** Rebuilding without `--seed` gives every team a
  different group. If they wanted the same groups with a different frontend, pass the
  seed from the old file (`generator.seed`) — or skip the rebuild entirely and just
  re-bundle the existing `result.json` against the new template.
- **The odds move.** A rebuild on Thursday is a different snapshot from Wednesday's and
  will produce different groups. Say so before doing it.

To re-bundle without re-pulling:

```bash
python bundle_frontend.py --result <existing result.json> --template <dir> --out dist/
```

---

## 5. Explain a result file

Everything is in there; read it rather than recomputing.

| Question | Where |
|---|---|
| Why did I get this group? | `grouping.summary`, `grouping.optimal`, `generator.seed` |
| Was the draw fair? | `teams[].total_odds` — all ≈ 1/n |
| Why is X missing? | `odds_snapshot.excluded[]`, with a reason each |
| What was X worth? | `golfers[].odds` — `raw`, `devigged`, `grouping_weight` |
| Where did the numbers come from? | `sources.kalshi` / `sources.espn`, plus `odds_snapshot.captured_at` |
| How is the winner decided? | `standings_rules`, and docs/FRONTEND-SPEC.md §6 |

`raw` is the quoted price, `devigged` divides the whole field by the observed book
sum, and `grouping_weight` is what the partitioner actually saw after exclusions —
those sum to 1.0 across every grouped golfer.

---

## 6. Two facts that shape every answer

**Kalshi will not answer a browser.** Its API allowlists request origins: `kalshi.com`
gets a 200, and every other origin — localhost, GitHub Pages, `file://` — gets a 403
with no CORS headers, preflight included. So the exported page cannot fetch live odds.
It always carries the snapshot, which is real and time-stamped, and shows live odds
only if `--kalshi-proxy` gave it a relay. Never promise live odds without one.

**ESPN publishes no field before the tournament starts.** A `pre` event returns zero
competitors, so a Wednesday-night build cannot finish the golfer→athlete join. That is
expected: the page redoes the join at runtime by name (three tiers, measured to resolve
every golfer ESPN actually lists). `deferred` in the build output is not a problem.

---

## 7. Repository map

| File | What |
|---|---|
| `build_competition.py` | the pipeline → result JSON |
| `bundle_frontend.py` | result JSON + template → self-contained page + zip |
| `league.py` | league file loading, validation, team ids |
| `espn_leaderboard.py` | ESPN event resolution, leaderboard parsing, the name join |
| `standings.py` | the standings rule, in Python, for testing |
| `frontend/template/` | the reference scoreboard — `lib.js` holds the rules |
| `docs/FRONTEND-SPEC.md` | the design brief and the full result-JSON schema |
| `groupers.py`, `group.py`, `kalshi_odds.py` | the engine; see README.md |

Do not edit the engine to change a competition. Every knob is a flag.

Tests: `python -m pytest tests/ -q`. The parity test proves `lib.js` and
`standings.py` agree; the render test drives the bundled page in a real browser.
Run both after touching the frontend or the standings rule.
