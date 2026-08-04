---
name: golf-pool
description: Run a golf pool competition end to end for a league in this repo - pull a tournament's odds from Kalshi, partition the field into equal-value groups, deal them to the league's teams, and export a self-contained scoreboard. Use when the user names a league and a tournament, asks to "build/run/draw the groups", "make this week's pool", "set up a league", "export the scoreboard", or mentions a Kalshi event ticker, an odds type (winner/top5/top10/makecut), or a league JSON. Also use to create or validate a new league file, to rebuild an existing competition from its result.json ("update the scoreboard", "the tournament has started", "redraw the groups", or "refresh the odds" - which routes here so §6 can explain that odds are read once, at the draw, and never re-read), or to match Kalshi golfer names to ESPN athletes - including settling the golfers a build reported as NEEDS REVIEW or unresolved, by filling in the decisions in a match-review.json and rebuilding.
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

**A normal week is two runs of that pipeline over one competition.** Before the
tournament ESPN has published no field, so the first run puts out a groups sheet —
rosters and the odds they were drawn on, nothing scored. Once play starts, the same
competition is rebuilt from its own result file and becomes a scoreboard. §1 is the
first run, §4 is the second; neither is a workaround for the other.

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
| The scoreboard now that play has started | §4 — it is a rebuild, not a new build |
| The golfers a build called NEEDS REVIEW settled | §4.2 |
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

### 1.2 Check the timing, and expect to build twice

Winner markets post **Sunday ~23:00Z of tournament week**, and markets keep being
*added* through Wednesday as the field firms up. **Pull Wednesday night.** A Sunday
pull groups an incomplete field, and it will not look incomplete.

If the field looks short (well under ~140 for a full-field event), say so before
building rather than after.

ESPN runs on the other clock: it publishes **zero competitors until the first tee
time** (§6). The build reads that off the leaderboard payload and picks its own mode.
There is no flag for this and nothing to decide:

| When it runs | `build_mode` | What comes out |
|---|---|---|
| Before the first tee time | `groups` | rosters and the odds they were drawn on. `live` is null, every golfer's `espn` is null, and the page fetches nothing at all |
| Once play has started | `live` | the same groups, an ESPN athlete id on every golfer the join settled, and a page that polls the leaderboard and ranks |

**A groups build is not a half-built one.** It is the groups sheet — the thing that
exists on Wednesday and the thing people want on Wednesday — and it is finished:
bundle it, send it, say that the ranking arrives when play does. Then run §4 against
that same result file on Thursday. That second run is what turns the competition into
a scoreboard, and it re-draws nothing.

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
| `--crest art/crest.png` | the league's badge, supplied at creation; beats the league file (§2) |
| `--banner art/banner.png` | the wide image across the top, same (§2) |
| `--no-crest` / `--no-banner` | build with none, whatever the league file says |
| `--exclude "Scottie Scheffler"` | drop a named golfer; repeatable |
| `--no-auto-exclude` | keep golfers over the fair-share threshold |
| `--match-review build/match-review.json` | the file reviewed name decisions are read from and written to; defaults to `match-review.json` beside `--output` (§4.2) |
| `--update-aliases` | keep the name bindings settled in that file as reusable aliases (§4.2) |

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
- the `ESPN:` line, which is where you find out which of the two builds this was. Three
  answers, and only the third is a problem:
  - `no field published` — the normal Wednesday answer. This is a groups build, nothing
    is scored, nothing needs fixing, and §4 is what adds the scoring later;
  - `N competitors in the field` — the join ran, and the `Match:` line underneath says
    how many Kalshi names it settled and by which tier (exact, alias, reviewed). The
    names it will not guess at come back as **NEEDS REVIEW**, listed, with a file to
    settle them in: that is §4.2, and it is a normal part of a live build rather than a
    failure;
  - `unreadable` — ESPN did not answer. The build falls back to a groups sheet, which is
    honest but is a groups sheet by accident on a day that should have had scoring.
    Re-run before handing anything over.

### 1.4 Bundle and hand over

```bash
python bundle_frontend.py --result build/result.json --out dist/
```

Writes `dist/<league>-<tournament>.html` (self-contained — opens from disk, no server)
and a matching `.zip` holding the page, `result.json`, and a manifest.

No `--template` is the right call: the default is the designed scoreboard
(`frontend/scoreboard/`), which is the page people want to be handed. There is a plain
reference at `frontend/template/` — `--template frontend/template` — and the only
reason to bundle it is diagnostic: if a number looks wrong on the designed page, the
reference draws the same data with no design in the way and says whether the page or
the file is at fault. Do not hand it over as the deliverable.

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
- **which of the two builds this is** — a groups sheet with the ranking still to come,
  or a live scoreboard — and, if it is live, how many golfers need a decision (§4.2);
- anything odd — a fuzzy tournament match, a short field.

If odds come up, say once that the page's odds are the snapshot from the draw and never
change on their own (§6). Do not offer live odds; there are none.

---

## 2. Define or check a league

A league file lives in `leagues/`. Two shapes work; prefer the first.

```json
{
  "league_name": "Sunday Fivesome",
  "tagline": "Season 4",
  "crest": "logos/example-crest.svg",
  "banner": "logos/example-banner.svg",
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

### The masthead

`crest`, `banner` and `tagline` fill the scoreboard's masthead — a crest beside the
name, a wide banner across the top, a line of small caps under the name. They are the
only part of the page that differs between leagues: the navy-and-gold chrome, the type
and the layout are the template's and are identical for every competition. Do not
restyle the page for a league; swap its two images.

The two images can also be handed in when the competition is created, which is the
usual route when somebody supplies art alongside the roster:

```bash
python build_competition.py --league leagues/wcw.json --tournament Wyndham \
    --crest art/crest.png --banner art/banner.png
```

`--crest` / `--banner` beat what the league file says; those paths resolve against the
working directory. If neither is given and the file says nothing, the build uses the
art the tool ships and prints a note saying so — so **a league with no art still gets a
finished-looking page, and you do not need to go and find some.** `--no-crest` /
`--no-banner` build with none, and `"crest": false` in the file is the standing form of
that. `tagline` has no default: do not invent one nobody asked for.

Paths in the file resolve against the league file and are inlined as data URIs like the
team logos, so **keep them small**: anything over 512 KB is refused and left as a path
that will not resolve in the export, and every inlined byte lands in every copy of the
page. A crest around 256 px and a banner around 720 px wide is the right order of
magnitude; a JPEG beats a PNG for a photographic banner by roughly five to one. The
banner is centre-cropped into a wide, short slot, so art with detail at the top and
bottom edges loses it.

Once built it is settled: the result file holds the data URIs, and a rebuild carries
them forward — including a deliberate absence. A rebuild never grows a crest the first
build did not have.

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
python build_competition.py --from-result build/result.json --output build/result.json
```

Everything comes back out of the file: league, both events, market, price mode, named
exclusions, seed, and any name decisions already applied. The groups and the odds at
creation are carried forward untouched, and anything typed on the command line overrides
what was recorded. `--from-result` also takes an exported `.zip` directly — that is
usually the copy the user still has.

**This is the second run of the normal week** (§1.2), and it is the same result file
both times: one competition, one file, brought up to date. Writing it back in place is
the ordinary thing to do — the rebuild carries the draw forward verbatim, records what
it was rebuilt from, and only writes once everything upstream of it succeeded. Give
`--output` a different path only when they want the old file kept as well; keep it in
the same directory either way, because the review file (§4.2) lives beside `--output`.

| They want | Add |
|---|---|
| The scoreboard, now play has started | *(nothing — this is the default)* |
| The golfers that came back NEEDS REVIEW settled | `--match-review build/match-review.json` (§4.2) |
| Those name bindings kept for next week | `--update-aliases` (§4.2) |
| A genuinely new draw | `--regroup` |
| The same groups, a new frontend | nothing — re-bundle (§4.3) |

### 4.1 Which one they actually want

**Default (`--from-result` alone).** Reads no odds at all. Redoes the ESPN join, updates
the tournament state, and rewrites the file. This is the Thursday-morning run and the
one that flips `build_mode` from `groups` to `live`: the field is posted, so every
golfer the join settles gets an athlete id and a headshot, the page starts polling and
ranking, and the names it will not guess at are listed for review (§4.2). It also works
after the tournament has finished, which a fresh build does not — settled markets quote
nothing, so a `--regroup` at that point fails by design.

It refuses one thing, loudly: a rebuild of a competition that **had** a field and now
finds none stops without writing. ESPN does not unpublish a field, so that is a failed
read rather than news, and rebuilding on it would null out a working scoreboard and look
like a normal build. The file you passed is untouched; try again, or pin the event with
`--espn-event <id>` if the lookup is what is failing.

**`--regroup`.** Pulls fresh odds and partitions again. **Every team gets a different
group.** Only do this if they have asked for a redraw and nobody is holding the old
groups. It refuses to overwrite the file it read unless given `--overwrite`.

### 4.2 Settle the golfers that came back NEEDS REVIEW

A live build joins the Kalshi names onto this week's ESPN field in three tiers, and all
three are exact: a decision somebody recorded, an explicit alias, or two normalised
names being equal. There is deliberately no fuzzy tier — a first-initial-and-last-name
rule binds Nicolas Echavarria to Nico Echavarria correctly and binds one of two
J. Smiths wrong, and nothing downstream can tell those two cases apart. So the names the
join will not guess at come back named rather than guessed:

```
  NEEDS REVIEW (12): Zachary Bauchou, Cameron Davis, Nicolas Echavarria, Jason Day, ...
Review -> build/match-review.json  (12 golfer(s) need a decision)
```

Expect a handful. On the measured Rocket Classic field, 139 of 151 Kalshi names resolved
outright, 12 came back for review, and 4 of those 12 turned out to be withdrawals.

**This is not a blocker.** The build already wrote a complete result file and it is
deployable exactly as it stands; the golfers left open simply score nothing, and their
teams score nothing from them. If the user asked for the groups, hand the build over and
mention the number. The review is what you do when the standings matter — an unresolved
golfer worth 4% sitting in somebody's roster is the whole scoreboard.

**Open the file and read it.** Each `pending` entry is one unsettled Kalshi name,
heaviest first, with the ESPN athletes nobody claimed offered as ranked suggestions:

```json
{
  "kalshi_name": "Nicolas Echavarria",
  "team": "Bogey Boys",
  "grouping_weight": 0.0121,
  "suggestions": [
    { "athlete_id": "4588361", "espn_name": "Nico Echavarria", "position": "T14",
      "confidence": 0.909, "why": "same first initial and last name" }
  ]
}
```

`espn_athletes_nobody_claimed` under it is the rest of the field with nobody on it — the
complete list of people an answer can be.

**Write one entry per pending name into `decisions`**, keyed by the exact `kalshi_name`
string:

```json
"decisions": {
  "Nicolas Echavarria": { "athlete_id": "4588361", "espn_name": "Nico Echavarria" },
  "Jason Day": { "absent": true, "note": "withdrew before the first round" }
}
```

Two kinds of answer, and they are genuinely different. `athlete_id` **binds**: this
Kalshi golfer is that ESPN athlete, and the page shows that athlete's scores under this
golfer's name. `absent` records that the golfer is **not in the field at all** — a
withdrawal. Both end with the golfer scoring nothing this week; the difference is that
the second is a fact somebody checked, and the file will say so instead of leaving them
looking missed. `athlete_id` is the only thing that binds — `espn_name` and `note` are
for readers, and `espn_name` is what an alias gets learned from.

Two ways to get this wrong, both easy:

- **Do not copy the top suggestion because it is the top suggestion.** Confidence ranks
  candidates, it does not settle them; the generator is the very rule the join refuses
  to trust. Read the two names against each other. Zachary/Zach Bauchou is obvious; two
  different golfers who happen to share a surname is not, and the file takes whatever
  you write.
- **Do not invent an `athlete_id`.** Use one that appears in this file — in a
  `suggestions` list or in `espn_athletes_nobody_claimed`. An id from anywhere else is
  not in this field, so the build prints the problem and ignores that decision: the name
  drops back to the automatic tiers it was already failing, and the golfer you meant to
  settle is exactly as unsettled as before, under a `decisions` entry that looks done.

**An empty `suggestions` list is an answer, not a gap.** It means nobody left in the
field resembles that name even loosely, and a Kalshi field and an ESPN field for the
same tournament are very nearly the same people — so check `espn_athletes_nobody_claimed`
once, and if they are not in it either, they withdrew. Record the absence.

Then rebuild, and the decisions take effect:

```bash
python build_competition.py --from-result build/result.json --output build/result.json \
  --match-review build/match-review.json --update-aliases
```

`--match-review` names the file. Leave it off and the build reads `match-review.json`
beside `--output` anyway, which is the same file — pass it when you want the intent on
the command line. The rebuild counts the settled names as `reviewed` on the `Match:`
line, records them in `sources.espn.match_decisions` so the next rebuild does not re-ask
a question somebody answered, and rewrites `match-review.json` with whatever is still
open.

`--update-aliases` is the part that pays next week. "Nicolas Echavarria is Nico
Echavarria" is true of every tournament, so it is written into `data/espn_aliases.json`
— the one repo file a build touches — and next month's build resolves that name with
nobody looking. Absences are deliberately not learned: "Jason Day withdrew" is true of
one week. Without the flag the run still prints what it would have kept and saves
nothing, so it is safe to see the list first and decide.

One refusal to know about: a review file recorded against a **different ESPN event** is
ignored wholesale, with a sentence saying so. Its athlete ids are ids in another field
and half those golfers may not even be playing this week. Delete it and let the build
write a fresh one.

### 4.3 After any of them

Re-bundle and re-send, exactly as §1.4. To re-bundle with no rebuild at all — a new
template against the same competition:

```bash
python bundle_frontend.py --result <existing result.json> --template <dir> --out dist/
```

### 4.4 Matching golfer names to ESPN on its own

The join runs inside every live build, but it is also a step you can run and read
without writing anything:

```bash
python espn_leaderboard.py --event 401811961 --match build/result.json \
  --aliases data/espn_aliases.json --decisions build/match-review.json
```

`--match` requires `--event`, because the join is against one published field and there
is exactly one: before the first tee time there is nothing to match against, and the
command says so rather than reaching for a substitute. `--season 2026 --find "wyndham"`
is how you get the id (§3).

It prints each Kalshi name, the ESPN athlete it resolved to and the tier that found it,
then the unresolved ones with the same ranked candidates the review file carries.
`--json` gives the matches, the report and the suggestions as data. It reads a result
file, a Kalshi odds file, a JSON list of names, or one name per line. Use it when a
build reports golfers needing review and the question is why, or to try a decision out
before committing it to the file.

---

## 5. Explain a result file

Everything is in there; read it rather than recomputing.

| Question | Where |
|---|---|
| Is this a groups sheet or a scoreboard? | `build_mode` — `groups` or `live`. Read it first; it says which half of the file exists |
| Why did I get this group? | `grouping.summary`, `grouping.optimal`, `generator.seed` |
| Was the draw fair? | `teams[].total_odds` — all ≈ 1/n |
| Why is X missing? | `odds_snapshot.excluded[]`, with a reason each |
| What was X worth? | `golfers[].odds` — `raw`, `devigged`, `grouping_weight` |
| Is X actually playing? | `golfers[].espn.in_field` — true, false (checked, and they are not), or null (nobody has looked) |
| Who is X on ESPN, and how do we know? | `golfers[].espn.athlete_id` / `.display_name`, and `.match` — `exact`, `alias`, `decision`, `absent` or `unresolved` |
| Which golfers still need a decision? | `sources.espn.match_report.unresolved`, beside `.absent` for the ones already settled (§4.2) |
| Which names did somebody settle by hand? | `sources.espn.match_decisions`; `sources.espn.aliases_applied` for the ones the alias file answered |
| Where did the numbers come from? | `sources.kalshi` / `sources.espn`, plus `odds_snapshot.captured_at` |
| Has this file been rebuilt? | `rebuilt_from` — null on a first build, else the mode and the count |
| How is the winner decided? | `standings_rules`, and docs/FRONTEND-SPEC.md §6 |

`raw` is the quoted price, `devigged` divides the whole field by the observed book
sum, and `grouping_weight` is what the partitioner actually saw after exclusions —
those sum to 1.0 across every grouped golfer.

In a `groups` build, `live`, `sources.espn.match_report` and every golfer's `espn` are
**null** — absent rather than empty, because there was no field to ask about and a shell
of nulls would read as a join that was tried and failed.
`sources.espn.field_size_at_build` is 0, and that is the reason it is a groups build.

---

## 6. Two facts that shape every answer

**Kalshi will not answer a browser, so the page has no live odds — by design.** Its API
allowlists request origins: `kalshi.com` gets a 200, and every other origin — localhost,
GitHub Pages, `file://` — gets a 403 with no CORS headers, preflight included. The
exported page therefore fetches **one** thing, the ESPN leaderboard, and carries its
odds baked in: real, time-stamped, and fixed as of the draw. There is no relay flag and
no live-odds setting to look for. **Never offer or promise live odds.**

There is no honest way to show prices moving at all, so do not look for one. A rebuild
used to be able to re-read the market and bake a second price in beside the drawn one;
that was removed, because a golfer showing two prices — the one his group was dealt on
and one from three days later — reads as a draw being adjusted after the fact. One
reading per competition, taken when the groups are drawn. `--regroup` prices a field
again only because it is dealing a completely new draw.

**ESPN publishes no field before the tournament starts.** A `pre` event returns zero
competitors, so on Wednesday night there is nobody to join the Kalshi names against — no
athlete ids, no headshots, and nothing to score. The build does not work around that. It
reads the field size off the payload, writes `build_mode: "groups"`, leaves `live` and
every golfer's `espn` block null, and produces a page that fetches nothing and ranks
nothing. In particular it does **not** match the field against last month's leaderboards
to recover identities: that answers a question the page cannot use — it needs this
week's scores — with a join nothing can check.

So a page built on Wednesday is the groups sheet: every team, every roster, the odds
each golfer was drawn at, no standings. It is complete, it is deployable, and it opens
from disk on a plane. The ranking arrives on the second run (§4), once the field exists
— the same file and the same groups, now with an ESPN athlete id on every golfer the
join settled and a handful listed for review (§4.2).

---

## 7. Repository map

| File | What |
|---|---|
| `build_competition.py` | the pipeline → result JSON, and the rebuild of one (§4) |
| `bundle_frontend.py` | result JSON + template → self-contained page + zip |
| `league.py` | league file loading, validation, team ids |
| `espn_leaderboard.py` | ESPN event resolution, leaderboard parsing, and the name join against one published field — three exact tiers, plus the suggester for whatever they leave |
| `match_review.py` | the worksheet those leftover names get settled in, and the aliases a settlement is worth keeping (§4.2) |
| `data/espn_aliases.json` | the learned Kalshi → ESPN name aliases. Written only by `--update-aliases`, created on first use, safe to hand-edit |
| `standings.py` | the standings rule, in Python, for testing |
| `frontend/scoreboard/` | the designed scoreboard — what a plain bundle produces |
| `frontend/template/` | the plain reference page, for telling a bad page from bad data |
| `frontend/lib.js` | the standings rule in JavaScript. One copy, inlined by both |
| `docs/FRONTEND-SPEC.md` | the design brief and the full result-JSON schema |
| `groupers.py`, `group.py`, `kalshi_odds.py` | the engine; see README.md |

Do not edit the engine to change a competition. Every knob is a flag.

Tests: `python -m pytest tests/ -q`. The parity test proves `lib.js` and
`standings.py` agree; `test_scoreboard_render.py` and `test_frontend_render.py` drive
the two bundled pages in a real browser. Run all three after touching the frontend or
the standings rule.
