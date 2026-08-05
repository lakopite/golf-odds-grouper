# Scoreboard spec

The design brief for the page a golf pool actually watches on Sunday afternoon.

It is a **single static HTML file**. No server, no build step, no backend, no
database. Everything it knows about the competition is baked into it at build time;
the most it ever fetches while running is the ESPN leaderboard, and a page built
before the field posted does not fetch even that (§2). It opens from a `file://` URL,
from a USB stick, from Dropbox, from anywhere.

Two implementations ship. `frontend/scoreboard/` is the designed page and the one
`bundle_frontend.py` produces by default; `frontend/template/` is a deliberately plain
reference that exists to prove the contract rather than to be the design — read it for
*what*, and ignore it for *how*. Both inline `frontend/lib.js`, which is where the rule
in §6 actually lives. See `frontend/README.md`.

> **The one thing to read first.** §4. Kalshi will not answer a browser, so **the page
> fetches at most one thing: the ESPN leaderboard.** There is no live-odds panel, no
> relay, no setting that turns one on. Odds are baked in and stated as of the moment
> they were captured. Any design that treats "live odds" as a thing the page can fetch
> is designing a panel that is permanently empty for every user.
>
> **The second thing.** §2. Whether the page is a scoreboard or a groups sheet is not a
> fact in the file — it is a fact about the clock, read off every poll. The page is
> built the night before and shows the draw until somebody tees off.

---

## 1. What the page is for

Ten to fifteen people each own a group of golfers in one tournament. The groups were
drawn to be equal in betting value, so nobody had a better draw than anybody else, and
now the whole week is settled by one question: **whose golfer is highest on the
leaderboard right now.**

The page answers that, and then answers the three follow-up questions in order:

1. **Who is winning?** — the league table.
2. **Why?** — which golfer of theirs is doing it, and by how much over the next team.
3. **What have I got?** — my golfers, where each of them stands, what they were worth
   when the groups were drawn.

Design for a phone held one-handed in front of a television, refreshed forty times on
a Sunday. Design for the fact that two teams are usually within one golfer of each
other and the interesting information is the *margin*, not the standings.

---

## 2. Where the data comes from

| | Baked in at build time | Fetched in the browser |
|---|---|---|
| Teams, logos, rosters | ✅ | — |
| Odds when the groups were drawn | ✅ | — |
| Which endpoints produced the numbers | ✅ | — |
| Grouping quality certificate | ✅ | — |
| Golfer → ESPN athlete id | ✅ — see §8 | — |
| Scores and positions | — | ✅ ESPN |

**ESPN is the only row on the right, and that is the whole of the page's network.**

### The file is never half-empty, and the clock is not in it

ESPN posts a tournament's field about **two days before the first round** — measured
2026-08-04 against the Wyndham, whose first round was the 6th: 147 competitors with
athlete ids, headshots and tee times, while the event was still `state: "pre"`. So a
competition drawn the night before is a complete one. Every golfer already carries an
athlete id, `DATA.live` is always an object, and `golfers[].espn` is always an object.
**Assume both are present.** (Before schema 4.0 they could be `null`; a file that old
is refused by the bundler rather than rendered.)

What a `pre` payload does **not** carry is a position. All 147 came back at `"-"`, with
`sortOrder` a dense 1..147. So a field is joinable long before it is rankable, and the
one question every page has to ask is which of those two it is looking at:

```js
const {meta, players} = GolfPool.parseLeaderboard(payload);
if (meta.started) { /* rank */ } else { /* draw the groups */ }
```

**`meta.started` is the gate, and it is not `state === "in"`.** `GolfPool.hasStarted`
(mirrored by `espn_leaderboard.has_started`, and the two are held together by
`tests/test_frontend_parity.py`) answers true on `state` in `"in"`/`"post"` **or** on
any golfer holding a real position. Two signals, either sufficient, because they fail
in opposite directions: the state is read off the event envelope, so a stale or missing
one would blank a board that is plainly live, and a golfer with a position is proof
from the field itself. A page that ranks on `players.length` alone will order the
league by ESPN's pre-tournament sort and print it as a leaderboard — complete, ordered,
with a leader and tie-breaks, and entirely invented. That is a far more convincing way
to be wrong than showing nothing.

This is also what makes a page change over **by itself**. It polls from the moment it
opens, days early, because that poll is how it learns the tournament has started; the
draw becomes a scoreboard while the tab is sitting open, with nothing rebuilt and no
second link. Anything that switches on it — the heading, the tab label, the column
headers — must therefore be set on every render and not once at load.

*A note on the word.* Two unrelated things in this project are called "live": the
`live` block in the result JSON, and the test suites that hit the real APIs. This
document always says which one it means. (A third, the `build_mode` value `"live"`, was
removed in 4.0 along with the mode itself.)

Everything baked in is one JSON object embedded in the page:

```html
<script id="competition-data" type="application/json">{ … }</script>
```

```js
const DATA = JSON.parse(document.getElementById('competition-data').textContent);
```

It is produced by `build_competition.py`; its full shape is §7. Nothing on the page
should re-derive a number the file already states — if a number seems to be missing,
it belongs in the file, not in the page.

---

## 3. ESPN — the leaderboard

Every page does this, including one opened days before the tournament: the poll is what
tells it play has started (§2). Until then it changes nothing on screen.

```
https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga&event=<id>
```

Unauthenticated. Sends `access-control-allow-origin: *`. Fetch it directly from the
browser; it works from `file://`. Poll it on `DATA.live.poll_interval_seconds`
(default 60).

**Use the URL the file gives you**, `DATA.live.espn_leaderboard_url` — the pinned
`event` id is already in it. Without that id the endpoint answers about whatever ESPN
thinks is current, which next Thursday is a different tournament. **Verify the id in
the response** against `DATA.live.espn_event_id` and refuse a payload that does not
match — scoring a league against the wrong tournament looks entirely normal until
someone notices the names.

### The four things in this payload that are not what they look like

Measured against a mid-tournament Round 2 payload (`espn-api/lb.json`, 147 competitors)
and the same tournament's final payload.

1. **`competitors[]` is not in rank order.** Sort it.

2. **`score.displayValue` is stale mid-round.** It counts *completed rounds only*, so
   a player halfway through round 2 shows their round-1 total. It was wrong for **42
   of 147** players. Never rank on it, never display it. The running total is the sum
   of the `linescores[].displayValue` values.

3. **`linescores[]` contains stubs** for rounds not yet played — no `value`, no
   `displayValue` key at all. Filter before summing. A withdrawn round reads `"-"`.

   That filtering is why the parsed `rounds[]` **must not be indexed from the end**. A
   golfer in the afternoon wave already has a linescore for the round in progress and its
   `displayValue` is `"-"`, so it drops out and their array stops at the round before —
   six of the 147 in `espn-api/lb.json`, whose `rounds` end at round 1 while round 2 is
   being played. `rounds[rounds.length - 1]` hands back *yesterday's* score to be printed
   under today's heading: a real number, in range, beside the right name, and wrong.
   Select the round by its `period`, and render nothing when there is no entry for it.

4. **`sortOrder` is the in-play rank**, and it is the only field that is. Zero
   inversions against the running total; the stale score field inverts 29 times. It is
   a total order `1..N` over the whole field, and it places every cut player (74–147)
   below every player who made the cut (1–73).

Also: the current round is `competitions[0].status.period`, **not** `event.status`.
And `status.position.isTie` is already a boolean — no need to parse the `T`.

### States the page has to handle

| State | What ESPN returns | What the page shows |
|---|---|---|
| Not started yet | `state: "pre"`, **the full field, every position `"-"`** | every roster with its odds at creation, "not started", **nothing ranked**. This is how every page spends its first day or two |
| No field at all | zero competitors | the same, but say so differently — an empty field once a tournament is under way is a bad answer, not an early one |
| In progress | `state: "in"`, positions moving | the full board |
| After the cut | 74 of 147 have position `"-"` | cut golfers visibly out, still listed |
| Withdrawn / DQ | `status.type` `STATUS_WD`/`STATUS_DQ`, no position | same treatment as cut, labelled |
| Finished | `state: "post"`, `completed: true` | final, stop implying it will change |
| ESPN unreachable | — | last-known board, plus a visible staleness marker |

### Never rank an empty board

This guard has nothing to do with the name join, and it is a separate thing from the
`meta.started` gate above — keep both. A page can be handed an empty board any
afternoon of the week: ESPN unreachable, a payload refused for the wrong event (above),
a fetch that fails on a train, a first poll that has not returned yet. All of those
arrive as zero competitors, and none of them is rare.

Everything the pool cares about is already in the file regardless: who holds whom, what
each golfer was worth, and that the draw came out even. Show all of it. What must
**not** appear is a ranking — running the standings rule over an empty leaderboard puts
every golfer in tier 2 (§6), which orders the teams by roster size and presents it as a
leaderboard. Positions read `—` until there are positions. The rosters and the odds
stay, because they are baked in and nothing the network did can affect them.

---

## 4. Kalshi — why the page never fetches odds

**Kalshi's API allowlists request origins.** Measured 2026-08-03:

| `Origin` sent | Response |
|---|---|
| *(none — curl, server-side)* | 200 |
| `https://kalshi.com` | 200, `ACAO: https://kalshi.com` |
| `https://elections.kalshi.com` | 200, echoed |
| `https://www.kalshi.com` | **403** |
| `http://localhost:8000` | **403** |
| `https://lakopite.github.io` | **403** |
| `null` (a `file://` page) | **403** |
| `OPTIONS` preflight from any other origin | **403** |

A browser always sends `Origin` on a cross-origin `fetch`. So **a static page cannot
read odds from Kalshi**, on any host, ever, without something in between.

This is not a bug to route around later. It is a fixed property of the design, and the
design answers it by not fetching odds at all.

### What this means for the page

**The page has exactly one network dependency: ESPN.** There is no odds request, no
relay setting, no "live odds unavailable" state to design, and no empty panel to fill.
The result file carries no Kalshi URL for the page to try, because trying it can only
fail. And ESPN is a dependency for the *ranking* only: the draw is baked in, so a page
that cannot reach anything still opens from disk with every roster and every price on
it. Losing the network must cost the positions and nothing else.

**The snapshot is the odds display.** Every golfer's price at the moment the groups
were drawn is in `DATA.golfers[].odds`, along with the book it came from and the time
it was captured. That is a real, checkable, permanent number, and it is the honest
answer to "what was he worth?" — the question the pool is actually asking. Show it,
and show `odds_snapshot.captured_at` beside it so it is never mistaken for a price
that is moving.

### There is exactly one price per golfer, and prices never move

Not "no live odds, but a rebuild can show movement". None. A rebuild does not re-read
Kalshi, there is no flag that makes it, and the result file has nowhere to put a second
reading: `golfers[].odds` carries `raw`, `devigged` and `grouping_weight`, all three
read at the same instant, and that instant is `odds_snapshot.captured_at`.

This used to work the other way. `--refresh-odds` re-read the market server-side and
baked a second price in beside the drawn one, which the page rendered as an arrow. It
was accurate and it was a mistake: a golfer showing two prices — the one his group was
dealt on and one from three days later — reads as a draw being adjusted after the fact.
Every reader who noticed it had to be talked back out of the same suspicion, and a
fairness claim that needs explaining is not doing its job.

So: **a page must never show a price moving**, and should not carry a column, a slot or
an empty state for one. If a result file from an older build still carries
`odds_snapshot.refreshed` or `golfers[].odds.current`, ignore both.

---

## 5. What the page must show

### 5.1 The league table — the headline

One row or card per team, in finishing order. Each carries:

- **Position** — `1`, `2`, `T3` … Ties are real and must be shown as ties (§6).
- **Team identity** — `team_name`, `player_name`, `team_logo` (already a data URI or a
  remote URL; may be null — design the fallback).
- **The golfer who is holding them up** — their best golfer's name and position. This
  is the single most-looked-at fact on the page after the position itself.
- **The margin** — `decided_at` says which golfer down the list separated this team
  from the one above it. "Won on their 3rd golfer" is the story of the afternoon and
  costs nothing to carry.
- **Depth** — how many of their golfers are still in it, out of how many they hold.
  Groups are uneven by design (28 to 32 golfers on a 150-golfer field split five ways),
  so this varies between teams and matters to the tie-break.
- **Their odds at creation** — `total_odds`, ~1/n by construction. Worth showing once
  as evidence the draw was fair, not repeated everywhere.

### 5.2 A team's golfers

Expanded, drilled into, or always visible — a design decision. Each golfer needs:

position · name · tournament total · this round's score · holes played in this round ·
what they were worth at creation

**Three of those are scores, they answer different questions, and a design that lets two
of them touch will be read as one number.** The tournament total is the sum of the
linescores (§3, item 2). This round's score is the linescore for
`competitions[0].status.period`. Holes played is `status.displayThru`. A golfer can be
six under for the week and three over for the morning, and `-6 · thru 12` reads as *six
under through twelve holes today* — a different golfer having a different week. **Label
every score with the round it belongs to, wherever two of them appear together**,
including the one-line summary of the team's leading golfer, which is exactly where the
shipped page had it wrong.

**Pick the round by its number**, never off the end of `rounds[]` — see §3, item 3, for
the six golfers that gets wrong on the checked-in payload — and **render a round nobody
has started as blank**. Not `E` and not an em dash: both of those say the golfer went
round in level par, and they have not been round at all. The same holds for the whole
field in the window between ESPN advancing `period` and the first group teeing off.

**Holes played, not holes remaining.** `displayThru` is a string. It counts up (`"18"` on
a completed round) in every payload measured here, but ESPN uses `"F"` on other events,
so `18 - thru` is arithmetic over a format this repo has not checked. Printing what the
field already says costs nothing and cannot be wrong.

Sorted by the rank key (§6), so the golfer carrying the team is always first. Cut,
withdrawn and unscoreable golfers stay listed — a team's roster does not shrink — but
must be visibly out of it. Before the first tee time nobody is out of anything yet, so
the "out" treatment belongs on `meta.started`, not on the golfer having no score and
not on the field being non-empty — the field is full for two days before it means
anything. Watch `player.statusShort` there too: for a golfer who has not teed off it is
a raw ISO tee time, where a cut golfer gets `"CUT"`. Their `status` is
`STATUS_SCHEDULED`, which is the thing to branch on.

`golfers[].espn.match` distinguishes the two ways a golfer can have no leaderboard row
at all: `"absent"` means somebody checked this week's field and confirmed the golfer is
not in it, `"unresolved"` means nobody has looked yet (§8). Both score nothing, so the
standings do not care; a page that shows the difference is telling the pool whether the
gap is a withdrawal or a build somebody still has to finish.

### 5.3 The odds panel

This panel is read by the league, not by whoever built the page, and everything on it
has to survive that. Anything a reader cannot act on belongs in the footer (§5.4) or
nowhere.

- Field size, book sum, and **capture time** for the snapshot, **in plain English**.
  The numbers may be exact; their captions may not be jargon. "ask prices · probability
  basis" is precise, and it is also the caption that got a scoreboard accused of hiding
  something. Say what the number means to somebody who has never priced a book.
- Who was **excluded and why** (`odds_snapshot.excluded[].reason` is `named` or
  `over_fair_share`). A golfer worth more than a whole group's fair share cannot be
  balanced around, so they were dropped — that is a fact the pool will argue about and
  the page should be able to settle.
- The **grouping certificate** (§`grouping`): the delta, whether it is provably optimal,
  and the group sizes.
- The **full draw**: every group as it was dealt, and every golfer in it with what they
  were worth at creation. The certificate is a claim about the draw; this is the
  working behind it, and it is the answer to "what did we all get" — which the league
  table cannot give, because it ranks and it hides each group behind a chevron.
- If `sources.kalshi.mutually_exclusive_outcomes` is **false** (Top 5 / Top 10 /
  MakeCut), the numbers are *share of N slots*, not probabilities, and the book sums
  toward N rather than 1. Do not label them "win probability" — or anything else that
  claims to be a probability. The safe framing, and the one the shipped page uses
  throughout, is what a golfer was *worth*: it is true of both market types, so no page
  has to detect which one it is holding. `sources.kalshi.market_label` in the footer
  (§5.4) names the market for anybody who wants to know which it was.

Deliberately **not** here: the name-join report and the list of API endpoints. Both are
provenance, both were read by nobody on a Sunday, and both crowded out the two cards
that answer the question the page exists for. The join's residue that still matters —
a count of unmatched golfers, and a per-golfer marker — belongs in the margin of the
standings (§8), and the events belong in the footer (§5.4 — which asks for *which*
Kalshi event and *which* ESPN event, not for the endpoint URLs; those are carried in
the result file and printed in the export's manifest, for checking server-side).

### 5.4 The provenance footer

Quiet, but present: which Kalshi event, which ESPN event, when the build ran, the
partition summary (`grouping.summary` — e.g. *"Delta 0.000858 (1 tick, 0.09% of the
field) — PROVEN OPTIMAL: no partition of this field does better"*), and the
`competition_id`. This is what makes "why did I get this group?" answerable.

### 5.5 The league's own art — and where it comes from

A badge beside the league name, a banner across the top, a tagline under the name.
These are the only per-league part of the design; everything else is the template's and
is the same for every competition.

The tagline is on `DATA.league.tagline`. **The two images are not on `DATA` at all.**
`DATA.league.logo` is a *slug* — the name of a directory of art — and a page that set
an `<img src>` from it would render a broken image with the word `wcw` under it.

The pictures arrive in their own element, which the bundler fills:

```html
<script id="league-art" type="application/json">/*__LEAGUE_ART_JSON__*/</script>
```
```js
var ART = JSON.parse(document.getElementById('league-art').textContent);
// { logo: "data:image/png;base64,…", banner: "data:image/png;base64,…" }
```

Either key may be absent and `{}` is ordinary — a league with a badge and no banner is
an ordinary league, and one with neither gets a masthead with its name in it, which the
page must draw as finished rather than as broken.

Why the split: the result JSON is the input to a rebuild and gets read by people, and
half a megabyte of base64 in it made it neither. The exported page still carries the
bytes, because the page is the thing that has to survive being emailed. So the two
artifacts in the zip deliberately differ, and only in this.

---

## 6. The standings rule — implement exactly

> Rank each team by the best leaderboard position it holds. Teams tied on that are
> separated by their next-best golfer, then the one after that, for as far as it
> takes. Groups are uneven, so a team can run out of golfers partway down — when it
> does, the team that still has one wins.

That is a **lexicographic comparison of each team's golfer ranks in ascending order**,
with the shorter roster padded by something worse than everything.

### Ranking one golfer

A golfer's rank is a **pair**, never a number:

| Tier | Meaning | Value |
|---|---|---|
| `0` | still in the tournament | the displayed position number (`T12` → `12`) |
| `1` | cut / withdrawn / disqualified | `sortOrder` |
| `2` | no ESPN athlete on this golfer — confirmed absent from the field, or not yet reviewed | `0` |
| `3` | *padding* — this team has no golfer this deep | `0` |

Compare pairs left to right.

Tier 0 uses the **displayed** position, which tied golfers share — that sharing is
what makes the tie-break fire at all. Tier 1 falls back to `sortOrder` because ESPN
publishes no position for a cut player (all 74 read `"-"`) while `sortOrder` still
ranks them sensibly among themselves and below everyone who made the cut. Tier 2 is a
golfer with no row on the leaderboard to read: either they are not in the field, or the
build could not settle their name and said so (§8). Measured on the Rocket Classic,
that was 12 of 151 before review and 4 of 151 after it — the difference is the whole
point of §8, and neither number changes the rule. Tier 2 ranks below every golfer who
is in the tournament but still ahead of holding nothing, because a team that drafted 12
golfers has drafted 12 golfers.

`standings_rules.golfer_rank_tiers` in the result file says the same thing in one line
each. A page **may** show it — if it does, it must show the file's words rather than
restating the rule and drifting — but it is a legend for a rule most readers never
question, and the shipped scoreboard leaves it out on those grounds. The field stays in
the file either way; it costs nothing and a page that wants it should not have to
re-derive it.

Do **not** collapse the tiers into one number. A cut golfer is not "position 74"; if
they were, a team of cut golfers could outrank a team holding someone in contention.

### Ranking the league

```
vector(team)  = sorted(golferRank(g) for g in team.golfers)
compare(a, b) = lexicographic over the vectors, padding the shorter with (3, 0)
```

Teams whose whole vectors are equal are **genuinely tied** and are reported tied —
`T1`, `T1`, `3` — not separated on something invented. Order tied teams
deterministically for rendering (the reference uses `team_id`), but never present that
order as a ranking.

### Verify against the golden file

`tests/fixtures/standings_golden.json` holds a five-team league over the real finished
field, built to hit every branch at once: a tie broken on the second golfer, a team of
nothing but cut golfers, a golfer who never teed off, uneven rosters, and a team that
runs out of golfers mid-comparison. Expected output:

| pos | team | decided on | vector |
|---|---|---|---|
| 1 | charlie | — | `[[0,1],[2,0]]` |
| 2 | alpha | golfer 1 | `[[0,5],[0,31]]` |
| 3 | bravo | golfer 2 | `[[0,5],[0,50],[1,74]]` |
| 4 | delta | golfer 1 | `[[1,79],[1,80]]` |
| 5 | echo | golfer 2 | `[[1,79]]` |

`frontend/lib.js` implements the rule and `standings.py` implements it again;
`tests/test_frontend_parity.py` runs both over the same payloads and fails if they
disagree. It sits above the template directories and every template pulls it in as
`../lib.js`, so there is exactly one copy. **Reuse it verbatim** and write only
presentation — then the parity test covers your page too.

---

## 7. The result JSON

Abridged; every field is present in a real file. Run
`python build_competition.py --league leagues/example-league.json --tournament wyndham`
for a complete one.

**Every block below is always present.** Since 4.0 there is one kind of build — it
cannot complete without a published ESPN field — so `live`, `golfers[].espn` and
`sources.espn.match_report` are never `null` because of when the build ran, and a
reader never has to establish which document they are holding. See §2.

```jsonc
{
  "schema_version": "4.0",   // 4.0 removed `build_mode` and the nulled-out half it
                             // selected. ESPN posts fields about two days early now,
                             // so the premise 2.0 split the file on -- no field before
                             // the first tee time -- is no longer true. There is one
                             // kind of build; `live` and `golfers[].espn` are always
                             // objects. The distinction moved to the page, which asks
                             // the leaderboard whether anybody has teed off. See §2.
                             // 3.0 replaced league.crest and league.banner -- two
                             // inlined data: URIs -- with league.logo, the NAME of a
                             // directory of art. The images are read at export and
                             // land in the page; this file carries none. See §5.5.
                             // 2.1 dropped odds_snapshot.refreshed, golfers[].odds
                             // .current and the "refresh-odds" rebuild mode: there is
                             // one price per golfer and no way to ask for a second.
                             // 2.0 split the file in two along `build_mode`, and the
                             // halves are not the same document. It also dropped
                             // live.name_match, golfers[].espn.source / .from_event,
                             // sources.espn.identities_from_history and
                             // .field_available_at_build: nothing matches names at
                             // runtime any more, and no identity comes from a
                             // tournament that is already over. See §8.
                             // (1.2 had dropped live.kalshi_markets_url and
                             // live.kalshi_proxy_url_template. See §4.)

  "competition_id": "uuid5, stable for this league+event+market",
  "generated_at": "2026-08-03T20:41:00+00:00",
  // poll_interval_seconds lives here as well as under `live`, because it is an input
  // somebody typed rather than a fact about the world. `live` mirrors it so the page
  // reads its own settings in one place.
  "generator": { "tool": "...", "git_commit": "e581c23", "seed": 42,
                 "poll_interval_seconds": 60 },

  // null on a first build. On a rebuild (--from-result), what it was rebuilt from and
  // how: mode is "refresh" | "regroup". A file carrying Wednesday's odds and Sunday's
  // leaderboard says so here. A rebuild is not part of a normal week any more — the
  // page ranks by itself — so this is usually null.
  "rebuilt_from": { "source_file": "build/result.json", "mode": "refresh",
                    "source_generated_at": "…", "source_schema_version": "2.0",
                    "first_built_at": "…", "rebuild_count": 2 },

  "league":   { "league_id": "uuid", "league_name": "Sunday Fivesome",
                "league_slug": "sunday-fivesome", "team_count": 5,
                // The masthead. Both are optional and both are NULL rather than
                // absent when unset -- a page that has to tell "this league has no
                // logo" from "this build predates logos" will get it wrong, and the
                // difference is worth nothing to anybody.
                // `logo` is a SLUG and NOT AN IMAGE: the name of a directory of art
                // under leagues/, holding logo.png and banner.png. THE PAGE DOES NOT
                // READ IT. The bundler resolves it and hands the page the pictures in
                // a separate element -- see §5.5 -- so this file stays a document
                // about a competition rather than an envelope for two PNGs.
                // A null here means this competition has no art, full stop; it is not
                // an invitation to substitute some.
                "logo": "sunday-fivesome",
                "tagline": "10th Anniversary" },

  "teams": [{
    "team_id": "uuid",  "team_name": "Bogey Boys",  "player_name": "Mo",
    "team_logo": "data:image/svg+xml;base64,…",     // or a URL, or null
    "group_index": 3,
    "golfer_ids": ["9fbf091c-…"], "golfer_names": ["Cameron Young"],
    "total_odds": 0.1998,        // sums to the golfers' grouping_weight exactly
    "golfer_count": 31
  }],

  "golfers": [{                  // strongest first
    "golfer_id": "9fbf091c-…",   // Kalshi golf_competitor UUID — stable across
                                 // events AND across market series. THE key.
    "name": "Cameron Young",
    "team_id": "uuid or null",   // null only if excluded
    "excluded": false,
    "kalshi": { "ticker": "KXPGATOUR-WYC26-CAME", "bid": 0.081, "ask": 0.09, "spread": 0.009 },
    "odds": {
      "raw": 0.09,               // as quoted, when the groups were drawn
      "devigged": 0.0688,        // ÷ observed book sum, whole field
      "grouping_weight": 0.0772  // what the partitioner saw; null if excluded.
                                 // Sums to 1.0 across every grouped golfer.
      // Three numbers, all read at the same instant, and there is never a fourth.
      // See §4: a rebuild does not go back to the market.
    },
    // Always an object: the build had a field or it stopped, so every golfer has been
    // looked for. A golfer no tier settled carries it with athlete_id, display_name,
    // headshot and country all null, and `match` says which way it went. See §8.
    "espn": {
      "athlete_id": "4425906", "display_name": "Cameron Young",
      "headshot": "https://a.espncdn.com/…", "country": "USA",
      // How this golfer was settled. The first three are matches, in the order the
      // tiers are tried; the last two both mean "no athlete" and are not the same
      // claim. "absent": somebody looked at this week's field and confirmed he is not
      // in it. "unresolved": nobody has looked.
      "match": "decision | alias | exact | absent | unresolved",
      // In THIS week's ESPN field. Three states, on purpose: true, false (checked, and
      // he is not), null (not checked). Folding null into false would report a
      // withdrawal the build has no evidence for.
      "in_field": true
    }
  }],

  // "pre" on a build made the night before, "in" once play starts. It is a record of
  // when the build ran, NOT the page's gate — the page reads `started` off each poll.
  // Dates and course survive a payload that omits them; the state does not, because a
  // run that did not read it does not know it.
  "tournament": { "name": "…", "season": 2026, "start": "…", "end": "…",
                  "state_at_build": "in", "course": { "name": "…", "par": 70 } },

  "sources": {
    "kalshi": { "event_ticker": "KXPGATOUR-WYC26", "series_ticker": "KXPGATOUR",
                "markets_endpoint": "https://…", "odds_type": "winner",
                "market_label": "Outright Winner",
                "mutually_exclusive_outcomes": true,   // false for top5/top10/makecut
                "price_mode": "ask", "price_level_structure": ["tapered_deci_cent"],
                // false, always. The page never requests this endpoint; it is here so a
                // reader can check the numbers server-side.
                "browser_reachable": false, "browser_note": "…403…" },
    "espn":   { "league": "pga", "event_id": "401811961",
                "leaderboard_endpoint": "https://…", "scoreboard_endpoint": "https://…",
                "browser_reachable": true, "browser_note": "…allow-origin: *…",
                // Never 0: a build with no field to join against stops instead.
                "field_size_at_build": 147,
                // Had anybody teed off when this ran? Normally false, because a pool is
                // drawn the night before. A record of the clock and nothing else — do
                // NOT gate the page on it; it is false for the life of the file while
                // the tournament it describes comes and goes. Use `meta.started` off
                // the poll (§2).
                "started_at_build": false,
                // `matched` can legitimately be lower than `requested` — see §8.
                // `requested` is the Kalshi field, odds_snapshot.field_size below.
                "match_report": { "espn_field_size": 147, "requested": 150,
                                  "matched": 146, "matched_decision": 6,
                                  "matched_alias": 1, "matched_exact": 139,
                                  "absent": [ … ],       // reviewed and confirmed out
                                  "unresolved": [ … ],   // nobody has looked yet
                                  // normalised full names two athletes in THIS field
                                  // share; refused rather than guessed
                                  "ambiguous_names": [],
                                  "problems": [ … ] },   // sentences, worth surfacing
                // The reviewed decisions this build applied, keyed by Kalshi name, so
                // the next rebuild does not re-ask a question somebody answered.
                "match_decisions": { "Jason Day": { "absent": true } },
                // The aliases that actually fired — not the whole alias file, which is
                // repo state rather than a fact about this competition.
                "aliases_applied": { "Zachary Bauchou": "Zach Bauchou" } }
  },

  "odds_snapshot": {
    "captured_at": "…",          // when the groups were drawn. Never moves on a rebuild.
    "price_mode": "ask", "field_size": 150,
    "raw_book_sum": 1.166, "liquidity": { … },
    "normalization": { "basis": "probability | share_of_n_slots", "note": "…" },
    "excluded": [{ "golfer_name": "…", "reason": "named | over_fair_share",
                   "raw_odds": 0.21, "devigged_odds": 0.18 }],
    // Whether the 1/n fair-share rule ran. An empty `excluded` means nobody was over
    // the line OR the rule was off, and only this says which.
    "auto_exclude": true,
    "fair_share_threshold": 0.2
  },

  "grouping": {
    "n_groups": 5, "grouped_golfers": 150,
    "delta": 0.000858, "delta_ticks": 1, "floor_ticks": 1,
    "optimal": true, "exact_grid": true,
    "dominant_golfers": [], "group_sizes": [28, 28, 31, 31, 32],
    "summary": "Delta 0.000858 (1 tick, 0.09% of the field) -- PROVEN OPTIMAL: …"
  },

  // What the page does while it is open: one endpoint, on a timer. There is no odds
  // endpoint here — see §4 — and no name-matching block either, in either mode: the
  // build has already written an ESPN athlete id onto every golfer it resolved, and
  // the page joins on that. See §8.
  //
  // Always present, so the page polls from the moment it opens — which is how it
  // notices the first tee time and starts ranking on its own (§2).
  // `espn_event_id` is the id to check an arriving payload against (§3).
  "live": {
    "espn_leaderboard_url": "https://…&event=401811961",
    "espn_event_id": "401811961",
    "poll_interval_seconds": 60
  },

  // The rule this file expects to be ranked by, in the file, so a page that wants to
  // show it does not have to restate it. Optional to render — see §6.
  // Tier 2 is "no ESPN athlete on this golfer: either confirmed absent from the field,
  // or not yet reviewed. Scores nothing either way." See §6.
  "standings_rules": { "description": "…",
                       "golfer_rank_tiers": { "0": "…", "1": "…", "2": "…", "3": "…" },
                       "comparison": "lexicographic …", "unresolved": "…" }
}
```

Typical size: ~100 KB for a 150-golfer field with five inlined SVG logos. A `groups`
file is a little smaller: three of its blocks are the word `null`.

---

## 8. Matching golfers to the leaderboard

Kalshi and ESPN publish the same golfers under different names, and Kalshi's stable
golfer UUID is not an ESPN id, so the two fields have to be joined by name somewhere.

**That somewhere is the build, and it is not the page.** A `live` build joins the Kalshi
names against the competitors ESPN published for this tournament and bakes the winning
`athlete_id` onto each golfer. At runtime the join is therefore a `Map` lookup on that
id:

```js
var byId = GolfPool.indexByAthleteId(parsed.players);   // athlete id -> ESPN player
var player = GolfPool.resolveGolfer(golfer, byId);      // -> player | null
```

`null` is a real answer and means "no leaderboard row to read": either the golfer is not
in the field, or the build could not settle their name. Both are tier 2 (§6) and both
are stated in the file — `golfers[].espn.match` and `.in_field`.

There is no name matching in `lib.js` any more. It used to carry a transliteration
table, a normaliser and a first-initial fallback mirrored character for character
against the Python, because a page built before the field existed had to finish the join
in the browser against a leaderboard the build had never seen. No page is built that way
now (§2), so the second copy of a subtle string algorithm is gone rather than
maintained, and a lookup on an integer cannot pick the wrong Smith.

### The three build-time tiers, and why there is no fourth

All three are exact or explicit. Tried in this order:

1. **`decision`** — a binding somebody reviewed and recorded for this competition, or a
   note that the golfer is genuinely not in the field. Always wins; somebody looked.
2. **`alias`** — an explicit Kalshi-name → ESPN-display-name entry from the alias file,
   reusable every week and learned from decisions.
3. **`exact`** — the two normalised display names are equal. Normalisation is NFKD, drop
   combining marks, transliterate the letters NFKD leaves alone (`ø→o`, `æ→ae`, `å→a`,
   `ł→l`, `ß→ss`), lowercase, hyphens and apostrophes to spaces, drop
   `jr/sr/ii/iii/iv/v`, drop non-letters, join runs of consecutive single letters
   (`C.T. Pan` ≡ `CT Pan`), collapse whitespace.

Measured on the Rocket Classic, 151 Kalshi names against 147 ESPN competitors:

| outcome | count |
|---|---|
| matched by `exact` | 139 |
| left open for review | 12 |
| — of those, really in the field | 8, all formal-vs-familiar first names: Zachary/Zach Bauchou, Cameron/Cam Davis, Kris/Kristoffer Ventura, Nicolas/Nico Echavarria, Matthew/Matt McCarty, Benjamin/Ben James, Jordan L./Jordan Smith, Hao-Tong/Haotong Li |
| — of those, genuinely absent | 4: Daniel Brown, Taylor Moore, Brooks Koepka, Jason Day, all withdrawn before play |

A first-initial-plus-last-name rule binds those eight correctly, and it was measured
collision-free inside that 147-player field. It used to be tier 4. It is not any more,
because "no collisions in one field" is exactly the measurement that cannot be
extrapolated: the week a field holds two J. Smiths, or a Cameron and a Carson Young, it
binds one of them to the wrong person, and nothing downstream — not the page, not the
standings, not the reader — can tell that case from the other 150. So the rule survives
as a **suggestion generator** and never as a match. An unresolved name comes back with
ranked candidates and the reason for each, a person settles it, and the settlement is
written down where it can be read before it takes effect.

### Two refusals, and why `matched` can be lower than `requested`

Both are in `sources.espn.match_report`, and neither is a failure:

- **Two Kalshi names resolving to one athlete.** One ESPN athlete cannot be on two
  teams, so the second name is left unresolved and `problems` says who already holds
  them. Reviewed decisions are applied in a first pass, so which name counts as "second"
  never depends on how Kalshi happened to sort its markets.
- **One normalised name shared by two athletes in the same field.** It is dropped from
  the index, listed in `ambiguous_names`, and *both* golfers come back unresolved. Zero
  such names in the measured field; the day there is one, a coin flip is the wrong way
  to decide which of two people is on somebody's team.

### `absent` and `unresolved` are different, and the page should know it

Both score nothing. Only one of them is a fact:

| `match` | `in_field` | Means |
|---|---|---|
| `"absent"` | `false` | somebody checked this week's field and confirmed the golfer is not in it |
| `"unresolved"` | `null` | nobody has looked yet |

An unresolved golfer is the build saying it does not know, out loud, by name. The
`unresolved` list in `match_report` is that list, and the build also writes it to a
`match-review.json` beside the result file with the unclaimed ESPN athletes and ranked
suggestions, for somebody to settle and rebuild from. What came back settled is in the
file too — `sources.espn.match_decisions` for the reviewed bindings and absences,
`sources.espn.aliases_applied` for the aliases that actually fired — so a page can say
how a golfer was settled without the review file being anywhere near it.

**A golfer left unresolved at build time is unscoreable for the life of that page.**
They carry no `athlete_id`, so there is nothing for `resolveGolfer` to look up, and
polling harder will not help: every poll re-reads the same leaderboard and finds the
same nothing. **The fix is a rebuild, not a refresh** — and there is no way for a page
to do it. That makes the review step load-bearing for correctness rather than tidiness,
and it makes `match_report.unresolved` worth surfacing somewhere quiet on the page: it
is the one defect in a scoreboard that the scoreboard cannot recover from on its own.

---

## 9. The template contract

`bundle_frontend.py` turns a template directory into the deliverable. A template is an
`index.html` plus local assets. Four things happen to it, and the fourth is optional:

1. **The data is injected.** The file must contain, verbatim:
   ```html
   <script id="competition-data" type="application/json">/*__COMPETITION_JSON__*/</script>
   ```
   The marker is replaced with the result JSON, escaped so a `</script>` or `<!--`
   inside the data cannot end the element. Missing marker → the build fails loudly.

2. **Local assets are inlined.** `<link rel=stylesheet href>` and `<script src>` become
   `<style>` / `<script>`; `<img src>` becomes a `data:` URI. A reference may point
   above the template directory — `../lib.js` is how both shipped templates share one
   copy of §6. Absolute URLs are left alone — but the page must still work when they
   fail to load, because it will routinely be opened with no network at all, so in
   practice a template should have none.

   Leave `src` **off** an `<img>` whose source arrives from the data at runtime. An
   empty `src=""` resolves to the template directory, and a directory is not an image.

3. **`{{tokens}}` are substituted**, HTML-escaped: `league_name`, `tournament`,
   `market`, `generated_at`, `team_count`, `competition_id`. This is how a `<title>`
   names the league without running JavaScript.

4. **The league's art is inlined**, if the template asks for it:
   ```html
   <script id="league-art" type="application/json">/*__LEAGUE_ART_JSON__*/</script>
   ```
   The marker is replaced with `{"logo": "data:…", "banner": "data:…"}` — only the keys
   that resolved, and `{}` for a league with no art. Read out of `leagues/<slug>/` at
   bundle time; see §5.5. Unlike the data marker this one is optional: a template that
   draws no masthead simply omits it and bundles unchanged.

Output: `<league>-<tournament>.html` (self-contained) and a matching `.zip` holding
that page, `result.json`, and a manifest.

**No frameworks, no CDNs, no imports, no build step.** A CDN `<script>` is not inlined
and turns the page into a brick the first time it is opened offline. One HTML file, one
`<style>`, `lib.js` plus your own script. If you want a framework, inline the whole
thing — but the reference does the job in ~250 lines of plain DOM and the data is a
single object, so weigh it.

---

## 10. Design notes

**Two audiences, one page.** The person checking who is winning wants one glance. The
person arguing about the draw wants the odds, the exclusions and the optimality proof.
Do not make the second lot invisible and do not make the first lot scroll.

**The margin is the content.** Two teams tied on a T4 golfer and separated on their
seventh is the whole afternoon. `decided_at` carries it; a design that shows only
positions throws away the most interesting number on the page.

**Uneven rosters are load-bearing, not an artefact.** The partitioner is free to give
one team 28 golfers and another 32 because forcing equal sizes costs real balance
(12 groups reach a 1-tick delta unconstrained and stall at 8 ticks with equal sizes).
"More golfers" is a real tie-break, so a design that implies every team has the same
number is lying about the rules.

**Refresh honestly.** Show when the board was last updated. When ESPN fails, keep the
last-known board and mark it stale rather than blanking or silently freezing.

**Design the empty states.** A page open before the first tee time, after ESPN goes
down, when a team's golfers have all been cut, when a logo is null. Every one of these
happens in a normal week, and the first one happens to every page: it is how each of
them spends its first day or two, because pools are drawn the night before. That state
is a groups sheet, not a scoreboard with the numbers missing, and it should look
finished rather than early (§2). Design the changeover too — the same page becomes a
leaderboard while somebody is looking at it, and nothing should jump or need a reload.
What is *not* on that list any more is live odds: there is no such state, because there
is no such fetch — the odds are simply there, dated.

**Accessibility.** Position and score must not be conveyed by colour alone — a page
read across a room is exactly where that fails. Support both colour schemes; the
reference honours `prefers-color-scheme`.

---

## 11. Checklist

- [ ] Reads its data from `#competition-data`; fetches nothing on load
- [ ] Contains `/*__COMPETITION_JSON__*/` inside that script tag
- [ ] No CDN, no external module, no build step
- [ ] **ESPN is the only host it ever requests** — no odds fetch, no relay, at all
- [ ] Assumes `DATA.live` and `golfers[].espn` are objects — 4.0 has no half-empty file
- [ ] **Ranks only when `meta.started` is true** (§2), never on `players.length` alone:
      a field is posted about two days early with no positions in it, and ranking that
      produces a complete invented league table
- [ ] Polls from load, days early if need be, and relabels itself when play starts —
      heading, tab and column headers set on every render, never once at load
- [ ] Polls ESPN on `live.poll_interval_seconds`, using `live.espn_leaderboard_url`
- [ ] Refuses a leaderboard whose event id is not `live.espn_event_id`
- [ ] Joins golfers to the board on `golfers[].espn.athlete_id` and nothing else — no
      name matching anywhere in the page
- [ ] Running total summed from `linescores`, never `score.displayValue`
- [ ] The tournament total, this round's score and holes played are each labelled with
      the round they belong to — no two of them adjacent and bare (§5.2)
- [ ] This round's score is selected out of `rounds[]` by `status.period`, never by
      taking the last element, and a round nobody has started renders blank — not `E`,
      not an em dash (§3 item 3, §5.2)
- [ ] Standings match `tests/fixtures/standings_golden.json`
- [ ] Ties shown as ties; `decided_at` surfaced
- [ ] Cut / WD golfers and golfers with no athlete id listed and visibly out
- [ ] Snapshot odds shown with capture time; exclusions shown with reasons
- [ ] **No price ever appears to move** — no movement column, no arrows, no slot for a
      second reading, and nothing rendered off `odds.current` or `odds_snapshot
      .refreshed` if an older file still carries them
- [ ] The odds view carries the four snapshot numbers **with plain-English captions**,
      the exclusions, the certificate, and the full draw group by group with every
      golfer's odds at creation — and no name-join report or endpoint list
- [ ] Which Kalshi event and which ESPN event are named somewhere on the page (§5.4)
- [ ] Handles `pre` (a full field, no positions), an empty field, ESPN down, null logos
- [ ] Opens from disk with no network at all: every roster and its odds shown, nothing
      ranked, and no error where the draw should be
- [ ] `player.statusShort` is never printed for a `STATUS_SCHEDULED` golfer — it is a
      raw ISO tee time
- [ ] The league's own art is read from `#league-art` and **not** from
      `DATA.league.logo`, which is a slug; the tagline is shown when `DATA.league`
      carries one, and the page looks finished with neither (§5.5)
- [ ] Readable on a phone; works in light and dark
- [ ] `python -m pytest tests/test_frontend_parity.py tests/test_scoreboard_render.py
      tests/test_frontend_render.py` passes
