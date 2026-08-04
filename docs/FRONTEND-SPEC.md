# Scoreboard spec

The design brief for the page a golf pool actually watches on Sunday afternoon.

It is a **single static HTML file**. No server, no build step, no backend, no
database. Everything it knows about the competition is baked into it at build time;
the most it ever fetches while running is the ESPN leaderboard, and a page built
before the field posted does not fetch even that (§2). It opens from a `file://` URL,
from a USB stick, from Dropbox, from anywhere.

A working reference implementation lives in `frontend/template/`. It is deliberately
plain — it exists to prove the contract, not to be the design. Read it for *what*, and
ignore it for *how*.

> **The one thing to read first.** §4. Kalshi will not answer a browser, so **the page
> fetches at most one thing: the ESPN leaderboard.** There is no live-odds panel, no
> relay, no setting that turns one on. Odds are baked in and stated as of the moment
> they were captured. Any design that treats "live odds" as a thing the page can fetch
> is designing a panel that is permanently empty for every user.
>
> **The second thing.** §2. `DATA.build_mode` says whether this page is a scoreboard or
> a groups sheet, and a groups sheet fetches nothing at all.

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
| Odds re-read at a later rebuild | ✅ when present — see §4 | — |
| Which endpoints produced the numbers | ✅ | — |
| Grouping quality certificate | ✅ | — |
| Golfer → ESPN athlete id | ✅ — see §8 | — |
| Scores and positions | — | ✅ ESPN |

**ESPN is the only row on the right, and that is the whole of the page's network.**

### `build_mode` — read it before anything else

ESPN publishes zero competitors until the first tee time, so half the pages this spec
describes are built before there is anything to score. That is not a degraded state to
design around; it is a different document, and `DATA.build_mode` says which one you
have:

| `build_mode` | `DATA.live` | `golfers[].espn` | What the page does |
|---|---|---|---|
| `"groups"` | `null` | `null` on every golfer | renders rosters and odds, requests nothing from anywhere, ranks nothing |
| `"live"` | an object (§7) | an object per golfer, carrying an ESPN athlete id when the build resolved one | polls ESPN, joins on that id, ranks |

Branch on `DATA.live` once, at the top, exactly as the reference does
(`var LIVE = DATA.live || null`). A `groups` page that starts a poll loop is asking a
question whose answer it could not use, and one that renders a spinner is apologising
for a fetch that was never going to happen.

*A note on the word.* Three unrelated things in this project are called "live": that
`build_mode` value, the `live` block in the result JSON, and the test suites that hit
the real APIs. This document always says which one it means.

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

Everything in this section is about a `live`-mode page. A `groups`-mode page has
`DATA.live === null` and makes no request at all (§2).

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

4. **`sortOrder` is the in-play rank**, and it is the only field that is. Zero
   inversions against the running total; the stale score field inverts 29 times. It is
   a total order `1..N` over the whole field, and it places every cut player (74–147)
   below every player who made the cut (1–73).

Also: the current round is `competitions[0].status.period`, **not** `event.status`.
And `status.position.isTie` is already a boolean — no need to parse the `T`.

### States the page has to handle

| State | What ESPN returns | What the page shows |
|---|---|---|
| Built before the field posted | nothing — a `groups` page never asks | every roster, with odds at creation; "not started"; **no positions and no ranking** |
| Not started yet | `state: "pre"`, **zero competitors** | the same: rosters and odds, **nothing ranked** |
| In progress | `state: "in"`, positions moving | the full board |
| After the cut | 74 of 147 have position `"-"` | cut golfers visibly out, still listed |
| Withdrawn / DQ | `status.type` `STATUS_WD`/`STATUS_DQ`, no position | same treatment as cut, labelled |
| Finished | `state: "post"`, `completed: true` | final, stop implying it will change |
| ESPN unreachable | — | last-known board, plus a visible staleness marker |

### Never rank an empty board

This guard has nothing to do with the name join and survives every change to it. Since
2.0 the Wednesday-night page is a `groups` page that does not fetch, so the *first* row
above is settled at build time — but a `live` page can be handed an empty board any
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

**The page has at most one network dependency: ESPN** — and a `groups` page has none
(§2). There is no odds request in either mode, no relay setting, no "live odds
unavailable" state to design, and no empty panel to fill. The result file carries no
Kalshi URL for the page to try, because trying it can only fail.

**The snapshot is the odds display.** Every golfer's price at the moment the groups
were drawn is in `DATA.golfers[].odds`, along with the book it came from and the time
it was captured. That is a real, checkable, permanent number, and it is the honest
answer to "what was he worth?" — the question the pool is actually asking. Show it,
and show `odds_snapshot.captured_at` beside it so it is never mistaken for a price
that is moving.

### Prices do still move — between builds

`build_competition.py --from-result <file> --refresh-odds` re-reads Kalshi
**server-side**, where it answers fine, and bakes the result in:
`odds_snapshot.refreshed` describes the re-read, and every golfer gains `odds.current`.
No network at load, because it is not a feed — it is a second snapshot, taken when
somebody rebuilt and re-sent the page.

Show it as movement against `odds.raw` (the price the groups were drawn on — **not**
`kalshi.ask`, which is a different number on any price mode but the default): an arrow
and a delta, never a second column of raw levels. "Cameron Young is shorter than when
you drafted him" is the interesting fact; a raw level beside `grouping_weight` is
actively misleading, because one is de-vigged and the other is not, so an unmoved
golfer reads as a jump.

Say what it is, too: the arrows will not change again until the next rebuild. Calling
it live would promise a number that cannot arrive.

`refreshed` is `null` on a page that was never rebuilt — the common case — and
`odds.current` is then `null` throughout, so the movement column is simply absent.
That is the default state and it needs no apology.

`refreshed.priced_since_the_draw` names golfers who were added to the market after the
draw and are therefore in nobody's group; `refreshed.no_longer_priced` names drawn
golfers whose market is gone, which usually means a withdrawal. Both are worth
surfacing — the pool will ask.

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

position · name · score to par · thru · what they were worth at creation · (movement
since the draw, on a rebuilt page — §4)

Sorted by the rank key (§6), so the golfer carrying the team is always first. Cut,
withdrawn and unscoreable golfers stay listed — a team's roster does not shrink — but
must be visibly out of it. On a `groups` page nobody is out of anything yet, so the
"out" treatment belongs on the board existing, not on the golfer having no score.

`golfers[].espn.match` distinguishes the two ways a golfer can have no leaderboard row
at all: `"absent"` means somebody checked this week's field and confirmed the golfer is
not in it, `"unresolved"` means nobody has looked yet (§8). Both score nothing, so the
standings do not care; a page that shows the difference is telling the pool whether the
gap is a withdrawal or a build somebody still has to finish.

### 5.3 The odds panel

- Field size, book sum, price mode, and **capture time** for the snapshot.
- Who was **excluded and why** (`odds_snapshot.excluded[].reason` is `named` or
  `over_fair_share`). A golfer worth more than a whole group's fair share cannot be
  balanced around, so they were dropped — that is a fact the pool will argue about and
  the page should be able to settle.
- On a rebuilt page, the re-read: when it happened, what the book summed to then, and
  that the movement it implies is frozen until the next rebuild (§4).
- If `sources.kalshi.mutually_exclusive_outcomes` is **false** (Top 5 / Top 10 /
  MakeCut), the numbers are *share of N slots*, not probabilities, and the book sums
  toward N rather than 1. Do not label them "win probability".

### 5.4 The provenance footer

Quiet, but present: which Kalshi event, which ESPN event, when the build ran, the
partition summary (`grouping.summary` — e.g. *"Delta 0.000858 (1 tick, 0.09% of the
field) — PROVEN OPTIMAL: no partition of this field does better"*), and the
`competition_id`. This is what makes "why did I get this group?" answerable.

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
each, so a page can show the rule it is running rather than restating it in its own
words and drifting.

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

`frontend/template/lib.js` implements the rule and `standings.py` implements it again;
`tests/test_frontend_parity.py` runs both over the same payloads and fails if they
disagree. **Reuse `lib.js` verbatim** and write only presentation — then the parity
test covers your page too.

---

## 7. The result JSON

Abridged; every field is present in a real file. Run
`python build_competition.py --league leagues/example-league.json --tournament wyndham`
for a complete one.

The example below is a `live`-mode file, because that is the one with everything in it.
**Three of these blocks are `null` in `groups` mode** — `live`, `golfers[].espn` and
`sources.espn.match_report` — and null rather than empty on purpose: an empty object
reads as a join that was attempted and came back with nothing, which is a different and
much worse fact than no join having been possible. See §2.

```jsonc
{
  "schema_version": "2.0",   // 2.0 split the file in two along `build_mode`, and the
                             // halves are not the same document. It also dropped
                             // live.name_match, golfers[].espn.source / .from_event,
                             // sources.espn.identities_from_history and
                             // .field_available_at_build: nothing matches names at
                             // runtime any more, and no identity comes from a
                             // tournament that is already over. See §8.
                             // (1.2 had dropped live.kalshi_markets_url and
                             // live.kalshi_proxy_url_template. See §4.)

  // "groups" or "live", and a fact about the ESPN leaderboard at build time rather
  // than a setting somebody chose. READ THIS FIRST: it says whether the rest of the
  // file describes a draw or a scoreboard.
  "build_mode": "live",

  "competition_id": "uuid5, stable for this league+event+market",
  "generated_at": "2026-08-03T20:41:00+00:00",
  // poll_interval_seconds lives here as well as under `live`, because it is an input
  // somebody typed: a groups build has no `live` block to keep it in, and it has to
  // survive the rebuild that turns that groups sheet into a scoreboard.
  "generator": { "tool": "...", "git_commit": "e581c23", "seed": 42,
                 "poll_interval_seconds": 60 },

  // null on a first build. On a rebuild (--from-result), what it was rebuilt from and
  // how: mode is "refresh" | "refresh-odds" | "regroup". A file carrying Wednesday's
  // odds and Sunday's leaderboard says so here — and a groups sheet rebuilt into a
  // scoreboard is exactly that file.
  "rebuilt_from": { "source_file": "build/result.json", "mode": "refresh",
                    "source_generated_at": "…", "source_schema_version": "2.0",
                    "first_built_at": "…", "rebuild_count": 2 },

  "league":   { "league_id": "uuid", "league_name": "Sunday Fivesome",
                "league_slug": "sunday-fivesome", "team_count": 5 },

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
      "grouping_weight": 0.0772, // what the partitioner saw; null if excluded.
                                 // Sums to 1.0 across every grouped golfer.
      "current": 0.095           // the same market at odds_snapshot.refreshed.at.
                                 // null unless the build re-read it (--refresh-odds).
                                 // Never feeds the grouping. Compare against `raw`.
    },
    // NULL on every golfer in groups mode — there was no field, so there is nothing
    // to say. In live mode every golfer has this block, and a golfer no tier settled
    // carries it with athlete_id, display_name, headshot and country all null. See §8.
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

  // "pre" on the groups build, "in" once play starts. Dates and course survive a
  // rebuild that could not reach ESPN; the state does not, because a run that could
  // not read it does not know it.
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
                // 0 in groups mode, and the reason it is a groups build.
                "field_size_at_build": 147,
                // NULL in groups mode: there was no join, so there is no report on one.
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
    // null unless a rebuild re-read the market. See §4.
    "refreshed": { "at": "…", "price_mode": "ask", "field_size": 151,
                   "raw_book_sum": 1.171, "matched": 149,
                   "no_longer_priced": ["…"], "priced_since_the_draw": ["…"] },
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
  // NULL in groups mode, and that is the whole instruction: fetch nothing, poll
  // nothing, rank nothing. `espn_event_id` is the id to check an arriving payload
  // against (§3).
  "live": {
    "espn_leaderboard_url": "https://…&event=401811961",
    "espn_event_id": "401811961",
    "poll_interval_seconds": 60
  },

  // The rule this file expects to be ranked by, in the file, so a page can show it.
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
`index.html` plus local assets. Three things happen to it:

1. **The data is injected.** The file must contain, verbatim:
   ```html
   <script id="competition-data" type="application/json">/*__COMPETITION_JSON__*/</script>
   ```
   The marker is replaced with the result JSON, escaped so a `</script>` or `<!--`
   inside the data cannot end the element. Missing marker → the build fails loudly.

2. **Local assets are inlined.** `<link rel=stylesheet href>` and `<script src>` become
   `<style>` / `<script>`; `<img src>` becomes a `data:` URI. Absolute URLs are left
   alone — but the page must still work when they fail to load, because it will
   routinely be opened with no network at all.

3. **`{{tokens}}` are substituted**, HTML-escaped: `league_name`, `tournament`,
   `market`, `generated_at`, `team_count`, `competition_id`. This is how a `<title>`
   names the league without running JavaScript.

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

**Design the empty states.** A page built before the field was posted, after ESPN goes
down, when a team's golfers have all been cut, when a logo is null. Every one of these
happens in a normal week, and the first is half of all the pages this repo produces —
it is a groups sheet, not a scoreboard with the numbers missing, and it should look
finished rather than early (§2). What is *not* on that list any more is live odds:
there is no such state, because there is no such fetch — the odds are simply there,
dated.

**Accessibility.** Position and score must not be conveyed by colour alone — a page
read across a room is exactly where that fails. Support both colour schemes; the
reference honours `prefers-color-scheme`.

---

## 11. Checklist

- [ ] Reads its data from `#competition-data`; fetches nothing on load
- [ ] Contains `/*__COMPETITION_JSON__*/` inside that script tag
- [ ] No CDN, no external module, no build step
- [ ] **ESPN is the only host it ever requests** — no odds fetch, no relay, at all
- [ ] Branches on `DATA.live` once: a `groups` page (`live: null`) starts no poll loop,
      issues no request at all, and says so rather than looking like it is waiting
- [ ] Polls ESPN on `live.poll_interval_seconds`, using `live.espn_leaderboard_url`
- [ ] Refuses a leaderboard whose event id is not `live.espn_event_id`
- [ ] Joins golfers to the board on `golfers[].espn.athlete_id` and nothing else — no
      name matching anywhere in the page
- [ ] Running total summed from `linescores`, never `score.displayValue`
- [ ] Standings match `tests/fixtures/standings_golden.json`
- [ ] Ties shown as ties; `decided_at` surfaced
- [ ] Cut / WD golfers and golfers with no athlete id listed and visibly out
- [ ] Snapshot odds shown with capture time; exclusions shown with reasons
- [ ] A rebuilt page shows `odds.current` as movement against `odds.raw`, and says the
      arrows are from a rebuild rather than a feed
- [ ] A page that was never rebuilt (`refreshed` null) simply has no movement column —
      not a blank one, not a spinner, not an apology
- [ ] Handles `pre` (zero competitors), ESPN down, null logos
- [ ] With no leaderboard: every roster and its odds are shown, and nothing is ranked
- [ ] Readable on a phone; works in light and dark
- [ ] `python -m pytest tests/test_frontend_parity.py tests/test_frontend_render.py` passes
