# Scoreboard spec

The design brief for the page a golf pool actually watches on Sunday afternoon.

It is a **single static HTML file**. No server, no build step, no backend, no
database. Everything it knows about the competition is baked into it at build time;
the only thing it fetches while running is the live leaderboard. It opens from a
`file://` URL, from a USB stick, from Dropbox, from anywhere.

A working reference implementation lives in `frontend/template/`. It is deliberately
plain — it exists to prove the contract, not to be the design. Read it for *what*, and
ignore it for *how*.

> **The one thing to read first.** §4. Kalshi will not answer a browser, so **the page
> fetches exactly one thing: the ESPN leaderboard.** There is no live-odds panel, no
> relay, no setting that turns one on. Odds are baked in and stated as of the moment
> they were captured. Any design that treats "live odds" as a thing the page can fetch
> is designing a panel that is permanently empty for every user.

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

| | Baked in at build time | Fetched live in the browser |
|---|---|---|
| Teams, logos, rosters | ✅ | — |
| Odds when the groups were drawn | ✅ | — |
| Odds re-read at a later rebuild | ✅ when present — see §4 | — |
| Which endpoints produced the numbers | ✅ | — |
| Grouping quality certificate | ✅ | — |
| Live scores and positions | — | ✅ ESPN |

**ESPN is the only row on the right, and that is the whole of the page's network.**

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

## 3. ESPN — the live leaderboard

```
https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga&event=<id>
```

Unauthenticated. Sends `access-control-allow-origin: *`. Fetch it directly from the
browser; it works from `file://`. Poll it on `DATA.live.poll_interval_seconds`
(default 60).

**Always pass the pinned `event` id** from `DATA.sources.espn.event_id`. Without it
the endpoint answers about whatever ESPN thinks is current, which next Thursday is a
different tournament. **Verify the id in the response** and refuse a payload that does
not match — scoring a league against the wrong tournament looks entirely normal until
someone notices the names.

### The four things in this payload that are not what they look like

Measured against a live Round 2 payload (`espn-api/lb.json`, 147 competitors) and the
same tournament's final payload.

1. **`competitors[]` is not in rank order.** Sort it.

2. **`score.displayValue` is stale mid-round.** It counts *completed rounds only*, so
   a player halfway through round 2 shows their round-1 total. It was wrong for **42
   of 147** players. Never rank on it, never display it. The live total is the sum of
   the `linescores[].displayValue` values.

3. **`linescores[]` contains stubs** for rounds not yet played — no `value`, no
   `displayValue` key at all. Filter before summing. A withdrawn round reads `"-"`.

4. **`sortOrder` is the live rank**, and it is the only field that is. Zero inversions
   against the live total; the stale score field inverts 29 times. It is a total order
   `1..N` over the whole field, and it places every cut player (74–147) below every
   player who made the cut (1–73).

Also: the current round is `competitions[0].status.period`, **not** `event.status`.
And `status.position.isTie` is already a boolean — no need to parse the `T`.

### States the page has to handle

| State | What ESPN returns | What the page shows |
|---|---|---|
| Before the tournament | `state: "pre"`, **zero competitors** | every roster, with odds at creation; "not started"; **no positions and no ranking** |
| In progress | `state: "in"`, positions live | the full board |
| After the cut | 74 of 147 have position `"-"` | cut golfers visibly out, still listed |
| Withdrawn / DQ | `status.type` `STATUS_WD`/`STATUS_DQ`, no position | same treatment as cut, labelled |
| Finished | `state: "post"`, `completed: true` | final, stop implying it will change |
| ESPN unreachable | — | last-known board, plus a visible staleness marker |

The pre-tournament case is not an edge case. Groups are drawn Wednesday night and the
page exists from that moment; for its first ~12 hours ESPN publishes **no field at
all**. It must be a designed state, not an empty div.

Everything the pool cares about at that point is already decided and already in the
file: who holds whom, what each golfer was worth, and that the draw came out even. Show
all of it. What must **not** appear is a ranking — running the standings rule over an
empty leaderboard puts every golfer in tier 2, which orders the teams by roster size and
presents it as a leaderboard. Positions read `—` until there are positions. The same
applies whenever the board is empty for any other reason: ESPN unreachable, or a payload
for the wrong tournament (§3). The rosters and the odds stay; the positions do not
appear.

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

**The page has one network dependency: ESPN.** There is no odds request, no relay
setting, no "live odds unavailable" state to design, and no empty panel to fill. The
result file carries no Kalshi URL for the page to try, because trying it can only fail.

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
withdrawn and absent golfers stay listed — a team's roster does not shrink — but must
be visibly out of it.

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
| `2` | priced by Kalshi, never in the ESPN field | `0` |
| `3` | *padding* — this team has no golfer this deep | `0` |

Compare pairs left to right.

Tier 0 uses the **displayed** position, which tied golfers share — that sharing is
what makes the tie-break fire at all. Tier 1 falls back to `sortOrder` because ESPN
publishes no position for a cut player (all 74 read `"-"`) while `sortOrder` still
ranks them sensibly among themselves and below everyone who made the cut. Tier 2 is a
golfer who never teed off; 4 of 151 in a measured field. They rank below every golfer
who is in the tournament but still ahead of holding nothing, because a team that
drafted 12 golfers has drafted 12 golfers.

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

```jsonc
{
  "schema_version": "1.2",   // 1.2 dropped live.kalshi_markets_url and
                             // live.kalshi_proxy_url_template. See §4.
  "competition_id": "uuid5, stable for this league+event+market",
  "generated_at": "2026-08-03T20:41:00+00:00",
  "generator": { "tool": "...", "git_commit": "e581c23", "seed": 42 },

  // null on a first build. On a rebuild (--from-result), what it was rebuilt from and
  // how: mode is "refresh" | "refresh-odds" | "regroup". A file carrying Wednesday's
  // odds and Sunday's leaderboard says so here.
  "rebuilt_from": { "source_file": "build/result.json", "mode": "refresh",
                    "source_generated_at": "…", "first_built_at": "…", "rebuild_count": 2 },

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
    "espn": {                    // athlete_id may be null; see §8
      "athlete_id": "4425906", "display_name": "Cameron Young",
      "headshot": "https://a.espncdn.com/…", "country": "USA",
      "match": "exact | initial_last | alias | unresolved | deferred",
      "source": "field | history | null",   // where the identity came from
      "from_event": { "event_id": "401811960", "name": "Rocket Classic", "end": "…" },
      "in_field": true           // in THIS week's ESPN field. null while it is
                                 // unpublished — unknown is not the same as no.
    }
  }],

  "tournament": { "name": "…", "season": 2026, "start": "…", "end": "…",
                  "state_at_build": "pre", "course": { "name": "…", "par": 70 } },

  "sources": {
    "kalshi": { "event_ticker": "KXPGATOUR-WYC26", "series_ticker": "KXPGATOUR",
                "markets_endpoint": "https://…", "odds_type": "winner",
                "market_label": "Outright Winner",
                "mutually_exclusive_outcomes": true,   // false for top5/top10/makecut
                "price_mode": "ask", "price_level_structure": ["tapered_deci_cent"],
                // false, always. The page never requests this endpoint; it is here so a
                // reader can check the numbers server-side.
                "browser_reachable": false, "browser_note": "…403…" },
    "espn":   { "event_id": "401811961", "leaderboard_endpoint": "https://…",
                "browser_reachable": true,
                "field_available_at_build": false,
                "identities_from_history": 146,        // see §8
                "match_report": { "matched": 146, "requested": 150,
                                  "from_field": 0, "from_history": 146,
                                  "not_in_field": [], "unresolved": [ … ],
                                  "history": { "scanned": [ … ], "athletes": 358,
                                               "unscanned_events": 30 } } }
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

  // What the page does while it is open. One endpoint, on a timer, and the name join
  // it needs to read the answer. There is no odds endpoint here — see §4.
  "live": {
    "espn_leaderboard_url": "https://…&event=401811961",
    "poll_interval_seconds": 60,
    "name_match": { "strategy": ["alias", "normalized_exact", "first_initial_and_last_name"],
                    "normalization": "…", "aliases": { "Zachary Bauchou": "Zach Bauchou" } }
  },

  "standings_rules": { "golfer_rank_tiers": { "0": "…", "1": "…", "2": "…", "3": "…" },
                       "comparison": "lexicographic …" }
}
```

Typical size: ~100 KB for a 150-golfer field with five inlined SVG logos.

---

## 8. Matching golfers to the leaderboard

Kalshi and ESPN publish the same golfers under different names, and Kalshi's stable
golfer UUID is not an ESPN id. The join is by name, and **the page has to be able to do
it** — a build run on Wednesday night has no ESPN field to join against, so
`golfers[].espn.athlete_id` is frequently null.

Three tiers, in order. `lib.js` implements all of them:

1. **Alias** — `DATA.live.name_match.aliases`, a hand-maintained override. Always wins.
2. **Normalised exact** — NFKD, drop combining marks, transliterate the letters NFKD
   leaves alone (`ø→o`, `æ→ae`, `å→a`, `ł→l`, `ß→ss`), lowercase, hyphens and
   apostrophes to spaces, drop `jr/sr/ii/iii/iv/v`, drop non-letters, join runs of
   consecutive single letters (`C.T. Pan` ≡ `CT Pan`), collapse whitespace.
3. **First initial + last name** — resolves every formal-vs-familiar first name a golf
   field throws up.

Measured on a real field, 151 Kalshi names against 147 ESPN competitors:

| tier | resolved |
|---|---|
| normalised exact | 139 |
| first initial + last | 8 — Zachary/Zach Bauchou, Cameron/Cam Davis, Kris/Kristoffer Ventura, Nicolas/Nico Echavarria, Matthew/Matt McCarty, Benjamin/Ben James, Jordan L./Jordan Smith, Hao-Tong/Haotong Li |
| unresolved | 4 — all genuinely absent from the field (withdrew before play) |

Collisions inside the ESPN field for tier 2: **zero**. When a key *is* ambiguous it is
dropped rather than guessed, and the golfer falls through to unresolved — which is
tier 2 of the rank key, and a legitimate display state ("not in the field").

### Identity before the tournament starts

The build does not have to leave `athlete_id` null just because this week's field is
empty. Those golfers played last week, so the build walks back through the season's
finished tournaments and matches against the union of their fields — the same three
tiers, over a wider set of athletes. Measured on the 2026 Wyndham with nothing
published: **146 of 150** identified from four earlier leaderboards, in 3.5 seconds.

Two things follow, and the page depends on both:

1. **What comes back is identity, never scoring.** An athlete id, a display name, a
   headshot, a country. Position, `sortOrder` and to-par describe a tournament that is
   over, and the standings rule ranks on exactly those fields — a golfer who won in July
   would show T1 on Thursday morning. `golfers[].espn.source` says `"history"` when this
   is where the identity came from, and `from_event` names the tournament.

2. **A baked `athlete_id` is a better runtime key than a name.** `lib.js` tries it first,
   so a pre-tournament build now hands the page an exact join instead of re-deriving one
   by name every poll. A golfer who does not tee off simply fails that lookup and falls
   through the name tiers to unresolved, which is correct.

Tier 2 needs one extra guard over a season's worth of athletes: a first-initial key that
is unique inside one field need not be unique across five hundred golfers. The union
index drops any key two athletes share, and the scan keeps widening the union while any
match still rests on tier 2 — so an ambiguous name comes back unresolved rather than
bound to the wrong person. Measured four tournaments back from the 2026 Wyndham: one
refused key, `c|young`, holding both Cameron Young and Carson Young.

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

**Design the empty states.** Before the field is posted, after ESPN goes down, when a
team's golfers have all been cut, when a logo is null. Every one of these happens in a
normal week. What is *not* on that list any more is live odds: there is no such state,
because there is no such fetch — the odds are simply there, dated.

**Accessibility.** Position and score must not be conveyed by colour alone — a page
read across a room is exactly where that fails. Support both colour schemes; the
reference honours `prefers-color-scheme`.

---

## 11. Checklist

- [ ] Reads its data from `#competition-data`; fetches nothing on load
- [ ] Contains `/*__COMPETITION_JSON__*/` inside that script tag
- [ ] No CDN, no external module, no build step
- [ ] **ESPN is the only host it ever requests** — no odds fetch, no relay, at all
- [ ] Polls ESPN on `live.poll_interval_seconds`, with the pinned `event` id
- [ ] Refuses a leaderboard whose event id is not the pinned one
- [ ] Live total summed from `linescores`, never `score.displayValue`
- [ ] Standings match `tests/fixtures/standings_golden.json`
- [ ] Ties shown as ties; `decided_at` surfaced
- [ ] Cut / WD / not-in-field golfers listed and visibly out
- [ ] Snapshot odds shown with capture time; exclusions shown with reasons
- [ ] A rebuilt page shows `odds.current` as movement against `odds.raw`, and says the
      arrows are from a rebuild rather than a feed
- [ ] A page that was never rebuilt (`refreshed` null) simply has no movement column —
      not a blank one, not a spinner, not an apology
- [ ] Handles `pre` (zero competitors), ESPN down, null logos
- [ ] With no leaderboard: every roster and its odds are shown, and nothing is ranked
- [ ] Readable on a phone; works in light and dark
- [ ] `python -m pytest tests/test_frontend_parity.py tests/test_frontend_render.py` passes
