# frontend-design/

The design pass, and what shipped from it.

`Frontend wireframe design/WCW Scoreboard.dc.html` is the source of record for how the
scoreboard looks. It is a hi-fi wireframe: a React component driven by `support.js`,
with every rule written as an inline `style` attribute, sample data for twelve teams and
a hundred golfers hard-coded in the class body, and editor props for switching between
the pre-tournament view and the ranked one, and between ESPN states. It is not a page you can hand anybody — it has no data
contract, it fetches a font CDN, and it needs a runtime that is not in this repository.

**`frontend/scoreboard/` is that design built for real data**, and it is what
`bundle_frontend.py` produces by default. Read the wireframe for intent; read the
template for what runs.

## What moved, and where it went

| In the wireframe | Shipped as |
|---|---|
| `TEAMS`, `FIELD` — hard-coded sample data | `DATA.teams`, `DATA.golfers` from the result file |
| `rank`, `cmp`, `cmpVec`, `build()` — the standings rule, re-derived | `frontend/lib.js`, verbatim, under the parity test |
| Sidebar crest, name, "10th Anniversary" | `DATA.league.crest` / `.league_name` / `.tagline` |
| `assets/wcw-banner.png` | `DATA.league.banner`, from the league file |
| `statusView()` — an editor prop with three settings | the real poll: last good response, staleness, HTTP and event-id errors |
| `notStarted` — a boolean prop | `meta.started` off each poll, read by `ranked()`. Not a fact in the file at all: the page crosses over by itself at the first tee time, so the two states are two moments of one page rather than two builds. The wireframe's paragraph callout did not ship — which of the several reasons there is nothing to rank is the status pill's job, in two words |
| Inline `style` on every node | `frontend/scoreboard/style.css`, as classes |
| The `narrow` flag and its resize listener | media queries at 760px |
| Google Fonts `<link>` | the same families named first, over a system stack |
| Sample "Excluded" and "Grouping certificate" cards | `odds_snapshot`, `grouping` |
| The full draw, group by group | `DATA.teams` and `DATA.golfers[].odds.grouping_weight` |
| Editor props (`buildMode`, `espnState`, `showAnnotations`) | state from the network, or facts in the file. Nothing is a setting — and `buildMode` is not a fact either any more: there is one build, and what the page draws comes from what the leaderboard last said |

Two things in the wireframe deliberately did **not** ship: the developer annotation
chips (`GolfPool.computeStandings()`, `build_mode: "groups"`) became factual badges
about the competition instead, and the hard-coded "12 teams · 105 golfers grouped" line
became the real counts.

Four things were later taken **out** of both, and the wireframe was edited to match:
the "Prices do not move here" card and the movement column it described (there is one
price per golfer now — see `docs/FRONTEND-SPEC.md` §4), the "Name join" card, the
"Where the numbers came from" card, and the not-started paragraph and rank-tier legend
on the standings view. All of them were accurate; all of them were read by the person
who built the page rather than by the league, and they crowded out the two cards and
the full draw that answer the question the page exists for.

## Changing the design

Edit the wireframe, then port the change into `frontend/scoreboard/`. The wireframe
cannot be bundled and the template cannot be opened in the design tool, so they are kept
in step by hand — which is the cost of a design tool whose output is a React runtime and
a deliverable that is one static file with no runtime at all.

`docs/FRONTEND-SPEC.md` is the contract both of them answer to, and
`tests/test_scoreboard_render.py` is what says the shipped page still answers it.

## The art

`assets/` holds the originals at full size: a 1024×1024 crest and a 1584×672 banner,
3.6 MB between them. Neither is usable as-is — a league's art is inlined into every copy
of the exported page, and the build refuses anything over 512 KB. Downscaled copies live
at `leagues/logos/wcw-crest.png` (256 px, 99 KB) and `leagues/logos/wcw-banner.png`
(720 px, 425 KB); point a league file's `crest` and `banner` at those.
