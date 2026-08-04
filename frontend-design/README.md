# frontend-design/

The design pass, and what shipped from it.

`Frontend wireframe design/WCW Scoreboard.dc.html` is the source of record for how the
scoreboard looks. It is a hi-fi wireframe: a React component driven by `support.js`,
with every rule written as an inline `style` attribute, sample data for twelve teams and
a hundred golfers hard-coded in the class body, and editor props for switching between
build modes and ESPN states. It is not a page you can hand anybody — it has no data
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
| `notStarted` — a boolean prop | `DATA.live === null`, and the three separate reasons a live page has nothing to rank |
| Inline `style` on every node | `frontend/scoreboard/style.css`, as classes |
| The `narrow` flag and its resize listener | media queries at 760px |
| Google Fonts `<link>` | the same families named first, over a system stack |
| Sample "Excluded", "Grouping certificate", "Name join" cards | `odds_snapshot`, `grouping`, `sources.espn.match_report` |
| Editor props (`buildMode`, `espnState`, `showAnnotations`) | facts in the file, or state from the network. Nothing is a setting |

Two things in the wireframe deliberately did **not** ship: the developer annotation
chips (`GolfPool.computeStandings()`, `build_mode: "groups"`) became factual badges
about the competition instead, and the hard-coded "12 teams · 105 golfers grouped" line
became the real counts.

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
