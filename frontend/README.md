# frontend/

Everything the exported page is made of. Nothing here runs in this directory — it is
input to `bundle_frontend.py`, which folds one of these templates around a result file
and writes a single self-contained `.html`.

```
frontend/
  lib.js            the standings rule and the ESPN parser. ONE copy, above both templates.
  scoreboard/       the designed page. The default.
  template/         the plain reference. Proves the contract; is not the design.
```

## The one copy of the rule

`lib.js` is the rule that decides the pool: how a golfer ranks, how a team's golfers
compare, what a tie is, and how ESPN's leaderboard payload is read. `standings.py`
implements the same rule in Python and `tests/test_frontend_parity.py` runs both over
the same payloads and fails if they disagree.

It sits **above** the template directories, and every template pulls it in with:

```html
<script src="../lib.js"></script>
```

which the bundler inlines like any other local asset. A template that copied the file
into its own directory would be a second implementation that nothing checks — one
implementation and one rumour. Write presentation; reuse this verbatim.

## The two templates

| | `scoreboard/` | `template/` |
|---|---|---|
| What it is | the design in `frontend-design/`, built for real data | the plainest thing that honours the contract |
| Bundled by default | ✅ | `--template frontend/template` |
| Views | Standings and Odds & the draw, as tabs | one scrolling page |
| Layout | league table with expandable groups, phone layout, light and dark | a stack of cards |
| Render suite | `tests/test_scoreboard_render.py` | `tests/test_frontend_render.py` |

Both make the same claims about the same data, and both suites check them. The
reference exists so that a page that looks wrong can be checked against a page with no
design in it: if the reference shows the same thing, the data says it.

## Adding a template

A template is a directory with an `index.html` in it. The whole contract is in
`bundle_frontend.py`'s docstring and `docs/FRONTEND-SPEC.md` §9; the short version:

1. **Carry the data marker**, verbatim, or the build fails loudly:
   ```html
   <script id="competition-data" type="application/json">/*__COMPETITION_JSON__*/</script>
   ```
2. **Reference assets locally.** `<link rel=stylesheet>`, `<script src>` and `<img src>`
   are inlined; a reference may point above the template directory (`../lib.js` does).
   Absolute URLs are left alone, which means they are still requests — and a page whose
   normal condition is being opened offline should have none. No CDN, no webfont link,
   no framework it did not inline.
3. **Leave `src` off** an `<img>` whose source arrives from the data at runtime. An
   empty `src=""` resolves to the template directory, and a directory is not an image.
4. **Use `{{tokens}}`** for anything that should be right before JavaScript runs:
   `league_name`, `tournament`, `market`, `generated_at`, `team_count`, `competition_id`.
5. **Carry the art element if you draw a masthead** — optional, unlike the data marker:
   ```html
   <script id="league-art" type="application/json">/*__LEAGUE_ART_JSON__*/</script>
   ```
   The bundler fills it with `{"logo": "data:…", "banner": "data:…"}`, read out of
   `leagues/<slug>/`. `DATA.league.logo` is that slug and not an image — a page that
   sets an `<img src>` from it renders a broken image with the word `wcw` under it.

Then point the bundler at it:

```bash
python bundle_frontend.py --result build/result.json --template frontend/mine --out dist/
```

## The three things a template has to get right

Everything else is taste. These are not:

- **Branch on `DATA.live` once.** Null means the page was built before ESPN published a
  field: it is a groups sheet, it fetches nothing, and it ranks nothing. Non-null means
  poll, join on `golfers[].espn.athlete_id`, and rank.
- **Never rank an empty board.** A live page can be handed zero competitors any
  afternoon — ESPN down, a payload refused for the wrong event, a first poll still in
  flight. Show every roster and every price; show no positions.
- **Say the odds are a snapshot.** Kalshi 403s every browser origin, so the page cannot
  fetch odds and never tries. The prices are the ones the groups were drawn on, dated.
