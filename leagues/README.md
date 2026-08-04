# Leagues

One file per league. A league is the roster of people who play the pool — it outlives
any single tournament, so it is written by hand and kept.

```json
{
  "league_name": "Sunday Fivesome",
  "tagline": "Season 4",
  "crest": "logos/example-crest.svg",
  "banner": "logos/example-banner.svg",
  "teams": [
    { "team_name": "Bogey Boys",     "player_name": "Mo",   "team_logo": "logos/bogey-boys.svg" },
    { "team_name": "Mulligan Mafia", "player_name": "Luis", "team_logo": null }
  ]
}
```

A bare JSON list of team objects works too and takes its league name from the filename.

| Field | |
|---|---|
| `league_name` | optional in the object form; defaults to the filename |
| `crest` | optional: a small square badge for the masthead. `false` for none |
| `banner` | optional: a wide image across the top of the page. `false` for none |
| `tagline` | optional: a short line under the league name, e.g. `10th Anniversary` |
| `team_name` | required, unique — the team id is derived from it |
| `player_name` | required — the human |
| `team_logo` | optional: a path relative to this file, an `https://` URL, or null |

**One team is one group.** Five teams, five groups.

## The masthead

Two images and a line of text. The crest sits beside the league name in the rail, the
banner runs across the top of the page, and the tagline goes under the name.

They are the *only* part of the page that changes between leagues. The navy, the gold,
the type and the layout belong to the scoreboard template and are the same for every
competition — which is why the two images live here rather than in the template, and
why nothing else does.

### Where the two images come from

In precedence order, highest first:

```bash
# 1. handed in when the competition is created — beats the file
python build_competition.py --league leagues/wcw.json --tournament Wyndham \
    --crest art/crest.png --banner art/banner.png

# 2. named in the league file, relative to it
{ "crest": "logos/wcw-crest.png", "banner": "logos/wcw-banner.png" }

# 3. neither — and the build uses the art the tool ships, and says so
```

`--crest` / `--banner` are the creation-time route: the images arrive beside the league
JSON, get inlined, and are baked into the exported page. A path typed there resolves
against the working directory, not against the league file, and a typo in one is an
error rather than a shrug — the build stops before it spends a minute on Kalshi.

### Saying "no art"

Unset means *"I did not supply any"* and gets the default. To mean *"this league has
none"*, say so: `"crest": false` in the file, or `--no-crest` for one build. Either way
the page renders with the league's name and nothing missing-looking where art would
have gone; it is a shape the design handles.

### Once built, it is settled

The result file records the art as a data URI, and a rebuild carries that forward —
including a deliberate absence. A page rebuilt on Thursday looks like the one sent on
Wednesday, and a rebuild never grows a crest the first build did not have.

## Team ids

Do not write them. Each is a UUIDv5 over (league, team name): stable across runs and
machines, with nothing to persist and no file that rewrites itself when you read it.
An explicit `team_id` in the file always wins if you ever need one.

The trade is that renaming a team mints a new id. That is right while nothing stores
history against the old one — and when something does, pin the current ids into the
file and they stop being derived:

```bash
python league.py leagues/my-league.json --write-ids
```

## Art

Team logos, the crest and the banner all work the same way: a local path is inlined
into the export as a data URI, so the exported page stays a single portable file.

Keep them small. Anything over 512 KB is refused and left as a path, which will not
resolve in the export — and every inlined byte lands in the result JSON *and* in every
page built from it. SVG is ideal and costs almost nothing; `logos/` holds five team
badges plus a crest and a banner as a worked example.

For photographic art there is no free lunch: a 256 px crest and a 720 px-wide banner
are the right order of magnitude, and a JPEG beats a PNG by roughly five to one on a
banner. `logos/wcw-crest.png` and `logos/wcw-banner.png` are checked in at those sizes,
and are what a build uses when it is offered nothing — between them they are about
520 KB, which is most of a default page's weight. Supplying lighter art is the way to
a lighter page.

The banner is drawn into a wide, short slot (184 px tall, full width) with
`object-fit: cover`, so it is centre-cropped top and bottom. Art whose meaning lives in
the middle band survives that; art with detail at the edges does not.

## Checking a file

```bash
python league.py leagues/my-league.json
```

Prints the teams and their ids, and fails with a sentence rather than a stack trace if
something is wrong.

## What is checked in

`example-league.json` and `logos/` only. Everything else in this directory is
gitignored — your rosters are yours.
