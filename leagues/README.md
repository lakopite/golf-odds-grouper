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
| `crest` | optional: a small square badge for the masthead |
| `banner` | optional: a wide image across the top of the page |
| `tagline` | optional: a short line under the league name, e.g. `10th Anniversary` |
| `team_name` | required, unique — the team id is derived from it |
| `player_name` | required — the human |
| `team_logo` | optional: a path relative to this file, an `https://` URL, or null |

**One team is one group.** Five teams, five groups.

## The masthead

`crest`, `banner` and `tagline` are the league's own identity, and they live here
rather than in the scoreboard template because a template that hard-coded one league's
crest would be that league's template. All three are optional and all three default to
null; a league with none of them gets a page that looks finished, with its name in the
masthead and nothing missing-looking where art would have gone.

They travel with the competition: a rebuild carries them forward out of the result
file, so a page rebuilt on Thursday looks like the one sent on Wednesday.

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
banner. `logos/wcw-crest.png` and `logos/wcw-banner.png` are checked in at those sizes.

## Checking a file

```bash
python league.py leagues/my-league.json
```

Prints the teams and their ids, and fails with a sentence rather than a stack trace if
something is wrong.

## What is checked in

`example-league.json` and `logos/` only. Everything else in this directory is
gitignored — your rosters are yours.
