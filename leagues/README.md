# Leagues

One file per league. A league is the roster of people who play the pool — it outlives
any single tournament, so it is written by hand and kept.

```json
{
  "league_name": "Sunday Fivesome",
  "teams": [
    { "team_name": "Bogey Boys",     "player_name": "Mo",   "team_logo": "logos/bogey-boys.svg" },
    { "team_name": "Mulligan Mafia", "player_name": "Luis", "team_logo": null }
  ]
}
```

A bare JSON list of team objects works too and takes its league name from the filename.

| Field | |
|---|---|
| `team_name` | required, unique — the team id is derived from it |
| `player_name` | required — the human |
| `team_logo` | optional: a path relative to this file, an `https://` URL, or null |

**One team is one group.** Five teams, five groups.

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

## Logos

A local path is inlined into the export as a data URI, so the exported page stays a
single portable file. Keep them small — anything over 512 KB is refused and left as a
path, because it would land in every copy of the result JSON and every page built from
it. SVG is ideal; `logos/` here holds five as a worked example.

## Checking a file

```bash
python league.py leagues/my-league.json
```

Prints the teams and their ids, and fails with a sentence rather than a stack trace if
something is wrong.

## What is checked in

`example-league.json` and `logos/` only. Everything else in this directory is
gitignored — your rosters are yours.
