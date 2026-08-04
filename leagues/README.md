# Leagues

One file per league. A league is the roster of people who play the pool — it outlives
any single tournament, so it is written by hand and kept.

```json
{
  "league_name": "Sunday Fivesome",
  "tagline": "Season 4",
  "logo": "example",
  "teams": [
    { "team_name": "Bogey Boys",     "player_name": "Mo",   "team_logo": "example/bogey-boys.svg" },
    { "team_name": "Mulligan Mafia", "player_name": "Luis", "team_logo": null }
  ]
}
```

A bare JSON list of team objects works too and takes its league name from the filename.

| Field | |
|---|---|
| `league_name` | optional in the object form; defaults to the filename |
| `logo` | optional: the name of a directory in here holding the league's art |
| `tagline` | optional: a short line under the league name, e.g. `10th Anniversary` |
| `team_name` | required, unique — the team id is derived from it |
| `player_name` | required — the human |
| `team_logo` | optional: a path relative to this file, an `https://` URL, or null |

**One team is one group.** Five teams, five groups.

## The masthead

Two images and a line of text: a badge beside the league name, a banner across the top
of the page, and the tagline under the name.

They are the *only* part of the page that changes between leagues. The navy, the gold,
the type and the layout belong to the scoreboard template and are the same for every
competition — which is why the two images live here rather than in the template, and
why nothing else does.

### `logo` is a slug, not a path

It names a directory in here, and that directory holds the art at fixed names:

```
leagues/wcw/logo.png        the square badge, around 256 px
leagues/wcw/banner.png      the wide image, around 720 px across
```
```json
{ "logo": "wcw" }
```

`.png`, `.jpg`, `.jpeg`, `.webp` and `.svg` all work — the *name* is fixed, the format
is yours. Both files are optional: a league with a badge and no banner is an ordinary
league and the page draws it, and a league with neither gets a masthead with its name
in it, which is a shape the design handles.

`--logo wcw` at creation beats what the file says, and `--no-logo` builds one
competition with no art at all:

```bash
python build_competition.py --league leagues/wcw.json --tournament Wyndham --logo wcw
```

A typo in `--logo` is an error rather than a shrug, and it fires before the build spends
a minute on Kalshi. A slug in the league file that names nothing is a warning: the build
finishes, and the page comes out with just the league's name.

### Where the images actually go

Nowhere until the export. This is the part worth understanding:

* the **league file** names the art: `"logo": "wcw"`
* the **result JSON** carries that same name, and no images at all
* the **exported page** has both images inlined as `data:` URIs

So `build/result.json` stays a readable document about a competition — a few hundred KB
of odds and groups — while `dist/<league>-<tournament>.html` is still one self-contained
file that opens from a USB stick with the wifi off. It used to be that both carried the
art, which made every result file mostly base64 and every rebuild a copy of it.

The images are read out of *this directory* at export time. That is why the art belongs
in the repository rather than on one laptop, and it is checked in even though the league
files themselves are not.

### Once built, it is settled

The result file records the slug, and a rebuild carries it forward — including a
deliberate absence. A page rebuilt on Thursday looks like the one sent on Wednesday, and
a rebuild never grows art the first build did not have. Replacing `leagues/wcw/logo.png`
and re-exporting *will* change the page, which is the one way to update the art of a
competition already drawn.

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

## Art, and what it costs

Team logos and the league's own art take different routes and it matters. A `team_logo`
is a path, and it is inlined into the **result JSON** at build time — so it is capped at
512 KB, and anything over that is refused and left as a path that will not resolve in
the export. SVG is ideal and costs almost nothing.

The league's `logo` and `banner` are not capped, because they only ever land in the
page. They are still most of that page's weight: a 2 MB banner becomes 2.7 MB of base64
in every copy of the HTML. A 256 px badge and a 720 px-wide banner are the right order
of magnitude, and a JPEG beats a PNG by roughly five to one on photographic art. The
export prints the size of anything over 1 MB rather than refusing it — a page too heavy
to email is a nuisance, not a wrong answer, but it should be one you chose.

The banner is drawn into a wide, short slot (184 px tall, full width) with
`object-fit: cover`, so it is centre-cropped top and bottom. Art whose meaning lives in
the middle band survives that; art with detail at the edges does not.

## Checking a file

```bash
python league.py leagues/my-league.json
```

Prints the teams and their ids, resolves the art slug to the files it actually finds,
and fails with a sentence rather than a stack trace if something is wrong.

## What is checked in

`example-league.json`, `example/` and the art directories of real leagues. Everything
else in here is gitignored — your rosters are yours, but your art has to be in the repo
for the export to find it.
