"""
league.py -- load a league definition and give every team a stable id.

A league is the roster of people playing the pool. It is the one thing that
outlives a single tournament, so it lives in a file the user writes by hand:

    leagues/<slug>.json

Two shapes are accepted. The object form carries a league name; the bare-list
form takes its name from the filename, which is enough for a one-league setup.

    {"league_name": "Sunday Fivesome", "teams": [ ...team objects... ]}
    [ ...team objects... ]

A team object is what the pool needs to draw a scoreboard row:

    {"team_name": "Bogey Boys", "player_name": "Mo", "team_logo": "logos/mo.png"}

The object form also carries the league's own identity, which the scoreboard puts
in its masthead:

    {"logo": "wcw", "tagline": "10th Anniversary"}

Both are optional. `tagline` is a line of text. `logo` is NOT a path -- it is a
slug naming a directory of art under `leagues/`, which holds one or both of:

    leagues/wcw/logo.png        the square badge beside the league name
    leagues/wcw/banner.png      the wide image across the top of the page

They are the one part of a league file that is about how the pool looks rather
than who is in it, and they live here rather than in the template because a
template that hard-coded one league's crest would be that league's template.
Everything else about the page -- the navy, the gold, the type, the layout --
belongs to the template and is the same for every league.

The slug may also be handed to `build_competition.py` as `--logo wcw` when the
competition is created, which beats what the file says.

WHY A SLUG AND NOT TWO PATHS
----------------------------
The art used to be two paths that a build read, base64'd, and wrote into the
result JSON. Every result file then carried half a megabyte of PNG it had no use
for, every rebuild copied it forward, and the one document that describes a
competition was mostly an envelope for two images.

A slug is a name, and a name is all any of it needed. The league file names the
art, the result file passes the name along, and the images are read exactly once
-- at export, by `bundle_frontend.py`, into the single page that actually has to
be portable. The exported HTML is still one file with nothing to fetch.

The cost is that the art has to be findable later, which is why it lives under
`leagues/<slug>/` at fixed names rather than wherever somebody happened to keep
it. It is a small price for a result file that stays readable.

There is no shipped default art any more, so unset means no art and nothing has
to distinguish it from a refusal.

WHY THE TEAM ID IS DERIVED RATHER THAN DRAWN
--------------------------------------------
Every team needs an id the rest of the system can key on. Drawing a random UUID
means writing it back into the user's file, and a file that rewrites itself on
read is a file people stop trusting. So the id is a UUIDv5 over
(league_id, team_name): stable across runs, stable across machines, and
computable from the file alone with nothing to persist.

The cost is that renaming a team mints a new id. That is the right trade while
nothing stores history against the old one -- and the moment something does,
`--write-ids` pins the current ids into the file and they stop being derived.
An explicit `team_id` in the file always wins.
"""

import json
import os
import re
import unicodedata
import uuid

# Fixed namespace so ids are reproducible on any machine, in any process. It is a
# constant, not a secret -- two people running the same league file must agree.
NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

REQUIRED_FIELDS = ("team_name", "player_name")
OPTIONAL_FIELDS = ("team_logo", "team_id", "color", "abbreviation")

# The league's own identity, for the scoreboard masthead. Both are optional and both
# are null when absent -- see the module docstring.
BRANDING_FIELDS = ("logo", "tagline")

# Where a league's art lives: leagues/<slug>/logo.png and leagues/<slug>/banner.png.
# One fixed place, because two programs have to agree on it -- the build validates the
# slug and the exporter reads the files, and they run minutes and a directory apart.
LEAGUES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leagues")

# The two images a slug names, and the extensions they may be saved under. The names
# are fixed so a slug is enough to find them; PNG comes first because that is what the
# checked-in art is, and SVG is last in the list and cheapest on the page.
ART_NAMES = ("logo", "banner")
ART_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".svg")


def slugify(text):
    """A filename-safe, url-safe token. Used for league slugs and export names."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "league"


def league_id_for(league_name):
    return str(uuid.uuid5(NAMESPACE, f"league:{league_name}"))


def team_id_for(league_id, team_name):
    return str(uuid.uuid5(NAMESPACE, f"team:{league_id}:{team_name}"))


def _fail(path, message):
    raise ValueError(f"{path}: {message}")


def load_league(path):
    """
    Read a league file and return it normalised, validated, and fully identified.

    Returns {"league_id", "league_name", "league_slug", "source_file", "crest",
    "banner", "tagline", "teams": [...]} where every team carries
    team_id / team_name / player_name / team_logo.

    Validation is loud and specific on purpose. A league file is hand-written, it is
    read once at the top of a build, and every mistake in it is cheap to state and
    expensive to discover three steps later -- a duplicate team name silently collapses
    two teams into one id, and a JSON object where a list belongs has a length, so it
    passes every downstream check before failing somewhere unrecognisable.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(
            f"{path} not found. A league file holds a list of teams, e.g.\n"
            '  [{"team_name": "Bogey Boys", "player_name": "Mo", "team_logo": "logos/mo.png"}]'
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}")

    default_name = os.path.splitext(os.path.basename(path))[0].replace("-", " ").replace("_", " ").title()

    if isinstance(raw, list):
        league_name, teams = default_name, raw
    elif isinstance(raw, dict):
        teams = raw.get("teams")
        if teams is None:
            _fail(path, 'object form needs a "teams" list. Bare-list form is also accepted.')
        league_name = raw.get("league_name") or raw.get("name") or default_name
    else:
        _fail(path, f"expected a JSON list of teams or an object with a teams list, got {type(raw).__name__}")

    if not isinstance(teams, list):
        _fail(path, f'"teams" must be a list, got {type(teams).__name__}')
    if not teams:
        _fail(path, "holds no teams")

    league_id = (raw.get("league_id") if isinstance(raw, dict) else None) or league_id_for(league_name)
    branding = _branding(path, raw if isinstance(raw, dict) else {})

    seen_names, seen_ids, out = set(), set(), []
    for i, t in enumerate(teams):
        where = f"teams[{i}]"
        if not isinstance(t, dict):
            _fail(path, f"{where} is a {type(t).__name__}, not an object")
        for field in REQUIRED_FIELDS:
            value = t.get(field)
            if not isinstance(value, str) or not value.strip():
                _fail(path, f"{where} needs a non-empty {field!r} (got {value!r})")

        team_name = t["team_name"].strip()
        if team_name in seen_names:
            _fail(path, f"{where} repeats team_name {team_name!r}; team names must be unique "
                        "because the team id is derived from them")
        seen_names.add(team_name)

        team_id = t.get("team_id") or team_id_for(league_id, team_name)
        if team_id in seen_ids:
            _fail(path, f"{where} repeats team_id {team_id!r}")
        seen_ids.add(team_id)

        unknown = set(t) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
        if unknown:
            # Not fatal: extra fields ride along into the result JSON so a frontend can
            # use them. But a typo'd "team_logos" would otherwise vanish without a word.
            print(f"note: {path} {where} carries unrecognised field(s) {sorted(unknown)}; "
                  "they will be passed through untouched.")

        team = {k: v for k, v in t.items() if k not in ("team_id",)}
        team["team_id"] = team_id
        team["team_name"] = team_name
        team["player_name"] = t["player_name"].strip()
        team.setdefault("team_logo", None)
        out.append(team)

    return {
        "league_id": league_id,
        "league_name": league_name,
        "league_slug": slugify(league_name),
        "source_file": path,
        **branding,
        "teams": out,
    }


def _branding(path, raw):
    """
    The two optional masthead fields, validated and always present as keys.

    Always present, and null when unset: a page that has to distinguish "this league
    has no logo" from "this build predates logos" is a page that will get it wrong,
    and the difference is worth nothing to anybody. An empty string is treated as
    unset for the same reason -- it is what a hand-edited file grows when somebody
    clears a value, and rendering an <img> with no src is worse than rendering none.

    That `logo` is a slug and not a path is checked here, and it is the whole reason
    this is not a two-line comprehension. `"logos/wcw-crest.png"` is what every league
    file said before the art moved into `leagues/<slug>/`; taken as a slug it names a
    directory nobody will ever create, and the only symptom would be a masthead that
    came out empty at the far end of a build. It is one comparison to say so instead.
    """
    out = {}
    for field in BRANDING_FIELDS:
        value = raw.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            out[field] = None
            continue
        if not isinstance(value, str):
            _fail(path, f"{field!r} must be a string or null, got {type(value).__name__}"
                        + (". `false` used to mean \"no art\"; unset means that now."
                           if value is False else ""))
        value = value.strip()
        if field == "logo" and value != slugify(value):
            _fail(path, f"'logo' names a directory of art under leagues/ -- "
                        f"leagues/<slug>/logo.png and leagues/<slug>/banner.png -- so it is a "
                        f"slug like \"wcw\", not a path. Got {value!r}.")
        out[field] = value
    return out


def art_files(slug, leagues_dir=None):
    """
    The art a `logo` slug names: {"logo": path or None, "banner": path or None}.

    One directory, two fixed names, whichever extension the file was saved under. Both
    are optional and either may be missing -- a league with a badge and no banner is an
    ordinary league and the page draws it -- so a slug that names nothing at all comes
    back as two Nones rather than as an exception. Only the caller knows whether that
    is worth a word: a build says so, a rebuild of a competition somebody deliberately
    left bare should not.

    `leagues_dir` exists for tests and for anyone keeping their leagues elsewhere; the
    default is the one place both the build and the exporter look.
    """
    found = {name: None for name in ART_NAMES}
    if not slug:
        return found
    directory = os.path.join(leagues_dir or LEAGUES_DIR, slug)
    for name in ART_NAMES:
        for ext in ART_EXTENSIONS:
            candidate = os.path.join(directory, name + ext)
            if os.path.isfile(candidate):
                found[name] = candidate
                break
    return found


def write_ids(path, league):
    """
    Pin the derived ids into the league file, in object form.

    Once written they are explicit and stop being derived, so a later rename keeps the
    team's identity. Opt-in: nothing else in this module writes to the user's file.
    """
    payload = {
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        # Written back only when set. This rewrites the user's file, and adding keys
        # they never typed is how a tool teaches people not to run it.
        **{f: league[f] for f in BRANDING_FIELDS if league[f] is not None},
        "teams": league["teams"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Validate a league file and show its teams.")
    ap.add_argument("league", help="path to the league JSON")
    ap.add_argument("--write-ids", action="store_true",
                    help="write the derived team ids back into the file, pinning them")
    ap.add_argument("--leagues-dir", default=LEAGUES_DIR,
                    help=f"where the art slugs live (default {LEAGUES_DIR})")
    args = ap.parse_args(argv)

    try:
        league = load_league(args.league)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"{league['league_name']}  ({league['league_id']})")
    if league.get("tagline"):
        print(f"  {league['tagline']}")
    if league["logo"]:
        # Resolved rather than repeated back. "logo: wcw" says nothing a reader could
        # not see in their own file; the two paths say whether the export will find
        # anything, which is the question they opened this for.
        found = art_files(league["logo"], args.leagues_dir)
        print(f"  logo: {league['logo']}")
        for name in ART_NAMES:
            print(f"    {name}: {found[name] or 'not found'}")
    print(f"{len(league['teams'])} teams -> {len(league['teams'])} groups\n")
    width = max(len(t["team_name"]) for t in league["teams"])
    for t in league["teams"]:
        logo = t.get("team_logo") or "-"
        print(f"  {t['team_name']:<{width}}  {t['player_name']:<16} {t['team_id']}  {logo}")

    if args.write_ids:
        write_ids(args.league, league)
        print(f"\nWrote ids into {args.league}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
