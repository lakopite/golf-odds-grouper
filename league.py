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

    {"crest": "logos/crest.png", "banner": "logos/banner.png", "tagline": "10th Anniversary"}

All three are optional and all three are null by default -- a league with no art
is a normal league and the page is designed for it. Paths are relative to the
league file and are inlined as data: URIs at build time, exactly as team logos
are, so an exported page still shows them on a plane. They are the one part of a
league file that is about how the pool looks rather than who is in it, and they
live here rather than in the template because a template that hard-coded one
league's crest would be that league's template.

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

# The league's own identity, for the scoreboard masthead. Every one of them is
# optional and every one of them is null when absent -- see the module docstring.
BRANDING_FIELDS = ("crest", "banner", "tagline")


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
    The three optional masthead fields, validated and always present as keys.

    Always present, and null when unset: a page that has to distinguish "this league
    has no crest" from "this build predates crests" is a page that will get it wrong,
    and the difference is worth nothing to anybody. An empty string is treated as
    unset for the same reason -- it is what a hand-edited file grows when somebody
    clears a value, and rendering an <img> with no src is worse than rendering none.
    """
    out = {}
    for field in BRANDING_FIELDS:
        value = raw.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            out[field] = None
            continue
        if not isinstance(value, str):
            _fail(path, f"{field!r} must be a string or null, got {type(value).__name__}")
        out[field] = value.strip()
    return out


def write_ids(path, league):
    """
    Pin the derived ids into the league file, in object form.

    Once written they are explicit and stop being derived, so a later rename keeps the
    team's identity. Opt-in: nothing else in this module writes to the user's file.
    """
    payload = {
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        # Written back only when set. This rewrites the user's file, and adding three
        # null keys they never typed is how a tool teaches people not to run it.
        **{f: league[f] for f in BRANDING_FIELDS if league.get(f)},
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
    args = ap.parse_args(argv)

    try:
        league = load_league(args.league)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"{league['league_name']}  ({league['league_id']})")
    if league.get("tagline"):
        print(f"  {league['tagline']}")
    for field in ("crest", "banner"):
        if league.get(field):
            print(f"  {field}: {league[field]}")
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
