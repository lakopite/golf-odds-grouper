#!/usr/bin/env python3
"""
build_competition.py -- one league + one tournament + one market -> one result file.

This is the step between the engine and the scoreboard. It resolves the tournament
on both APIs, pulls the odds, runs the partitioner, deals the groups out to the
league's teams, and writes a single JSON file that is the source of truth for that
competition:

    python build_competition.py --league leagues/sunday-fivesome.json \
                                --tournament "Wyndham Championship" \
                                --odds winner

Everything the scoreboard needs is in that file, and everything needed to audit it
later is in it too: which endpoints were read, when, at what prices, who was
excluded and why, and how good the partition was. Nothing in the frontend re-derives
a number that this file already states.

TWO BUILDS, AND THE ESPN LEADERBOARD DECIDES WHICH
---------------------------------------------------
ESPN publishes ZERO competitors until the first tee time. That single fact splits
this program in half, and the split is read off the payload rather than off a flag:

  groups   ESPN has published no field. There is nobody to join the Kalshi names
           against and no score to show, so the build does not try: no ESPN block on
           any golfer, `live` is null, and the page that gets built fetches nothing
           at all. What it has is everything that is already decided -- the teams,
           the groups, and the odds the groups were drawn on. That is a complete,
           deployable, honest artifact: it is the groups sheet, and the groups sheet
           is what exists on Wednesday.

  live     ESPN has published a field. Now the Kalshi names can be joined against
           THIS week's actual competitors, every match is against a person who is
           really in this tournament, and the page can rank. A golfer the join does
           not resolve carries no athlete id, scores nothing and is not counted --
           and, crucially, is NAMED, so the gap is a thing somebody can close rather
           than a silence.

So the normal week is two runs of this program against one competition: build the
groups before the tournament, rebuild once it starts.

    python build_competition.py --league ... --tournament ...   # Wednesday, groups
    python build_competition.py --from-result build/result.json # Thursday, live

SETTLING THE NAMES THE JOIN WILL NOT GUESS AT
----------------------------------------------
The join is exact and explicit in all three of its tiers -- there is no fuzzy match,
for the reasons in espn_leaderboard's docstring -- so a live build leaves a handful of
names open. It writes them to a review file, each beside the ESPN athletes nobody
claimed and a ranked suggestion or two, and the next build reads that file back and
applies whatever has been filled in. See match_review.py. The review is the step a
model is good at and a regular expression is not, and it happens where the answer can
be read before it takes effect.

REBUILDING ONE
--------------
Because the result file describes the whole competition, it is also the input to the
next build of it:

    python build_competition.py --from-result build/result.json

That reads the league, the tournament on both APIs, the market, the price mode, the
hand-picked exclusions, the seed and any reviewed name decisions out of the file,
carries the groups and the odds at creation forward untouched, and redoes the parts
that have a "now" -- above all the ESPN join, which is the whole difference between
the two builds above. It does not re-read the odds, ever: they were read once, when
the groups were drawn, and that reading is the competition. `--regroup` is the one
that deals again, and it says so.

WHY IT ALL GOES IN ONE FILE
---------------------------
The scoreboard is a static page with no backend, so at runtime it can reach exactly
two things: the ESPN leaderboard, and whatever is baked into it. Odds move and Kalshi
will not answer a browser at all (its API allowlists origins -- see
docs/FRONTEND-SPEC.md), so the odds AT CREATION TIME have to be carried, not fetched.
A file that carries them but not the endpoint they came from is a number nobody can
check, so the endpoint is carried too.

WHAT IT DOES NOT DO
-------------------
It does not write into the repository. The result file and the bundle it feeds are
artifacts of one run and belong to the user, not to the source tree. The one thing a
run can teach the repo is a golfer name alias, and that is opt-in (`--update-aliases`)
and prints what it learned either way.
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import subprocess
import sys
import time
import urllib.parse
import uuid
import zipfile
from datetime import datetime, timezone

import espn_leaderboard
import group as grouper_cli
import groupers
import kalshi_odds
import league as league_mod
import match_review

_HERE = os.path.dirname(os.path.abspath(__file__))

# 1.1 added, all additive: `rebuilt_from`, `odds_snapshot.refreshed`,
# `odds_snapshot.auto_exclude`, `golfers[].odds.current`, and `golfers[].espn` gaining
# source / from_event / in_field.
# 1.2 REMOVED `live.kalshi_markets_url` and `live.kalshi_proxy_url_template`. Kalshi
# 403s every browser origin, so a page could only read them through a relay somebody
# had to run; nothing does, and a slot for a thing that never happens is a promise the
# page kept having to explain. Odds are baked, full stop.
# 2.0 SPLIT the file in two along `build_mode`, and it is a breaking change because the
# halves are not the same document. A "groups" build has `live: null` and `espn: null`
# on every golfer -- not empty, ABSENT, because before the first tee time there is no
# field and every previous attempt to answer anyway (an identity recovered from last
# month's leaderboard, a name match the browser would retry) was answering a question
# the page cannot use. A "live" build carries `golfers[].espn.athlete_id` for everyone
# it resolved, which is what lets the page join by id and drop name matching entirely.
# Also REMOVED: `golfers[].espn.source` / `.from_event`, `sources.espn.
# identities_from_history`, `sources.espn.field_available_at_build` (say `build_mode`),
# and `live.name_match` (nothing matches names at runtime any more). ADDED:
# `build_mode`, `sources.espn.match_decisions`, and `golfers[].espn.match` gaining
# "absent", which is a fact somebody checked rather than a name that was missed.
# 2.1 REMOVED `odds_snapshot.refreshed`, `golfers[].odds.current`, and the
# "refresh-odds" value of `rebuilt_from.mode`. The odds are read once, at the moment
# the groups are drawn, and that reading IS the competition. A second reading taken
# days later was a price nobody was dealt on, printed beside one they were, and every
# reader who noticed had to be talked back out of thinking the draw had moved. The
# groups are worth what they were worth on Wednesday; there is now exactly one price
# per golfer in this file and no way to ask for another.
SCHEMA_VERSION = "2.1"

# The market a competition is priced off. The Kalshi series ticker is the whole of the
# difference between them -- every series returns the identical market shape.
ODDS_TYPES = {
    "winner":  {"series": "KXPGATOUR",     "label": "Outright Winner",  "exclusive": True},
    "top5":    {"series": "KXPGATOP5",     "label": "Top 5 Finishers",  "exclusive": False},
    "top10":   {"series": "KXPGATOP10",    "label": "Top 10 Finishers", "exclusive": False},
    "makecut": {"series": "KXPGAMAKECUT",  "label": "To Make the Cut",  "exclusive": False},
}
DEFAULT_ODDS_TYPE = "winner"
ALIAS_FILE = "data/espn_aliases.json"

# Everything a rebuild reads back out of a result file rather than off the command
# line, by argparse dest. See apply_result_defaults.
REBUILD_INPUTS = ("tournament", "kalshi_event", "odds", "price", "espn_event",
                  "espn_league", "season", "seed", "exclude", "auto_exclude",
                  "poll_interval")

# Local logos are inlined so the exported bundle is one portable file. A logo bigger
# than this is a mistake rather than a choice -- a 2 MB PNG lands in every copy of the
# result JSON and every copy of the HTML built from it.
MAX_INLINE_LOGO_BYTES = 512 * 1024

# The art a competition gets when nobody supplies any.
#
# The scoreboard's chrome -- the navy, the gold, the type -- is the template's and is
# the same for every league. These two files are the only part of the masthead that is
# a picture rather than a rule, which is why they are the only part a league overrides:
# with `--crest` / `--banner` at creation, or with `crest` / `banner` in the league
# file. Both are checked in at the sizes the page draws them at, 256 px and 720 px.
_ART = os.path.join(_HERE, "leagues", "logos")
DEFAULT_CREST = os.path.join(_ART, "wcw-crest.png")
DEFAULT_BANNER = os.path.join(_ART, "wcw-banner.png")

# Decimal places kept on every probability in the result file. Deep enough that the
# rounding is far below the tick grid the odds live on, shallow enough that the JSON
# reads as numbers rather than as float noise -- and applied consistently, so summing a
# team's golfers gives that team's total exactly.
WEIGHT_PRECISION = 10


# ---------------------------------------------------------------------------
# Resolving the tournament on both APIs
# ---------------------------------------------------------------------------

def resolve_kalshi_event(query, series):
    """
    Find the Kalshi event for a tournament in a given series.

    Accepts a full ticker (`KXPGATOUR-WYC26`), a bare suffix (`WYC26`), or a name
    ("wyndham"). Event codes are not derivable -- 2026 Wyndham is WYC26 and 2025 was
    WC25 -- so a name has to be resolved against the live list rather than constructed.

    Returns (best, ranked). Both, because a near-miss that the caller can show beats a
    confident wrong pick: grouping the wrong tournament produces a completely valid
    looking answer.
    """
    events = kalshi_odds.events_for(series)
    rows = [{
        "event_ticker": e.get("event_ticker"),
        "title": e.get("title"),
        "tournament": e.get("sub_title") or e.get("title"),
    } for e in events if e.get("event_ticker")]

    q = (query or "").strip()
    if not q:
        return None, rows

    upper = q.upper()
    for r in rows:
        if r["event_ticker"] == upper or r["event_ticker"].endswith(f"-{upper}"):
            return r, [dict(r, score=1.0)]

    scored = []
    for r in rows:
        score = espn_leaderboard.score_name(q, r["tournament"] or "")
        if score >= 1.0:
            return dict(r, score=1.0), [dict(r, score=1.0)]
        if score > 0:
            scored.append(dict(r, score=score))
    scored.sort(key=lambda r: -r["score"])
    return (scored[0] if scored else None), scored


def resolve_espn_event(query, season, league="pga"):
    try:
        return espn_leaderboard.resolve_event(query, season, league)
    except Exception as exc:                       # noqa: BLE001 -- reported, not swallowed
        print(f"!! ESPN event lookup failed: {exc}", file=sys.stderr)
        return None, []


# ---------------------------------------------------------------------------
# The ESPN join
# ---------------------------------------------------------------------------

def read_espn_field(args, espn_event):
    """
    This week's ESPN field, and the build mode it decides.
    -> (mode, meta, players, error).

    "live" the moment ESPN lists a single competitor, "groups" until then. Read off the
    payload rather than off the event's `state` string, because the field is the thing
    that actually gates the work: a build can only join names against competitors that
    exist, whatever ESPN is currently calling the event.

    An ESPN that cannot be reached at all yields "groups" too, and that is the right
    outcome for a FIRST build: a groups build is complete and correct on its own terms,
    where a live build missing its leaderboard would be a scoreboard with every golfer
    unresolved, which reads as 150 withdrawals. It is emphatically NOT the right
    outcome for a rebuild of a competition that already had a field -- see
    refuse_downgrade, which is where that gets caught.

    Returns the error as well, because "ESPN published nothing" and "ESPN did not
    answer" are the same empty list and completely different facts.
    """
    meta, players, error = None, [], None
    if espn_event:
        try:
            payload = espn_leaderboard.fetch_leaderboard(espn_event["event_id"], args.espn_league)
            meta, players = espn_leaderboard.parse_leaderboard(payload)
        except Exception as exc:                   # noqa: BLE001 -- reported, not fatal here
            error = str(exc)
            print(f"!! could not read the ESPN leaderboard: {exc}", file=sys.stderr)
    else:
        error = "no ESPN event was resolved for this tournament"

    state = (meta or {}).get("state") or (espn_event or {}).get("state") or "unknown"
    if players:
        print(f"ESPN:    {len(players)} competitors in the field [{state}] -- building with "
              "live scoring")
    elif error:
        print("ESPN:    unreadable -- building the groups only. Nothing is scored.")
    else:
        print(f"ESPN:    no field published [{state}] -- normal before the first tee time. "
              "Building the groups; re-run --from-result once play starts to add scoring.")
    return ("live" if players else "groups"), meta, players, error


def refuse_downgrade(espn, prior_mode, source_file):
    """
    Refuse to rebuild a live competition into a groups sheet.

    Once ESPN has published a field it does not unpublish it, so a rebuild that finds
    no field where the last one found 150 competitors has not learned something -- it
    has failed to ask. Writing the file anyway would null `live` and every golfer's
    athlete id, and the page built from it would show a finished tournament as "not
    started yet": a total loss of the scoreboard, presented as a normal build.

    A rebuild is cheap and repeatable, so the right move is to stop. The previous
    result file is untouched and still correct.
    """
    if prior_mode != "live" or espn["mode"] == "live":
        return
    raise SystemExit(
        f"{source_file} was built with a published ESPN field, and this run found none"
        + (f" ({espn['error']})" if espn["error"] else " (ESPN returned zero competitors)")
        + ".\nESPN does not withdraw a field once it has posted it, so this is a failed "
        "read rather than a change in the world. Rebuilding on it would produce a page "
        "with no scoring at all and no sign that anything went wrong, so nothing has been "
        "written -- the file you passed is untouched and still correct. Try again, or pass "
        "--espn-event <id> if the event lookup is what is failing.")


def join_field(names, players, aliases, decisions):
    """
    Join the Kalshi field onto this week's ESPN field, and say what is left over.

    Everything printed here is either a number somebody should sanity-check or a name
    somebody should act on. In particular the two piles at the bottom are kept apart on
    purpose: `absent` has been looked at and `unresolved` has not, and only the second
    is a reason to do anything.
    """
    matches, report = espn_leaderboard.match_field(
        names, players, aliases=aliases, decisions=decisions)

    print(f"Match:   {report['matched']}/{report['requested']} joined to ESPN "
          f"({report['matched_exact']} exact, {report['matched_alias']} alias, "
          f"{report['matched_decision']} reviewed)")
    for problem in report["problems"]:
        print(f"!! {problem}", file=sys.stderr)
    if report["ambiguous_names"]:
        print(f"  {len(report['ambiguous_names'])} name(s) are shared by two athletes in this "
              "field and were refused rather than guessed: "
              + ", ".join(report["ambiguous_names"]))
    if report["absent"]:
        print(f"  reviewed and confirmed not in the field ({len(report['absent'])}): "
              + ", ".join(report["absent"][:8])
              + (" ..." if len(report["absent"]) > 8 else "")
              + ". They score nothing, which is correct.")
    if report["unresolved"]:
        print(f"  NEEDS REVIEW ({len(report['unresolved'])}): "
              + ", ".join(report["unresolved"][:8])
              + (" ..." if len(report["unresolved"]) > 8 else ""))
    return matches, report


def espn_stage(args, espn_event, field, weight_by_name, team_name_of, tournament_name,
               aliases, recorded_decisions=None):
    """
    The whole ESPN half of a build: read the field, let it decide the mode, join what
    there is to join, and assemble the review of whatever is left over.

    Shared verbatim by build() and rebuild(), because the ESPN side of a competition
    depends on the tournament and on the clock and on nothing either of them decided.
    That is the entire reason a rebuild exists: same draw, later clock.

    Reviewed decisions come from two places and both are wanted. The result file
    carries the ones already applied, so a rebuild does not re-ask a question somebody
    answered last time; the review file carries whatever was filled in since, and wins
    on a conflict because it is the newer statement. Decisions for names that are not
    in this Kalshi field are dropped -- they are left over from a different draw and
    binding them would be binding somebody else's competition.
    """
    mode, meta, players, error = read_espn_field(args, espn_event)
    event_id = (espn_event or {}).get("event_id")
    if mode == "groups":
        return {"mode": mode, "meta": meta, "players": players, "matches": {},
                "report": None, "decisions": {}, "review": None, "error": error}

    path = match_review.review_path(args.output, args.match_review)
    from_file, notes = match_review.load(path, event_id)
    for note in notes:
        print(f"!! {note}", file=sys.stderr)

    names = [g["golfer_name"] for g in field]
    known = set(names)
    offered = {**(recorded_decisions or {}), **from_file}
    decisions = {name: d for name, d in offered.items() if name in known}
    if from_file:
        print(f"Review:  {len(from_file)} decision(s) read from {path}")
    # Normally none: a competition's Kalshi field is the same field on every rebuild of
    # it. When it is not -- Kalshi respelled a golfer, or this file belongs to a
    # different draw -- the decision is correctly dropped, and saying so out loud is the
    # difference between "somebody settled that name and it stopped applying" and a
    # golfer who quietly went back to scoring nothing.
    if len(decisions) < len(offered):
        stale = sorted(set(offered) - known)
        print(f"  {len(stale)} reviewed decision(s) are for golfers who are not in this "
              f"Kalshi field and were not applied: {', '.join(stale[:8])}"
              + (" ..." if len(stale) > 8 else ""))

    matches, report = join_field(names, players, aliases, decisions)
    return {
        "mode": mode, "meta": meta, "players": players, "matches": matches,
        "report": report, "decisions": decisions, "error": error,
        "review": {
            "path": path,
            "tournament": tournament_name,
            "espn": {
                "event_id": event_id,
                "league": args.espn_league,
                "field_size": len(players),
                "leaderboard_endpoint": espn_leaderboard.leaderboard_url(
                    event_id, args.espn_league),
            },
            "golfers": [{"name": n, "grouping_weight": weight_by_name.get(n),
                         "team": team_name_of.get(n)} for n in names],
            "matches": matches, "report": report, "players": players,
            "decisions": decisions,
        },
    }


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def pull_odds(event_ticker, price):
    """
    Fetch the field with its diagnostics intact.

    kalshi_odds.fetch_golfers() cleans the bid/ask/spread off, and the result file
    wants them: "Cameron Young was 0.0900 ask against a 0.0810 bid" is the auditable
    version of "Cameron Young was 0.09".
    """
    markets = kalshi_odds.markets_for(event_ticker)
    if not markets:
        raise SystemExit(f"{event_ticker} returned zero markets. Confirm the request "
                         "succeeded before concluding the tournament is not posted.")
    golfers = kalshi_odds.to_golfers(markets, price=price)
    if not golfers:
        raise SystemExit(
            f"{event_ticker} returned {len(markets)} markets but none are active. "
            "Settled markets quote bid=0.0000/ask=1.0000 and are filtered out. "
            "Re-pull during tournament week."
        )
    tick = {m.get("price_level_structure") for m in markets if m.get("price_level_structure")}
    return markets, grouper_cli.sort_field(golfers), sorted(tick)


def resolve_exclusions(golfers, n_groups, named, auto):
    """
    The exclusion set, and a reason for each name.

    A result file that says "excluded: Scottie Scheffler" and nothing else is a
    decision with no argument attached. Everything here records which rule fired.
    """
    excluded = []
    named = list(named or [])
    known = {g["golfer_name"] for g in golfers}

    for name in named:
        if name not in known:
            print(f"!! warning: excluded golfer {name!r} is not in this field, so it excluded nothing")
            continue
        excluded.append({"golfer_name": name, "reason": "named"})

    if auto:
        remaining = [g for g in golfers if g["golfer_name"] not in {e["golfer_name"] for e in excluded}]
        for name in grouper_cli.auto_exclusions(remaining, n_groups):
            excluded.append({"golfer_name": name, "reason": "over_fair_share"})

    return excluded


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def prior_build_mode(result):
    """
    The mode a previous result file was built in.

    Files written before 2.0 carry no `build_mode` and have to be inferred, which is
    exact rather than a guess: they recorded the ESPN field size at build time, and
    having a field is the whole of what the mode means.
    """
    if result.get("build_mode"):
        return result["build_mode"]
    espn = (result.get("sources") or {}).get("espn") or {}
    return "live" if espn.get("field_size_at_build") else "groups"


def build(args, league=None, rebuilt_from=None, prior_mode=None, recorded_decisions=None):
    started = time.time()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    league = league or league_mod.load_league(args.league)
    teams = league["teams"]
    n_groups = len(teams)
    print(f"League: {league['league_name']} -- {n_groups} teams")

    odds_type = args.odds
    if odds_type in ODDS_TYPES:
        series = ODDS_TYPES[odds_type]["series"]
        market_label = ODDS_TYPES[odds_type]["label"]
        exclusive = ODDS_TYPES[odds_type]["exclusive"]
    else:
        # A raw series ticker. Supported because Kalshi adds series faster than this
        # table gets updated, and the shape is identical whatever the prefix.
        series, market_label, exclusive = odds_type, odds_type, False
        odds_type = "custom"

    # -- tournament, on both APIs -------------------------------------------
    if args.kalshi_event:
        kalshi_event = {"event_ticker": args.kalshi_event, "tournament": args.tournament, "title": None}
    else:
        kalshi_event, ranked = resolve_kalshi_event(args.tournament, series)
        if not kalshi_event:
            raise SystemExit(
                f"no {series} event on Kalshi matches {args.tournament!r}. "
                "Run `python kalshi_odds.py --list-events` to see what is posted; winner "
                "markets appear Sunday ~23:00Z of tournament week."
            )
        if len(ranked) > 1 and ranked[0].get("score", 0) < 1.0:
            print(f"note: matched {kalshi_event['event_ticker']} ({kalshi_event['tournament']}); "
                  f"runners-up were {', '.join(r['event_ticker'] for r in ranked[1:4])}")
    event_ticker = kalshi_event["event_ticker"]
    tournament_name = kalshi_event.get("tournament") or args.tournament
    print(f"Kalshi:  {event_ticker}  {tournament_name}  [{market_label}]")

    season = args.season or datetime.now(timezone.utc).year
    if args.espn_event:
        espn_event = {"event_id": str(args.espn_event), "name": tournament_name}
    else:
        espn_event, espn_ranked = resolve_espn_event(tournament_name, season, args.espn_league)
        if not espn_event:
            print(f"!! no ESPN {args.espn_league} event in {season} matches {tournament_name!r}. "
                  "The scoreboard will fall back to ESPN's current event, which may be the "
                  "wrong tournament. Pass --espn-event <id> to pin it.", file=sys.stderr)
        elif espn_ranked and espn_ranked[0].get("score", 1.0) < 1.0:
            print(f"note: ESPN matched {espn_event['event_id']} ({espn_event['name']}) on a "
                  f"partial name match. Pass --espn-event to override.")
    if espn_event:
        # A pinned event was never looked up, so it has no state to report. The join
        # below prints the state it actually finds either way.
        state = espn_event.get("state")
        print(f"ESPN:    {espn_event['event_id']}  {espn_event.get('name')}"
              + (f"  [{state}]" if state else ""))

    # -- odds ----------------------------------------------------------------
    markets, field, tick_structures = pull_odds(event_ticker, args.price)
    raw_sum = sum(g["odds"] for g in field)
    liquidity = kalshi_odds.liquidity_report(field)
    print(f"Odds:    {len(field)} golfers, {args.price} book sums to {raw_sum:.4f}, "
          f"{liquidity['two_sided_quotes']} two-sided")
    if not exclusive and raw_sum > 1.6:
        print(f"note: a {market_label} book is not a probability distribution -- its outcomes "
              "are not mutually exclusive, so the de-vig gives share-of-N-slots. Groups still "
              "balance; read the numbers as weights.")

    if len(field) < n_groups:
        raise SystemExit(f"only {len(field)} golfers for {n_groups} teams")

    # -- exclusions ----------------------------------------------------------
    over = grouper_cli.golfers_over_threshold(field, n_groups)
    if over:
        print(f"Above the 1/{n_groups} fair share ({1/n_groups:.4f}): "
              + ", ".join(f"{g['golfer_name']} {g['odds']:.4f}" for g in over))
    excluded = resolve_exclusions(field, n_groups, args.exclude, args.auto_exclude)
    excluded_names = {e["golfer_name"] for e in excluded}
    if excluded:
        print("Excluded: " + ", ".join(f"{e['golfer_name']} ({e['reason']})" for e in excluded))

    devigged = {g["golfer_name"]: g["odds"] for g in grouper_cli.normalize_probabilities(field)}
    if excluded_names:
        weighted = grouper_cli.odds_to_conditional(field, excluded_names)
    else:
        weighted = grouper_cli.normalize_probabilities(field)

    if len(weighted) < n_groups:
        raise SystemExit(f"only {len(weighted)} golfers left after excluding {len(excluded)}, "
                         f"which cannot fill {n_groups} groups")

    # -- the partition -------------------------------------------------------
    print(f"Grouping {len(weighted)} golfers into {n_groups} groups...")
    groups, report = groupers.partition(weighted, n_groups, time_limit=args.time_limit)
    print(grouper_cli.describe_partition(report))
    for line in grouper_cli.describe_dominant(report, n_groups):
        print(line)

    # -- the deal ------------------------------------------------------------
    # Which team gets which group is random by design; the seed makes a whole run
    # reproducible, and it goes in the result file so a run can be re-created.
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    order = list(range(n_groups))
    random.Random(seed).shuffle(order)
    team_groups = {teams[i]["team_id"]: groups[order[i]] for i in range(n_groups)}

    # -- ESPN ----------------------------------------------------------------
    aliases = load_aliases(args.alias_file)
    weight_by_name = {g["golfer_name"]: g["odds"] for g in weighted}
    team_name_of = {g["golfer_name"]: team["team_name"]
                    for team in teams for g in team_groups[team["team_id"]]}
    # A --regroup deals new groups out of the same tournament, so the reviewed name
    # decisions are still about the same people and come with it. Dropping them would
    # make a regroup unresolve golfers somebody had already settled, for no reason
    # except that the partitioner ran again.
    espn = espn_stage(args, espn_event, field, weight_by_name, team_name_of,
                      tournament_name, aliases, recorded_decisions=recorded_decisions)
    refuse_downgrade(espn, prior_mode, args.from_result)

    # -- assemble ------------------------------------------------------------
    result = assemble(
        now=now, args=args, league=league, teams=teams, team_groups=team_groups,
        odds_type=odds_type, series=series, market_label=market_label, exclusive=exclusive,
        event_ticker=event_ticker, tournament_name=tournament_name, season=season,
        espn_event=espn_event, espn=espn,
        field=field, devigged=devigged, weighted=weighted, excluded=excluded,
        liquidity=liquidity, raw_sum=raw_sum, tick_structures=tick_structures,
        auto_exclude=args.auto_exclude,
        report=report, groups=groups, order=order, seed=seed,
        rebuilt_from=rebuilt_from,
    )

    finish(result, args, espn, aliases, started)
    return result


def finish(result, args, espn, aliases, started):
    """
    Inline the logos, write the files, learn what the run learned, and show the groups.

    Shared by build() and rebuild() because it is the same ending either way: the only
    thing that differs between the two is how the numbers above it were arrived at.
    """
    # Logos are resolved against the league file's own directory: a logo path in a
    # league file is relative to that file, not to wherever the build was run from. A
    # rebuild has no league file -- its logos are already data: URIs and pass straight
    # through -- so anything still relative there is resolved against the result file.
    base = args.league or args.from_result or "."
    league_dir = os.path.dirname(os.path.abspath(base))
    for team in result["teams"]:
        team["team_logo"] = inline_logo(team.get("team_logo"), league_dir)
    # The masthead art settles first -- which of the command line, the league file and
    # the shipped default wins -- and only then gets inlined, so there is one path
    # through the base64 for all three.
    resolve_league_art(result, args)
    for field in ("crest", "banner"):
        result["league"][field] = inline_logo(result["league"].get(field), league_dir, field)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResult -> {args.output}  ({os.path.getsize(args.output) // 1024} KB)")

    if espn["review"]:
        written = match_review.write(espn["review"]["path"],
                                     **{k: v for k, v in espn["review"].items() if k != "path"})
        pending = espn["report"]["unresolved"]
        if written and pending:
            print(f"Review -> {written}  ({len(pending)} golfer(s) need a decision)")
            print("  Each one is listed with the ESPN athletes nobody claimed and a ranked "
                  "suggestion or two. Fill in `decisions` and rebuild with --from-result; "
                  "until then those golfers score nothing.")
        elif written:
            print(f"Review -> {written}  (nothing open; it records the decisions applied)")

    learned = match_review.learned_aliases(espn["decisions"], espn["matches"], aliases)
    if learned:
        print(f"{len(learned)} reviewed decision(s) are worth keeping as name aliases: "
              + ", ".join(f"{k} -> {v}" for k, v in list(learned.items())[:5]))
        if args.update_aliases:
            save_aliases(args.alias_file, {**aliases, **learned})
            print(f"  written to {args.alias_file} (the one repo file a build touches). "
                  "Next tournament resolves them with nobody looking.")
        else:
            print(f"  not saved. Re-run with --update-aliases to pin them into {args.alias_file}.")

    print_groups(result)
    print(f"\nBuilt in {time.time() - started:.1f}s")


# ---------------------------------------------------------------------------
# Rebuilding from a result file
# ---------------------------------------------------------------------------

def load_result(path):
    """
    Read a result file, refusing anything that is not one.

    An exported `.zip` is accepted directly, because that is usually the copy a user
    still has: the page is for reading and the JSON inside the zip is for re-running,
    and making somebody unzip it first to get at the second is a step with no thought
    in it. bundle_frontend.py writes it as `result.json`.
    """
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                if "result.json" not in z.namelist():
                    raise SystemExit(f"{path} is a zip with no result.json in it. An export from "
                                     "bundle_frontend.py has one; unzip it and check.")
                result = json.loads(z.read("result.json").decode("utf-8"))
        else:
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found. --from-result takes a result JSON written by "
                         "this tool (build/result.json), or an exported .zip holding one.")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")

    # Every block a rebuild goes on to dereference. Checking six of them and then dying
    # on a KeyError in the seventh is the same damaged file reported twice as badly.
    missing = [k for k in ("schema_version", "generated_at", "generator", "league", "teams",
                           "golfers", "tournament", "sources", "odds_snapshot", "grouping")
               if k not in result]
    if missing:
        raise SystemExit(f"{path} is missing {', '.join(missing)}, so it is not a result file "
                         "from this tool. Pass the result.json, not the league file or the "
                         "Kalshi capture.")

    # A 1.x file rebuilds into a 2.0 one without any special handling, and it is worth
    # saying why rather than adding a gate that would only ever fire on a file nobody
    # has: a rebuild takes the draw, the odds and the tournament out of the file and
    # derives the whole ESPN half fresh from this week's leaderboard. Nothing 1.x
    # recorded about golfer identities is carried forward, which is exactly what should
    # happen -- 1.x could record an athlete id recovered from an earlier tournament and
    # never checked against this week's field.
    return result


def league_from_result(result):
    """
    The league, back out of a result file.

    A rebuild does not need the original league file and should not depend on it: the
    result carries every team with the id it was dealt under, and those ids are what the
    groups are keyed on. Re-reading the league file instead would silently mint new ids
    the moment somebody had renamed a team, and hand every player a different group.
    """
    league = result["league"]
    derived = ("group_index", "golfer_ids", "golfer_names", "total_odds", "golfer_count")
    return {
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        "league_slug": league.get("league_slug") or league_mod.slugify(league["league_name"]),
        "source_file": league.get("source_file"),
        # Carried forward like the logos are: by the time a result file exists these
        # are data: URIs, and a rebuild that dropped them would quietly un-brand a
        # page somebody has already seen. `.get` because a file written before
        # branding existed has no such keys and is still a perfectly good rebuild.
        **{f: league.get(f) for f in league_mod.BRANDING_FIELDS},
        "teams": [{k: v for k, v in t.items() if k not in derived} for t in result["teams"]],
    }


def rebuild(args, result):
    """
    A new build of an existing competition: same teams, same groups, same odds at
    creation -- everything else brought up to date.

    What moves between two runs of the same competition is what the world did, not what
    the pool decided. So the draw is carried forward verbatim and the run re-reads the
    parts that have a "now": the ESPN join (a Wednesday build has no field to join
    against and a Thursday one does) and the tournament's state. Kalshi is not read at
    all. The odds were read once, when the groups were drawn, and a rebuild that went
    back for a second reading would be putting a price nobody was dealt on next to the
    one everybody was.

    Re-partitioning is deliberately NOT what this does. Rebuilding a live competition
    and quietly dealing everyone new golfers is the single most destructive thing this
    tool could do; --regroup asks for it explicitly and goes through build().
    """
    started = time.time()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    league = league_mod.load_league(args.league) if args.league else league_from_result(result)
    recorded = {t["team_id"]: t for t in result["teams"]}
    if len(league["teams"]) != len(recorded) or any(
            t["team_id"] not in recorded for t in league["teams"]):
        raise SystemExit(
            f"{args.league} does not describe the league in {args.from_result}: the team ids "
            "do not line up, so there is no way to say which team holds which group. Team ids "
            "are derived from team names, so a renamed team is a new team. Rebuild without "
            "--league to use the teams recorded in the result file.")
    teams = league["teams"]
    print(f"League: {league['league_name']} -- {len(teams)} teams "
          f"(from {args.from_result}, built {result['generated_at']})")

    kalshi = result["sources"]["kalshi"]
    espn_source = result["sources"].get("espn") or {}
    tournament = result["tournament"]
    print(f"Kalshi:  {kalshi['event_ticker']}  {tournament['name']}  [{kalshi['market_label']}]")

    # -- the draw, carried forward -------------------------------------------
    field = [{"golfer_name": g["name"], "odds": g["odds"]["raw"], "golfer_id": g.get("golfer_id"),
              "_bid": (g.get("kalshi") or {}).get("bid"),
              "_ask": (g.get("kalshi") or {}).get("ask"),
              "_spread": (g.get("kalshi") or {}).get("spread"),
              "_ticker": (g.get("kalshi") or {}).get("ticker")}
             for g in result["golfers"]]
    devigged = {g["name"]: g["odds"]["devigged"] for g in result["golfers"]}
    weighted = [{"golfer_name": g["name"], "odds": g["odds"]["grouping_weight"],
                 "golfer_id": g.get("golfer_id")}
                for g in result["golfers"] if g["odds"].get("grouping_weight") is not None]
    weight_of = {g["golfer_name"]: g["odds"] for g in weighted}
    excluded = [{"golfer_name": e["golfer_name"], "reason": e["reason"]}
                for e in result["odds_snapshot"]["excluded"]]

    team_groups, order, orphans = {}, [], []
    for team in teams:
        row = recorded[team["team_id"]]
        orphans += [n for n in row["golfer_names"] if n not in weight_of]
        team_groups[team["team_id"]] = [
            {"golfer_name": name, "golfer_id": gid, "odds": weight_of.get(name, 0.0)}
            for name, gid in zip(row["golfer_names"], row["golfer_ids"])]
        order.append(row["group_index"])
    if orphans:
        # Silently weighting them zero would rebuild a file whose team totals no longer
        # add up to 1.0 and whose draw no longer looks even -- on evidence that is
        # simply missing rather than wrong.
        raise SystemExit(
            f"{args.from_result} puts {len(orphans)} golfer(s) in a team who carry no "
            f"grouping weight in golfers[]: {', '.join(sorted(set(orphans))[:5])}. The file has "
            "been edited or truncated; rebuilding from it would report totals that are not "
            "the totals anybody was dealt.")

    print(f"Draw:    {len(weighted)} golfers in {len(teams)} groups, "
          f"{len(excluded)} excluded, carried forward unchanged")

    season = args.season or tournament["season"]
    espn_event = None
    if args.espn_event:
        espn_event = {"event_id": str(args.espn_event), "name": tournament["name"]}
    elif espn_source.get("event_id"):
        espn_event = {"event_id": str(espn_source["event_id"]), "name": tournament["name"],
                      "start": tournament.get("start")}
    else:
        espn_event, _ = resolve_espn_event(tournament["name"], season, args.espn_league)
        if not espn_event:
            print(f"!! still no ESPN {args.espn_league} event in {season} matching "
                  f"{tournament['name']!r}. Pass --espn-event <id>.", file=sys.stderr)

    # Aliases the file already knows about come first, so a rebuild from an exported
    # zip on a machine with no alias file still resolves the names the first build did.
    # The repo's own file wins on a conflict: it is the one somebody maintains.
    aliases = {**(espn_source.get("aliases_applied") or {}), **load_aliases(args.alias_file)}
    weight_by_name = {g["golfer_name"]: g["odds"] for g in weighted}
    team_name_of = {g["golfer_name"]: team["team_name"]
                    for team in teams for g in team_groups[team["team_id"]]}
    espn = espn_stage(args, espn_event, field, weight_by_name, team_name_of,
                      tournament["name"], aliases,
                      recorded_decisions=espn_source.get("match_decisions"))
    refuse_downgrade(espn, prior_build_mode(result), args.from_result)

    # -- assemble ------------------------------------------------------------
    rebuilt_from = {
        "source_file": args.from_result,
        "source_generated_at": result["generated_at"],
        "source_schema_version": result.get("schema_version"),
        "mode": "refresh",
        "rebuild_count": ((result.get("rebuilt_from") or {}).get("rebuild_count") or 0) + 1,
        "first_built_at": ((result.get("rebuilt_from") or {}).get("first_built_at")
                           or result["generated_at"]),
    }
    out = assemble(
        now=now, captured_at=result["odds_snapshot"]["captured_at"], args=args, league=league,
        teams=teams, team_groups=team_groups,
        odds_type=kalshi["odds_type"], series=kalshi["series_ticker"],
        market_label=kalshi["market_label"], exclusive=kalshi["mutually_exclusive_outcomes"],
        event_ticker=kalshi["event_ticker"], tournament_name=tournament["name"], season=season,
        espn_event=espn_event, espn=espn,
        field=field, devigged=devigged, weighted=weighted, excluded=excluded,
        liquidity=result["odds_snapshot"]["liquidity"],
        raw_sum=result["odds_snapshot"]["raw_book_sum"],
        price_mode=kalshi["price_mode"],
        auto_exclude=result["odds_snapshot"].get("auto_exclude", args.auto_exclude),
        tick_structures=kalshi.get("price_level_structure") or [],
        report=result["grouping"], order=order,
        seed=result["generator"].get("seed"),
        tournament_prior=tournament,
        rebuilt_from=rebuilt_from,
    )

    finish(out, args, espn, aliases, started)
    return out


def assemble(**k):
    """
    The result file.

    Grouped by what a reader is asking. `league` / `teams` / `golfers` answer "who has
    whom". `odds_snapshot` and `grouping` answer "why". `sources` answers "where did
    this come from and can I check it". `live` answers "what does the page fetch while
    it runs" -- ESPN, or in a groups build nothing at all, and then it is null rather
    than an empty object, because there is no polling to configure.

    `build_mode` is the key to reading the rest; see the module docstring.
    """
    field, devigged, weighted = k["field"], k["devigged"], k["weighted"]
    espn = k["espn"]
    is_live = espn["mode"] == "live"
    match_report = espn["report"] or {}
    absent = set(match_report.get("absent") or [])
    weight_by_name = {g["golfer_name"]: g["odds"] for g in weighted}
    excluded_names = {e["golfer_name"] for e in k["excluded"]}
    captured_at = k.get("captured_at") or k["now"]
    # The price mode the CARRIED snapshot was captured at, which is not this run's
    # --price. A rebuild passes the mode recorded in the file, so a later run under a
    # different --price cannot relabel Wednesday's ask prices as mids.
    price_mode = k.get("price_mode") or k["args"].price
    prior = k.get("tournament_prior") or {}
    prior_course = prior.get("course") or {}
    team_of = {}
    for team_id, golfers in k["team_groups"].items():
        for g in golfers:
            team_of[g["golfer_name"]] = team_id

    golfers_out = []
    for g in field:
        name = g["golfer_name"]
        hit = (espn["matches"] or {}).get(name)
        player = hit["player"] if hit else None
        golfers_out.append({
            "golfer_id": g.get("golfer_id"),
            "name": name,
            "team_id": team_of.get(name),
            "excluded": name in excluded_names,
            "kalshi": {
                "ticker": g.get("_ticker"),
                "bid": g.get("_bid"),
                "ask": g.get("_ask"),
                "spread": g.get("_spread"),
            },
            "odds": {
                "raw": g["odds"],
                "devigged": round(devigged.get(name, 0.0), WEIGHT_PRECISION),
                # Three numbers, all of them read at the same instant, and there will
                # never be a fourth: the groups were drawn on these and stay drawn on
                # them for as long as the file exists.
                "grouping_weight": (round(weight_by_name[name], WEIGHT_PRECISION)
                                    if name in weight_by_name else None),
            },
            # Null in a groups build, and null rather than a shell of nulls: before the
            # first tee time this golfer has no ESPN presence to describe, and a block
            # of empty fields reads as a join that was attempted and failed.
            "espn": ({
                "athlete_id": player["athlete_id"] if player else None,
                "display_name": player["name"] if player else None,
                "headshot": player["headshot"] if player else None,
                "country": player["country"] if player else None,
                # How this golfer was settled. The last two both score nothing and are
                # emphatically not the same claim: "absent" means somebody looked at
                # this week's field and confirmed the golfer is not in it, "unresolved"
                # means nobody has looked. See match_review.py.
                "match": hit["match"] if hit else ("absent" if name in absent
                                                   else "unresolved"),
                # True if they are in this week's field, False if that was checked and
                # they are not, None if it has not been checked. Three states because
                # there are three, and folding the third into False would report a
                # withdrawal this build has no evidence for.
                "in_field": True if hit else (False if name in absent else None),
            } if is_live else None),
        })
    golfers_out.sort(key=lambda g: -g["odds"]["raw"])

    teams_out = []
    for i, team in enumerate(k["teams"]):
        members = k["team_groups"][team["team_id"]]
        # Summed from the SAME rounded weights the golfers carry, so a reader adding up
        # a team's golfers gets the team's total rather than a number 1e-8 away from it.
        weights = [round(weight_by_name[g["golfer_name"]], WEIGHT_PRECISION) for g in members]
        teams_out.append({
            **{key: value for key, value in team.items()},
            "group_index": k["order"][i],
            "golfer_ids": [g.get("golfer_id") for g in members],
            "golfer_names": [g["golfer_name"] for g in members],
            "total_odds": round(sum(weights), WEIGHT_PRECISION),
            "golfer_count": len(members),
        })

    markets_endpoint = (f"{kalshi_odds.BASE}/markets?"
                        + urllib.parse.urlencode({"event_ticker": k["event_ticker"], "limit": 500}))
    espn_event = k["espn_event"] or {}
    leaderboard = espn_leaderboard.leaderboard_url(espn_event.get("event_id"), k["args"].espn_league)

    return {
        "schema_version": SCHEMA_VERSION,
        # "groups" or "live", and which one is a fact about the ESPN leaderboard at
        # build time rather than a setting. Read this first: it says whether the rest
        # of the file describes a draw or a scoreboard.
        "build_mode": espn["mode"],
        "competition_id": str(uuid.uuid5(
            league_mod.NAMESPACE,
            f"competition:{k['league']['league_id']}:{k['event_ticker']}:{k['odds_type']}")),
        "generated_at": k["now"],
        # The inputs that shaped this run rather than facts about the world, which is
        # why the poll interval is here and not only under `live`: `live` is null in a
        # groups build, and a setting somebody typed should survive the rebuild that
        # turns that groups sheet into a scoreboard. `live` mirrors it when there is a
        # page to configure, so the page still reads it in one line.
        "generator": {
            "tool": "golf-odds-grouper/build_competition.py",
            "git_commit": _git_commit(),
            "seed": k["seed"],
            "poll_interval_seconds": k["args"].poll_interval,
        },
        # Null on a first build. On a rebuild, what it was rebuilt from and how -- so a
        # file that carries Wednesday's odds and Sunday's leaderboard says so itself.
        "rebuilt_from": k.get("rebuilt_from"),

        "league": {
            "league_id": k["league"]["league_id"],
            "league_name": k["league"]["league_name"],
            "league_slug": k["league"]["league_slug"],
            "source_file": k["league"]["source_file"],
            "team_count": len(k["teams"]),
            # The masthead. Paths here, and for the two images possibly `false` or
            # nothing at all; finish() settles which of the command line, this file and
            # the shipped default wins, then turns whatever won into a data: URI in the
            # same pass that does it for the team logos.
            "crest": k["league"].get("crest"),
            "banner": k["league"].get("banner"),
            "tagline": k["league"].get("tagline"),
        },
        "teams": teams_out,
        "golfers": golfers_out,

        # Dates and course are static facts about the tournament, so a rebuild that
        # could not reach ESPN keeps the ones it was given rather than nulling them --
        # the scoreboard's "first round <date>" line is the pre-tournament state this
        # whole file exists to serve. `state_at_build` is NOT static and gets no such
        # fallback: if this run could not read the state, it does not know it.
        "tournament": {
            "name": k["tournament_name"],
            "season": int(k["season"]),
            "start": (espn_event.get("start") or (espn["meta"] or {}).get("start")
                      or prior.get("start")),
            "end": (espn_event.get("end") or (espn["meta"] or {}).get("end")
                    or prior.get("end")),
            "state_at_build": espn_event.get("state") or (espn["meta"] or {}).get("state"),
            "course": {
                "name": (espn["meta"] or {}).get("course") or prior_course.get("name"),
                "par": (espn["meta"] or {}).get("par") or prior_course.get("par"),
            },
        },

        "sources": {
            "kalshi": {
                "base_url": kalshi_odds.BASE,
                "series_ticker": k["series"],
                "event_ticker": k["event_ticker"],
                "markets_endpoint": markets_endpoint,
                "odds_type": k["odds_type"],
                "market_label": k["market_label"],
                "mutually_exclusive_outcomes": k["exclusive"],
                "price_mode": price_mode,
                "price_level_structure": k["tick_structures"],
                "browser_reachable": False,
                "browser_note": (
                    "Kalshi allowlists request origins: a GET carrying "
                    "Origin: https://kalshi.com returns 200, every other origin returns 403 "
                    "with no CORS headers (localhost, GitHub Pages and file:// all measured "
                    "2026-08-03). The scoreboard therefore never fetches odds: the prices "
                    "the groups were drawn on are baked in, and they are the only prices "
                    "this competition has."
                ),
            },
            "espn": {
                "league": k["args"].espn_league,
                "event_id": espn_event.get("event_id"),
                "leaderboard_endpoint": leaderboard,
                "scoreboard_endpoint": espn_leaderboard.scoreboard_url(k["season"], k["args"].espn_league),
                "browser_reachable": True,
                "browser_note": "ESPN sends access-control-allow-origin: * -- fetch it directly.",
                # Zero in a groups build, and the reason it is a groups build.
                "field_size_at_build": len(espn["players"]),
                # Null in a groups build. There was no join, so there is no report on
                # one -- as opposed to a join that found nothing, which is a different
                # and much worse thing for a file to be unable to distinguish.
                "match_report": espn["report"],
                # The reviewed decisions this build applied, carried so the next rebuild
                # does not re-ask a question somebody already answered. Absences belong
                # to this tournament and live only here; name bindings are also worth
                # keeping globally, and --update-aliases is what puts them in the alias
                # file.
                "match_decisions": espn["decisions"] or {},
                # The aliases that actually fired, so a rebuild from an exported zip on
                # a machine with no alias file resolves the same names. Not the whole
                # alias file: that is repo state, not a fact about this competition.
                "aliases_applied": {
                    name: hit["player"]["name"]
                    for name, hit in (espn["matches"] or {}).items()
                    if hit["match"] == "alias"
                },
            },
        },

        "odds_snapshot": {
            "captured_at": captured_at,
            "price_mode": price_mode,
            "field_size": len(field),
            "raw_book_sum": round(k["raw_sum"], 6),
            "liquidity": k["liquidity"],
            "normalization": {
                "method": "divide by the observed book sum, then rescale over survivors",
                "basis": "probability" if k["exclusive"] else "share_of_n_slots",
                "note": (
                    "grouping_weight sums to 1.0 across every golfer that was grouped. "
                    "devigged is the same de-vig over the WHOLE field, before exclusions."
                ),
            },
            # Recorded because it is a decision this run made and the next one cannot
            # infer: a file with no over_fair_share exclusions either had nobody over
            # the line or had the rule switched off, and those rebuild differently.
            "auto_exclude": bool(k["auto_exclude"]),
            "excluded": [
                {**e,
                 "raw_odds": next((g["odds"] for g in field if g["golfer_name"] == e["golfer_name"]), None),
                 "devigged_odds": round(devigged.get(e["golfer_name"], 0.0), WEIGHT_PRECISION)}
                for e in k["excluded"]
            ],
            "fair_share_threshold": round(1 / len(k["teams"]), WEIGHT_PRECISION),
        },

        "grouping": {
            "method": "Karmarkar-Karp differencing + local search over moves and swaps",
            "n_groups": len(k["teams"]),
            "grouped_golfers": len(weighted),
            "delta": k["report"]["delta"],
            "delta_ticks": k["report"]["delta_ticks"],
            "floor_ticks": k["report"]["floor_ticks"],
            "optimal": k["report"]["optimal"],
            "exact_grid": k["report"]["exact_grid"],
            "tick_value": k["report"]["tick_value"],
            "field_ticks": k["report"]["field_ticks"],
            "fair_share_ticks": k["report"]["fair_share_ticks"],
            "dominant_golfers": k["report"]["dominant_golfers"],
            "group_sizes": k["report"]["group_sizes"],
            "summary": grouper_cli.describe_partition(k["report"]),
        },

        # What the page does while it is open. ESPN, on a timer, and nothing else --
        # Kalshi 403s every browser origin, so the odds are baked rather than fetched
        # and there is no second endpoint here to configure. See sources.kalshi.
        #
        # NULL in a groups build, and that is the whole instruction: fetch nothing, poll
        # nothing, rank nothing. There is no field to score against, so a page that
        # polled would be asking a question whose answer it could not use. It also means
        # a groups page needs no network at all -- it opens from disk on a plane.
        #
        # There is no name-matching block here in either mode. A live build has already
        # written an ESPN athlete id onto every golfer it resolved, and the page joins
        # on that id: exact, and incapable of quietly picking the wrong Smith.
        "live": ({
            "espn_leaderboard_url": leaderboard,
            "espn_event_id": espn_event.get("event_id"),
            "poll_interval_seconds": k["args"].poll_interval,
        } if is_live else None),

        "standings_rules": {
            "description": (
                "Rank each team by the best leaderboard position it holds; break ties on the "
                "next-best golfer, and so on. A team that runs out of golfers loses to one "
                "that has not."
            ),
            "golfer_rank_tiers": {
                "0": "still in the tournament -- rank on the displayed position number (T12 -> 12)",
                "1": "cut, withdrawn or disqualified -- no ESPN position, rank on sortOrder",
                "2": ("no ESPN athlete on this golfer: either confirmed absent from the field, "
                      "or not yet reviewed. Scores nothing either way"),
                "3": "padding: this team has no golfer this deep",
            },
            "comparison": "lexicographic over each team's (tier, value) pairs sorted ascending",
            "unresolved": "teams equal on the whole vector are reported tied, not separated",
        },
    }


# ---------------------------------------------------------------------------
# Aliases, logos, plumbing
# ---------------------------------------------------------------------------

def load_aliases(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("aliases", data) if isinstance(data, dict) else {}


def save_aliases(path, aliases):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Kalshi golfer name -> ESPN displayName. This file IS one of the three "
                     "tiers the build-time join uses, and the only one a person maintains. "
                     "Learned by --update-aliases from decisions settled in a match review; "
                     "safe to edit by hand."),
            "aliases": dict(sorted(aliases.items())),
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")


def inline_logo(value, base_dir, what="logo"):
    """
    Turn a local image path into a data: URI so the export is one portable file.

    Used for team logos and for the league's crest and banner; `what` names which, so
    the warnings say what is missing rather than calling a banner a logo.

    http(s) URLs and existing data: URIs pass through. A missing file is a warning
    rather than an error: a league is still perfectly playable without a crest.
    """
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    if not os.path.exists(path):
        print(f"!! {what} not found: {value} (looked in {path}) -- the page will render without it")
        return None
    size = os.path.getsize(path)
    if size > MAX_INLINE_LOGO_BYTES:
        print(f"!! {what} {value} is {size // 1024} KB, over the {MAX_INLINE_LOGO_BYTES // 1024} KB "
              "inline limit. Left as a path, which will not resolve in the exported page: "
              "shrink it, save it as a JPEG, or host it and use a URL instead.")
        return value
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


def resolve_league_art(result, args):
    """
    Settle the crest and the banner before they are inlined.

    The two images are the only per-league part of the masthead, and they can arrive
    three ways. In precedence order, highest first:

      1. `--no-crest` / `--no-banner` -- this competition has none, say no more.
      2. `--crest PATH` / `--banner PATH`, handed in beside the league file when the
         competition is created. Resolved against the working directory, like every
         other path typed on a command line, and made absolute here so the inliner
         does not later resolve it against the league file's directory instead.
      3. `crest` / `banner` in the league file, resolved against that file. `false`
         there is the standing form of (1): a league that never wants art.
      4. Nothing -- and then DEFAULT_CREST / DEFAULT_BANNER, so a page built by
         somebody who supplied no art still looks like the design rather than like a
         league whose art failed to load.

    Rule 4 fires only for a build that read a league file. A rebuild carries forward
    what the result file already recorded, null included: the first build settled this
    question, and a rebuild that re-answered it would put a crest on a page somebody
    had already sent round without one.
    """
    from_league_file, defaulted = bool(args.league), []
    for field, typed, cleared, fallback in (
            ("crest", args.crest, args.no_crest, DEFAULT_CREST),
            ("banner", args.banner, args.no_banner, DEFAULT_BANNER)):
        if cleared:
            value = None
        elif typed:
            value = os.path.abspath(typed)
        else:
            value = result["league"].get(field)
            if value is False:
                value = None
            elif value is None and from_league_file:
                value = fallback
                defaulted.append(field)
        result["league"][field] = value

    # Worth a line. The default is half a megabyte of PNG that lands in the result
    # file and in every page built from it, so somebody who did not know they were
    # getting it should find out here rather than from the size of the export.
    if defaulted:
        print(f"note: no {' or '.join(defaulted)} supplied, using the default. "
              f"{' '.join(f'--{f} PATH' for f in defaulted)} to supply your own, "
              f"{' '.join(f'--no-{f}' for f in defaulted)} for none.")
    return result


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                              # noqa: BLE001 -- provenance is nice, not required
        return None


def print_groups(result):
    teams = sorted(result["teams"], key=lambda t: -t["total_odds"])
    by_id = {g["golfer_id"] or g["name"]: g for g in result["golfers"]}
    print("\n------------- GROUPS -------------")
    for t in teams:
        print(f"{t['team_name']} ({t['player_name']}) -- {t['golfer_count']} golfers, "
              f"total {t['total_odds']:.4f}")
        names = []
        for gid, name in zip(t["golfer_ids"], t["golfer_names"]):
            g = by_id.get(gid or name)
            names.append(f"{name} {g['odds']['grouping_weight']:.4f}" if g else name)
        print("   " + ", ".join(names))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", help="path to the league JSON (not needed with --from-result)")
    ap.add_argument("--from-result", metavar="PATH",
                    help="rebuild an existing competition from its result JSON. Every input "
                         "is taken from the file -- league, tournament, market, price, "
                         "exclusions, seed -- and anything given on the command line wins. "
                         "The groups and the odds at creation are carried forward untouched; "
                         "the ESPN join is redone.")
    ap.add_argument("--regroup", action="store_true",
                    help="with --from-result: pull fresh odds and PARTITION AGAIN. Every team "
                         "gets a different group. Not what you want mid-tournament.")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --regroup to write over the result file it read")
    ap.add_argument("--crest", metavar="PATH",
                    help="the league's badge for the masthead, handed in beside the league "
                         "file. Beats a `crest` in that file. A local path is inlined into "
                         "the export; an https:// URL is left alone. Around 256 px square. "
                         "Unset, a build from a league file uses the shipped default.")
    ap.add_argument("--banner", metavar="PATH",
                    help="the wide image across the top of the page, same rules as --crest. "
                         "Around 720 px wide.")
    ap.add_argument("--no-crest", action="store_true",
                    help="build this competition with no crest, whatever the league file says")
    ap.add_argument("--no-banner", action="store_true", help="likewise, with no banner")
    ap.add_argument("--tournament", help="tournament name or Kalshi event code, e.g. 'Wyndham' or WYC26")
    ap.add_argument("--odds", default=DEFAULT_ODDS_TYPE,
                    help=f"{'/'.join(ODDS_TYPES)} or a raw Kalshi series ticker (default {DEFAULT_ODDS_TYPE})")
    ap.add_argument("--kalshi-event", help="pin the Kalshi event ticker and skip the name lookup")
    ap.add_argument("--espn-event", help="pin the ESPN event id and skip the name lookup")
    ap.add_argument("--espn-league", default=espn_leaderboard.DEFAULT_LEAGUE)
    ap.add_argument("--season", type=int, help="season for the ESPN lookup (default: this year)")
    ap.add_argument("--price", default=kalshi_odds.DEFAULT_PRICE, choices=list(kalshi_odds.PRICE_MODES))
    ap.add_argument("--exclude", action="append", metavar="NAME", help="golfer to exclude; repeatable")
    ap.add_argument("--auto-exclude", dest="auto_exclude", action="store_true", default=True,
                    help="drop golfers over the 1/teams fair share (default on)")
    ap.add_argument("--no-auto-exclude", dest="auto_exclude", action="store_false")
    ap.add_argument("--time-limit", type=float, default=groupers.DEFAULT_TIME_LIMIT)
    ap.add_argument("--seed", type=int, help="seed the deal of groups to teams")
    ap.add_argument("--output", default="build/result.json")
    ap.add_argument("--alias-file", default=ALIAS_FILE)
    ap.add_argument("--match-review", metavar="PATH",
                    help="the review file for golfer names the join will not guess at. "
                         "Read for decisions before the join and rewritten after it. "
                         "Defaults to match-review.json beside --output. Only a live "
                         "build has anything to review.")
    ap.add_argument("--update-aliases", action="store_true",
                    help="write name bindings settled in the review file back to the alias "
                         "file, so the next tournament resolves them automatically")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds the scoreboard should wait between ESPN polls (default 60)")
    return ap


def typed_options(argv, args):
    """
    Which of a rebuild's inputs were actually typed, by dest name.

    argparse cannot tell `--price ask` from a `--price` nobody passed that defaulted to
    "ask", and here the difference decides an outcome: a rebuild fills every unset
    option from the result file, so without this, somebody rebuilding a Top 5
    competition and typing `--odds winner` would silently get Top 5 back -- their
    instruction dropped because it happened to equal the default.

    Only the four inputs whose default is a value somebody might type need asking
    about, and they are found by parsing a second time with those defaults replaced by
    a sentinel. Everything else a rebuild fills defaults to None, which nobody types.

    Probing narrowly is not just economy. `--exclude` is an `append` option, and
    argparse appends to whatever the default is -- hand it a sentinel and the parser
    itself raises AttributeError the moment somebody excludes a golfer.
    """
    ambiguous = [dest for dest in REBUILD_INPUTS
                 if build_parser().get_default(dest) is not None]
    probe, sentinel = build_parser(), object()
    probe.set_defaults(**{dest: sentinel for dest in ambiguous})
    probed = vars(probe.parse_args(argv))
    return ({dest for dest in ambiguous if probed[dest] is not sentinel}
            | {dest for dest in REBUILD_INPUTS
               if dest not in ambiguous and getattr(args, dest) is not None})


def apply_result_defaults(args, result, typed=()):
    """
    Fill in every argument the result file already answers.

    A result file is a complete description of a competition: which league, which
    tournament on both APIs, which market at which price, who was excluded by hand,
    which seed dealt the groups. Re-typing all of that to rebuild is how a rebuild turns
    into a different competition -- so it is read rather than re-typed.

    Only arguments the user did not type are filled (`typed`, from typed_options).
    Anything given on the command line wins, which is what makes a rebuild-with-one-
    change a one-flag operation.
    """
    kalshi = result["sources"]["kalshi"]
    espn = result["sources"].get("espn") or {}
    # Keyed by REBUILD_INPUTS, and read back through it below, so the two cannot drift
    # apart without raising.
    defaults = {
        "tournament": result["tournament"]["name"],
        "kalshi_event": kalshi["event_ticker"],
        # A custom series was given as a raw ticker; the label is the ticker itself.
        "odds": kalshi["series_ticker"] if kalshi["odds_type"] == "custom" else kalshi["odds_type"],
        "price": kalshi["price_mode"],
        "espn_event": espn.get("event_id"),
        "espn_league": espn.get("league"),
        "season": result["tournament"]["season"],
        "seed": result["generator"].get("seed"),
        # Only the named ones. `over_fair_share` is a rule, not a decision, and it is
        # re-derived against whatever field this run reads -- but WHETHER the rule was
        # on is a decision, and it comes back too. A 1.0 file does not record it and
        # falls through to the flag's own default, which is what it was built under.
        "exclude": [e["golfer_name"] for e in result["odds_snapshot"]["excluded"]
                    if e["reason"] == "named"] or None,
        "auto_exclude": result["odds_snapshot"].get("auto_exclude"),
        # From `generator`, not from `live`. A groups build has no `live` block at all,
        # and reading it there loses the setting on exactly the rebuild that first needs
        # it. Pre-2.0 files only have the `live` copy, so fall back to it.
        "poll_interval": (result["generator"].get("poll_interval_seconds")
                          or (result.get("live") or {}).get("poll_interval_seconds")),
    }
    filled = []
    for name in REBUILD_INPUTS:
        value = defaults[name]
        if value is None or name in typed:
            continue
        setattr(args, name, value)
        filled.append(name)

    # --exclude is an append option and a typed one replaces the recorded list rather
    # than adding to it, which is the same rule every other option follows. Silently
    # re-admitting golfers the pool had already excluded by hand is not, so it is said.
    if "exclude" in typed and defaults["exclude"]:
        print(f"note: --exclude replaces the {len(defaults['exclude'])} exclusion(s) recorded "
              f"in the file ({', '.join(defaults['exclude'])}). Name them again to keep them.")
    return filled


def check_art_options(parser, args):
    """
    Catch a bad `--crest` / `--banner` now rather than forty seconds into a build.

    A path typed on the command line is a thing somebody meant, so a typo in one is an
    error and not the shrug `inline_logo` gives a league file's missing art. Getting it
    at the top matters because everything between here and the inliner is network: the
    Kalshi fetch and the ESPN join both run first, and finding out afterwards that the
    banner was `bannner.png` means running them again.
    """
    for field in ("crest", "banner"):
        value, cleared = getattr(args, field), getattr(args, f"no_{field}")
        if value and cleared:
            parser.error(f"--{field} and --no-{field} ask for opposite things")
        if value and not value.startswith(("http://", "https://", "data:")) \
                and not os.path.isfile(value):
            parser.error(f"--{field} {value} is not a file")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    check_art_options(parser, args)

    if args.from_result:
        result = load_result(args.from_result)
        apply_result_defaults(args, result, typed_options(argv, args))
        same_file = os.path.abspath(args.output) == os.path.abspath(args.from_result)
        if args.regroup and same_file and not args.overwrite:
            parser.error(
                f"--regroup would overwrite {args.from_result} with a different draw, and that "
                "file is the record of the groups people already have. Give --output a new "
                "path, or --overwrite if replacing it is the point.")
        try:
            if args.regroup:
                build(args, league=(league_mod.load_league(args.league) if args.league
                                    else league_from_result(result)),
                      prior_mode=prior_build_mode(result),
                      recorded_decisions=((result["sources"].get("espn") or {})
                                          .get("match_decisions")),
                      rebuilt_from={
                          "source_file": args.from_result,
                          "source_generated_at": result["generated_at"],
                          "source_schema_version": result.get("schema_version"),
                          "mode": "regroup",
                          "rebuild_count": ((result.get("rebuilt_from") or {})
                                            .get("rebuild_count") or 0) + 1,
                          "first_built_at": ((result.get("rebuilt_from") or {})
                                             .get("first_built_at") or result["generated_at"]),
                      })
            else:
                rebuild(args, result)
        except ValueError as exc:
            raise SystemExit(str(exc))
        return 0

    if not args.league:
        parser.error("give --league PATH, or --from-result PATH to rebuild an existing one")
    if not args.tournament and not args.kalshi_event:
        parser.error("give --tournament NAME or --kalshi-event TICKER")
    if args.regroup:
        parser.error("--regroup only means something with --from-result")

    try:
        build(args)
    except ValueError as exc:
        # A bad league file is the common case here, and it already carries a sentence
        # naming the file and the field. A traceback would bury it.
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
