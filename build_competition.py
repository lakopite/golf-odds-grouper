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

ONE BUILD, THE NIGHT BEFORE
---------------------------
Both APIs are posted by Wednesday night, so one run does the whole job.

Kalshi posts winner markets Sunday ~23:00Z of tournament week and keeps ADDING them
through Wednesday as the field firms up. ESPN posts its field about two days before
the first round -- measured 2026-08-04 against the Wyndham, whose first round was the
6th: 147 competitors with athlete ids, headshots and tee times, while the event was
still `pre`. So Wednesday night is the first moment both are complete, and it is also
when a pool wants to be drawn. The timings agree; there is nothing to work around.

    python build_competition.py --league ... --tournament ...   # and that is it

This did not use to be true. ESPN published zero competitors until the first tee time,
which split the program in half: a Wednesday "groups" build that could join no names
and score nothing, and a Thursday rebuild that turned it into a scoreboard. Both modes
are gone, along with `build_mode` itself. A build now joins the Kalshi field onto the
real ESPN one and writes an athlete id for every golfer it settles, so the page has
everything it needs before anyone tees off.

WHICH LEAVES ONE DISTINCTION, AND IT MOVED TO THE PAGE
------------------------------------------------------
A field is joinable long before it is rankable. A `pre` payload lists all 147 players
and gives not one of them a position, so nothing built on Wednesday can rank anything
-- but nothing needs to be rebuilt for it to start. The exported page polls ESPN,
reads `started` off the payload, shows the draw until somebody tees off and ranks from
then on. The transition happens in the browser, on its own, while the page is open.

That is the whole of what the second build used to do, and it is now free.

A golfer the join does not resolve carries no athlete id, scores nothing, and is
NAMED, so the gap is something somebody can close rather than a silence.

SETTLING THE NAMES THE JOIN WILL NOT GUESS AT
----------------------------------------------
The join is exact and explicit in all three of its tiers -- there is no fuzzy match,
for the reasons in espn_leaderboard's docstring -- so a build leaves a handful of
names open. It writes them to a review file, each beside the ESPN athletes nobody
claimed and a ranked suggestion or two, and the next build reads that file back and
applies whatever has been filled in. See match_review.py. The review is the step a
model is good at and a regular expression is not, and it happens where the answer can
be read before it takes effect.

AND THE DEAL WAITS FOR THE ANSWER
----------------------------------
Some of those open names are not names at all: they are golfers who withdrew. Measured
on the Rocket Classic, 151 Kalshi markets against 147 ESPN competitors left 12 open, and
4 of the 12 -- Daniel Brown, Taylor Moore, Brooks Koepka, Jason Day -- were simply not
playing. From here the two look identical, and only one of them is a hole.

So the join runs BEFORE the partition and the deal, and an unresolved name stops the run:

    read ESPN field -> pull odds -> JOIN -> gate
      -> exclusions (`withdrawn` beside `named` and `over_fair_share`)
      -> partition -> deal

It used to run last, forty lines after the deal, which meant a golfer who had withdrawn
was partitioned, weighted and dealt onto somebody's card before anybody found out --
carrying a full share of a group that could never score it, on a team whose `total_odds`
said the draw was even. There was no way back either: a rebuild deliberately never
re-partitions, so the only route from "he withdrew" to "deal without him" was --regroup,
which re-deals every team. In a five-team pool the fair share is 20%, so a Koepka-sized
hole is a real piece of one group, and which group is decided by a coin toss.

Stopping costs one re-run and settles the question where it can be read. `--deal-anyway`
is the other honest answer -- somebody looked and still cannot say which it is -- and it
deals them in at full weight, exactly as this did before the gate existed.

REBUILDING ONE
--------------
Because the result file describes the whole competition, it is also the input to the
next build of it:

    python build_competition.py --from-result build/result.json

That reads the league, the tournament on both APIs, the market, the price mode, the
hand-picked exclusions, the seed and any reviewed name decisions out of the file,
carries the groups and the odds at creation forward untouched, and redoes the ESPN
join. It does not re-read the odds, ever: they were read once, when the groups were
drawn, and that reading is the competition. `--regroup` is the one that deals again,
and it says so.

A rebuild is no longer part of a normal week -- the page ranks by itself once play
starts, so nothing has to be re-run to make scoring appear. What it is for now is
settling names: fill in the review file and rebuild, and the golfers the join would
not guess at get their athlete ids. Also a late withdrawal, a redraw, or an ESPN
event that was pinned wrong the first time.

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
# 3.0 REPLACED `league.crest` and `league.banner` -- two base64 data: URIs, routinely
# half a megabyte of PNG between them -- with `league.logo`, the NAME of a directory of
# art under `leagues/`. This file is the document that describes a competition, and it
# had become mostly an envelope for two images that nothing in it read: every rebuild
# copied them forward, every diff was unreadable, and every copy of the file paid for
# them again. The images are now read once, at export, by bundle_frontend.py, which
# inlines them into the one artifact that has to be portable. The exported page is
# unchanged -- still a single file with nothing to fetch. A 2.x file rebuilds, and its
# inlined art is dropped with a line saying so; name the slug and it comes back.
# 4.0 REMOVED `build_mode`, and with it the half of this file that could be null. ESPN
# now posts the field about two days before the first round (measured 2026-08-04 on the
# Wyndham: 147 competitors while the event was still `pre`), so the premise 2.0 split
# the file on -- no field before the first tee time -- is simply no longer true. There
# is one kind of build. `live` and `golfers[].espn` are ALWAYS objects; a reader no
# longer has to check which document it is holding before reading it. The distinction
# the mode used to carry did not disappear, it moved to where it belongs: the page asks
# the leaderboard whether anybody has teed off and shows the draw until somebody has.
# A 3.x file rebuilds into a 4.0 one, and a groups-mode 3.x file gains the ESPN half it
# never had. One key is ADDED, `sources.espn.started_at_build`: it records whether play
# had begun when the build ran, which is the only thing `build_mode` said that was worth
# keeping, and it is a record rather than an instruction -- it is false for the life of
# a normally-built file while the tournament it describes comes and goes. Nothing reads
# it to decide anything; the page asks the leaderboard.
# 4.1 ADDED a third value to `odds_snapshot.excluded[].reason`: "withdrawn", for a golfer
# a review confirmed is not in this week's ESPN field. Additive -- no key is added,
# removed or retyped, and a 4.0 reader reads a 4.1 file correctly on every key it knows.
# What changes is which combinations occur. Before this, a golfer with `espn.in_field:
# false` had always been dealt: the join ran after the partition, so a withdrawal was
# discovered with the golfer already on somebody's card at a full `grouping_weight`, and
# the team's `total_odds` claimed a share of the pool that one of its golfers could not
# score. Now the join runs before the deal, a confirmed absence joins `named` and
# `over_fair_share` in the exclusion set, and that golfer comes out shaped like every
# other excluded one -- `excluded: true`, `grouping_weight: null`, no `team_id`. The
# reason is a new word for a shape readers already handle. A golfer who withdraws AFTER
# the draw is untouched and still carries their weight on the card they were dealt to:
# a rebuild does not re-deal, so the totals stay the totals people were dealt.
SCHEMA_VERSION = "4.1"

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

# Team logos are inlined so the exported bundle is one portable file. A logo bigger
# than this is a mistake rather than a choice -- it lands in every copy of the result
# JSON and every copy of the HTML built from it.
#
# The league's own art is NOT inlined here and this limit does not reach it: a slug is
# a name, the pictures it names are read at export, and they land in the page alone.
# That is the whole point of the slug -- see league.py.
MAX_INLINE_LOGO_BYTES = 512 * 1024

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

def read_espn_field(args, espn_event) -> tuple:
    """
    This week's ESPN field. -> (meta, players).

    A published field is a PRECONDITION of building, not a mode selector. ESPN posts it
    about two days before the first round, so any run made when a pool is actually
    drawn -- the night before -- has one to read. Not finding one means something is
    wrong with this run, and the three things it can be are worth telling apart, so
    they get three different sentences.

    This is deliberately fatal. It used to fall back to a groups sheet, which was the
    right answer when a Wednesday field genuinely did not exist yet; now it would hand
    somebody a page that can never score, built from a failed read, on a night when the
    field was there for the asking. Stopping costs one re-run. The alternative costs
    the scoreboard, silently.

    A rebuild gets one extra sentence, and it is the important one. Somebody whose
    tournament is halfway through its final round and whose rebuild just died needs to
    be told, in the same breath, that their existing file is untouched -- otherwise the
    reasonable reading of any error here is "the scoreboard is gone".
    """
    intact = ("\nNothing has been written: " + args.from_result + " is untouched and still "
              "correct." if args.from_result else "")

    if not espn_event:
        raise SystemExit(
            "no ESPN event was resolved for this tournament, so there is no field to join "
            "the Kalshi names against and nothing this competition could ever score.\n"
            "Find the event id with `python espn_leaderboard.py --season <year> --find "
            "<name>` and pass it as --espn-event <id>." + intact)

    try:
        payload = espn_leaderboard.fetch_leaderboard(espn_event["event_id"], args.espn_league)
        meta, players = espn_leaderboard.parse_leaderboard(payload)
    except Exception as exc:                       # noqa: BLE001 -- turned into a clean exit
        raise SystemExit(
            f"could not read the ESPN leaderboard for event {espn_event['event_id']}: {exc}\n"
            "The field is what every golfer's athlete id comes from, so a build without it "
            "would produce a page that can never score, and this run has stopped rather than "
            "write one. Try again." + intact)

    state = (meta or {}).get("state") or espn_event.get("state") or "unknown"
    if not players:
        raise SystemExit(
            f"ESPN lists no competitors for event {espn_event['event_id']} [{state}].\n"
            "ESPN posts a tournament's field about two days before the first round, so an "
            "empty field usually means one of three things: the event id is wrong (check it "
            "with `python espn_leaderboard.py --season <year> --find <name>`), this is being "
            "run more than a couple of days out, or ESPN is having a bad morning. All three "
            "are worth fixing before drawing a pool -- the athlete ids in this join are the "
            "only key the exported page has." + intact)

    # Said out loud because it is the one thing about this run that the clock decides.
    # A build made the night before is correct and complete and still cannot rank
    # anything, and a reader who does not know that reads the page as broken.
    print(f"ESPN:    {len(players)} competitors in the field [{state}]"
          + ("" if meta.get("started") else
             " -- not started, so the page shows the draw until the first tee time and "
             "ranks from then on by itself"))
    return meta, players


def check_pinned_event(espn_event, meta, tournament_name, pinned):
    """
    A pinned `--espn-event` names its own tournament. Say so if it is a different one.

    This got more dangerous rather than less. A wrong id used to produce an obviously
    empty groups build most weeks, because most weeks the pinned event had no field
    posted either; now any id with competitors on it builds a complete, confident,
    entirely wrong scoreboard, joined against the wrong 150 people. Nothing downstream
    can tell -- every number in the file is internally consistent.

    A note rather than a refusal, because the name match is fuzzy and the pin is the
    override of last resort: somebody pinning an id has usually just been let down by
    the name lookup, and refusing on the same lookup would leave them nowhere.
    """
    if not pinned or not meta:
        return
    found = meta.get("event")
    if not found or not tournament_name:
        return
    if espn_leaderboard.score_name(tournament_name, found) > 0:
        return
    print(f"!! --espn-event {espn_event['event_id']} is ESPN's {found!r}, and the Kalshi "
          f"event is {tournament_name!r}. Those do not look like the same tournament.\n"
          "   Every golfer in this build is about to be joined against that field, and a "
          "build against the wrong one comes out looking completely normal. Check the id "
          "with `python espn_leaderboard.py --season <year> --find <name>` before handing "
          "the page over.", file=sys.stderr)


def refuse_a_collapsed_join(espn, prior_espn, source_file):
    """
    Refuse a rebuild whose join fell off a cliff against the last one.

    This is what is left of the old refuse_downgrade, and it covers the case that one
    did not. Reading NO field is now fatal everywhere, for every build -- but reading a
    complete and WRONG one is not, and on a rebuild that is the more dangerous of the
    two. A mistyped `--espn-event`, or an event lookup that drifted to a different
    tournament, returns 150 real competitors who are the wrong 150 people. Every number
    that comes out is internally consistent and every athlete id is a valid id in
    somebody else's field.

    The file remembers what the join used to manage, so there is something to compare
    against: a rebuild that resolves a small fraction of what it resolved last time has
    almost certainly been pointed at the wrong tournament. Withdrawals move this number
    by ones and twos. Nothing legitimate halves it.

    Skipped when there is nothing to compare against -- a pre-4.0 file built before the
    field existed recorded no join at all, and upgrading one is exactly the case where
    the count is supposed to go up from nothing.
    """
    was = ((prior_espn or {}).get("match_report") or {}).get("matched")
    now = (espn["report"] or {}).get("matched", 0)
    if not was or now >= was / 2:
        return
    raise SystemExit(
        f"this rebuild joined {now} of the Kalshi field to ESPN, where {source_file} "
        f"joined {was}.\nA field moves by a golfer or two between builds; it does not "
        "halve. The likeliest cause is that this run read a different tournament's "
        f"leaderboard -- check the ESPN event id, and `python espn_leaderboard.py "
        "--season <year> --find <name>` if you need to look it up. Writing on it would "
        "put a valid athlete id from the wrong field onto most of the pool.\n"
        f"Nothing has been written: {source_file} is untouched and still correct. Pass "
        "--espn-event <id> to override the recorded one deliberately.")


def report_withdrawals_on_cards(espn, teams, team_groups):
    """
    Name the teams carrying a golfer who has since been confirmed out of the field.

    A withdrawal before the draw is handled by leaving the golfer out of it -- see
    gate_on_unresolved. This is the other case, and it is the one a pool actually meets
    most weeks: the groups were dealt on Wednesday, somebody pulled out on Thursday, and
    a review has now confirmed it. The card does not change. That is the pool's
    convention and it is the only answer consistent with a rebuild that never re-deals;
    dropping the golfer and rescaling one team's total would report a total nobody was
    dealt, which this file refuses to do everywhere else.

    So the answer is "nothing happens", and the reason this function exists is that
    "nothing happens" was previously indistinguishable from "nobody noticed". A team
    losing a fifth of its group is worth a sentence naming the team, the golfer and the
    size of the hole, and worth saying that --regroup is the thing that would re-deal
    around it -- so that leaving the draw alone is visibly a decision rather than an
    oversight.
    """
    absent = set(espn["report"]["absent"])
    if not absent:
        return
    for team in teams:
        members = team_groups[team["team_id"]]
        total = sum(g["odds"] for g in members)
        hit = [g for g in members if g["golfer_name"] in absent]
        if not hit:
            continue
        share = sum(g["odds"] for g in hit)
        print(f"!! {team['team_name']} holds {len(hit)} golfer(s) a review has confirmed are not "
              "in this week's ESPN field: "
              + ", ".join(f"{g['golfer_name']} {g['odds']:.4f}" for g in hit)
              + f".\n   That is {share:.2%} of the {total:.2%} they were dealt, and it stays on "
              "their card: people were told which golfers they own, so a rebuild does not take "
              "one back. Those golfers score nothing, which is what a withdrawal costs the team "
              "that drew them. --regroup re-deals every team around the field that is left.",
              file=sys.stderr)


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


def espn_stage(args, espn_event, espn_field, field, weight_by_name, tournament_name,
               aliases, team_name_of=None, recorded_decisions=None):
    """
    The whole ESPN half of a build: read the field, join it, and assemble the review of
    whatever is left over.

    Shared verbatim by build() and rebuild(), because the ESPN side of a competition
    depends on the tournament and on nothing the pool decided. That is why a rebuild
    can settle a name without touching the draw.

    The field arrives already read, from read_espn_field, because reading it is the
    precondition of the whole run and a precondition belongs before the work it gates
    -- see build().

    Nothing here needs the draw, and that is the point. This used to run last, after the
    partition and the deal, and `team_name_of` was the only reason it had to -- one
    column in the review file. It is now optional, and a build calls this BEFORE it
    deals, so that a golfer who turns out to have withdrawn can be left out of the draw
    rather than discovered inside it. A rebuild still passes the teams, because a rebuild
    has the draw in front of it from the file it was handed.

    `weight_by_name` is what each golfer is worth, and the two callers have different
    numbers to offer. A rebuild passes the grouping weights the pool was dealt on. A
    build cannot: the grouping weights depend on the exclusion set, which depends on the
    answers to the very questions this stage is asking. So it passes the de-vig over the
    whole field, which is a FLOOR on the same quantity -- removing anybody scales every
    survivor up -- and orders the review file identically, because rescaling is monotone.

    Reviewed decisions come from two places and both are wanted. The result file
    carries the ones already applied, so a rebuild does not re-ask a question somebody
    answered last time; the review file carries whatever was filled in since, and wins
    on a conflict because it is the newer statement. Decisions for names that are not
    in this Kalshi field are dropped -- they are left over from a different draw and
    binding them would be binding somebody else's competition.
    """
    team_name_of = team_name_of or {}
    meta, players = espn_field
    event_id = (espn_event or {}).get("event_id")

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
        "meta": meta, "players": players, "matches": matches,
        "report": report, "decisions": decisions,
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


def resolve_exclusions(golfers, n_groups, named, auto, withdrawn=()):
    """
    The exclusion set, and a reason for each name.

    A result file that says "excluded: Scottie Scheffler" and nothing else is a
    decision with no argument attached. Everything here records which rule fired.

    Three rules now, and they are applied in this order for a reason.

    `named` is somebody's instruction and goes first. `withdrawn` is a fact about the
    world -- a review looked at this week's published ESPN field and confirmed the
    golfer is not in it -- and goes second. `over_fair_share` is a rule measured against
    whatever is left, so it has to go last: taking a golfer out redistributes their
    probability over everybody who remains, which can push the next one over 1/N.
    grouper_cli.auto_exclusions iterates that cascade to a fixed point, but it can only
    iterate over the field it is handed. Give it the withdrawals afterwards and it would
    have balanced the pool around a golfer who is not playing.

    A withdrawn name is never warned about the way an unknown `named` one is: it came out
    of the join, which read it from this same field.
    """
    excluded = []
    named = list(named or [])
    known = {g["golfer_name"] for g in golfers}

    for name in named:
        if name not in known:
            print(f"!! warning: excluded golfer {name!r} is not in this field, so it excluded nothing")
            continue
        excluded.append({"golfer_name": name, "reason": "named"})

    already = {e["golfer_name"] for e in excluded}
    for name in withdrawn or ():
        if name in known and name not in already:
            excluded.append({"golfer_name": name, "reason": "withdrawn"})

    if auto:
        remaining = [g for g in golfers if g["golfer_name"] not in {e["golfer_name"] for e in excluded}]
        for name in grouper_cli.auto_exclusions(remaining, n_groups):
            excluded.append({"golfer_name": name, "reason": "over_fair_share"})

    return excluded


# ---------------------------------------------------------------------------
# The gate: a name nobody has settled stops the deal
# ---------------------------------------------------------------------------

def write_review(espn):
    """
    Write the review worksheet. -> the path written, or None if there was nothing to say.

    Shared, because there are now two ways out of a run and both owe the reviewer this
    file. finish() writes it at the end of a build that completed; gate_on_unresolved
    writes it on the way out of one that stopped, which is the run that needs it most --
    stopping somebody without handing them the worksheet that answers the question would
    be a refusal with no way to act on it.
    """
    if not espn["review"]:
        return None
    return match_review.write(espn["review"]["path"],
                              **{k: v for k, v in espn["review"].items() if k != "path"})


def gate_on_unresolved(espn, devigged, args):
    """
    Stop before the deal when a Kalshi golfer cannot be found in this week's ESPN field.

    An unresolved name is two different things wearing one silence: a golfer whose name
    this join cannot spell, or a golfer who is not playing. Both come back with no
    athlete id and no row. Only a person can tell them apart, and until somebody has,
    dealing is a coin toss between "this golfer scores normally" and "one team is
    carrying a hole worth a slice of a group".

    Which is why this stops the DEAL rather than the run. A rebuild is not gated: it does
    not deal, the golfers are already on cards, and refusing to refresh a working
    scoreboard over an open name would be strictly worse than what this replaced.
    `--regroup` goes through build() and is gated, because a regroup deals.

    Nothing has been written when this fires, including the odds. The Kalshi pull has to
    come first -- the join needs the Kalshi names -- so a stopped run spends it and keeps
    nothing, and the next run reads the market again. That is correct rather than
    unfortunate: the prices a competition is worth are the prices at the moment it was
    actually dealt, and this run dealt nothing.

    `--deal-anyway` is the second honest answer and not merely an escape hatch. Before
    the first tee time "no row on the leaderboard" is genuinely not yet the same fact as
    "withdrew" -- ESPN's field still moves, and a golfer Kalshi already prices may be an
    alternate ESPN has not listed. Somebody who looked twice and still cannot say should
    NOT record an absence: a wrong one now takes the golfer out of the draw entirely
    rather than merely leaving them scoreless, and stops anybody looking again. Dealing
    them in at full weight is what this tool did before the gate existed, and it is the
    right answer to "I do not know".
    """
    pending = list(espn["report"]["unresolved"])
    if not pending:
        return

    shown = ", ".join(f"{name} {devigged.get(name, 0.0):.4f}" for name in pending[:8])
    share = sum(devigged.get(name, 0.0) for name in pending)

    if args.deal_anyway:
        print(f"note: --deal-anyway, so the {len(pending)} unsettled golfer(s) go into the draw "
              f"at their full weight: {shown}" + (" ..." if len(pending) > 8 else ""))
        print(f"  That is {share:.2%} of the book being dealt without anybody having confirmed "
              "those golfers are playing. Whichever of them are not, their teams carry the hole "
              "-- the right call when a field is still moving and an absence would be a guess, "
              "and the wrong one if nobody has looked yet.")
        return

    written = write_review(espn)
    where = written or match_review.review_path(args.output, args.match_review)
    raise SystemExit(
        f"{len(pending)} Kalshi golfer(s) are not in this week's ESPN field and nobody has said "
        f"why:\n  {shown}" + (" ..." if len(pending) > 8 else "")
        + f"\n  -- {share:.2%} of the priced book between them.\n\n"
        "Each one is either a name this join will not guess at or a golfer who is not playing, "
        "and from here those are the same silence. Dealing now would put whichever of them "
        "withdrew onto somebody's card at a full share of a group they cannot score, and there "
        "is no way back afterwards: a rebuild never re-deals, so the only route from 'he "
        "withdrew' to 'deal without him' is --regroup, which re-deals every team. So this run "
        "has stopped before the deal.\n\n"
        "Nothing has been written. No groups were drawn, and the odds this run read were not "
        "kept -- the next run reads the market again, and those are the prices the pool is "
        f"dealt on.\n\nSettle them in {where} and run this again:\n"
        '  {"athlete_id": "..."}  binds this golfer to that ESPN athlete\n'
        '  {"absent": true}       records that they are not in the field, and deals without them\n'
        "Record an absence only when you can say what happened -- before the first tee time "
        "ESPN's field is still moving, and an absence now takes a golfer out of the draw rather "
        "than merely leaving them scoreless. If you looked and still cannot tell, pass "
        "--deal-anyway: they go into the draw at full weight, which is what this did before "
        "this check existed.")


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

def build(args, league=None, rebuilt_from=None, recorded_decisions=None):
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
                  "There is no field to join the Kalshi names against, so this build is about "
                  "to stop. Pass --espn-event <id> to pin it.", file=sys.stderr)
        elif espn_ranked and espn_ranked[0].get("score", 1.0) < 1.0:
            print(f"note: ESPN matched {espn_event['event_id']} ({espn_event['name']}) on a "
                  f"partial name match. Pass --espn-event to override.")
    if espn_event:
        # A pinned event was never looked up, so it has no state to report. The read
        # below prints the state it actually finds either way.
        state = espn_event.get("state")
        print(f"ESPN:    {espn_event['event_id']}  {espn_event.get('name')}"
              + (f"  [{state}]" if state else ""))

    # -- the ESPN field, which is a precondition ------------------------------
    # Read here, before anything expensive, because that is what a precondition is for.
    # It used to be read at the end, next to the join that consumes it, which was fine
    # while a missing field only downgraded the build. Now it stops the run -- and a
    # run that is going to stop should not first spend a Kalshi pull, a partition and a
    # deal to get there, nor make somebody wait for all three to be told the event id
    # was wrong.
    espn_field = read_espn_field(args, espn_event)
    check_pinned_event(espn_event, espn_field[0], tournament_name, args.espn_event)

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

    devigged = {g["golfer_name"]: g["odds"] for g in grouper_cli.normalize_probabilities(field)}

    # -- ESPN, which now comes before the draw --------------------------------
    # The join used to run last, after the partition and the deal, and its only tie to
    # them was one column in the review file. Everything it needs is here: the Kalshi
    # names, and what each of them is worth over the whole field. So it runs here, and a
    # golfer it cannot find is a question that gets answered BEFORE anybody is dealt
    # anything -- which is the whole of this change. See gate_on_unresolved.
    #
    # A --regroup deals new groups out of the same tournament, so the reviewed name
    # decisions are still about the same people and come with it. Dropping them would
    # make a regroup unresolve golfers somebody had already settled, for no reason
    # except that the partitioner ran again.
    aliases = load_aliases(args.alias_file)
    espn = espn_stage(args, espn_event, espn_field, field, devigged, tournament_name,
                      aliases, recorded_decisions=recorded_decisions)
    gate_on_unresolved(espn, devigged, args)

    # -- exclusions ----------------------------------------------------------
    # Measured over the whole priced book, which is what this line claims to be about --
    # a golfer worth more than a group's share of what Kalshi priced. The rule itself is
    # applied further down against the field that is actually left, so a withdrawal can
    # push somebody over the line who is not named here; the "Excluded:" line below is
    # where every golfer who was dropped says which rule dropped them.
    over = grouper_cli.golfers_over_threshold(field, n_groups)
    if over:
        print(f"Above the 1/{n_groups} fair share ({1/n_groups:.4f}): "
              + ", ".join(f"{g['golfer_name']} {g['odds']:.4f}" for g in over))
    withdrawn = list(espn["report"]["absent"])
    excluded = resolve_exclusions(field, n_groups, args.exclude, args.auto_exclude,
                                  withdrawn=withdrawn)
    excluded_names = {e["golfer_name"] for e in excluded}
    if excluded:
        print("Excluded: " + ", ".join(f"{e['golfer_name']} ({e['reason']})" for e in excluded))
    gone = [e["golfer_name"] for e in excluded if e["reason"] == "withdrawn"]
    if gone:
        print(f"  {len(gone)} of them because a review confirmed they are not in this week's "
              f"ESPN field, worth {sum(devigged.get(n, 0.0) for n in gone):.2%} of the book "
              "between them. They are left out of the draw rather than dealt onto a card they "
              "cannot score on. Their share goes back to everybody else, so the fair-share rule "
              "was re-measured over the field that is left.")

    if excluded_names:
        weighted = grouper_cli.odds_to_conditional(field, excluded_names)
    else:
        weighted = grouper_cli.normalize_probabilities(field)

    if len(weighted) < n_groups:
        raise SystemExit(f"only {len(weighted)} golfers left after excluding {len(excluded)} "
                         f"({len(gone)} of them withdrawn), which cannot fill {n_groups} groups")

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
    Inline the team logos, write the files, learn what the run learned, show the groups.

    Shared by build() and rebuild() because it is the same ending either way: the only
    thing that differs between the two is how the numbers above it were arrived at.
    """
    # Team logos are resolved against the league file's own directory: a logo path in a
    # league file is relative to that file, not to wherever the build was run from. A
    # rebuild has no league file -- its logos are already data: URIs and pass straight
    # through -- so anything still relative there is resolved against the result file.
    base = args.league or args.from_result or "."
    league_dir = os.path.dirname(os.path.abspath(base))
    for team in result["teams"]:
        team["team_logo"] = inline_logo(team.get("team_logo"), league_dir)
    # The league's own art is settled but not read. It goes into the file as the slug
    # it arrived as, and bundle_frontend.py turns it into pictures at export.
    resolve_league_logo(result, args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResult -> {args.output}  ({os.path.getsize(args.output) // 1024} KB)")

    if espn["review"]:
        written = write_review(espn)
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
    # A pre-3.0 file carries its art as two inlined images and no slug. They are not
    # carried forward -- 3.0 has nowhere to put them, and the whole reason for the
    # change is that a result file should not hold pictures -- so it is said out loud.
    # Un-branding a page somebody has already sent round is exactly the kind of quiet
    # change this tool goes out of its way not to make.
    if not league.get("logo") and (league.get("crest") or league.get("banner")):
        print("note: this result file predates the art slug and carries its crest and banner "
              "inline. They are dropped. Put `\"logo\": \"<slug>\"` in the league file (art in "
              "leagues/<slug>/logo.png and banner.png) and rebuild with --league, or pass "
              "--logo <slug>, to get the masthead back.")
    return {
        "league_id": league["league_id"],
        "league_name": league["league_name"],
        "league_slug": league.get("league_slug") or league_mod.slugify(league["league_name"]),
        "source_file": league.get("source_file"),
        # Carried forward, because a rebuild that dropped them would quietly un-brand a
        # page somebody has already seen. `.get` because a file written before branding
        # existed has no such keys and is still a perfectly good rebuild.
        **{f: league.get(f) for f in league_mod.BRANDING_FIELDS},
        "teams": [{k: v for k, v in t.items() if k not in derived} for t in result["teams"]],
    }


def rebuild(args, result):
    """
    A new build of an existing competition: same teams, same groups, same odds at
    creation -- everything else brought up to date.

    What moves between two runs of the same competition is what the world did, not what
    the pool decided. So the draw is carried forward verbatim and the run re-reads the
    ESPN join and the tournament's state. Kalshi is not read at all. The odds were read
    once, when the groups were drawn, and a rebuild that went back for a second reading
    would be putting a price nobody was dealt on next to the one everybody was.

    Scoring is NOT a reason to run this. The page polls ESPN and starts ranking on its
    own at the first tee time; nothing has to be rebuilt for that. What this is for is
    the join: names settled in the review file, a golfer who has since withdrawn, or an
    ESPN event that was pinned to the wrong id.

    Re-partitioning is deliberately NOT what this does. Rebuilding a competition and
    quietly dealing everyone new golfers is the single most destructive thing this tool
    could do; --regroup asks for it explicitly and goes through build().

    Which decides what happens to a golfer who withdraws AFTER the draw, and the answer
    is: nothing. They keep the card they were dealt to, they carry the weight they were
    dealt at, and they score nothing -- which is how a pool handles a withdrawal, and the
    only answer consistent with a rebuild that never re-deals. Dropping them and
    renormalizing their team's total would move one team's number and not the others', so
    the file would report a total nobody was dealt; this function already refuses a file
    that does that (see `orphans` below). What a rebuild does add is a sentence naming
    the team that is carrying the hole, because until now that discovery was silent.

    A rebuild is also NOT gated on unresolved names, and that asymmetry is deliberate.
    The gate in build() protects the deal, and there is no deal here -- the golfers are
    already on cards. Refusing to refresh a working scoreboard over an open name would be
    strictly worse than the behaviour it replaced.
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
    espn_field = read_espn_field(args, espn_event)
    check_pinned_event(espn_event, espn_field[0], tournament["name"], args.espn_event)
    espn = espn_stage(args, espn_event, espn_field, field, weight_by_name,
                      tournament["name"], aliases, team_name_of=team_name_of,
                      recorded_decisions=espn_source.get("match_decisions"))
    refuse_a_collapsed_join(espn, espn_source, args.from_result)
    report_withdrawals_on_cards(espn, teams, team_groups)

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
    it runs" -- ESPN, and only ESPN.

    Every block here is always present. There is no longer a version of this file with
    half of it nulled out: a build cannot complete without a published ESPN field, so
    a reader never has to establish which document they are holding before reading it.
    """
    field, devigged, weighted = k["field"], k["devigged"], k["weighted"]
    espn = k["espn"]
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
            # Always an object: a build has a published field or it has stopped, so
            # every golfer has been looked for. What varies is whether they were found,
            # and `match` says which of the four ways that went.
            "espn": {
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
            },
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
        "competition_id": str(uuid.uuid5(
            league_mod.NAMESPACE,
            f"competition:{k['league']['league_id']}:{k['event_ticker']}:{k['odds_type']}")),
        "generated_at": k["now"],
        # The inputs that shaped this run rather than facts about the world. The poll
        # interval is one -- somebody typed it -- and `live` mirrors it below so the
        # page reads its own settings in one place rather than two.
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
            # The masthead. `logo` is the NAME of a directory of art under leagues/ --
            # never a path and never an image. finish() settles whether the command
            # line or the league file said it; bundle_frontend.py reads the pictures it
            # names, at export, into the page. Null means this competition has no art,
            # which is a shape the design draws rather than a gap to fill in.
            "logo": k["league"].get("logo"),
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
                # Never zero: a build with no field to join against does not get this
                # far. See read_espn_field.
                "field_size_at_build": len(espn["players"]),
                # Had anybody teed off when this was built? A record of the clock, not
                # an instruction -- the page asks the leaderboard the same question on
                # every poll and believes the answer it gets. False here is the normal
                # case, because a pool is drawn the night before.
                "started_at_build": bool((espn["meta"] or {}).get("started")),
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
        # Always present. A page built the night before polls from the moment it opens,
        # which is the point: it is watching for the first tee time. Until then it draws
        # the groups sheet from the data it already has, and the poll is what tells it
        # to stop. Nothing is re-run to make that happen.
        #
        # There is no name-matching block here. The build has already written an ESPN
        # athlete id onto every golfer it resolved, and the page joins on that id:
        # exact, and incapable of quietly picking the wrong Smith.
        "live": {
            "espn_leaderboard_url": leaderboard,
            "espn_event_id": espn_event.get("event_id"),
            "poll_interval_seconds": k["args"].poll_interval,
        },

        "standings_rules": {
            "description": (
                "Rank each team by the best leaderboard position it holds; break ties on the "
                "next-best golfer, and so on. A team that runs out of golfers loses to one "
                "that has not."
            ),
            # This rule reads a LEADERBOARD, so nothing may run it before play starts.
            # Every golfer in a pre-tournament field lands in tier 1 and comes out
            # ranked on a sortOrder that is not a leaderboard at all -- confidently,
            # and meaninglessly. The gate is upstream, in the page: see `started` in
            # espn_leaderboard.has_started and lib.js's hasStarted.
            "golfer_rank_tiers": {
                "0": "still in the tournament -- rank on the displayed position number (T12 -> 12)",
                "1": ("in the field with no ESPN position: cut, withdrawn, disqualified, or "
                      "not yet teed off. Rank on sortOrder"),
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


def inline_logo(value, base_dir):
    """
    Turn a local team logo into a data: URI so the export is one portable file.

    http(s) URLs and existing data: URIs pass through. A missing file is a warning
    rather than an error: a team is still perfectly playable without a badge, and the
    page draws its initials instead.
    """
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    if not os.path.exists(path):
        print(f"!! logo not found: {value} (looked in {path}) -- the page will render without it")
        return None
    size = os.path.getsize(path)
    if size > MAX_INLINE_LOGO_BYTES:
        print(f"!! logo {value} is {size // 1024} KB, over the {MAX_INLINE_LOGO_BYTES // 1024} KB "
              "inline limit. Left as a path, which will not resolve in the exported page: "
              "shrink it, save it as a JPEG, or host it and use a URL instead.")
        return value
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


def resolve_league_logo(result, args, leagues_dir=None):
    """
    Settle which art slug this competition carries, and say what it found.

    The slug is the only per-league part of the masthead, and it arrives three ways.
    In precedence order, highest first:

      1. `--no-logo` -- this competition has none, say no more.
      2. `--logo SLUG`, handed in when the competition is created.
      3. `logo` in the league file, which is where it usually comes from. A rebuild has
         no league file and takes it from the result file it was handed, which is the
         same answer given once and kept.

    There is no fourth rule. The tool used to ship a default pair and fill them in for
    any league that supplied none, which meant a page could open wearing another
    league's crest; a league with no art now gets a masthead with its name in it, and
    that is a shape the design draws.

    Nothing is read. The slug goes into the result file as a name and stays one until
    export. What this does do is LOOK -- one stat per image -- because a slug that names
    nothing is worth a line here rather than a blank masthead noticed by somebody who
    has already been sent the page.
    """
    if args.no_logo:
        result["league"]["logo"] = None
    elif args.logo:
        result["league"]["logo"] = args.logo

    slug = result["league"]["logo"]
    if not slug:
        return result
    found = league_mod.art_files(slug, leagues_dir)
    missing = [name for name in league_mod.ART_NAMES if not found[name]]
    if len(missing) == len(league_mod.ART_NAMES):
        print(f"!! logo {slug!r} names no art: "
              f"{os.path.join(leagues_dir or league_mod.LEAGUES_DIR, slug)} holds no "
              f"{' or '.join(n + '.png' for n in league_mod.ART_NAMES)}. The page will render "
              "with the league's name and nothing else.")
    elif missing:
        print(f"note: {slug} has no {missing[0]} image; the page draws the half that is there.")
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
    ap.add_argument("--logo", metavar="SLUG",
                    help="the league's art, as the name of a directory under leagues/ holding "
                         "logo.png and banner.png -- `--logo wcw` for leagues/wcw/. Beats a "
                         "`logo` in the league file. The images are read at export and inlined "
                         "into the page, never into this file.")
    ap.add_argument("--no-logo", action="store_true",
                    help="build this competition with no art, whatever the league file says")
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
                         "Defaults to match-review.json beside --output.")
    ap.add_argument("--deal-anyway", dest="deal_anyway", action="store_true",
                    help="deal even though some Kalshi golfers were not found in the ESPN "
                         "field. A build normally stops there, because an unfound golfer is "
                         "either a name it will not guess at or somebody who withdrew, and "
                         "dealing puts whichever of them withdrew onto a card at full weight. "
                         "Pass this when you have looked and still cannot say -- an absence "
                         "recorded wrongly is worse. Ignored by a rebuild, which does not deal.")
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
        # From `generator`, which is where an input somebody typed belongs. A 3.x file
        # built before the first tee time has no `live` block at all, so reading it
        # there would drop the setting on exactly the rebuild that upgrades that file;
        # pre-2.0 files have only the `live` copy, so it is still the fallback.
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
    Catch a bad `--logo` now rather than forty seconds into a build.

    A slug typed on the command line is a thing somebody meant, so a typo in one is an
    error and not the shrug a league file's `logo` gets. Getting it at the top matters
    because everything between here and the end of the build is network: the Kalshi
    fetch and the ESPN join both run first, and finding out afterwards that the slug was
    `wwc` means running them again.
    """
    if args.logo and args.no_logo:
        parser.error("--logo and --no-logo ask for opposite things")
    if not args.logo:
        return
    if args.logo != league_mod.slugify(args.logo):
        parser.error(f"--logo {args.logo} is a slug -- the name of a directory under leagues/ "
                     "holding logo.png and banner.png, like `wcw` -- not a path")
    if not any(league_mod.art_files(args.logo).values()):
        parser.error(f"--logo {args.logo} names no art: "
                     f"{os.path.join(league_mod.LEAGUES_DIR, args.logo)} holds no logo.png or "
                     "banner.png")


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
        # Said rather than refused, because the flag is valid on the other half of this
        # branch: --regroup deals, so --regroup --deal-anyway means something. A plain
        # rebuild deals nothing, so there is nothing for it to override -- and somebody
        # who typed it was expecting an effect and should be told they did not get one.
        if args.deal_anyway and not args.regroup:
            print("note: --deal-anyway has no effect on a rebuild. It overrides the check that "
                  "stops a DEAL when a golfer cannot be found in the ESPN field, and a rebuild "
                  "does not deal -- the golfers named below are already on cards and stay there.")
        try:
            if args.regroup:
                build(args, league=(league_mod.load_league(args.league) if args.league
                                    else league_from_result(result)),
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
