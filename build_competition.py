#!/usr/bin/env python3
"""
build_competition.py -- one league + one tournament + one market -> one result file.

This is the step between the engine and the scoreboard. It resolves the tournament
on both APIs, pulls the odds, runs the partitioner, deals the groups out to the
league's teams, joins the field onto ESPN, and writes a single JSON file that is the
source of truth for that competition:

    python build_competition.py --league leagues/sunday-fivesome.json \
                                --tournament "Wyndham Championship" \
                                --odds winner

Everything the scoreboard needs is in that file, and everything needed to audit it
later is in it too: which endpoints were read, when, at what prices, who was
excluded and why, and how good the partition was. Nothing in the frontend re-derives
a number that this file already states.

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
from datetime import datetime, timezone

import espn_leaderboard
import group as grouper_cli
import groupers
import kalshi_odds
import league as league_mod

SCHEMA_VERSION = "1.0"

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

# Local logos are inlined so the exported bundle is one portable file. A logo bigger
# than this is a mistake rather than a choice -- a 2 MB PNG lands in every copy of the
# result JSON and every copy of the HTML built from it.
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

def build(args):
    started = time.time()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    league = league_mod.load_league(args.league)
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
        print(f"ESPN:    {espn_event['event_id']}  {espn_event.get('name')}  "
              f"[{espn_event.get('state', '?')}]")

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

    # -- ESPN join -----------------------------------------------------------
    aliases = load_aliases(args.alias_file)
    espn_players, espn_meta, match_report, matches = [], None, None, {}
    if espn_event:
        try:
            payload = espn_leaderboard.fetch_leaderboard(espn_event["event_id"], args.espn_league)
            espn_meta, espn_players = espn_leaderboard.parse_leaderboard(payload)
        except Exception as exc:                   # noqa: BLE001
            print(f"!! could not read the ESPN leaderboard: {exc}", file=sys.stderr)
    if espn_players:
        matches, match_report = espn_leaderboard.match_field(
            [g["golfer_name"] for g in field], espn_players, aliases)
        print(f"ESPN join: {match_report['matched']}/{match_report['requested']} matched "
              f"({match_report['matched_exact']} exact, {match_report['matched_initial_last']} "
              f"initial+last, {match_report['matched_alias']} alias); "
              f"{len(match_report['unresolved'])} unresolved")
        if match_report["unresolved"]:
            print("  not in the ESPN field: " + ", ".join(match_report["unresolved"][:8])
                  + (" ..." if len(match_report["unresolved"]) > 8 else ""))
    else:
        print("ESPN join: deferred -- the field is not posted yet "
              f"({(espn_meta or {}).get('state', 'unknown')}). The scoreboard matches by name "
              "at runtime using the same two tiers.")

    # -- assemble ------------------------------------------------------------
    result = assemble(
        now=now, args=args, league=league, teams=teams, team_groups=team_groups,
        odds_type=odds_type, series=series, market_label=market_label, exclusive=exclusive,
        event_ticker=event_ticker, tournament_name=tournament_name, season=season,
        espn_event=espn_event, espn_meta=espn_meta, matches=matches, match_report=match_report,
        espn_field_size=len(espn_players),
        field=field, devigged=devigged, weighted=weighted, excluded=excluded,
        liquidity=liquidity, raw_sum=raw_sum, tick_structures=tick_structures,
        report=report, groups=groups, order=order, seed=seed, aliases=aliases,
    )

    # Logos last, and against the league file's own directory: a logo path in a league
    # file is relative to that file, not to wherever the build was run from.
    league_dir = os.path.dirname(os.path.abspath(args.league))
    for team in result["teams"]:
        team["team_logo"] = inline_logo(team.get("team_logo"), league_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResult -> {args.output}  ({os.path.getsize(args.output) // 1024} KB)")

    learned = learn_aliases(matches, aliases)
    if learned:
        print(f"{len(learned)} new name alias(es) resolved by initial+last: "
              + ", ".join(f"{k} -> {v}" for k, v in list(learned.items())[:5]))
        if args.update_aliases:
            save_aliases(args.alias_file, {**aliases, **learned})
            print(f"  written to {args.alias_file} (the one repo file a build touches)")
        else:
            print(f"  not saved. Re-run with --update-aliases to pin them into {args.alias_file}.")

    print_groups(result)
    print(f"\nBuilt in {time.time() - started:.1f}s")
    return result


def assemble(**k):
    """
    The result file.

    Grouped by what a reader is asking. `league` / `teams` / `golfers` answer "who has
    whom". `odds_snapshot` and `grouping` answer "why". `sources` answers "where did
    this come from and can I check it". `live` answers "what may the page fetch".
    """
    field, devigged, weighted = k["field"], k["devigged"], k["weighted"]
    weight_by_name = {g["golfer_name"]: g["odds"] for g in weighted}
    excluded_names = {e["golfer_name"] for e in k["excluded"]}
    team_of = {}
    for team_id, golfers in k["team_groups"].items():
        for g in golfers:
            team_of[g["golfer_name"]] = team_id

    golfers_out = []
    for g in field:
        name = g["golfer_name"]
        hit = (k["matches"] or {}).get(name)
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
                "grouping_weight": (round(weight_by_name[name], WEIGHT_PRECISION)
                                    if name in weight_by_name else None),
            },
            "espn": {
                "athlete_id": player["athlete_id"] if player else None,
                "display_name": player["name"] if player else None,
                "headshot": player["headshot"] if player else None,
                "country": player["country"] if player else None,
                # "deferred" and "unresolved" are different facts. Deferred means the
                # field did not exist yet, which is normal before Thursday and is
                # finished by the page at runtime. Unresolved means the field existed
                # and this golfer is not in it -- they withdrew.
                "match": hit["match"] if hit else (
                    "unresolved" if k["espn_field_size"] else "deferred"),
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
        "generator": {
            "tool": "golf-odds-grouper/build_competition.py",
            "git_commit": _git_commit(),
            "seed": k["seed"],
        },

        "league": {
            "league_id": k["league"]["league_id"],
            "league_name": k["league"]["league_name"],
            "league_slug": k["league"]["league_slug"],
            "source_file": k["league"]["source_file"],
            "team_count": len(k["teams"]),
        },
        "teams": teams_out,
        "golfers": golfers_out,

        "tournament": {
            "name": k["tournament_name"],
            "season": int(k["season"]),
            "start": espn_event.get("start") or (k["espn_meta"] or {}).get("start"),
            "end": espn_event.get("end") or (k["espn_meta"] or {}).get("end"),
            "state_at_build": espn_event.get("state") or (k["espn_meta"] or {}).get("state"),
            "course": {
                "name": (k["espn_meta"] or {}).get("course"),
                "par": (k["espn_meta"] or {}).get("par"),
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
                "price_mode": k["args"].price,
                "price_level_structure": k["tick_structures"],
                "browser_reachable": False,
                "browser_note": (
                    "Kalshi allowlists request origins: a GET carrying "
                    "Origin: https://kalshi.com returns 200, every other origin returns 403 "
                    "with no CORS headers (localhost, GitHub Pages and file:// all measured "
                    "2026-08-03). A static page cannot read live odds from Kalshi directly; "
                    "it needs a relay. See live.kalshi_proxy_url_template."
                ),
            },
            "espn": {
                "league": k["args"].espn_league,
                "event_id": espn_event.get("event_id"),
                "leaderboard_endpoint": leaderboard,
                "scoreboard_endpoint": espn_leaderboard.scoreboard_url(k["season"], k["args"].espn_league),
                "browser_reachable": True,
                "browser_note": "ESPN sends access-control-allow-origin: * -- fetch it directly.",
                "field_available_at_build": bool(k["espn_field_size"]),
                "field_size_at_build": k["espn_field_size"],
                "match_report": k["match_report"],
            },
        },

        "odds_snapshot": {
            "captured_at": k["now"],
            "price_mode": k["args"].price,
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

        "live": {
            "espn_leaderboard_url": leaderboard,
            "poll_interval_seconds": k["args"].poll_interval,
            "kalshi_markets_url": markets_endpoint,
            "kalshi_proxy_url_template": k["args"].kalshi_proxy or None,
            "name_match": {
                "strategy": ["alias", "normalized_exact", "first_initial_and_last_name"],
                "normalization": ("NFKD, drop combining marks, lowercase, hyphen and apostrophe "
                                  "to space, drop jr/sr/ii/iii/iv/v, drop non-letters, collapse spaces"),
                "aliases": k["aliases"],
            },
        },

        "standings_rules": {
            "description": (
                "Rank each team by the best leaderboard position it holds; break ties on the "
                "next-best golfer, and so on. A team that runs out of golfers loses to one "
                "that has not."
            ),
            "golfer_rank_tiers": {
                "0": "still in the tournament -- rank on the displayed position number (T12 -> 12)",
                "1": "cut, withdrawn or disqualified -- no ESPN position, rank on sortOrder",
                "2": "priced by Kalshi but never in the ESPN field",
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
            "note": ("Kalshi golfer name -> ESPN displayName, for names the two-tier match "
                     "cannot settle on its own. Learned by --update-aliases; safe to edit."),
            "aliases": dict(sorted(aliases.items())),
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")


def learn_aliases(matches, known):
    """Every non-exact match is worth remembering; an exact one teaches nothing."""
    return {
        name: hit["player"]["name"]
        for name, hit in (matches or {}).items()
        if hit["match"] == "initial_last" and name not in known
    }


def inline_logo(value, base_dir):
    """
    Turn a local logo path into a data: URI so the export is one portable file.

    http(s) URLs and existing data: URIs pass through. A missing file is a warning
    rather than an error: a league is still perfectly playable without a crest.
    """
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    if not os.path.exists(path):
        print(f"!! logo not found: {value} (looked in {path}) -- the team will render without one")
        return None
    size = os.path.getsize(path)
    if size > MAX_INLINE_LOGO_BYTES:
        print(f"!! logo {value} is {size // 1024} KB, over the {MAX_INLINE_LOGO_BYTES // 1024} KB "
              "inline limit. Left as a path; host it and use a URL instead.")
        return value
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


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
    ap.add_argument("--league", required=True, help="path to the league JSON")
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
    ap.add_argument("--update-aliases", action="store_true",
                    help="write newly learned golfer name aliases back to the alias file")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds the scoreboard should wait between ESPN polls (default 60)")
    ap.add_argument("--kalshi-proxy", metavar="URL_TEMPLATE",
                    help="CORS relay for live odds, with {url} where the encoded Kalshi URL goes. "
                         "Without one the scoreboard shows the snapshot only -- Kalshi will not "
                         "answer a browser.")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.tournament and not args.kalshi_event:
        build_parser().error("give --tournament NAME or --kalshi-event TICKER")

    try:
        build(args)
    except ValueError as exc:
        # A bad league file is the common case here, and it already carries a sentence
        # naming the file and the field. A traceback would bury it.
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
