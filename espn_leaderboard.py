#!/usr/bin/env python3
"""
espn_leaderboard.py -- resolve an ESPN golf event and read its live leaderboard.

ESPN is the scoring source. Kalshi prices the field; ESPN says who is beating whom.
Both are unauthenticated GETs, and ESPN sends `access-control-allow-origin: *`, so
the frontend can poll this endpoint directly from the browser. (Kalshi cannot --
see the CORS note in docs/FRONTEND-SPEC.md.)

    https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga&event=<id>
    https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates=<YYYY>

WHAT THE PAYLOAD ACTUALLY MEANS -- measured, not assumed
--------------------------------------------------------
Measured against a live Rocket Classic R2 payload (147 competitors, espn-api/lb.json)
and the same event's final payload:

  * `competitors[]` is NOT in rank order. Sort it yourself.

  * `sortOrder` is the live rank and is a TOTAL order over the whole field, 1..N.
    Zero inversions against the live to-par total; 29 inversions against
    `score.displayValue`. It also puts every cut player (74..147) below every
    made-cut player (1..73), so it needs no special-casing to stay sane after a cut.

  * `score.displayValue` counts COMPLETED ROUNDS ONLY. Mid-round it was wrong for
    42 of 147 players. Never rank on it. The live total is the sum of the
    linescore `displayValue`s.

  * `linescores[]` carries STUB entries for rounds not yet played -- no `value`,
    no `displayValue`. Filter before summing. A withdrawn round reads "-".

  * `status.position.displayName` is live-accurate and carries ties ("T1", "T8"),
    and is "-" for every cut player. It is the right thing to DISPLAY and the right
    thing to rank the league on; `sortOrder` is the tie-break underneath it.

  * Current round is `competitions[0].status.period`, not `event.status`.

  * A `pre` event PUBLISHES ITS FIELD, and publishes it early. Measured 2026-08-04
    against the Wyndham Championship (event 401811961, first round 2026-08-06):
    `state` is "pre", `period` is 0, and `competitors[]` already holds all 147
    players with athlete ids, headshots, countries and tee times. Two days out.
    This used to be the opposite -- a `pre` event returned zero competitors -- and
    that one fact used to split this program into a groups build and a live one.
    It does not any more: the join can be made the night before, so it is.

  * What a `pre` event does NOT publish is a POSITION. All 147 came back with
    `position.displayName == "-"`, no linescores worth summing, and `thru` 0 --
    while `sortOrder` was still a dense 1..147. So the field is joinable long
    before it is rankable, and ranking on what a `pre` payload offers would order
    the league by ESPN's pre-tournament sort and present it as a leaderboard.
    `meta["started"]` below is the guard, and it is the only thing that decides
    whether a position is ever shown.

THE NAME JOIN
-------------
Kalshi's `custom_strike.golf_competitor` UUID is stable across events and across
market series, so it is the right key for a golfer -- but it is not an ESPN id, and
ESPN publishes no Kalshi id. The join is by name, against ONE published field, in
three tiers that are each either exact or explicit:

    decision   a reviewed binding recorded against this competition. Binds a Kalshi
               name to an ESPN athlete id, or records that the golfer is genuinely
               not in the field. Always wins, because somebody looked.
    alias      an explicit Kalshi-name -> ESPN-display-name mapping from the alias
               file. Reusable across tournaments; learned from decisions.
    exact      the two normalised display names are equal.

There is deliberately NO fuzzy tier. Measured on Rocket Classic, 151 Kalshi markets
against 147 ESPN competitors, `exact` alone resolves 139. The 8 it does not are
formal-vs-familiar first names -- Zachary/Zach Bauchou, Cameron/Cam Davis,
Kris/Kristoffer Ventura, Nicolas/Nico Echavarria, Matthew/Matt McCarty,
Benjamin/Ben James, Jordan L./Jordan Smith, Hao-Tong/Haotong Li -- and a
first-initial-and-last-name rule would bind all 8 correctly. It would also bind the
day a field holds two J. Smiths, and nothing downstream could tell the two cases
apart. So that rule is kept as a SUGGESTION (see suggest_matches) and never as a
match: an unresolved name comes back with ranked candidates attached, a reviewer
settles it, and the settlement is recorded where it can be read.

The remaining 4 of 151 -- Daniel Brown, Taylor Moore, Brooks Koepka, Jason Day --
were genuinely absent from the ESPN field, withdrawn before play. A review confirms
that too, and "confirmed absent" is a different fact from "we could not find him".
Only the first is knowable without looking.

WHEN THERE IS NO FIELD TO JOIN AGAINST
--------------------------------------
A field that comes back empty used to be the ordinary state of a Wednesday, and the
build put out a groups-and-odds file with no ESPN block in it. It is now an error.
ESPN posts a field about two days before the first round, so an empty one means the
event id is wrong, the run is very early, or the read failed -- and a build that went
ahead anyway would produce a page with no athlete ids, which can never score however
long anybody waits. build_competition.read_espn_field stops instead.

The window has a far edge, and it is real: measured 2026-08-04, the tournament two
days out had its 147 competitors and the one nine days out had none. So a pool cannot
be drawn a week early. It can be drawn the night before, which is when pools are
drawn.

Matching a Kalshi field against LAST month's leaderboard to recover athlete ids is
still not attempted, and now never needs to be: it answers a question nobody asked
(the page needs scores, and those come from this week) at the cost of a join whose
correctness cannot be checked against anything.
"""

import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

LEADERBOARD_URL = "https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/{league}/scoreboard"
DEFAULT_LEAGUE = "pga"
UA = "Mozilla/5.0 (compatible; golf-odds-grouper/1.0)"

# Suffixes that are part of a legal name but never part of an identity.
_SUFFIXES = r"\b(jr|sr|ii|iii|iv|v)\b\.?"

# Letters NFKD does not decompose, because they are letters in their own right rather
# than a base plus an accent. A golf field is full of them -- Rasmus Højgaard and
# Thorbjørn Olesen are both in the measured Wyndham field -- and one source writing
# "Hojgaard" while the other writes "Højgaard" is exactly the miss this table prevents.
_TRANSLITERATE = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a",
    "ð": "d", "Ð": "d", "þ": "th", "Þ": "th", "ł": "l", "Ł": "l",
    "đ": "d", "Đ": "d", "ß": "ss", "œ": "oe", "Œ": "oe", "ı": "i",
})

# Words that appear in half the golf calendar. They are stripped before a tournament
# name is scored, so "Wyndham" decides the match rather than "Championship".
_GENERIC_TOURNAMENT_WORDS = {
    "the", "championship", "championships", "classic", "open", "invitational",
    "tournament", "of", "at", "in", "and", "cup", "golf", "presented", "by",
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url, **params):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ESPN returned HTTP {exc.code} for {url}") from exc


def leaderboard_url(event_id=None, league=DEFAULT_LEAGUE):
    params = {"league": league}
    if event_id:
        params["event"] = str(event_id)
    return f"{LEADERBOARD_URL}?{urllib.parse.urlencode(params)}"


def scoreboard_url(season, league=DEFAULT_LEAGUE):
    return f"{SCOREBOARD_URL.format(league=league)}?{urllib.parse.urlencode({'dates': season})}"


def fetch_leaderboard(event_id=None, league=DEFAULT_LEAGUE):
    """Raw leaderboard payload. No event id means 'whatever ESPN thinks is current'."""
    params = {"league": league}
    if event_id:
        params["event"] = str(event_id)
    return _get(LEADERBOARD_URL, **params)


def season_events(season, league=DEFAULT_LEAGUE):
    """
    Every event in a season: id, name, dates, state.

    This is the only endpoint that lists a whole season -- the leaderboard endpoint
    answers about one event at a time and defaults to the current one.
    """
    data = _get(SCOREBOARD_URL.format(league=league), dates=str(season))
    out = []
    for e in data.get("events") or []:
        out.append({
            "event_id": e.get("id"),
            "name": e.get("name"),
            "short_name": e.get("shortName"),
            "start": e.get("date"),
            "end": e.get("endDate"),
            "state": ((e.get("status") or {}).get("type") or {}).get("state"),
        })
    return out


def season_calendar(season, league=DEFAULT_LEAGUE, on=None):
    """
    Every event in a season -- id, name, start, end -- for 12 KB instead of 35 MB.

    `scoreboard?dates=<YYYY>` embeds every competitor of every event that has already
    been played: 35 MB in August 2026, and it grows all season. The same endpoint asked
    for ONE DAY returns only that day's events -- but `leagues[0].calendar` still lists
    the WHOLE season regardless. Measured 2026-08-03: `dates=20250701` returned all 49
    events of the 2025 season in a 12 KB response.

    The calendar carries no status, which is why season_events() still exists: resolving
    a tournament by name wants the state as well. It carries dates, and dates are all
    that is needed to find the tournaments that came BEFORE this one.

    `on` is the day to anchor the request on -- any date inside the season. The default
    is 1 July, which is mid-season on every golf calendar and cheap on most of them; the
    response is larger on a day with a tournament in it, and the calendar is identical
    either way.
    """
    data = _get(SCOREBOARD_URL.format(league=league), dates=str(on or f"{season}0701"))
    rows = []
    for e in ((data.get("leagues") or [{}])[0].get("calendar") or []):
        if isinstance(e, dict) and e.get("id"):
            rows.append({
                "event_id": str(e["id"]),
                "name": e.get("label"),
                "start": e.get("startDate"),
                "end": e.get("endDate"),
            })
    rows.sort(key=lambda e: e.get("start") or "")
    return rows


# ---------------------------------------------------------------------------
# Event resolution
# ---------------------------------------------------------------------------

def tournament_tokens(text):
    """
    (distinctive tokens, all tokens) for a tournament name.

    Both, because dropping the generic words is right until the query IS generic
    words: "The Open Championship" is three of them and nothing else, so a scorer that
    only ever looks at the distinctive set can never match it at all. Also drops a bare
    year, so "2026 Wyndham Championship" and "Wyndham Championship" agree.
    """
    words = [w for w in normalize_name(re.sub(r"\b(19|20)\d{2}\b", " ", str(text or ""))).split() if w]
    return set(words) - _GENERIC_TOURNAMENT_WORDS, set(words)


def score_name(query, candidate):
    """
    0..1 on how well a tournament name matches a query.

    Jaccard over distinctive tokens, so "wyndham" scores 1.0 against "2026 Wyndham
    Championship" and 0 against everything else on the calendar. When the query has no
    distinctive tokens at all it falls back to the full token set, halved -- a match on
    generic words is a real signal but never a confident one.
    """
    want_key, want_all = tournament_tokens(query)
    have_key, have_all = tournament_tokens(candidate)
    if normalize_name(query) == normalize_name(candidate):
        return 1.0
    if want_key and have_key:
        overlap = want_key & have_key
        return len(overlap) / len(want_key | have_key) if overlap else 0.0
    if want_all and have_all:
        overlap = want_all & have_all
        return 0.5 * len(overlap) / len(want_all | have_all) if overlap else 0.0
    return 0.0


def ambiguous(ranked, margin=0.05):
    """
    True when the top two candidates are too close to pick between.

    A tournament resolved wrong produces a completely valid-looking grouping of the
    wrong field, so this exists to make the caller stop and ask rather than commit.
    """
    return len(ranked) > 1 and (ranked[0]["score"] - ranked[1]["score"]) < margin


def resolve_event(query, season, league=DEFAULT_LEAGUE, events=None):
    """
    Find the ESPN event whose name best matches `query`.

    Returns (best, ranked). `ranked` is every candidate that scored above zero, so a
    caller can show the near-misses instead of silently committing to a guess -- and
    `ambiguous(ranked)` says whether it should.
    """
    events = season_events(season, league) if events is None else events
    if not events:
        raise RuntimeError(f"ESPN listed no {league} events for season {season}")

    scored = []
    for e in events:
        score = score_name(query, e.get("name") or "")
        if score >= 1.0:
            return dict(e, score=1.0), [dict(e, score=1.0)]
        if score > 0:
            scored.append(dict(e, score=score))

    scored.sort(key=lambda e: (-e["score"], e.get("start") or ""))
    return (scored[0] if scored else None), scored


# ---------------------------------------------------------------------------
# Leaderboard parsing
# ---------------------------------------------------------------------------

def to_par(value):
    """'E' -> 0, '-2' -> -2, '+3' -> 3, '-' or missing -> None."""
    s = str(value).strip()
    if s == "E":
        return 0
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return None


def fmt_par(n):
    return "E" if n == 0 else format(n, "+d")


def position_number(display_name):
    """
    'T12' -> 12, '3' -> 3, '-' or None -> None.

    None is the honest answer for a cut, withdrawn or disqualified player: ESPN gives
    them no position, and inventing one would put them in the standings ahead of
    someone still playing.
    """
    if not display_name:
        return None
    m = re.match(r"^T?(\d+)$", str(display_name).strip())
    return int(m.group(1)) if m else None


def has_started(state, players):
    """
    Has anybody teed off? -> bool.

    ESPN posts the field about two days before the first round, so "there are players
    in this payload" stopped meaning "this tournament is under way" and a page that
    conflated the two would rank the league off a pre-tournament sort order. This is
    the one question that separates them, and `frontend/lib.js` asks it the same way.

    Two signals, either of which is enough, because they fail in opposite directions.
    `state` is ESPN's own answer and the one to trust -- but it is read off the event
    envelope, and an envelope that is missing or stale would silently rank a field
    nobody has played. A golfer holding an actual position is proof from the field
    itself. Requiring BOTH would blank the board on a good payload with an odd
    envelope; requiring neither is what this exists to prevent.
    """
    return state in ("in", "post") or any(p.get("position_number") is not None
                                          for p in players)


def parse_leaderboard(payload):
    """
    -> (meta, players). `players` is sorted by sortOrder, i.e. by live rank.

    Every field the standings need is computed here rather than read off the payload,
    because the fields ESPN publishes for them are stale mid-round -- and, since ESPN
    began posting fields early, because a payload full of players is not by itself a
    tournament in progress. See the module docstring for the measurements.
    """
    events = payload.get("events") or []
    if not events:
        return None, []
    ev = events[0]
    comp = (ev.get("competitions") or [{}])[0]
    course = (ev.get("courses") or [{}])[0]

    meta = {
        "event_id": ev.get("id"),
        "event": ev.get("name"),
        "short_name": ev.get("shortName"),
        "course": course.get("name"),
        "par": course.get("shotsToPar"),
        "round": (comp.get("status") or {}).get("period"),
        "detail": ((comp.get("status") or {}).get("type") or {}).get("detail"),
        "state": ((ev.get("status") or {}).get("type") or {}).get("state"),
        "completed": bool(((ev.get("status") or {}).get("type") or {}).get("completed")),
        "start": ev.get("date"),
        "end": ev.get("endDate"),
        "purse": ev.get("displayPurse"),
    }

    players = []
    for c in comp.get("competitors") or []:
        athlete = c.get("athlete") or {}
        st = c.get("status") or {}
        pos = st.get("position") or {}

        rounds = []
        for ls in c.get("linescores") or []:
            if "displayValue" not in ls:
                continue                       # stub for a round not yet played
            v = to_par(ls.get("displayValue"))
            if v is None:
                continue                       # "-": no score for that round
            rounds.append({"round": ls.get("period"), "to_par": v, "strokes": ls.get("value")})

        live = sum(r["to_par"] for r in rounds) if rounds else None
        players.append({
            "athlete_id": athlete.get("id"),
            "name": athlete.get("displayName"),
            "short_name": athlete.get("shortName"),
            "headshot": (athlete.get("headshot") or {}).get("href"),
            "flag": (athlete.get("flag") or {}).get("href"),
            "country": (athlete.get("flag") or {}).get("alt"),
            "amateur": bool(c.get("amateur")),
            "sort_order": c.get("sortOrder", 9999),
            "position": pos.get("displayName"),
            "position_number": position_number(pos.get("displayName")),
            "tied": bool(pos.get("isTie")),
            "thru": st.get("displayThru") or st.get("thru"),
            "tee_time": st.get("teeTime"),
            "round": st.get("period"),
            "status": (st.get("type") or {}).get("name"),
            "status_short": st.get("displayValue"),
            "to_par": live,
            "to_par_display": fmt_par(live) if live is not None else "-",
            "stale_to_par": (c.get("score") or {}).get("displayValue"),
            "rounds": rounds,
        })

    players.sort(key=lambda p: p["sort_order"])
    # Derived rather than read, and derived HERE so there is one answer to it. A field
    # exists from about two days out; a leaderboard does not exist until somebody hits
    # a ball. Everything that ranks is gated on this. See has_started.
    meta["started"] = has_started(meta["state"], players)
    return meta, players


# ---------------------------------------------------------------------------
# The name join
# ---------------------------------------------------------------------------

def normalize_name(name):
    """
    Fold a display name to its comparable core.

    Strips accents, case, punctuation, generational suffixes and hyphenation, and
    transliterates the letters NFKD leaves alone. Two rules are less obvious:

    Hyphens become spaces rather than nothing, so "Hao-Tong Li" and "Haotong Li" meet
    at the tier-2 key rather than only one of them reaching it.

    Runs of consecutive single letters are joined, so "C.T. Pan" and "CT Pan" agree,
    as do "J.J. Spaun"/"JJ Spaun" and "J.T. Poston"/"JT Poston". Only CONSECUTIVE
    singles join: "Jordan L. Smith" keeps its middle initial as its own token, which
    is what lets tier 2 drop it and match "Jordan Smith".
    """
    s = str(name or "").translate(_TRANSLITERATE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace("'", "").replace("’", "")
    s = re.sub(_SUFFIXES, " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(_join_initials(s.split()))


def _join_initials(parts):
    out, prev_single = [], False
    for part in parts:
        single = len(part) == 1
        if single and prev_single:
            out[-1] += part
        else:
            out.append(part)
        prev_single = single
    return out


def initial_last_key(name):
    """
    (first initial, last name) -- the tier-2 key.

    Resolves every formal-vs-familiar first name a golf field throws up (Zachary/Zach,
    Cameron/Cam, Matthew/Matt, Benjamin/Ben, Nicolas/Nico, Kristoffer/Kris) and any
    stray middle initial, without the false positives a pure last-name match invites.
    Measured collisions inside a real 147-player field: zero.
    """
    parts = normalize_name(name).split()
    if len(parts) < 2:
        return None
    return f"{parts[0][0]}|{parts[-1]}"


def build_index(players):
    """
    Two ways to reach an athlete in a published field: by ESPN id, and by normalised
    display name.

    A normalised name that two athletes in the SAME field share is dropped rather than
    resolved to whichever the payload listed first. It has not happened in a measured
    field, and the day it does, a coin flip is the wrong way to decide which of two
    people is on somebody's team. Both come back unresolved and go to review.
    """
    by_id, exact = {}, {}
    for p in players:
        if p.get("athlete_id"):
            by_id[str(p["athlete_id"])] = p
        if p.get("name"):
            exact.setdefault(normalize_name(p["name"]), []).append(p)
    return {
        "by_id": by_id,
        "exact": {k: v[0] for k, v in exact.items() if len(v) == 1},
        "ambiguous": sorted(k for k, v in exact.items() if len(v) > 1),
    }


def match_golfer(name, index, aliases=None):
    """
    -> (espn_player | None, how). `how` is one of alias / exact / unresolved.

    An alias maps a Kalshi name to an ESPN display name and is checked first, so a
    name somebody has already settled stays settled. Both tiers are exact matches on a
    normalised string; neither guesses.
    """
    aliases = aliases or {}
    target = aliases.get(name) or aliases.get(normalize_name(name))
    if target:
        hit = index["exact"].get(normalize_name(target))
        if hit:
            return hit, "alias"

    hit = index["exact"].get(normalize_name(name))
    if hit:
        return hit, "exact"

    return None, "unresolved"


def apply_decision(name, decision, index):
    """
    One reviewed decision -> (player | None, how, problem).

    A decision is `{"athlete_id": ...}` to bind, or `{"absent": true}` to record that
    the golfer is not in this field. `espn_name` alone is accepted too, because a
    reviewer editing the file by hand has the name in front of them and not the id.

    `problem` is a sentence, not an exception. A decision that names an athlete who is
    not in the field is a mistake worth shouting about, but it is not a reason to
    abandon a build of 150 golfers -- so it is reported, and the name falls through to
    the automatic tiers, which may well resolve it correctly on their own.
    """
    if decision.get("absent"):
        return None, "absent", None

    athlete_id = decision.get("athlete_id")
    if athlete_id is not None:
        hit = index["by_id"].get(str(athlete_id))
        if hit:
            return hit, "decision", None
        return None, None, (f"{name}: reviewed decision names ESPN athlete {athlete_id}, who is "
                            "not in this field. Ignored.")

    espn_name = decision.get("espn_name")
    if espn_name:
        hit = index["exact"].get(normalize_name(espn_name))
        if hit:
            return hit, "decision", None
        return None, None, (f"{name}: reviewed decision names ESPN player {espn_name!r}, who is "
                            "not in this field under that name. Ignored.")

    return None, None, (f"{name}: reviewed decision carries neither athlete_id, espn_name nor "
                        "absent, so it decides nothing. Ignored.")


def match_field(names, players, aliases=None, decisions=None):
    """
    Join Kalshi golfer names onto THIS WEEK'S published ESPN field.

    Tiers, in the order they are tried: a reviewed `decision`, an explicit `alias`,
    then a normalised `exact` match. See the module docstring for why there is no
    fourth.

    Returns (matches, report). `matches` maps the Kalshi name to the ESPN player and
    the tier that found it. `report` counts the tiers and, more usefully, splits what
    is left over into two piles that mean different things:

        absent       reviewed and confirmed not to be in the field. They withdrew.
                     A verified fact.
        unresolved   nobody has looked yet. Might be a withdrawal, might be a name
                     this join cannot spell. Not a fact, a question.

    Both score nothing and neither is counted, so they are the same to the standings
    and completely different to a reader deciding whether the build is finished.

    Done in two passes, and the order matters. One ESPN athlete cannot be on two
    teams, so when two Kalshi names reach the same person the second is refused --
    and which one is "second" must not depend on how Kalshi happened to sort its
    markets. Every reviewed decision is settled first, then the automatic tiers fill
    in around what is left. So a decision always beats a name that merely spells the
    same, which is the right way round: somebody looked at one of them.
    """
    index = build_index(players)
    decisions = decisions or {}
    matches, absent, problems = {}, [], []
    counts = {"decision": 0, "alias": 0, "exact": 0}
    claimed = {}
    pending = []

    def claim(name, player, how):
        """-> True if this golfer took the athlete, False if somebody already had them."""
        athlete_id = str(player.get("athlete_id"))
        if athlete_id in claimed:
            problems.append(f"{name}: resolves to ESPN athlete {athlete_id} "
                            f"({player.get('name')}), who is already held by "
                            f"{claimed[athlete_id]!r}. Left unresolved.")
            return False
        claimed[athlete_id] = name
        counts[how] += 1
        matches[name] = {"player": player, "match": how}
        return True

    # Pass 1: what somebody decided.
    for name in names:
        decision = decisions.get(name)
        if not decision:
            pending.append(name)
            continue
        player, how, problem = apply_decision(name, decision, index)
        if problem:
            # The decision decides nothing, but the golfer might still resolve on
            # their own -- refusing to try would turn one typo into a golfer who
            # scores nothing all week.
            problems.append(problem)
            pending.append(name)
        elif how == "absent":
            absent.append(name)
        elif not claim(name, player, how):
            pending.append(name)

    # Pass 2: what the two exact tiers can settle around them.
    unresolved = []
    for name in pending:
        player, how = match_golfer(name, index, aliases)
        if player is None or not claim(name, player, how):
            unresolved.append(name)

    return matches, {
        "espn_field_size": len(players),
        "requested": len(names),
        "matched": len(matches),
        **{f"matched_{k}": v for k, v in counts.items()},
        "absent": absent,
        "unresolved": unresolved,
        "ambiguous_names": index["ambiguous"],
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Suggesting, for the names the join will not guess at
# ---------------------------------------------------------------------------

# Below this, a suggestion is noise, and a review file padded with noise is a review
# file nobody reads.
SUGGESTION_FLOOR = 0.45

# Spelling similarity ALONE has to clear a much higher bar than a structural signal
# does. Measured against the 147-player field: "Xavier Quetzalcoatl" scores 0.47 on
# letters against "Daniel Azallion", who shares not one thing with it. Two unrelated
# golfers land in the 0.4-0.55 range routinely, so anything that shares no part of a
# name has to look like a typo of it -- not merely have some letters in common.
SPELLING_FLOOR = 0.75


def _score_suggestion(query, candidate):
    """
    (score, why) for one candidate against one unmatched name.

    The structural signals come first and set a floor, because they are the ones that
    are actually diagnostic: "same first initial and last name" is the whole of the
    formal-vs-familiar problem (Zachary/Zach, Cameron/Cam, Nicolas/Nico) and it is
    worth ranking above a closer-looking string that shares no name part. Spelling
    similarity breaks ties underneath, and stands alone for the transliteration misses
    that share no whole token at all.

    `why` is the point of the exercise as much as the score is. A reviewer confirming
    a binding wants to see the reason it was proposed, not a number.
    """
    q, c = normalize_name(query), normalize_name(candidate)
    ratio = difflib.SequenceMatcher(None, q, c).ratio()
    qp, cp = q.split(), c.split()

    key = initial_last_key(query)
    if key and key == initial_last_key(candidate):
        return round(max(0.90, ratio), 3), "same first initial and last name"
    if qp and cp and qp[-1] == cp[-1]:
        return round(max(0.70, ratio), 3), "same last name"
    if qp and cp and qp[0] == cp[0]:
        return round(max(0.55, ratio), 3), "same first name"
    shared = sorted(set(qp) & set(cp))
    if shared:
        return round(max(0.50, ratio), 3), "shares " + ", ".join(shared)
    if ratio >= SPELLING_FLOOR:
        return round(ratio, 3), "similar spelling"
    # Nothing in common and it does not even look like a typo. Scoring it zero is how a
    # golfer who is simply not in this field comes back with an EMPTY suggestion list,
    # which is the clearest thing the review file can say about them.
    return 0.0, "nothing in common"


def suggest_matches(name, candidates, limit=3, floor=SUGGESTION_FLOOR):
    """
    The ESPN athletes a reviewer should look at first for one unmatched Kalshi name.

    Ranked, capped, and each carrying the reason it is here. This is the whole of what
    replaced the first-initial-and-last-name match tier: the same signal, offered
    rather than taken.

    `candidates` should be the athletes NOBODY has matched yet. Proposing a golfer who
    already belongs to somebody is how a review file talks a reviewer into a swap.
    """
    scored = []
    for player in candidates:
        if not player.get("name"):
            continue
        score, why = _score_suggestion(name, player["name"])
        if score >= floor:
            scored.append({
                "athlete_id": player.get("athlete_id"),
                "espn_name": player["name"],
                "position": player.get("position"),
                "confidence": score,
                "why": why,
            })
    scored.sort(key=lambda s: (-s["confidence"], s["espn_name"] or ""))
    return scored[:limit]


def unclaimed(players, matches):
    """
    The athletes in the field that no Kalshi golfer resolved to, in leaderboard order.

    Half of a review. A reviewer given only the unmatched Kalshi names is guessing;
    given both lists side by side, the answer is usually obvious -- and the list being
    SHORT is itself the evidence that nothing was missed, because a Kalshi field and an
    ESPN field of the same tournament are very nearly the same people.
    """
    taken = {str(hit["player"].get("athlete_id")) for hit in (matches or {}).values()
             if hit.get("player")}
    return [p for p in players if str(p.get("athlete_id")) not in taken]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_names(path):
    """
    Golfer names out of whatever file is to hand.

    A result file from build_competition.py, a Kalshi odds file from kalshi_odds.py, a
    bare JSON list, or one name per line. All four are things a user already has, and
    guessing wrong is cheap to notice, so this reads the shape rather than asking for it.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):
        rows = data.get("golfers")
        if rows is None:
            raise ValueError(f"{path}: JSON object with no golfers list")
        return [r["name"] if "name" in r else r["golfer_name"] for r in rows if isinstance(r, dict)]
    if isinstance(data, list):
        if all(isinstance(r, str) for r in data):
            return list(data)
        return [r.get("name") or r.get("golfer_name") for r in data if isinstance(r, dict)]
    raise ValueError(f"{path}: expected a list of names or an object with a golfers list")


def _load_map(path, key):
    """A `{key: {...}}` block out of a JSON file, or {} if there is no file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return (data.get(key) or {}) if isinstance(data, dict) else {}


def print_match_report(matches, report, names, players=()):
    """
    The join, name by name, with the tier that found each one -- and for the ones no
    tier found, the candidates a reviewer should look at.

    The suggestions are the useful half of this output. "12 unresolved" is a number;
    "Nicolas Echavarria -> Nico Echavarria, same first initial and last name" is a
    decision somebody can make in two seconds.
    """
    width = max((len(n) for n in names), default=10)
    for name in names:
        hit = matches.get(name)
        if not hit:
            print(f"  {name:<{width}}  {'--':<26} {'':<10} "
                  + ("absent (reviewed)" if name in report.get("absent", []) else "unresolved"))
            continue
        print(f"  {name:<{width}}  {hit['player']['name']:<26} "
              f"{str(hit['player']['athlete_id']):<10} {hit['match']}")

    print(f"\n{report['matched']}/{report['requested']} matched "
          f"({report.get('matched_exact', 0)} exact, {report.get('matched_alias', 0)} alias, "
          f"{report.get('matched_decision', 0)} reviewed)")
    for problem in report.get("problems") or []:
        print(f"!! {problem}")
    if report.get("absent"):
        print(f"reviewed and confirmed absent ({len(report['absent'])}): "
              + ", ".join(report["absent"]))
    if report["unresolved"]:
        print(f"\nunresolved ({len(report['unresolved'])}) -- nobody has looked at these yet:")
        free = unclaimed(players, matches)
        for name in report["unresolved"]:
            print(f"  {name}")
            for s in suggest_matches(name, free):
                print(f"      -> {s['espn_name']:<26} {str(s['athlete_id']):<10} "
                      f"{s['confidence']:.2f}  {s['why']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", help="ESPN event id, e.g. 401811961")
    ap.add_argument("--league", default=DEFAULT_LEAGUE)
    ap.add_argument("--season", help="list a season's events, e.g. 2026")
    ap.add_argument("--find", help="resolve a tournament name against --season")
    ap.add_argument("--file", help="read a saved payload instead of the API")
    ap.add_argument("--calendar", action="store_true",
                    help="list --season from the cheap calendar endpoint (no fields, 12 KB)")
    ap.add_argument("--match", metavar="PATH",
                    help="match golfer names onto the ESPN athletes in --event's field. "
                         "Reads a result file, a Kalshi odds file, a JSON list of names, "
                         "or one name per line.")
    ap.add_argument("--aliases", metavar="PATH", help="alias file for --match")
    ap.add_argument("--decisions", metavar="PATH",
                    help="a match-review file for --match, so reviewed bindings are applied")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.match:
        if not args.event:
            ap.error("--match needs --event: the join is against one published field, and "
                     "before the first tee time there is no field to join against. "
                     "`--season YYYY --find <name>` finds the event id.")
        try:
            names = read_names(args.match)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        aliases = _load_map(args.aliases, "aliases")
        decisions = _load_map(args.decisions, "decisions")

        meta, players = parse_leaderboard(fetch_leaderboard(args.event, args.league))
        print(f"{(meta or {}).get('event', args.event)}: {len(players)} in the field")
        if not players:
            print("ESPN has published no competitors for this event yet, so there is nothing "
                  "to match against. Re-run after the first tee time.")
            return 1

        matches, report = match_field(names, players, aliases=aliases, decisions=decisions)
        if args.json:
            print(json.dumps({
                "matches": {n: {"match": h["match"], "athlete_id": h["player"]["athlete_id"],
                                "espn_name": h["player"]["name"]} for n, h in matches.items()},
                "report": report,
                "suggestions": {n: suggest_matches(n, unclaimed(players, matches))
                                for n in report["unresolved"]},
            }, indent=2))
            return 0
        print_match_report(matches, report, names, players)
        return 0 if report["matched"] else 1

    if args.calendar:
        if not args.season:
            ap.error("--calendar needs --season")
        for e in season_calendar(args.season, args.league):
            print(f"  {e['event_id']:<12} {(e['start'] or '')[:10]}  {e['name']}")
        return 0

    if args.season and not args.event and not args.find:
        for e in season_events(args.season, args.league):
            print(f"  {e['event_id']:<12} {e['state']:<5} {e['start'][:10]}  {e['name']}")
        return 0

    if args.find:
        if not args.season:
            ap.error("--find needs --season")
        best, ranked = resolve_event(args.find, args.season, args.league)
        if not best:
            print(f"no ESPN event in {args.season} matches {args.find!r}")
            return 1
        print(f"best: {best['event_id']}  {best['name']}  ({best['state']}, {best['start'][:10]})")
        for e in ranked[1:4]:
            print(f"  also: {e['event_id']}  {e['name']}  score {e['score']:.2f}")
        return 0

    payload = json.load(open(args.file)) if args.file else fetch_leaderboard(args.event, args.league)
    meta, players = parse_leaderboard(payload)
    if not meta:
        print("no events in payload -- between tournaments")
        return 1
    if args.json:
        print(json.dumps({"meta": meta, "players": players}, indent=2))
        return 0

    print(f"{meta['event']} @ {meta['course']} (par {meta['par']})  [{meta['state']}]")
    print(f"{meta['detail']}  {len(players)} in the field\n")
    print(f"{'POS':<6} {'PLAYER':<26} {'TOT':>5} {'THRU':>5}  ROUNDS")
    print("-" * 74)
    for p in players[:args.top]:
        rs = ", ".join(f"{fmt_par(r['to_par'])} R{r['round']}" for r in p["rounds"])
        print(f"{str(p['position']):<6} {str(p['name']):<26} {p['to_par_display']:>5} "
              f"{str(p['thru']):>5}  {rs or '--'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
