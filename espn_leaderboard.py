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

  * A `pre` event returns ZERO competitors. The field does not exist until play
    starts, which is why the golfer->athlete join cannot be finished at build time
    and the frontend has to be able to do it too.

THE NAME JOIN
-------------
Kalshi's `custom_strike.golf_competitor` UUID is stable across events and across
market series, so it is the right key for a golfer -- but it is not an ESPN id, and
ESPN publishes no Kalshi id. The join is by name, in two tiers.

Measured on Rocket Classic, 151 Kalshi markets against 147 ESPN competitors:
    tier 1  normalised exact                      139
    tier 2  first initial + last name               8   (Zachary/Zach Bauchou,
              Cameron/Cam Davis, Kris/Kristoffer Ventura, Nicolas/Nico Echavarria,
              Matthew/Matt McCarty, Benjamin/Ben James, Jordan L./Jordan Smith,
              Hao-Tong/Haotong Li)
    unresolved                                      4   (Daniel Brown, Taylor Moore,
              Brooks Koepka, Jason Day -- all genuinely absent from the ESPN field,
              withdrawn before play)
Tier 2 had ZERO collisions within the ESPN field. So the two tiers together resolve
every golfer ESPN actually lists, and everything left over is a golfer who is not
playing rather than a golfer we failed to find.

Tier 3 is a manual alias, which always wins. That is the escape hatch for the day a
field contains two J. Smiths.

WHEN THERE IS NO FIELD TO JOIN AGAINST
--------------------------------------
A `pre` event returns zero competitors, so on Wednesday night there is nothing to
match. That does not mean the golfers are unknown: a Kalshi field is drawn from the
same tour that played last week, so almost every name in it appears in an EARLIER
event's leaderboard, and those payloads are up and finished. `match_history()` walks
back through the season and joins against the union of the fields it finds.

Measured 2026-08-03, the 150-golfer Kalshi Wyndham field against an ESPN event that
had published nothing at all:

    1 earlier tournament    147 athletes    123 of 150 resolved
    2                       183            130
    3                       282            143
    4                       358            146   <- the default
    12                      ~900           146   (no further gain; 3.5s -> 6.5s)

The four it never resolves are golfers with no PGA Tour start this season -- they are
absent from ESPN, not missed by the matcher. The whole scan costs about 3.5 seconds.

What comes back from an earlier tournament is an IDENTITY -- athlete id, display
name, headshot, country -- and deliberately nothing else. See identity().
"""

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

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


def finished_before(calendar, cutoff=None, exclude_ids=()):
    """
    The events that had finished by `cutoff`, newest first.

    Compared on the DATE alone. ESPN writes calendar times as `2026-07-30T07:00Z` and
    the rest of this project writes `2026-08-03T21:00:00+00:00`; both truncate to the
    same ten characters, and tournaments are days apart, so the day is the only part of
    the comparison that has ever meant anything.
    """
    cutoff = (cutoff or datetime.now(timezone.utc).isoformat())[:10]
    exclude = {str(i) for i in exclude_ids}
    done = [e for e in calendar
            if e.get("end") and e["end"][:10] < cutoff and e["event_id"] not in exclude]
    done.sort(key=lambda e: e["end"], reverse=True)
    return done


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


def parse_leaderboard(payload):
    """
    -> (meta, players). `players` is sorted by sortOrder, i.e. by live rank.

    Every field the standings need is computed here rather than read off the payload,
    because the fields ESPN publishes for them are stale mid-round. See the module
    docstring for the measurements.
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
    """Two lookup tiers over an ESPN field, with ambiguous tier-2 keys dropped."""
    exact, initial = {}, {}
    for p in players:
        if not p.get("name"):
            continue
        exact.setdefault(normalize_name(p["name"]), []).append(p)
        key = initial_last_key(p["name"])
        if key:
            initial.setdefault(key, []).append(p)
    return {
        "exact": {k: v[0] for k, v in exact.items() if len(v) == 1},
        "initial_last": {k: v[0] for k, v in initial.items() if len(v) == 1},
        "ambiguous": sorted(k for k, v in initial.items() if len(v) > 1),
    }


def match_golfer(name, index, aliases=None):
    """
    -> (espn_player | None, how). `how` is one of alias / exact / initial_last / unresolved.

    An alias maps a Kalshi name to an ESPN display name and is checked first, so a
    field with two J. Smiths can be settled by hand and stay settled.
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

    key = initial_last_key(name)
    if key:
        hit = index["initial_last"].get(key)
        if hit:
            return hit, "initial_last"

    return None, "unresolved"


def match_field(names, players, aliases=None):
    """
    Join a list of Kalshi golfer names onto an ESPN field.

    Returns (matches, report). `matches` maps the Kalshi name to the ESPN player plus
    how it was found; `report` counts each tier and names what is left over, because
    "4 unresolved" and "which 4" are different facts and only the second is actionable.
    """
    index = build_index(players)
    matches, counts, unresolved = {}, {"alias": 0, "exact": 0, "initial_last": 0, "unresolved": 0}, []
    for name in names:
        player, how = match_golfer(name, index, aliases)
        counts[how] += 1
        if player is None:
            unresolved.append(name)
        else:
            matches[name] = {"player": player, "match": how}
    report = {
        "espn_field_size": len(players),
        "requested": len(names),
        "matched": len(matches),
        **{f"matched_{k}": v for k, v in counts.items() if k != "unresolved"},
        "unresolved": unresolved,
        "ambiguous_keys": index["ambiguous"],
    }
    return matches, report


# ---------------------------------------------------------------------------
# The join when this week's field does not exist yet
# ---------------------------------------------------------------------------

# What survives from an earlier tournament. Everything else a parsed player carries --
# position, sortOrder, to_par, thru, rounds, status -- describes a week that is over.
IDENTITY_FIELDS = ("athlete_id", "name", "short_name", "headshot", "flag", "country", "amateur")


def identity(player, event=None):
    """
    The part of an ESPN player that is the PERSON rather than the week.

    This is the whole safety argument for matching against an earlier tournament. A
    player object pulled out of last month's leaderboard carries last month's position,
    to-par and sortOrder, and the standings rule ranks on exactly those fields -- so
    handing one to the scoreboard would show a golfer who won in July sitting at T1 on
    Thursday morning of a tournament that has not started. Identity is stable and
    scoring is not, so the scoring fields are dropped here rather than carried onward
    and hoped about.

    `event` is stamped on so the result file can say which tournament answered.
    """
    out = {k: player.get(k) for k in IDENTITY_FIELDS}
    out["from_event"] = ({"event_id": event.get("event_id"), "name": event.get("name"),
                          "end": event.get("end")} if event else None)
    return out


def match_history(names, season, league=DEFAULT_LEAGUE, aliases=None, max_events=4,
                  cutoff=None, exclude_ids=(), calendar=None, fetch=None, log=None):
    """
    Resolve golfer names against the season's EARLIER tournaments.

    Walks back from `cutoff`, newest first, accumulating a union of every athlete it
    sees and re-running the two-tier match over that union after each event. Stops when
    every name has an exact match, or after `max_events` leaderboards -- a first-initial
    match keeps the scan going, because widening the union is the only thing that can
    prove that match wrong.

    Matching over the union rather than event by event is what keeps tier 2 honest. A
    first-initial-and-last-name key is measured collision-free inside ONE field, but a
    season is five hundred golfers and that guarantee does not survive the widening.
    build_index() drops a key the moment two athletes share it, so a name that is
    ambiguous anywhere in the scanned history comes back unresolved instead of bound to
    a coin flip -- and because the whole union is re-matched each round, a name resolved
    early is re-checked against everything found later.

    It is not hypothetical. Four tournaments back from the 2026 Wyndham the union holds
    both Cameron Young and Carson Young, so `c|young` is refused. Cameron Young was
    already resolved on tier 1, so nothing was lost -- but a source writing "Cam Young"
    would come back unresolved rather than silently bound to Carson.

    Returns (matches, report). Each match carries the tier that found it, the identity,
    and the event that answered.
    """
    fetch = fetch or fetch_leaderboard
    pending = list(names)
    report = {
        "requested": len(pending),
        "matched": 0,
        "unresolved": list(pending),
        "scanned": [],
        "athletes": 0,
        "ambiguous_keys": [],
        "events_available": 0,
        "unscanned_events": 0,
    }
    if not pending or max_events <= 0:
        return {}, report

    calendar = season_calendar(season, league) if calendar is None else calendar
    earlier = finished_before(calendar, cutoff=cutoff, exclude_ids=exclude_ids)
    report["events_available"] = len(earlier)

    seen, union, matches, sub = set(), [], {}, None
    for event in earlier[:max_events]:
        try:
            _, players = parse_leaderboard(fetch(event["event_id"], league))
        except Exception as exc:                       # noqa: BLE001 -- reported, not fatal
            if log:
                log(f"!! could not read {event['name']} ({event['event_id']}): {exc}")
            report["scanned"].append({**event, "field_size": None, "error": str(exc)})
            continue

        fresh = 0
        for p in players:
            if p.get("athlete_id") and p["athlete_id"] not in seen:
                seen.add(p["athlete_id"])
                union.append(identity(p, event))
                fresh += 1
        report["scanned"].append({**event, "field_size": len(players), "new_athletes": fresh})

        matches, sub = match_field(pending, union, aliases)
        if log:
            log(f"   {event['name']}: {len(players)} played, {len(sub['unresolved'])} "
                f"of {len(pending)} still unmatched")
        # Stopping the moment everything is matched would defeat the ambiguity check for
        # exactly the matches that need it: a tier-2 key that is unique in one field can
        # stop being unique two tournaments back, and the run would already have stopped
        # and taken the guess. So an all-exact answer ends the scan, and an answer
        # leaning on a first-initial key keeps widening the union that could refute it.
        if not sub["unresolved"] and not any(h["match"] == "initial_last"
                                             for h in matches.values()):
            break

    report["athletes"] = len(union)
    if sub:
        report["matched"] = len(matches)
        report["unresolved"] = sub["unresolved"]
        report["ambiguous_keys"] = sub["ambiguous_keys"]
    # How much history was left on the table. With names still unresolved, this is the
    # difference between "nobody else has played this season" and "look further back".
    report["unscanned_events"] = max(0, len(earlier) - len(report["scanned"]))
    return {name: {**hit, "source": "history", "event": hit["player"]["from_event"]}
            for name, hit in matches.items()}, report


def match_field_and_history(names, players, season, league=DEFAULT_LEAGUE, aliases=None,
                            max_events=4, cutoff=None, exclude_ids=(), calendar=None,
                            fetch=None, log=None):
    """
    The whole join, in the order it should be attempted: this week's field first, the
    season's earlier tournaments for whoever is left.

    Both halves are worth keeping apart in the report. A name this week's field cannot
    answer for means the golfer is not playing -- they withdrew -- and that is a fact
    the pool wants stated. A name resolved out of history is an identity with no
    scoring attached, which is exactly what a build run before the first tee time can
    honestly know.
    """
    matches, report = match_field(names, players or [], aliases)
    for hit in matches.values():
        hit["source"], hit["event"] = "field", None

    report["from_field"] = len(matches)
    report["from_history"] = 0
    # Only meaningful once a field exists: before that, nobody is "not in" it.
    report["not_in_field"] = list(report["unresolved"]) if players else []
    report["history"] = None

    if report["unresolved"] and max_events > 0:
        found, history = match_history(
            report["unresolved"], season, league, aliases=aliases, max_events=max_events,
            cutoff=cutoff, exclude_ids=exclude_ids, calendar=calendar, fetch=fetch, log=log)
        matches.update(found)
        report["history"] = history
        report["from_history"] = len(found)
        report["unresolved"] = history["unresolved"]

    counts = {"alias": 0, "exact": 0, "initial_last": 0}
    for hit in matches.values():
        counts[hit["match"]] = counts.get(hit["match"], 0) + 1
    report["matched"] = len(matches)
    report.update({f"matched_{k}": v for k, v in counts.items()})
    return matches, report


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


def print_match_report(matches, report, names):
    """The join, name by name, with the tier and the tournament that answered it."""
    width = max((len(n) for n in names), default=10)
    for name in names:
        hit = matches.get(name)
        if not hit:
            print(f"  {name:<{width}}  --")
            continue
        where = hit.get("event")
        via = hit["match"] + (f" @ {where['name']}" if where else "")
        print(f"  {name:<{width}}  {hit['player']['name']:<26} "
              f"{str(hit['player']['athlete_id']):<10} {via}")

    print(f"\n{report['matched']}/{report['requested']} matched "
          f"({report.get('matched_exact', 0)} exact, {report.get('matched_initial_last', 0)} "
          f"initial+last, {report.get('matched_alias', 0)} alias)")
    if report.get("from_history"):
        scanned = (report.get("history") or {}).get("scanned") or []
        print(f"{report['from_history']} resolved from {len(scanned)} earlier tournament(s): "
              + ", ".join(e["name"] for e in scanned))
    if report.get("not_in_field"):
        print(f"not in this week's field ({len(report['not_in_field'])}): "
              + ", ".join(report["not_in_field"]))
    if report["unresolved"]:
        print(f"unresolved ({len(report['unresolved'])}): " + ", ".join(report["unresolved"]))


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
                    help="match golfer names onto ESPN athletes. Reads a result file, a "
                         "Kalshi odds file, a JSON list of names, or one name per line.")
    ap.add_argument("--history", type=int, default=4, metavar="N",
                    help="earlier tournaments to fall back on when the field is not posted "
                         "yet (default 4; 0 disables)")
    ap.add_argument("--aliases", metavar="PATH", help="alias file for --match")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.match:
        if not args.season:
            ap.error("--match needs --season (the season whose tournaments to look back through)")
        try:
            names = read_names(args.match)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        aliases = {}
        if args.aliases and os.path.exists(args.aliases):
            with open(args.aliases, encoding="utf-8") as f:
                loaded = json.load(f)
            aliases = loaded.get("aliases", loaded) if isinstance(loaded, dict) else {}

        players, meta = [], None
        if args.event:
            meta, players = parse_leaderboard(fetch_leaderboard(args.event, args.league))
            print(f"{(meta or {}).get('event', args.event)}: {len(players)} in the field"
                  + ("" if players else " -- not posted yet, falling back on earlier tournaments"))
        matches, report = match_field_and_history(
            names, players, args.season, args.league, aliases=aliases,
            max_events=args.history, cutoff=(meta or {}).get("start"),
            exclude_ids=[args.event] if args.event else (), log=print)
        if args.json:
            print(json.dumps({"matches": matches, "report": report}, indent=2))
            return 0
        print_match_report(matches, report, names)
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
