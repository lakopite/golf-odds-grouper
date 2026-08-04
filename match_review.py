#!/usr/bin/env python3
"""
match_review.py -- the file where the leftover golfer names get settled.

The Kalshi->ESPN join resolves a name three ways and all three are exact: a decision
somebody recorded, an explicit alias, or two normalised display names being equal.
Anything else is left alone rather than guessed at (see espn_leaderboard's module
docstring for why). That leaves a handful of names per tournament -- 12 of 151 on the
measured Rocket Classic field -- which somebody has to look at.

This is that somebody's worksheet, and it round-trips:

    build writes it   ->   a reviewer fills in `decisions`   ->   build reads it back

The build writes every unmatched Kalshi name beside the ESPN athletes nobody claimed,
with ranked suggestions and the reason for each. The reviewer -- in practice Claude,
driving the golf-pool skill, which is what "LLM matching" means here -- adds one entry
per name to `decisions`. The next build applies them, records them in the result file,
and rewrites this file with whatever is still open.

WHY A FILE AND NOT AN API CALL
-------------------------------
The pipeline is two unauthenticated GETs and no credentials. Putting a model call
inside it would add a key, a network dependency and a per-build cost to a program that
currently runs anywhere, and would bind golfers to athletes with nothing written down
in between. A file costs one extra step and buys three things: the decisions are
visible before they take effect, they are still there next week, and a build with no
model anywhere near it behaves identically as long as the file is filled in.

TWO KINDS OF ANSWER
-------------------
A name can be settled two ways, and they are genuinely different:

    {"athlete_id": "4588361"}      this Kalshi name IS that ESPN athlete
    {"absent": true}               this golfer is not in the field. They withdrew.

Both end with the golfer scoring nothing this week -- an absent golfer has no
leaderboard row to read. The difference is that the second is a fact somebody checked
and the first tells the scoreboard whose scores to show. A build that has neither only
knows it could not find the name, which is not the same as either.
"""

import json
import os
from datetime import datetime, timezone

import espn_leaderboard

SCHEMA = "golf-pool/match-review/1"

# How many candidates to offer per unmatched name. Three is enough for the answer to
# be in the list on every measured field, and short enough that a reviewer reads all
# of them rather than skimming the first.
SUGGESTIONS_PER_NAME = 3

HOW_TO_USE = [
    "This file settles the golfer names the automatic join would not guess at.",
    "For each entry in `pending`, add one key to `decisions`, under the EXACT "
    "`kalshi_name` string:",
    '  "Nicolas Echavarria": {"athlete_id": "4588361", "espn_name": "Nico Echavarria"}',
    "     -- this Kalshi golfer is that ESPN athlete. `athlete_id` is what binds; "
    "`espn_name` is for readers and for learning a reusable alias.",
    '  "Jason Day": {"absent": true, "note": "withdrew before the first round"}',
    "     -- this golfer is not in the field at all. They will score nothing, which is "
    "correct, and the file will say it was checked rather than missed.",
    "Only use an `athlete_id` that appears in `pending[].suggestions` or in "
    "`espn_athletes_nobody_claimed`. An id from anywhere else is not in this field and "
    "will be refused.",
    "A suggestion is a proposal, not an answer: confirm it against the name, do not "
    "copy the highest confidence. If neither list has the golfer in it, they are "
    "absent -- an ESPN field and a Kalshi field for the same tournament are very "
    "nearly the same people.",
    "Then re-run the build with --match-review pointing at this file. Decisions "
    "already applied are listed below and do not need re-entering.",
]


def review_path(output_path, explicit=None):
    """
    Where the review file goes: beside the result file it belongs to.

    A competition's result and its open questions travel together -- the same
    directory gets handed over, zipped, or thrown away as one thing.
    """
    if explicit:
        return explicit
    return os.path.join(os.path.dirname(os.path.abspath(output_path)), "match-review.json")


def load(path, espn_event_id=None):
    """
    Read the decisions out of a review file. -> (decisions, notes).

    `notes` are sentences for the caller to print. A review file is hand-edited by
    definition, so every way it can be wrong is reported rather than raised: a build
    that drops 8 reviewed bindings because the ninth had a typo has turned a small
    problem into a large one.

    A file recorded against a DIFFERENT ESPN event is refused outright. Its athlete ids
    are ids in another field, and the golfers whose names they are attached to may not
    even be playing this week -- applying them would bind real people to the wrong
    tournament, which is exactly the failure this whole path exists to prevent.
    """
    if not path or not os.path.exists(path):
        return {}, []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return {}, [f"{path} is not valid JSON ({exc}), so no reviewed decisions were "
                    "applied. Fix the file or delete it."]
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return {}, [f"{path} is not a {SCHEMA} file, so no reviewed decisions were applied."]

    recorded_event = str((data.get("espn") or {}).get("event_id") or "")
    if espn_event_id and recorded_event and recorded_event != str(espn_event_id):
        return {}, [f"{path} was written for ESPN event {recorded_event}, and this build is "
                    f"event {espn_event_id}. Its athlete ids belong to another field, so "
                    "none of its decisions were applied. Point --match-review somewhere "
                    "else, or delete it and let this build write a fresh one."]

    decisions, notes = {}, []
    for name, decision in (data.get("decisions") or {}).items():
        if not decision:
            continue                                   # an unfilled stub, not an error
        if not isinstance(decision, dict):
            notes.append(f"{path}: the decision for {name!r} is {type(decision).__name__}, "
                         "not an object. Ignored.")
            continue
        decisions[name] = decision
    return decisions, notes


def write(path, *, tournament, espn, golfers, matches, report, players,
          decisions=None, now=None):
    """
    Write the worksheet. Returns the path written, or None if there was nothing to say.

    `golfers` is the Kalshi field as the build knows it -- name, grouping weight, and
    which team holds them -- because the third of those decides how much a reviewer
    should care. An unresolved golfer worth 0.0004 in nobody's group is noise; one
    worth 4% sitting in a team's roster is the whole scoreboard.

    Nothing is written when every name resolved and no decision was needed. A file
    whose only content is "nothing to do" is a file somebody has to open to find that
    out.
    """
    pending = list(report.get("unresolved") or [])
    decisions = dict(decisions or {})
    if not pending and not decisions:
        return None

    meta = {g["name"]: g for g in golfers}
    free = espn_leaderboard.unclaimed(players, matches)
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    for name in pending:
        row = meta.get(name) or {}
        rows.append({
            "kalshi_name": name,
            "team": row.get("team"),
            "grouping_weight": row.get("grouping_weight"),
            "suggestions": espn_leaderboard.suggest_matches(
                name, free, limit=SUGGESTIONS_PER_NAME),
        })
    # Heaviest first: if a reviewer only gets through half the list, it should be the
    # half that moves the standings.
    rows.sort(key=lambda r: -(r["grouping_weight"] or 0))

    payload = {
        "schema": SCHEMA,
        "generated_at": now,
        "tournament": tournament,
        "espn": espn,
        "how_to_use": HOW_TO_USE,
        "decisions": decisions,
        "pending": rows,
        "espn_athletes_nobody_claimed": [
            {"athlete_id": p.get("athlete_id"), "espn_name": p.get("name"),
             "position": p.get("position")}
            for p in free
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def learned_aliases(decisions, matches, known):
    """
    The reusable half of a review.

    "Nicolas Echavarria is Nico Echavarria" is true every week and is worth keeping in
    the alias file, where next month's build resolves it with nobody looking.
    "Jason Day withdrew" is true of one tournament and belongs only to this
    competition, so it is deliberately not learned.

    Taken from the MATCH rather than from the decision, so the alias records the ESPN
    display name that actually resolved rather than whatever the reviewer typed
    alongside the id.
    """
    out = {}
    for name, decision in (decisions or {}).items():
        if decision.get("absent"):
            continue
        hit = (matches or {}).get(name)
        if not hit or hit.get("match") != "decision":
            continue
        espn_name = hit["player"].get("name")
        if espn_name and known.get(name) != espn_name:
            out[name] = espn_name
    return out
