"""
Live checks against the real ESPN endpoints.

Skipped unless ESPN_LIVE=1, so the default suite stays offline and deterministic:

    ESPN_LIVE=1 python -m pytest tests/test_live_espn.py -v

The offline fixtures can prove the parser is self-consistent. They cannot prove that a
one-day scoreboard request still carries the whole season's calendar, or that a `pre`
event still returns zero competitors -- and both of those are load-bearing. The first is
the difference between a 12 KB request and a 35 MB one; the second is the fact the whole
program is now shaped around, because it is what splits a build into a groups sheet and
a scoreboard.
"""

import os
from datetime import datetime, timezone

import pytest

import espn_leaderboard as espn

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("ESPN_LIVE") != "1",
        reason="live API test; set ESPN_LIVE=1 to run",
    ),
]

SEASON = datetime.now(timezone.utc).year


@pytest.fixture(scope="module")
def calendar():
    return espn.season_calendar(SEASON)


def test_one_day_still_returns_the_whole_season(calendar):
    """
    The measurement season_calendar() exists for. If this ever drops to a handful of
    events, ESPN has started scoping the calendar to the requested window and the cheap
    path is gone -- the fix is to go back to season_events(), which is correct and large.
    """
    assert len(calendar) > 30
    assert all(e["event_id"] and e["start"] for e in calendar)
    months = {e["start"][5:7] for e in calendar}
    assert len(months) > 6, "a season, not a window"


def test_the_calendar_agrees_with_the_scoreboard_on_event_ids(calendar):
    """
    The calendar is the cheap list and the scoreboard is the authoritative one. They have
    to be the same events, or `--calendar` hands somebody an id that means something else
    and they pin a build to the wrong tournament.
    """
    listed = {e["event_id"] for e in espn.season_events(SEASON)}
    assert listed & {e["event_id"] for e in calendar} == listed


def test_a_pre_tournament_event_still_publishes_no_field(calendar):
    """
    The premise the two build modes rest on. A build asks the leaderboard who is playing
    and gets nobody, so it does not pretend to know: it writes the groups and no scoring.

    If ESPN ever starts publishing fields in advance, this fails -- and the right
    response is to celebrate, because every build becomes a live one and the second step
    of the week goes away.
    """
    now = datetime.now(timezone.utc).isoformat()
    upcoming = [e for e in calendar if (e["start"] or "") > now]
    if not upcoming:
        pytest.skip("no upcoming event in this season's calendar")
    meta, players = espn.parse_leaderboard(espn.fetch_leaderboard(upcoming[0]["event_id"]))
    assert meta["state"] == "pre"
    assert players == []


def test_a_finished_event_still_returns_a_field_with_athlete_ids(calendar):
    """
    The other half of the same premise, and the one the join depends on. Every competitor
    ESPN lists must carry an athlete id, because that id is the only key the exported
    page has -- a field of players without ids is a scoreboard that can rank nobody.
    """
    now = datetime.now(timezone.utc).isoformat()
    done = [e for e in calendar if (e.get("end") or "") < now]
    assert done, "no tournament has finished this season, which cannot be right"
    _, players = espn.parse_leaderboard(espn.fetch_leaderboard(done[-1]["event_id"]))
    assert len(players) > 50
    assert all(p["athlete_id"] for p in players)


def test_a_real_field_joins_to_itself_with_no_review_left_over(calendar):
    """
    End to end, live: take the names off a finished tournament, hand them back to the
    join as if they were a Kalshi field, and expect every one to come home exactly.

    This is the floor. The two automatic tiers are both exact string comparisons, so a
    field matched against ITSELF must leave nothing unresolved and nothing ambiguous. If
    it does not, normalisation has started folding two different people together, and
    that is the one failure mode the exact tier is not allowed to have.
    """
    now = datetime.now(timezone.utc).isoformat()
    done = [e for e in calendar if (e.get("end") or "") < now]
    _, players = espn.parse_leaderboard(espn.fetch_leaderboard(done[-1]["event_id"]))
    names = [p["name"] for p in players]

    matches, report = espn.match_field(names, players)
    assert report["unresolved"] == []
    assert report["ambiguous_names"] == []
    assert report["problems"] == []
    assert report["matched"] == len(names)
    assert all(hit["player"]["athlete_id"] for hit in matches.values())
