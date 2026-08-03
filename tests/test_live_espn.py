"""
Live checks against the real ESPN endpoints.

Skipped unless ESPN_LIVE=1, so the default suite stays offline and deterministic:

    ESPN_LIVE=1 python -m pytest tests/test_live_espn.py -v

The offline fixtures can prove the parser is self-consistent. They cannot prove that a
one-day scoreboard request still carries the whole season's calendar, or that a `pre`
event still returns zero competitors -- and both of those are load-bearing. The first
is the difference between a 12 KB request and a 35 MB one; the second is the reason the
name join has to be able to work without a field at all.
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
    path is gone -- the fix is to go back to season_events(), not to look further back.
    """
    assert len(calendar) > 30
    assert all(e["event_id"] and e["start"] for e in calendar)
    months = {e["start"][5:7] for e in calendar}
    assert len(months) > 6, "a season, not a window"


def test_the_calendar_agrees_with_the_scoreboard_on_event_ids(calendar):
    """
    The calendar is the cheap list and the scoreboard is the authoritative one. They
    have to be the same events, or a history scan fetches leaderboards for ids that
    mean something else.
    """
    listed = {e["event_id"] for e in espn.season_events(SEASON)}
    assert listed & {e["event_id"] for e in calendar} == listed


def test_finished_events_still_return_their_fields(calendar):
    done = espn.finished_before(calendar)
    assert done, "no tournament has finished this season, which cannot be right"
    _, players = espn.parse_leaderboard(espn.fetch_leaderboard(done[0]["event_id"]))
    assert len(players) > 50
    assert all(p["athlete_id"] for p in players)


def test_a_pre_tournament_event_still_publishes_no_field(calendar):
    """
    The premise of the whole history fallback. If ESPN ever starts publishing fields in
    advance, this fails -- and the right response is to celebrate and simplify.
    """
    now = datetime.now(timezone.utc).isoformat()
    upcoming = [e for e in calendar if (e["start"] or "") > now]
    if not upcoming:
        pytest.skip("no upcoming event in this season's calendar")
    meta, players = espn.parse_leaderboard(espn.fetch_leaderboard(upcoming[0]["event_id"]))
    assert meta["state"] == "pre"
    assert players == []


def test_history_identifies_a_field_that_has_not_teed_off(calendar):
    """
    End to end, live: take the names off a finished tournament, pretend they are a
    Kalshi field for a tournament that has not started, and resolve them out of history.
    """
    done = espn.finished_before(calendar)
    _, players = espn.parse_leaderboard(espn.fetch_leaderboard(done[0]["event_id"]))
    names = [p["name"] for p in players[:40]]

    matches, report = espn.match_history(names, SEASON, max_events=4,
                                         exclude_ids=[done[0]["event_id"]])
    assert report["matched"] > len(names) * 0.8, report["unresolved"]
    for hit in matches.values():
        assert hit["player"]["athlete_id"]
        assert "position_number" not in hit["player"], "history must carry no scoring"
