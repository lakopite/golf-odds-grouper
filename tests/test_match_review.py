"""
Tests for match_review.py -- the file the leftover golfer names get settled in.

This module is load-bearing for correctness rather than for tidiness, which is a change
from how the old fuzzy tier worked. A golfer left unresolved at build time carries no
ESPN athlete id, the exported page joins on that id and nothing else, and there is no
runtime name-rescue any more. So a name that never gets settled scores nothing for the
whole life of that page.

That puts the weight on two things, and most of what is below is about them: that a
reviewer is given enough to answer with, and that a wrong or stale answer is refused
rather than applied.
"""

import json

import pytest

import espn_leaderboard as espn
import match_review


FIELD = [
    {"athlete_id": "4588361", "name": "Nico Echavarria", "position": "T4"},
    {"athlete_id": "9478", "name": "Scottie Scheffler", "position": "1"},
    {"athlete_id": "4602673", "name": "Zach Bauchou", "position": "T9"},
]
NAMES = ["Nicolas Echavarria", "Scottie Scheffler", "Zachary Bauchou", "Jason Day"]


def golfers(names=NAMES, team="Bogey Boys"):
    """The Kalshi field as the build knows it: name, weight, and who holds them."""
    return [{"name": n, "grouping_weight": round(0.4 - 0.1 * i, 2), "team": team}
            for i, n in enumerate(names)]


def espn_block(event_id="401811961"):
    return {"event_id": event_id, "league": "pga", "field_size": len(FIELD),
            "leaderboard_endpoint": "https://site.web.api.espn.com/x"}


def write(tmp_path, decisions=None, names=NAMES, players=FIELD, applied=None):
    """Run the join and write the review file it leaves behind. -> (path, payload)."""
    matches, report = espn.match_field(names, players, decisions=decisions)
    path = match_review.write(
        str(tmp_path / "match-review.json"), tournament="Wyndham Championship",
        espn=espn_block(), golfers=golfers(names), matches=matches, report=report,
        players=players, decisions=applied if applied is not None else (decisions or {}))
    return path, (json.loads(open(path).read()) if path else None)


# ---------------------------------------------------------------------------
# Where the file goes
# ---------------------------------------------------------------------------

def test_the_review_lands_beside_the_result_it_belongs_to(tmp_path):
    """
    A competition's answers and its open questions travel together. The same directory
    gets handed over, zipped or thrown away as one thing, so the review file has no
    business living somewhere else by default.
    """
    out = str(tmp_path / "build" / "result.json")
    assert match_review.review_path(out) == str(tmp_path / "build" / "match-review.json")


def test_an_explicit_path_wins(tmp_path):
    out = str(tmp_path / "build" / "result.json")
    assert match_review.review_path(out, "/elsewhere/review.json") == "/elsewhere/review.json"


# ---------------------------------------------------------------------------
# Reading one back
# ---------------------------------------------------------------------------

def test_no_file_is_not_an_error(tmp_path):
    """
    Most builds have no review file, because most builds resolve everything. Treating
    its absence as a problem would put a warning on the normal case.
    """
    assert match_review.load(str(tmp_path / "nothing.json")) == ({}, [])
    assert match_review.load(None) == ({}, [])


def test_a_file_that_is_not_json_is_reported_rather_than_raised(tmp_path):
    """
    This file is hand-edited by definition, so a trailing comma is a normal Tuesday. A
    traceback in the middle of a build would lose the fifty things the build had already
    done correctly.
    """
    path = tmp_path / "match-review.json"
    path.write_text('{"decisions": {,}}')
    decisions, notes = match_review.load(str(path))
    assert decisions == {}
    assert len(notes) == 1 and "not valid JSON" in notes[0]


def test_a_file_that_is_not_a_review_file_is_refused(tmp_path):
    path = tmp_path / "match-review.json"
    path.write_text(json.dumps({"decisions": {"Jason Day": {"absent": True}}}))
    decisions, notes = match_review.load(str(path))
    assert decisions == {}
    assert match_review.SCHEMA in notes[0]


def test_a_review_written_for_another_tournament_is_refused_whole(tmp_path):
    """
    The guard that matters most in this file. Its athlete ids are ids in ANOTHER field:
    4588361 is Nico Echavarria at the Wyndham and somebody else entirely in a different
    event's numbering, and the golfers it names may not be playing this week at all.

    Applying any of it would bind real people to the wrong tournament, which is the
    exact failure the whole review path exists to prevent. So the file is refused whole
    rather than filtered -- there is no subset of it that is known to be safe.
    """
    path = tmp_path / "match-review.json"
    path.write_text(json.dumps({
        "schema": match_review.SCHEMA,
        "espn": {"event_id": "401811961"},
        "decisions": {"Nicolas Echavarria": {"athlete_id": "4588361"},
                      "Jason Day": {"absent": True}},
    }))
    decisions, notes = match_review.load(str(path), espn_event_id="999999")
    assert decisions == {}
    assert "401811961" in notes[0] and "999999" in notes[0]

    # The same file against the event it was written for is fine.
    decisions, notes = match_review.load(str(path), espn_event_id="401811961")
    assert set(decisions) == {"Nicolas Echavarria", "Jason Day"} and notes == []


def test_a_review_with_no_event_recorded_is_taken_at_face_value(tmp_path):
    """
    A hand-written file that never carried an event id cannot be proved wrong, and
    refusing it would make the file impossible to write by hand. The names still have to
    be in this Kalshi field for anything to happen -- build_competition drops the rest.
    """
    path = tmp_path / "match-review.json"
    path.write_text(json.dumps({"schema": match_review.SCHEMA,
                                "decisions": {"Jason Day": {"absent": True}}}))
    decisions, notes = match_review.load(str(path), espn_event_id="401811961")
    assert decisions == {"Jason Day": {"absent": True}} and notes == []


def test_an_unfilled_stub_is_not_an_error(tmp_path):
    """
    A reviewer part-way through has answered some and not others. The unanswered ones
    are the normal state of this file, not a complaint to print.
    """
    path = tmp_path / "match-review.json"
    path.write_text(json.dumps({
        "schema": match_review.SCHEMA,
        "decisions": {"Jason Day": None, "Zachary Bauchou": {}, "Nicolas Echavarria":
                      {"athlete_id": "4588361"}},
    }))
    decisions, notes = match_review.load(str(path))
    assert set(decisions) == {"Nicolas Echavarria"}
    assert notes == []


def test_a_decision_that_is_not_an_object_is_reported(tmp_path):
    """`"Jason Day": "absent"` is somebody's reasonable guess at the format, and wrong."""
    path = tmp_path / "match-review.json"
    path.write_text(json.dumps({"schema": match_review.SCHEMA,
                                "decisions": {"Jason Day": "absent"}}))
    decisions, notes = match_review.load(str(path))
    assert decisions == {}
    assert "not an object" in notes[0] and "Jason Day" in notes[0]


# ---------------------------------------------------------------------------
# Writing one
# ---------------------------------------------------------------------------

def test_nothing_open_and_nothing_decided_writes_no_file(tmp_path):
    """
    A file whose only content is "nothing to do" is a file somebody has to open to find
    that out. The build that resolved everything automatically should leave no trace.
    """
    path, payload = write(tmp_path, names=[p["name"] for p in FIELD])
    assert path is None and payload is None
    assert not (tmp_path / "match-review.json").exists()


def test_decisions_alone_still_write_the_file(tmp_path):
    """
    Once everything is settled the file stops being a worksheet and starts being the
    record of what was decided, which is worth keeping where the next build can read it.
    """
    decisions = {"Nicolas Echavarria": {"athlete_id": "4588361"},
                 "Zachary Bauchou": {"athlete_id": "4602673"},
                 "Jason Day": {"absent": True}}
    path, payload = write(tmp_path, decisions=decisions)
    assert path is not None
    assert payload["pending"] == []
    assert payload["decisions"] == decisions


def test_the_heaviest_golfers_come_first(tmp_path):
    """
    A reviewer who only gets through half the list should get through the half that
    moves the standings. An unresolved golfer worth 0.04% is noise; one worth 4% sitting
    in somebody's roster is the scoreboard.
    """
    path, payload = write(tmp_path)
    weights = [row["grouping_weight"] for row in payload["pending"]]
    assert weights == sorted(weights, reverse=True)


def test_each_pending_golfer_carries_their_team_and_their_weight(tmp_path):
    """Both are how a reviewer decides how much to care before reading a single name."""
    _, payload = write(tmp_path)
    for row in payload["pending"]:
        assert row["team"] == "Bogey Boys"
        assert isinstance(row["grouping_weight"], float)


def test_the_right_answer_is_offered_with_the_reason_it_was_offered(tmp_path):
    """
    The formal-vs-familiar case, end to end through the file. This is what the deleted
    fuzzy tier used to bind silently; here it is proposed, with its reason, and somebody
    confirms it.
    """
    _, payload = write(tmp_path)
    row = next(r for r in payload["pending"] if r["kalshi_name"] == "Nicolas Echavarria")
    assert row["suggestions"][0]["espn_name"] == "Nico Echavarria"
    assert row["suggestions"][0]["athlete_id"] == "4588361"
    assert row["suggestions"][0]["why"] == "same first initial and last name"
    assert row["suggestions"][0]["position"] == "T4"


def test_a_golfer_who_is_simply_not_here_is_offered_nobody(tmp_path):
    """
    Which is the answer. An ESPN field and a Kalshi field for one tournament are very
    nearly the same people, so an empty suggestion list beside a short unclaimed list
    says "they withdrew" more clearly than any guess would.
    """
    _, payload = write(tmp_path)
    row = next(r for r in payload["pending"] if r["kalshi_name"] == "Jason Day")
    assert row["suggestions"] == []


def test_suggestions_are_capped(tmp_path):
    _, payload = write(tmp_path)
    assert all(len(r["suggestions"]) <= match_review.SUGGESTIONS_PER_NAME
               for r in payload["pending"])


def test_an_athlete_somebody_already_holds_is_not_offered_to_anybody_else(tmp_path):
    """
    Half of a review is the list of ESPN athletes nobody claimed. Proposing a golfer who
    already belongs to a team is how a review file talks a reviewer into a swap, and the
    swap would take one golfer off somebody's roster to put them on another.
    """
    _, payload = write(tmp_path)
    free = {a["espn_name"] for a in payload["espn_athletes_nobody_claimed"]}
    assert "Scottie Scheffler" not in free, "he matched exactly; he is not available"
    assert free == {"Nico Echavarria", "Zach Bauchou"}
    for row in payload["pending"]:
        assert all(s["espn_name"] in free for s in row["suggestions"])


def test_the_file_says_which_event_it_was_written_for(tmp_path):
    """Without this, load() has nothing to check a stale file against."""
    _, payload = write(tmp_path)
    assert payload["espn"]["event_id"] == "401811961"
    assert payload["schema"] == match_review.SCHEMA
    assert payload["tournament"] == "Wyndham Championship"
    assert payload["how_to_use"], "the next agent to open this has to be told the format"


def test_what_is_written_is_what_load_reads_back(tmp_path):
    """
    The round trip, which is the whole mechanism: build writes, a reviewer fills in, the
    next build reads. A file only one half of that can parse is not a round trip.
    """
    path, payload = write(tmp_path)
    payload["decisions"] = {"Nicolas Echavarria": {"athlete_id": "4588361"},
                            "Jason Day": {"absent": True, "note": "withdrew"}}
    with open(path, "w") as f:
        json.dump(payload, f)

    decisions, notes = match_review.load(path, espn_event_id="401811961")
    assert notes == []
    matches, report = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert matches["Nicolas Echavarria"]["player"]["name"] == "Nico Echavarria"
    assert report["absent"] == ["Jason Day"]
    assert report["unresolved"] == ["Zachary Bauchou"], "the one nobody answered"


def test_a_real_field_produces_a_review_short_enough_to_read(espn_players, tmp_path):
    """
    Against the checked-in 147-player leaderboard rather than three synthetic golfers,
    because the size of this file is the question a reviewer cares about most. If a
    normal tournament left forty names open, nobody would work through it and the
    athlete ids would never get filled in. Four, each with the answer attached or
    visibly absent, is a job somebody does before their coffee goes cold.
    """
    # Kalshi's spelling for three of them, ESPN's for the rest, plus one golfer who is
    # not in this field at all. The three are REPLACED rather than added: adding them
    # would leave the ESPN athlete claimed by his own exact name, and an athlete
    # somebody already holds is correctly never offered to anybody else.
    formal = {"Nico Echavarria": "Nicolas Echavarria", "Zach Bauchou": "Zachary Bauchou",
              "Matt McCarty": "Matthew McCarty"}
    subset = espn_players[:120]
    assert set(formal) <= {p["name"] for p in subset}, "the fixture has to contain them"
    names = [formal.get(p["name"], p["name"]) for p in subset] + ["Brooks Koepka"]
    matches, report = espn.match_field(names, espn_players)
    path = match_review.write(
        str(tmp_path / "match-review.json"),
        tournament="Rocket Classic", espn=espn_block(), golfers=golfers(names),
        matches=matches, report=report, players=espn_players)
    payload = json.loads(open(path).read())

    assert len(payload["pending"]) == 4, [r["kalshi_name"] for r in payload["pending"]]
    offered = {r["kalshi_name"]: [s["espn_name"] for s in r["suggestions"]]
               for r in payload["pending"]}
    assert offered["Nicolas Echavarria"][0] == "Nico Echavarria"
    assert offered["Zachary Bauchou"][0] == "Zach Bauchou"
    assert offered["Matthew McCarty"][0] == "Matt McCarty"
    assert offered["Brooks Koepka"] == [], "not in this field, and nothing pretends he is"


# ---------------------------------------------------------------------------
# What a review is worth keeping
# ---------------------------------------------------------------------------

def test_a_settled_name_becomes_a_reusable_alias():
    """
    "Nicolas Echavarria is Nico Echavarria" is true every week, so settling it once
    should settle it for good -- otherwise the same handful of names comes back to the
    review file at every tournament and the step stops getting done.
    """
    decisions = {"Nicolas Echavarria": {"athlete_id": "4588361"}}
    matches, _ = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert match_review.learned_aliases(decisions, matches, {}) == {
        "Nicolas Echavarria": "Nico Echavarria"}


def test_a_withdrawal_is_not_worth_keeping():
    """
    "Jason Day is not in the field" is true of one tournament and false of the next one
    he enters. It belongs to this competition and travels in the result file; putting it
    in the alias file would make it permanent, and wrong by Thursday next week.
    """
    decisions = {"Jason Day": {"absent": True, "note": "withdrew"}}
    matches, _ = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert match_review.learned_aliases(decisions, matches, {}) == {}


def test_an_alias_already_known_is_not_relearned():
    decisions = {"Nicolas Echavarria": {"athlete_id": "4588361"}}
    matches, _ = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert match_review.learned_aliases(
        decisions, matches, {"Nicolas Echavarria": "Nico Echavarria"}) == {}


def test_the_alias_records_the_name_that_resolved_not_the_one_that_was_typed():
    """
    A reviewer writes `espn_name` for their own benefit and can get it slightly wrong --
    a missing accent, a different spelling -- while the athlete_id beside it is right.
    The id is what bound, so the id's display name is what the alias has to record, or
    the alias will not fire next week.
    """
    decisions = {"Nicolas Echavarria": {"athlete_id": "4588361", "espn_name": "N. Echavarria"}}
    matches, _ = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert match_review.learned_aliases(decisions, matches, {}) == {
        "Nicolas Echavarria": "Nico Echavarria"}


def test_a_decision_that_never_bound_teaches_nothing():
    """A refused athlete id resolved nobody, so there is no display name to remember."""
    decisions = {"Nicolas Echavarria": {"athlete_id": "99999999"}}
    matches, report = espn.match_field(NAMES, FIELD, decisions=decisions)
    assert report["problems"], "the bad id is reported"
    assert match_review.learned_aliases(decisions, matches, {}) == {}
