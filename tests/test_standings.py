"""
Tests for standings.py -- the rule the pool is actually settled by.

Half of these are hand-built two-team cases, because the rule's whole content is in
its edge cases: what a tie does, what a cut golfer does, and what happens when one
team runs out of golfers. The other half run it over the real 147-player leaderboard,
because a rule that only works on toy input is not a rule.
"""

import json
import os

import pytest

import espn_leaderboard as espn
import standings


def player(position=None, sort_order=1, to_par=0, name="X"):
    """A minimal parsed ESPN player. position=None means cut / WD / DQ."""
    return {
        "name": name,
        "athlete_id": name,
        "position": position,
        "position_number": espn.position_number(position),
        "sort_order": sort_order,
        "to_par": to_par,
    }


def league(*rosters):
    """Teams named A, B, C... each holding the golfers given."""
    teams, by_key = [], {}
    for i, roster in enumerate(rosters):
        name = chr(ord("A") + i)
        golfers = []
        for j, p in enumerate(roster):
            key = f"{name}{j}"
            golfers.append({"golfer_id": key, "name": key})
            if p is not None:
                by_key[key] = p
        teams.append({"team_id": name, "golfers": golfers})
    return teams, by_key


def order(rows):
    return [r["team_id"] for r in rows]


# ---------------------------------------------------------------------------
# The primary rule
# ---------------------------------------------------------------------------

def test_best_golfer_wins():
    teams, index = league(
        [player("T5", 5), player("2", 2)],          # A holds 2nd
        [player("3", 3), player("T5", 5)],          # B holds 3rd
    )
    rows = standings.compute(teams, index)
    assert order(rows) == ["A", "B"]
    assert rows[0]["rank"] == 1 and rows[1]["rank"] == 2


def test_a_team_is_ranked_on_its_best_golfer_not_its_average():
    """One golfer in contention beats five in the middle. That is the game."""
    teams, index = league(
        [player("1", 1), player("T140", 140), player("T140", 141)],
        [player("T10", 10), player("T10", 11), player("T10", 12), player("T10", 13)],
    )
    assert order(standings.compute(teams, index)) == ["A", "B"]


# ---------------------------------------------------------------------------
# The tie-break, which is the whole reason the vector exists
# ---------------------------------------------------------------------------

def test_tie_on_the_best_golfer_breaks_on_the_second():
    teams, index = league(
        [player("T1", 1), player("T20", 20)],
        [player("T1", 2), player("T9", 9)],
    )
    rows = standings.compute(teams, index)
    assert order(rows) == ["B", "A"]
    assert rows[1]["decided_at"] == 2               # separated on the 2nd golfer


def test_the_tie_break_walks_as_deep_as_it_has_to():
    shared = [player("T1", 1), player("T4", 4), player("T9", 9)]
    teams, index = league(shared + [player("T30", 30)], shared + [player("T25", 25)])
    rows = standings.compute(teams, index)
    assert order(rows) == ["B", "A"]
    assert rows[1]["decided_at"] == 4


def test_running_out_of_golfers_loses_to_still_having_one():
    """The user's rule, stated exactly: 'the person who has more golfers wins'."""
    teams, index = league(
        [player("T1", 1)],
        [player("T1", 2), player("T80", 80)],
    )
    rows = standings.compute(teams, index)
    assert order(rows) == ["B", "A"]
    assert rows[1]["decided_at"] == 2


def test_extra_golfers_do_not_help_if_the_tie_breaks_first():
    teams, index = league(
        [player("T1", 1), player("2", 2)],
        [player("T1", 2), player("T50", 50), player("T51", 51), player("T52", 52)],
    )
    assert order(standings.compute(teams, index)) == ["A", "B"]


def test_identical_teams_are_reported_tied_not_separated():
    teams, index = league(
        [player("T1", 1), player("T7", 7)],
        [player("T1", 2), player("T7", 8)],
    )
    rows = standings.compute(teams, index)
    assert all(r["tied"] for r in rows)
    assert {r["rank"] for r in rows} == {1}
    assert [r["position"] for r in rows] == ["T1", "T1"]


def test_a_tie_for_first_is_followed_by_third_not_second():
    teams, index = league(
        [player("T1", 1)],
        [player("T1", 2)],
        [player("9", 9)],
    )
    rows = standings.compute(teams, index)
    assert [r["position"] for r in rows] == ["T1", "T1", "3"]


# ---------------------------------------------------------------------------
# Cut, withdrawn, and never in the field
# ---------------------------------------------------------------------------

def test_any_golfer_still_playing_beats_every_cut_golfer():
    teams, index = league(
        [player(None, 74), player(None, 75)],       # both cut
        [player("T70", 70)],                        # made the cut, barely
    )
    assert order(standings.compute(teams, index)) == ["B", "A"]


def test_cut_golfers_are_ordered_among_themselves_by_sort_order():
    """ESPN gives all 74 cut players the position '-'; sortOrder still ranks them."""
    teams, index = league([player(None, 80)], [player(None, 75)])
    assert order(standings.compute(teams, index)) == ["B", "A"]


def test_a_golfer_who_never_teed_off_ranks_below_a_cut_one():
    teams, index = league([None], [player(None, 147)])
    assert order(standings.compute(teams, index)) == ["B", "A"]


def test_a_golfer_who_never_teed_off_still_beats_having_no_golfer():
    """A team that drafted 12 golfers drafted 12 golfers, withdrawals included."""
    teams, index = league(
        [player("T1", 1), None],
        [player("T1", 2)],
    )
    rows = standings.compute(teams, index)
    assert order(rows) == ["A", "B"]
    assert rows[1]["decided_at"] == 2


def test_a_team_with_no_resolvable_golfers_does_not_crash():
    teams, index = league([None, None], [player("1", 1)])
    rows = standings.compute(teams, index)
    assert order(rows) == ["B", "A"]
    assert rows[1]["counting"] == 0 and rows[1]["to_par"] is None


def test_an_empty_roster_ranks_last():
    teams, index = league([], [player(None, 147)])
    assert order(standings.compute(teams, index)) == ["B", "A"]


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

def test_compare_reports_the_depth_that_decided_it():
    assert standings.compare([(0, 1), (0, 5)], [(0, 1), (0, 9)]) == (-1, 2)
    assert standings.compare([(0, 1)], [(0, 2)]) == (-1, 1)
    assert standings.compare([(0, 1)], [(0, 1)]) == (0, None)


def test_compare_is_antisymmetric():
    a, b = [(0, 1), (1, 90)], [(0, 1), (0, 40)]
    left, at_left = standings.compare(a, b)
    right, at_right = standings.compare(b, a)
    assert left == -right and at_left == at_right


def test_padding_is_worse_than_every_real_tier():
    assert standings.PADDING > standings.golfer_rank(None)
    assert standings.PADDING > standings.golfer_rank(player(None, 9999))
    assert standings.PADDING > standings.golfer_rank(player("T1", 1))


# ---------------------------------------------------------------------------
# Detail carried for display
# ---------------------------------------------------------------------------

def test_golfers_come_back_in_leaderboard_order():
    teams, index = league([player("T30", 30), player("2", 2), player(None, 90)])
    rows = standings.compute(teams, index)
    assert [d["rank"] for d in rows[0]["golfers"]] == [(0, 2), (0, 30), (1, 90)]
    assert rows[0]["best"]["rank"] == (0, 2)


def test_counting_excludes_cut_golfers_and_to_par_does_not():
    teams, index = league([player("2", 2, to_par=-5), player(None, 90, to_par=3)])
    row = standings.compute(teams, index)[0]
    assert row["counting"] == 1 and row["roster"] == 2 and row["in_field"] == 2
    assert row["to_par"] == -2


# ---------------------------------------------------------------------------
# Over the real field
# ---------------------------------------------------------------------------

def test_over_the_real_leaderboard(espn_players):
    """
    Five teams dealt the real 147-player field round-robin. The ranking must be total,
    every team must be accounted for, and the leader's best golfer must be the one
    ESPN ranks first among that team's golfers.
    """
    teams = [{"team_id": f"T{i}", "golfers": []} for i in range(5)]
    for i, p in enumerate(espn_players):
        teams[i % 5]["golfers"].append({"golfer_id": p["athlete_id"], "name": p["name"]})

    rows = standings.compute(teams, standings.index_players(espn_players))
    assert len(rows) == 5
    assert sorted(r["team_id"] for r in rows) == [f"T{i}" for i in range(5)]
    assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)

    # Round-robin over a sortOrder-sorted field means team 0 holds the outright leader.
    assert rows[0]["team_id"] == "T0"
    assert rows[0]["best"]["espn"]["sort_order"] == 1
    assert rows[0]["rank"] == 1


def test_every_golfer_on_the_real_field_lands_in_tier_zero_or_one(espn_players):
    for p in espn_players:
        tier, _ = standings.golfer_rank(p)
        assert tier in (0, 1)
        assert (tier == 0) == (p["position_number"] is not None)


# ---------------------------------------------------------------------------
# The golden vector -- the contract with the frontend
# ---------------------------------------------------------------------------

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "standings_golden.json")


def golden_case(players):
    """
    A deliberately awkward five-team league over the finished real field.

    Built to reach every branch of the rule at once: uneven rosters, a genuine tie on
    the best golfer that has to break on the second, a golfer who never teed off, a
    team of nothing but cut golfers, and a team that runs out of golfers mid-comparison.

    Teams are described by their SHAPE rather than by index, so the case stays
    meaningful if the fixture is ever recaptured. tools/make_golden.py regenerates the
    expected output.
    """
    made = [p for p in players if p["position_number"] is not None]
    cut = [p for p in players if p["position_number"] is None]

    shared = {}
    for p in made:
        shared.setdefault(p["position_number"], []).append(p)
    tied = next(v for k, v in sorted(shared.items()) if len(v) >= 2)

    return {
        "teams": [
            # Tie on the best golfer; alpha's second golfer is better, so alpha wins at depth 2.
            {"team_id": "alpha", "golfers": [{"name": tied[0]["name"]}, {"name": made[30]["name"]}]},
            {"team_id": "bravo", "golfers": [{"name": tied[1]["name"]}, {"name": made[50]["name"]},
                                             {"name": cut[0]["name"]}]},
            # Holds the outright leader, plus a golfer who never teed off.
            {"team_id": "charlie", "golfers": [{"name": made[0]["name"]}, {"name": "Tiger Woods"}]},
            # Nothing but cut golfers; echo holds the same best golfer and nobody behind them.
            {"team_id": "delta", "golfers": [{"name": cut[5]["name"]}, {"name": cut[6]["name"]}]},
            {"team_id": "echo", "golfers": [{"name": cut[5]["name"]}]},
        ],
        "by_name": {p["name"]: p for p in players},
    }


def test_golden_vector_matches(espn_final_players):
    case = golden_case(espn_final_players)
    rows = standings.compute(case["teams"], case["by_name"])
    actual = [{"team_id": r["team_id"], "position": r["position"], "rank": r["rank"],
               "decided_at": r["decided_at"], "counting": r["counting"],
               "vector": [list(v) for v in r["vector"]]} for r in rows]
    with open(GOLDEN) as f:
        assert actual == json.load(f)["expected"]


def test_the_golden_case_actually_exercises_the_hard_paths(espn_final_players):
    """A golden file that only covers the easy path is a golden file that proves nothing."""
    case = golden_case(espn_final_players)
    rows = standings.compute(case["teams"], case["by_name"])
    by_id = {r["team_id"]: r for r in rows}

    tiers = {rank[0] for row in rows for rank in row["vector"]}
    assert tiers == {0, 1, 2}                        # playing, cut, and absent all present
    assert len({len(r["vector"]) for r in rows}) > 1                    # uneven rosters

    # alpha and bravo tie on their best golfer and separate on the second.
    assert by_id["alpha"]["vector"][0] == by_id["bravo"]["vector"][0]
    assert by_id["bravo"]["decided_at"] == 2
    assert by_id["alpha"]["rank"] < by_id["bravo"]["rank"]

    # delta and echo share a best golfer; echo has nobody behind them and loses on it.
    assert by_id["delta"]["vector"][0] == by_id["echo"]["vector"][0]
    assert by_id["echo"]["decided_at"] == 2
    assert by_id["delta"]["rank"] < by_id["echo"]["rank"]

    # charlie holds the outright leader, so charlie leads.
    assert by_id["charlie"]["rank"] == 1


def test_cut_and_playing_over_the_finished_field(espn_final_players):
    """
    73 made the cut and 74 did not, and ESPN gives all 74 the position "-". The tiers
    have to sort that out, and sortOrder has to keep every cut player behind every
    player who made it.
    """
    playing = [p for p in espn_final_players if standings.golfer_rank(p)[0] == 0]
    cut = [p for p in espn_final_players if standings.golfer_rank(p)[0] == 1]
    assert len(playing) == 73 and len(cut) == 74
    assert max(p["sort_order"] for p in playing) < min(p["sort_order"] for p in cut)
