"""
Parity between standings.py and frontend/template/lib.js.

The rule that decides the pool exists twice: in Python, where it can be tested, and in
JavaScript, where it actually runs. Two implementations of a rule is one implementation
and one rumour unless something checks them against each other, so this feeds both the
same ESPN payload and the same league and fails if they disagree about anything --
the finishing order, the tie flags, or the depth each tie broke at.

Needs node on PATH. Skipped without it rather than failing, because the Python side is
still worth testing on a machine that has no JavaScript runtime.
"""

import json
import os
import shutil
import subprocess

import pytest

import espn_leaderboard
import standings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "frontend", "template", "lib.js")
FINAL = os.path.join(ROOT, "tests", "fixtures", "espn_final_with_cut.json")
MID = os.path.join(ROOT, "espn-api", "lb.json")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


def run_node(script, payload):
    """Run a snippet with GolfPool in scope and the payload on stdin as JSON."""
    source = f"""
const fs = require('fs');
const GolfPool = require({json.dumps(LIB)});
const INPUT = JSON.parse(fs.readFileSync(0, 'utf8'));
{script}
"""
    proc = subprocess.run(["node", "-e", source], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Leaderboard parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", [MID, FINAL], ids=["mid-round", "final-with-cut"])
def test_leaderboard_parsing_agrees(fixture):
    with open(fixture) as f:
        payload = json.load(f)

    _, py_players = espn_leaderboard.parse_leaderboard(payload)
    js_players = run_node("""
      const {players} = GolfPool.parseLeaderboard(INPUT);
      console.log(JSON.stringify(players.map(p => ({
        athleteId: p.athleteId, name: p.name, sortOrder: p.sortOrder,
        position: p.position === undefined ? null : p.position,
        positionNumber: p.positionNumber, toPar: p.toPar, tied: p.tied,
      }))));
    """, payload)

    assert len(js_players) == len(py_players)
    for py, js in zip(py_players, js_players):
        assert js["athleteId"] == py["athlete_id"]
        assert js["name"] == py["name"]
        assert js["sortOrder"] == py["sort_order"]
        assert js["positionNumber"] == py["position_number"]
        assert js["toPar"] == py["to_par"]
        assert js["tied"] == py["tied"]


# ---------------------------------------------------------------------------
# The join -- one key, and both sides use it
# ---------------------------------------------------------------------------
#
# There is no name-matching parity left to check, and that is the improvement rather
# than a gap. lib.js used to carry a second copy of the transliteration table, the
# normaliser and a first-initial fallback, and this file existed partly to stop the two
# drifting. Now the build writes an ESPN athlete id onto every golfer it resolved and
# both sides look that id up, so there is one implementation of the join and it is a
# dictionary.
#
# What still needs checking is that both sides look it up the SAME WAY, including when
# they are given a golfer they cannot resolve. See
# test_a_golfer_the_build_could_not_settle_scores_on_neither_side.


def golfer(name, player=None):
    """
    A golfer as a result file writes one: a Kalshi name, and the ESPN athlete the build
    bound it to -- or an explicit null when the build bound it to nobody.

    Both keys are set because the two implementations are handed the same object and
    each reads its own: standings.py takes `espn`, lib.js takes `espn.athlete_id`. If
    they ever disagree about which golfer is which, that is what these tests are for.
    """
    return {"name": name, "espn": {"athlete_id": player["athlete_id"] if player else None}}


# ---------------------------------------------------------------------------
# The standings rule -- the thing that actually matters
# ---------------------------------------------------------------------------

def _js_standings(payload, teams):
    return run_node("""
      const {players} = GolfPool.parseLeaderboard(INPUT.payload);
      const index = GolfPool.indexByAthleteId(players);
      const byTeam = new Map();
      const teams = INPUT.teams.map(function (t) {
        byTeam.set(t.team_id, t.golfers);
        return {team_id: t.team_id};
      });
      const rows = GolfPool.computeStandings(teams, byTeam, function (g) {
        return GolfPool.resolveGolfer(g, index);
      });
      console.log(JSON.stringify(rows.map(r => ({
        team_id: r.team.team_id, position: r.position, rank: r.rank,
        decided_at: r.decidedAt === undefined ? null : r.decidedAt,
        counting: r.counting, vector: r.vector, to_par: r.toPar,
      }))));
    """, {"payload": payload, "teams": teams})


def _py_standings(payload, teams):
    _, players = espn_leaderboard.parse_leaderboard(payload)
    rows = standings.compute(teams, standings.index_players(players))
    return [{
        "team_id": r["team_id"], "position": r["position"], "rank": r["rank"],
        "decided_at": r["decided_at"], "counting": r["counting"],
        "vector": [list(v) for v in r["vector"]], "to_par": r["to_par"],
    } for r in rows]


def test_both_sides_resolve_a_golfer_whose_kalshi_name_is_not_the_espn_one(espn_final_payload):
    """
    The case the deleted matcher existed for, and the one that would expose either side
    quietly keeping a name fallback the other does not have.

    These golfers carry names ESPN has never heard of. Only the athlete id can find
    them, so if either implementation is secretly matching on the name it either scores
    nobody here or scores somebody the other side did not.
    """
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = [
        {"team_id": "a", "golfers": [golfer("A Golfer Nobody Calls That", players[0]),
                                     golfer("Another Alias Entirely", players[9])]},
        {"team_id": "b", "golfers": [golfer("Third Made-Up Name", players[1])]},
    ]
    js, py = _js_standings(espn_final_payload, teams), _py_standings(espn_final_payload, teams)
    assert js == py
    assert [r["team_id"] for r in py] == ["a", "b"], "the ids resolved, so 'a' holds the leader"
    assert all(r["counting"] for r in py)


def test_a_golfer_the_build_could_not_settle_scores_on_neither_side(espn_final_payload):
    """
    A null athlete id is a real answer: the build looked at this week's field and bound
    this golfer to nobody. The page has no way to reconsider, and this reference
    implementation must not either -- even though the name it carries IS in the field.

    That is the asymmetry this test exists to hold shut. lib.js cannot fall back to a
    name because it has no name matcher left; standings.py could, and must not, or the
    two would disagree about exactly the golfers a review was meant to settle.
    """
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    unsettled = dict(golfer(players[0]["name"], None))       # right name, no id
    teams = [
        {"team_id": "a", "golfers": [unsettled]},
        {"team_id": "b", "golfers": [golfer(players[5]["name"], players[5])]},
    ]
    js, py = _js_standings(espn_final_payload, teams), _py_standings(espn_final_payload, teams)
    assert js == py
    assert py[0]["team_id"] == "b", "the settled golfer wins; the unsettled one scores nothing"
    assert py[1]["counting"] == 0


def as_a_result_file_would(teams, players):
    """
    Bind a by-name league onto the field, the way a build does before either side reads
    it.

    test_standings.py and tools/make_golden.py describe their leagues by name, which is
    the right shape for a fixture a person maintains. Neither implementation is given
    names any more, so the parity harness does the binding the build would have done --
    once, in one place, so both sides are handed the identical object.
    """
    by_name = {p["name"]: p for p in players}
    return [{**t, "golfers": [golfer(g["name"], by_name.get(g["name"])) for g in t["golfers"]]}
            for t in teams]


def test_standings_agree_on_the_golden_case(espn_final_payload):
    from test_standings import golden_case
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = as_a_result_file_would(golden_case(players)["teams"], players)
    assert _js_standings(espn_final_payload, teams) == _py_standings(espn_final_payload, teams)


def test_js_matches_the_checked_in_golden_file(espn_final_payload):
    """
    Belt and braces: the JS is compared against the file on disk, not just against a
    freshly computed Python answer. If both implementations drift the same way, this
    still catches it.
    """
    from test_standings import golden_case, GOLDEN
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = as_a_result_file_would(golden_case(players)["teams"], players)
    with open(GOLDEN) as f:
        expected = json.load(f)["expected"]
    actual = [{k: v for k, v in row.items() if k != "to_par"}
              for row in _js_standings(espn_final_payload, teams)]
    assert actual == expected


@pytest.mark.parametrize("fixture", [MID, FINAL], ids=["mid-round", "final-with-cut"])
@pytest.mark.parametrize("n_teams", [2, 5, 13])
def test_standings_agree_on_round_robin_leagues(fixture, n_teams):
    """
    The whole field dealt out round-robin, at three league sizes. Round-robin over a
    rank-sorted field produces exactly the near-ties the rule is built to resolve --
    adjacent teams share their best golfer's neighbourhood all the way down.
    """
    with open(fixture) as f:
        payload = json.load(f)
    _, players = espn_leaderboard.parse_leaderboard(payload)

    teams = [{"team_id": f"team-{i}", "golfers": []} for i in range(n_teams)]
    for i, p in enumerate(players):
        teams[i % n_teams]["golfers"].append(golfer(p["name"], p))

    assert _js_standings(payload, teams) == _py_standings(payload, teams)


def test_standings_agree_when_golfers_are_missing_from_the_field(espn_final_payload):
    """A draw made on Wednesday can hold golfers who never teed off on Thursday."""
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = [
        {"team_id": "a", "golfers": [golfer(players[0]["name"], players[0]),
                                     golfer("Tiger Woods", None)]},
        {"team_id": "b", "golfers": [golfer(players[0]["name"], players[0])]},
        {"team_id": "c", "golfers": [golfer("Nobody At All", None)]},
        {"team_id": "d", "golfers": []},
    ]
    assert _js_standings(espn_final_payload, teams) == _py_standings(espn_final_payload, teams)


def test_standings_agree_on_a_pre_tournament_payload():
    """ESPN publishes zero competitors before play starts. Neither side may crash."""
    payload = {"events": [{"id": "1", "name": "Not Started",
                           "status": {"type": {"state": "pre"}},
                           "competitions": [{"status": {"period": 0}, "competitors": []}]}]}
    teams = [{"team_id": "a", "golfers": [golfer("Cameron Young", {"athlete_id": "1"})]},
             {"team_id": "b", "golfers": [golfer("Rory McIlroy", {"athlete_id": "2"}),
                                          golfer("Jon Rahm", {"athlete_id": "3"})]}]
    assert _js_standings(payload, teams) == _py_standings(payload, teams)
