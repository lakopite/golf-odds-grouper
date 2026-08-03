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
# Name matching
# ---------------------------------------------------------------------------

NAMES = [
    "Cameron Young", "Zachary Bauchou", "Cameron Davis", "Kris Ventura",
    "Nicolas Echavarria", "Matthew McCarty", "Benjamin James", "Jordan L. Smith",
    "Hao-Tong Li", "Rasmus Højgaard", "Thorbjørn Olesen", "Erik Van Rooyen",
    "C.T. Pan", "Séamus Power", "Tiger Woods", "Davis Love III",
]


def test_normalization_agrees():
    js = run_node("console.log(JSON.stringify(INPUT.map(n => GolfPool.normalizeName(n))));", NAMES)
    assert js == [espn_leaderboard.normalize_name(n) for n in NAMES]


def test_initial_last_key_agrees():
    js = run_node("console.log(JSON.stringify(INPUT.map(n => GolfPool.initialLastKey(n))));", NAMES)
    assert js == [espn_leaderboard.initial_last_key(n) for n in NAMES]


@pytest.mark.parametrize("fixture", [MID, FINAL], ids=["mid-round", "final-with-cut"])
def test_the_join_agrees_over_the_whole_field(fixture):
    """Every name in the field, plus the awkward ones, matched by both sides."""
    with open(fixture) as f:
        payload = json.load(f)
    _, players = espn_leaderboard.parse_leaderboard(payload)
    names = NAMES + [p["name"] for p in players]

    js = run_node("""
      const {players} = GolfPool.parseLeaderboard(INPUT.payload);
      const index = GolfPool.buildIndex(players);
      console.log(JSON.stringify(INPUT.names.map(function (n) {
        const hit = GolfPool.matchGolfer({name: n}, index, {});
        return [hit.how, hit.player ? hit.player.athleteId : null];
      })));
    """, {"payload": payload, "names": names})

    index = espn_leaderboard.build_index(players)
    expected = []
    for name in names:
        player, how = espn_leaderboard.match_golfer(name, index)
        expected.append([how, player["athlete_id"] if player else None])
    assert js == expected


# ---------------------------------------------------------------------------
# The standings rule -- the thing that actually matters
# ---------------------------------------------------------------------------

def _js_standings(payload, teams):
    return run_node("""
      const {players} = GolfPool.parseLeaderboard(INPUT.payload);
      const index = GolfPool.buildIndex(players);
      const byTeam = new Map();
      const teams = INPUT.teams.map(function (t) {
        byTeam.set(t.team_id, t.golfers);
        return {team_id: t.team_id};
      });
      const rows = GolfPool.computeStandings(teams, byTeam, function (g) {
        return GolfPool.matchGolfer(g, index, INPUT.aliases || {}).player;
      });
      console.log(JSON.stringify(rows.map(r => ({
        team_id: r.team.team_id, position: r.position, rank: r.rank,
        decided_at: r.decidedAt === undefined ? null : r.decidedAt,
        counting: r.counting, vector: r.vector, to_par: r.toPar,
      }))));
    """, {"payload": payload, "teams": teams})


def _py_standings(payload, teams):
    _, players = espn_leaderboard.parse_leaderboard(payload)
    matches, _ = espn_leaderboard.match_field(
        [g["name"] for t in teams for g in t["golfers"]], players)
    rows = standings.compute(teams, standings.index_players(players, matches))
    return [{
        "team_id": r["team_id"], "position": r["position"], "rank": r["rank"],
        "decided_at": r["decided_at"], "counting": r["counting"],
        "vector": [list(v) for v in r["vector"]], "to_par": r["to_par"],
    } for r in rows]


def test_standings_agree_on_the_golden_case(espn_final_payload):
    from test_standings import golden_case
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = golden_case(players)["teams"]
    assert _js_standings(espn_final_payload, teams) == _py_standings(espn_final_payload, teams)


def test_js_matches_the_checked_in_golden_file(espn_final_payload):
    """
    Belt and braces: the JS is compared against the file on disk, not just against a
    freshly computed Python answer. If both implementations drift the same way, this
    still catches it.
    """
    from test_standings import golden_case, GOLDEN
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = golden_case(players)["teams"]
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
        teams[i % n_teams]["golfers"].append({"name": p["name"]})

    assert _js_standings(payload, teams) == _py_standings(payload, teams)


def test_standings_agree_when_golfers_are_missing_from_the_field(espn_final_payload):
    """A build made on Wednesday can hold golfers who withdrew before Thursday."""
    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    teams = [
        {"team_id": "a", "golfers": [{"name": players[0]["name"]}, {"name": "Tiger Woods"}]},
        {"team_id": "b", "golfers": [{"name": players[0]["name"]}]},
        {"team_id": "c", "golfers": [{"name": "Nobody At All"}]},
        {"team_id": "d", "golfers": []},
    ]
    assert _js_standings(espn_final_payload, teams) == _py_standings(espn_final_payload, teams)


def test_standings_agree_on_a_pre_tournament_payload():
    """ESPN publishes zero competitors before play starts. Neither side may crash."""
    payload = {"events": [{"id": "1", "name": "Not Started",
                           "status": {"type": {"state": "pre"}},
                           "competitions": [{"status": {"period": 0}, "competitors": []}]}]}
    teams = [{"team_id": "a", "golfers": [{"name": "Cameron Young"}]},
             {"team_id": "b", "golfers": [{"name": "Rory McIlroy"}, {"name": "Jon Rahm"}]}]
    assert _js_standings(payload, teams) == _py_standings(payload, teams)
