"""
Tests for league.py.

The league file is hand-written, so most of what matters here is what happens when it
is wrong. A duplicate team name silently collapses two teams into one id; a JSON
object where a list belongs has a length, so it passes every downstream check before
failing somewhere unrecognisable. Both should fail here, loudly, by name.
"""

import json

import pytest

import league


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def test_ids_are_deterministic_across_processes():
    a = league.league_id_for("Sunday Fivesome")
    b = league.league_id_for("Sunday Fivesome")
    assert a == b
    assert league.team_id_for(a, "Bogey Boys") == league.team_id_for(b, "Bogey Boys")


def test_ids_are_namespaced_by_league():
    """The same team name in two leagues must not be the same team."""
    one = league.league_id_for("League One")
    two = league.league_id_for("League Two")
    assert league.team_id_for(one, "Bogey Boys") != league.team_id_for(two, "Bogey Boys")


def test_ids_are_uuids():
    import uuid
    value = league.team_id_for(league.league_id_for("L"), "T")
    assert uuid.UUID(value).version == 5


def test_explicit_team_id_wins(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps([
        {"team_name": "Alpha", "player_name": "Ann", "team_id": "pinned-id"},
    ]))
    assert league.load_league(str(path))["teams"][0]["team_id"] == "pinned-id"


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_bare_list_takes_its_name_from_the_filename(tmp_path):
    path = tmp_path / "sunday-fivesome.json"
    path.write_text(json.dumps([{"team_name": "Alpha", "player_name": "Ann"}]))
    assert league.load_league(str(path))["league_name"] == "Sunday Fivesome"


def test_object_form_carries_the_league_name(league_file):
    path, _ = league_file
    loaded = league.load_league(path)
    assert loaded["league_name"] == "Test League"
    assert loaded["league_slug"] == "test-league"
    assert len(loaded["teams"]) == 4


def test_missing_logo_becomes_none_rather_than_absent(league_file):
    path, _ = league_file
    teams = {t["team_name"]: t for t in league.load_league(path)["teams"]}
    assert teams["Bravo"]["team_logo"] is None
    assert teams["Alpha"]["team_logo"] == "logos/a.png"


def test_unknown_fields_ride_along(tmp_path, capsys):
    path = tmp_path / "l.json"
    path.write_text(json.dumps([{"team_name": "A", "player_name": "Ann", "motto": "fore"}]))
    team = league.load_league(str(path))["teams"][0]
    assert team["motto"] == "fore"
    assert "motto" in capsys.readouterr().out       # and it says so, in case it is a typo


# ---------------------------------------------------------------------------
# Failure modes, each of which used to be silent somewhere
# ---------------------------------------------------------------------------

def test_duplicate_team_names_are_refused(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps([
        {"team_name": "Alpha", "player_name": "Ann"},
        {"team_name": "Alpha", "player_name": "Ben"},
    ]))
    with pytest.raises(ValueError, match="repeats team_name"):
        league.load_league(str(path))


def test_object_where_a_list_belongs(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"Alpha": "Ann"}))
    with pytest.raises(ValueError, match="teams"):
        league.load_league(str(path))


@pytest.mark.parametrize("teams,pattern", [
    ([], "no teams"),
    ([{"player_name": "Ann"}], "team_name"),
    ([{"team_name": "Alpha"}], "player_name"),
    ([{"team_name": "  ", "player_name": "Ann"}], "team_name"),
    ([{"team_name": "Alpha", "player_name": 7}], "player_name"),
    (["Alpha"], "not an object"),
])
def test_bad_team_entries(tmp_path, teams, pattern):
    path = tmp_path / "l.json"
    path.write_text(json.dumps(teams))
    with pytest.raises(ValueError, match=pattern):
        league.load_league(str(path))


def test_missing_file_says_what_the_file_should_hold(tmp_path):
    with pytest.raises(ValueError, match="team_name"):
        league.load_league(str(tmp_path / "nope.json"))


def test_invalid_json_is_reported_as_invalid_json(tmp_path):
    path = tmp_path / "l.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        league.load_league(str(path))


# ---------------------------------------------------------------------------
# write_ids
# ---------------------------------------------------------------------------

def test_write_ids_pins_them_and_reloads_identically(league_file):
    path, _ = league_file
    first = league.load_league(path)
    league.write_ids(path, first)
    second = league.load_league(path)
    assert [t["team_id"] for t in first["teams"]] == [t["team_id"] for t in second["teams"]]
    assert json.loads(open(path).read())["teams"][0]["team_id"] == first["teams"][0]["team_id"]


def test_pinned_ids_survive_a_rename(league_file):
    """The point of write_ids: a rename after pinning keeps the team's identity."""
    path, _ = league_file
    league.write_ids(path, league.load_league(path))
    pinned = league.load_league(path)["teams"][0]["team_id"]

    payload = json.loads(open(path).read())
    payload["teams"][0]["team_name"] = "Alpha Renamed"
    open(path, "w").write(json.dumps(payload))

    assert league.load_league(path)["teams"][0]["team_id"] == pinned


def test_slugify():
    assert league.slugify("Sunday Fivesome") == "sunday-fivesome"
    assert league.slugify("2026 Wyndham Championship!") == "2026-wyndham-championship"
    assert league.slugify("Café Crème") == "cafe-creme"
    assert league.slugify("---") == "league"
