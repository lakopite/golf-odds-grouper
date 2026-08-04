"""
Tests for league.py.

The league file is hand-written, so most of what matters here is what happens when it
is wrong. A duplicate team name silently collapses two teams into one id; a JSON
object where a list belongs has a length, so it passes every downstream check before
failing somewhere unrecognisable. Both should fail here, loudly, by name.
"""

import json
import os

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


# ---------------------------------------------------------------------------
# The league's own identity
#
# The scoreboard has a masthead, and what goes in it is a fact about the league rather
# than about the template -- a template that hard-coded one league's crest would be
# that league's template.
#
# `logo` is a SLUG, not a path: the name of a directory under leagues/ holding logo.png
# and banner.png. The league file names the art, the result file passes the name along,
# and the images are read once at export. Nothing in this half of the pipeline opens an
# image at all.
# ---------------------------------------------------------------------------

def test_branding_is_read_off_the_object_form(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({
        "league_name": "WCW", "logo": "wcw", "tagline": "10th Anniversary",
        "teams": [{"team_name": "A", "player_name": "Ann"}]}))
    loaded = league.load_league(str(path))
    assert loaded["logo"] == "wcw"
    assert loaded["tagline"] == "10th Anniversary"


@pytest.mark.parametrize("payload", [
    [{"team_name": "A", "player_name": "Ann"}],
    {"league_name": "X", "teams": [{"team_name": "A", "player_name": "Ann"}]},
    {"league_name": "X", "logo": "  ", "tagline": None,
     "teams": [{"team_name": "A", "player_name": "Ann"}]},
], ids=["bare-list", "no-branding", "blank-branding"])
def test_branding_keys_are_always_present_and_null_when_unset(tmp_path, payload):
    """
    Null rather than absent. A page that has to tell "this league has no logo" from
    "this build predates logos" will get it wrong, and the difference is worth nothing
    to anybody. A blank string counts as unset -- it is what a hand-edited file grows
    when somebody clears a value, and an <img> with no src is worse than no <img>.
    """
    path = tmp_path / "l.json"
    path.write_text(json.dumps(payload))
    loaded = league.load_league(str(path))
    assert [loaded[f] for f in league.BRANDING_FIELDS] == [None, None]


@pytest.mark.parametrize("field", ["logo", "tagline"])
def test_branding_that_is_not_a_string_is_refused(tmp_path, field):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"league_name": "X", field: {"src": "a.png"},
                                "teams": [{"team_name": "A", "player_name": "Ann"}]}))
    with pytest.raises(ValueError, match=f"'{field}' must be a string or null"):
        league.load_league(str(path))


def test_a_path_where_a_slug_belongs_is_refused(tmp_path):
    """
    "logos/wcw-crest.png" is what every league file said before the art moved into
    leagues/<slug>/, and it is what anybody who has seen one will type. Taken as a slug
    it names a directory nobody will ever create, and the only symptom would be an
    empty masthead at the far end of a build somebody has already waited for.
    """
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"league_name": "X", "logo": "logos/wcw-crest.png",
                                "teams": [{"team_name": "A", "player_name": "Ann"}]}))
    with pytest.raises(ValueError, match="slug like"):
        league.load_league(str(path))


def test_false_no_longer_means_anything_and_says_so(tmp_path):
    """It used to mean "this league has none, do not fill in the default". There is no
    default any more, so unset means that -- and a file that still says `false` should
    hear why rather than be told a bool is not a string."""
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"league_name": "X", "logo": False,
                                "teams": [{"team_name": "A", "player_name": "Ann"}]}))
    with pytest.raises(ValueError, match="unset means that now"):
        league.load_league(str(path))


def test_write_ids_keeps_the_branding_and_invents_none(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"league_name": "X", "tagline": "Season 4",
                                "teams": [{"team_name": "A", "player_name": "Ann"}]}))
    league.write_ids(str(path), league.load_league(str(path)))
    written = json.loads(path.read_text())
    assert written["tagline"] == "Season 4"
    # It rewrites the user's file. Adding a null key they never typed is how a tool
    # teaches people not to run it.
    assert "logo" not in written
    assert league.load_league(str(path))["tagline"] == "Season 4"


# ---------------------------------------------------------------------------
# From a slug to two files
# ---------------------------------------------------------------------------

def art_dir(tmp_path, *names):
    """A leagues directory with one slug in it, holding the files named."""
    d = tmp_path / "wcw"
    d.mkdir()
    for name in names:
        (d / name).write_bytes(b"\x89PNG")
    return str(tmp_path)


def test_a_slug_resolves_to_the_two_images_it_names(tmp_path):
    found = league.art_files("wcw", art_dir(tmp_path, "logo.png", "banner.png"))
    assert found["logo"].endswith(os.path.join("wcw", "logo.png"))
    assert found["banner"].endswith(os.path.join("wcw", "banner.png"))


def test_either_image_may_be_missing(tmp_path):
    """A league with a badge and no banner is an ordinary league and the page draws it,
    so half an answer is an answer rather than an error."""
    found = league.art_files("wcw", art_dir(tmp_path, "logo.png"))
    assert found["logo"] and found["banner"] is None


def test_a_slug_that_names_nothing_is_two_nones(tmp_path):
    assert league.art_files("nope", str(tmp_path)) == {"logo": None, "banner": None}
    assert league.art_files(None, str(tmp_path)) == {"logo": None, "banner": None}


def test_the_extension_is_whatever_the_file_was_saved_as(tmp_path):
    """SVG costs a page almost nothing and is the right answer for a drawn crest; PNG
    is what a generated one arrives as. The name is fixed, the extension is not."""
    found = league.art_files("wcw", art_dir(tmp_path, "logo.svg", "banner.jpg"))
    assert found["logo"].endswith(".svg") and found["banner"].endswith(".jpg")


def test_the_checked_in_example_league_resolves(tmp_path):
    """The worked example is the one league anybody can run without writing a file, so
    its art has to be findable from a clean checkout."""
    example = league.load_league(os.path.join(league.LEAGUES_DIR, "example-league.json"))
    found = league.art_files(example["logo"])
    assert found["logo"] and found["banner"]


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
