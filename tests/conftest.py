import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


@pytest.fixture
def kalshi_raw():
    """A trimmed live GET /markets payload: 18 active markets plus one finalized."""
    return load_fixture("kalshi_markets_raw.json")


@pytest.fixture
def kalshi_markets(kalshi_raw):
    return kalshi_raw["markets"]


@pytest.fixture
def kalshi_odds_file():
    """The converted odds file that kalshi_odds.py --write-odds-file emits."""
    return load_fixture("kalshi_golfers_file.json")


@pytest.fixture
def dk_payload():
    """The legacy DraftKings eventgroup envelope."""
    return load_fixture("dk_data.json")


@pytest.fixture
def participants():
    return load_fixture("participants.json")


# The ESPN payload is read from espn-api/ rather than copied into fixtures/. It is
# 584 KB and already checked in, and a second copy is a second thing to keep in step
# with reality -- the measurements in espn_leaderboard.py's docstring are against
# THIS file, so the tests need to be too.
ESPN_PAYLOAD = os.path.join(ROOT, "espn-api", "lb.json")


@pytest.fixture
def espn_payload():
    """A live Rocket Classic Round 2 leaderboard: 147 competitors, mid-round."""
    with open(ESPN_PAYLOAD) as f:
        return json.load(f)


@pytest.fixture
def espn_players(espn_payload):
    import espn_leaderboard
    return espn_leaderboard.parse_leaderboard(espn_payload)[1]


@pytest.fixture
def espn_final_payload():
    """
    The same tournament finished: 73 made the cut, 74 did not.

    The mid-round payload cannot exercise the cut path at all -- nobody is cut in
    round 2 -- and the cut path is where the standings rule has its second tier.
    Trimmed of statistics, links and hole detail; every field the parser reads is
    verbatim.
    """
    return load_fixture("espn_final_with_cut.json")


@pytest.fixture
def espn_final_players(espn_final_payload):
    import espn_leaderboard
    return espn_leaderboard.parse_leaderboard(espn_final_payload)[1]


@pytest.fixture
def league_file(tmp_path):
    """A four-team league written to a temp path, returned as (path, payload)."""
    payload = {
        "league_name": "Test League",
        "teams": [
            {"team_name": "Alpha", "player_name": "Ann", "team_logo": "logos/a.png"},
            {"team_name": "Bravo", "player_name": "Ben"},
            {"team_name": "Charlie", "player_name": "Cat"},
            {"team_name": "Delta", "player_name": "Dee"},
        ],
    }
    path = tmp_path / "test-league.json"
    path.write_text(json.dumps(payload))
    return str(path), payload


# ---------------------------------------------------------------------------
# The browser, and the competition the render suites drive it against
#
# Two templates ship and each has its own render suite -- the same claims about the
# same data, checked against two different sets of selectors. Everything up to the
# bundling is identical, so it lives here rather than in whichever suite happened to
# be written first.
# ---------------------------------------------------------------------------

ESPN_EVENT_ID = "401811960"
LEADERBOARD_GLOB = "**/site.web.api.espn.com/**"


def chromium_executable():
    """
    A browser Playwright did not download itself.

    Environments that pre-install Chromium often pin a different build number than the
    Playwright package expects, and the default launch then fails on a path that does
    not exist. Point at whatever is actually on disk before giving up.

    Two rules, both learned by getting them wrong. Candidates are filtered down to a
    file that is actually EXECUTABLE, because `pw-browsers` holds a directory per build
    (`chromium-1194`, `chromium_headless_shell-1194`) alongside the binary, and handing
    Playwright a directory fails exactly like handing it nothing -- the suite then
    reports "no chromium available" on a machine carrying three of them, which is the
    worst way for a browser test to be wrong: it never runs and never says so. And the
    per-build directories are searched directly, rather than trusting `chromium` to be
    the binary; here it happens to be a symlink to one, but that is this image's
    convention and not a guarantee.
    """
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), "/opt/pw-browsers"]
    candidates = [os.environ.get("CHROMIUM_PATH")]
    for root in roots:
        if root:
            candidates += sorted(glob.glob(os.path.join(root, "chromium*", "chrome-linux", "chrome")))
            candidates += sorted(glob.glob(os.path.join(root, "chromium*", "**", "headless_shell"),
                                           recursive=True))
    candidates += ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.fixture(scope="session")
def browser():
    """
    One browser for the whole run. Skipped, not failed, without Playwright or a binary:
    the Python side is still worth testing on a machine with no browser on it.
    """
    api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:                            # noqa: BLE001 -- retried below
            path = chromium_executable()
            if not path:
                pytest.skip("no chromium available")
            try:
                b = p.chromium.launch(executable_path=path)
            except Exception as exc:                 # noqa: BLE001
                pytest.skip(f"no chromium available: {exc}")
        yield b
        b.close()


@pytest.fixture
def serve_espn(espn_final_payload):
    """
    A route handler that answers the leaderboard with the checked-in finished field,
    CORS header and all. Call it with a payload to answer with something else.
    """
    def handler(payload=None):
        body = json.dumps(espn_final_payload if payload is None else payload)
        return lambda route: route.fulfill(
            status=200, content_type="application/json",
            headers={"access-control-allow-origin": "*"}, body=body)
    return handler


@pytest.fixture
def rocket_classic(espn_final_payload):
    """
    A four-team league over the real finished Rocket Classic field, as a result file.

    Built here rather than by build_competition.py because that needs a live Kalshi
    pull, and the point of the render suites is the page rather than the pull. The shape
    is the shape build_competition.py emits -- test_build_competition.py holds that to
    account separately.
    """
    import espn_leaderboard
    from test_build_competition import golfer_name, live_stage, make_result

    _, players = espn_leaderboard.parse_leaderboard(espn_final_payload)
    names = [golfer_name(i) for i in range(len(players))]
    result = make_result(n_teams=4, n_golfers=len(players), espn=live_stage(names))

    # Re-label the synthetic field with the real one, keeping the odds and the deal.
    # Round-robin by rank, so team 0 holds the winner and the leaderboard order of the
    # teams is known in advance.
    #
    # The athlete id is the point. There is no runtime name join any more, and the baked
    # id is the only key the page has, so bake the REAL one.
    ordered = sorted(result["golfers"], key=lambda g: -g["odds"]["raw"])
    for golfer, player in zip(ordered, players):
        golfer["name"] = player["name"]
        golfer["espn"] = {"athlete_id": player["athlete_id"], "display_name": player["name"],
                          "headshot": player["headshot"], "country": player["country"],
                          "match": "exact", "in_field": True}
    by_id = {g["golfer_id"]: g for g in result["golfers"]}
    for team in result["teams"]:
        team["golfer_names"] = [by_id[gid]["name"] for gid in team["golfer_ids"]]

    result["sources"]["espn"]["event_id"] = ESPN_EVENT_ID
    url = espn_leaderboard.leaderboard_url(ESPN_EVENT_ID)
    result["sources"]["espn"]["leaderboard_endpoint"] = url
    result["live"]["espn_leaderboard_url"] = url
    result["live"]["espn_event_id"] = ESPN_EVENT_ID
    result["tournament"]["name"] = "Rocket Classic"
    return {"result": result, "players": players}


@pytest.fixture
def groups_result(rocket_classic):
    """
    The same competition as the build makes it before the first tee time: `live` is
    null, no ESPN block on any golfer, and nothing for the page to fetch.
    """
    result = json.loads(json.dumps(rocket_classic["result"]))
    result["build_mode"] = "groups"
    result["live"] = None
    result["sources"]["espn"]["field_size_at_build"] = 0
    result["sources"]["espn"]["match_report"] = None
    for golfer in result["golfers"]:
        golfer["espn"] = None
    return result


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits a real API; runs only when KALSHI_LIVE=1 / ESPN_LIVE=1",
    )
