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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: hits the real Kalshi API; runs only when KALSHI_LIVE=1"
    )
