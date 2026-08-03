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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: hits the real Kalshi API; runs only when KALSHI_LIVE=1"
    )
