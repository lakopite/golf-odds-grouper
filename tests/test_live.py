"""
Live checks against the real Kalshi API.

Skipped unless KALSHI_LIVE=1, so the default suite stays offline and deterministic:

    KALSHI_LIVE=1 python -m pytest tests/test_live.py -v

These exist because the offline fixtures can only prove the code is self-consistent.
They cannot prove the endpoint still answers, still returns money as strings, or still
synthesizes a YES ask on an untraded golfer. Run them before trusting a season's pull.
"""

import os

import pytest

import kalshi_odds

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("KALSHI_LIVE") != "1",
        reason="live API test; set KALSHI_LIVE=1 to run",
    ),
]

SERIES = "KXPGATOUR"


@pytest.fixture(scope="module")
def live_event():
    return kalshi_odds.latest_event(SERIES)


def test_the_endpoint_still_answers(live_event):
    assert live_event["event_ticker"].startswith(SERIES + "-")
    assert live_event["active_count"] > 0


def test_a_full_field_arrives_in_one_page(live_event):
    """limit=500 covers any golf field, so no cursor loop should be needed."""
    assert 50 < len(live_event["markets"]) < 500


def test_money_fields_are_still_strings(live_event):
    m = live_event["markets"][0]
    assert isinstance(m["yes_ask_dollars"], str)
    assert isinstance(m["yes_bid_dollars"], str)


def test_every_active_market_still_quotes_an_ask(live_event):
    """
    The structural guarantee the ask price mode rests on. If this fails, Kalshi
    stopped synthesizing the YES ask off the NO book and to_golfers() will start
    raising -- which is the intended behaviour, not a bug to paper over.
    """
    golfers = kalshi_odds.to_golfers(live_event["markets"], price="ask")
    assert all(g["odds"] > 0 for g in golfers)


def test_the_bid_book_is_still_the_wrong_input(live_event):
    """Documented measurement: the bid side sums under 1.0 and leaves golfers at zero."""
    bids = kalshi_odds.to_golfers(live_event["markets"], price="bid")
    asks = kalshi_odds.to_golfers(live_event["markets"], price="ask")
    assert sum(g["odds"] for g in bids) < sum(g["odds"] for g in asks)
    assert any(g["odds"] == 0 for g in bids)


def test_the_ask_book_carries_an_overround(live_event):
    """A Winner book should sum above 1.0. Below 1.0 means the field is stale or thin."""
    total = sum(g["odds"] for g in kalshi_odds.to_golfers(live_event["markets"]))
    assert 1.0 < total < 2.0


def test_the_winner_market_still_ticks_in_deci_cents(live_event):
    active = [m for m in live_event["markets"] if m["status"] == "active"]
    assert active[0]["price_level_structure"] == "tapered_deci_cent"


def test_golfer_ids_are_present_and_unique(live_event):
    golfers = kalshi_odds.to_golfers(live_event["markets"])
    ids = [g["golfer_id"] for g in golfers]
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_fetch_golfers_resolves_the_current_event_unaided():
    event_ticker, golfers, report = kalshi_odds.fetch_golfers(series=SERIES)
    assert event_ticker.startswith(SERIES + "-")
    assert report["golfers"] == len(golfers)
    assert golfers[0]["odds"] >= golfers[-1]["odds"]


def test_events_carry_a_ticker_but_no_usable_date():
    """
    Documents why latest_event() does not trust list position. If a date field ever
    appears here, latest_event() can be simplified to read it.
    """
    events = kalshi_odds.events_for(SERIES)
    assert len(events) > 1
    assert all(e.get("event_ticker") for e in events)
    assert not any(e.get("strike_date") for e in events)


def test_the_chosen_event_is_the_newest_tradeable_one():
    """latest_event() picks on market created_time; check nothing newer is being missed."""
    chosen = kalshi_odds.latest_event(SERIES)
    newest = max((m.get("created_time") or "")
                 for m in chosen["markets"] if m.get("status") == "active")
    assert newest == chosen["newest_market"]

    for e in kalshi_odds.events_for(SERIES)[:4]:
        if e["event_ticker"] == chosen["event_ticker"]:
            continue
        active = [m for m in kalshi_odds.markets_for(e["event_ticker"])
                  if m.get("status") == "active"]
        if active:
            assert max((m.get("created_time") or "") for m in active) <= newest


def test_a_full_field_comes_back_in_a_single_page():
    """limit=500 covers any golf field, so the cursor loop should never engage."""
    chosen = kalshi_odds.latest_event(SERIES)
    assert len(chosen["markets"]) == len(kalshi_odds.markets_for(chosen["event_ticker"], max_pages=1))
