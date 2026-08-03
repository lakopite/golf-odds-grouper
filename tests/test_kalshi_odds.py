"""Tests for the Kalshi client's parsing layer. No network -- see test_live.py for that."""

import pytest

import kalshi_odds


# ---------------------------------------------------------------------------
# _f  --  the string-coercion trap
# ---------------------------------------------------------------------------

def test_the_fixture_still_holds_money_as_strings(kalshi_markets):
    """
    Pins the fixture, not the API. If Kalshi ever switches to numbers, only
    test_live.py::test_money_fields_are_still_strings can tell us.
    """
    assert isinstance(kalshi_markets[0]["yes_ask_dollars"], str)


def test_f_coerces_string_money():
    assert kalshi_odds._f("0.0110") == pytest.approx(0.011)


def test_f_returns_zero_not_a_string_for_junk():
    for junk in (None, "", "n/a", {}):
        assert kalshi_odds._f(junk) == 0.0


def test_the_book_is_summed_numerically_not_concatenated(kalshi_markets):
    """The trap _f exists for: + on Kalshi's string money is silent and wrong."""
    total = sum(g["odds"] for g in kalshi_odds.to_golfers(kalshi_markets))
    assert isinstance(total, float) and total > 0


# ---------------------------------------------------------------------------
# to_golfers
# ---------------------------------------------------------------------------

def test_default_price_mode_is_ask():
    assert kalshi_odds.DEFAULT_PRICE == "ask"


def test_ask_is_the_default_used_by_to_golfers(kalshi_markets):
    default = kalshi_odds.to_golfers(kalshi_markets)
    explicit = kalshi_odds.to_golfers(kalshi_markets, price="ask")
    assert default == explicit


def test_finalized_markets_are_dropped(kalshi_markets):
    """A settled market quotes bid=0.0000/ask=1.0000 and would poison every sum."""
    names = [g["golfer_name"] for g in kalshi_odds.to_golfers(kalshi_markets)]
    assert "Settled Golfer" not in names
    assert len(names) == sum(1 for m in kalshi_markets if m["status"] == "active")


def test_include_closed_keeps_the_settled_market(kalshi_markets):
    golfers = kalshi_odds.to_golfers(kalshi_markets, active_only=False)
    settled = next(g for g in golfers if g["golfer_name"] == "Settled Golfer")
    assert settled["odds"] == 1.0


def test_every_active_market_quotes_an_ask(kalshi_markets):
    """
    The structural claim: Kalshi synthesizes the YES ask off the NO book, so the
    ask side never empties even on a golfer who has never traded.
    """
    golfers = kalshi_odds.to_golfers(kalshi_markets, price="ask")
    assert all(g["odds"] > 0 for g in golfers)


def test_bid_leaves_a_third_of_the_field_at_zero(kalshi_markets):
    """The measured reason bid is disqualified as a grouping input."""
    golfers = kalshi_odds.to_golfers(kalshi_markets, price="bid")
    zeros = [g for g in golfers if g["odds"] == 0]
    assert zeros, "fixture should contain golfers with no bid"
    assert sum(g["odds"] for g in golfers) < sum(
        g["odds"] for g in kalshi_odds.to_golfers(kalshi_markets, price="ask")
    )


def test_missing_ask_raises_rather_than_defaulting_to_zero(kalshi_markets):
    broken = [dict(m) for m in kalshi_markets if m["status"] == "active"]
    broken[0]["yes_ask_dollars"] = "0.0000"
    with pytest.raises(ValueError, match="quotes no ask"):
        kalshi_odds.to_golfers(broken)


def test_strict_false_is_the_only_way_past_a_missing_ask(kalshi_markets):
    broken = [dict(m) for m in kalshi_markets if m["status"] == "active"]
    broken[0]["yes_ask_dollars"] = "0.0000"
    golfers = kalshi_odds.to_golfers(broken, strict=False)
    assert golfers[0]["odds"] == 0.0


def test_unknown_price_mode_raises(kalshi_markets):
    with pytest.raises(ValueError, match="unknown price mode"):
        kalshi_odds.to_golfers(kalshi_markets, price="vwap")


def test_mid_sits_between_bid_and_ask(kalshi_markets):
    two_sided = [
        m for m in kalshi_markets
        if m["status"] == "active" and kalshi_odds._f(m["yes_bid_dollars"]) > 0
    ]
    mids = {g["golfer_name"]: g["odds"] for g in kalshi_odds.to_golfers(two_sided, price="mid")}
    for m in two_sided:
        bid, ask = kalshi_odds._f(m["yes_bid_dollars"]), kalshi_odds._f(m["yes_ask_dollars"])
        assert bid <= mids[m["yes_sub_title"]] <= ask


def test_golfer_id_is_carried_through(kalshi_markets):
    golfers = kalshi_odds.to_golfers(kalshi_markets)
    ids = [g["golfer_id"] for g in golfers]
    assert all(ids), "every golfer should carry custom_strike.golf_competitor"
    assert len(set(ids)) == len(ids), "golfer ids must be unique within an event"


def test_golfer_id_survives_a_market_with_no_custom_strike(kalshi_markets):
    stripped = [dict(m) for m in kalshi_markets if m["status"] == "active"]
    stripped[0].pop("custom_strike")
    assert kalshi_odds.to_golfers(stripped)[0]["golfer_id"] is None


def test_clean_golfers_drops_diagnostics_and_keeps_the_id(kalshi_markets):
    clean = kalshi_odds.clean_golfers(kalshi_odds.to_golfers(kalshi_markets))
    assert set(clean[0]) == {"golfer_name", "odds", "golfer_id"}


def test_prices_are_probabilities_not_fractional_odds(kalshi_markets):
    """A YES contract settles at $1.00, so the quote IS the implied probability."""
    for g in kalshi_odds.to_golfers(kalshi_markets):
        assert 0 < g["odds"] <= 1


# ---------------------------------------------------------------------------
# liquidity_report
# ---------------------------------------------------------------------------

def test_liquidity_report_counts_one_sided_quotes(kalshi_markets):
    golfers = kalshi_odds.to_golfers(kalshi_markets)
    rep = kalshi_odds.liquidity_report(golfers)
    assert rep["golfers"] == len(golfers)
    assert rep["two_sided_quotes"] + rep["one_sided_or_empty"] == rep["golfers"]
    assert rep["one_sided_or_empty"] > 0, "fixture should contain no-bid golfers"


def test_liquidity_report_sum_matches_the_book(kalshi_markets):
    golfers = kalshi_odds.to_golfers(kalshi_markets)
    rep = kalshi_odds.liquidity_report(golfers)
    assert rep["probability_sum"] == pytest.approx(sum(g["odds"] for g in golfers), abs=1e-4)


# ---------------------------------------------------------------------------
# get()  --  the 429 rule
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(kalshi_odds.time, "sleep", lambda _s: None)


def test_get_backs_off_then_succeeds(monkeypatch, no_sleep):
    fake = _FakeSession([_Resp(429), _Resp(429), _Resp(200, {"events": [1]})])
    monkeypatch.setattr(kalshi_odds, "session", lambda: fake)
    assert kalshi_odds.get("/events") == {"events": [1]}
    assert fake.calls == 3


def test_persistent_429_raises_and_never_returns_empty(monkeypatch, no_sleep):
    fake = _FakeSession([_Resp(429)] * 4)
    monkeypatch.setattr(kalshi_odds, "session", lambda: fake)
    with pytest.raises(RuntimeError, match="rate limited"):
        kalshi_odds.get("/events")


def test_non_200_is_not_swallowed(monkeypatch, no_sleep):
    fake = _FakeSession([_Resp(500)])
    monkeypatch.setattr(kalshi_odds, "session", lambda: fake)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        kalshi_odds.get("/events")


# ---------------------------------------------------------------------------
# latest_event  --  must never quietly report "no tournament"
# ---------------------------------------------------------------------------

def _dated(markets, created):
    return [{**m, "created_time": created} for m in markets]


def test_latest_event_skips_events_with_no_active_markets(monkeypatch, no_sleep, kalshi_markets):
    settled_only = [m for m in kalshi_markets if m["status"] != "active"]
    monkeypatch.setattr(
        kalshi_odds, "events_for",
        lambda s, limit=200: [{"event_ticker": "OLD"}, {"event_ticker": "NEW"}],
    )
    monkeypatch.setattr(
        kalshi_odds, "markets_for",
        lambda t: settled_only if t == "OLD" else kalshi_markets,
    )
    assert kalshi_odds.latest_event("KXPGATOUR")["event_ticker"] == "NEW"


def test_latest_event_picks_by_market_age_not_list_position(monkeypatch, no_sleep, kalshi_markets):
    """
    /events carries no date field, so list order is an assumption rather than a fact.
    A stale event with one stray active market must not win just by sitting at the head.
    """
    stale = _dated(kalshi_markets[:2], "2026-06-01T00:00:00Z")
    current = _dated(kalshi_markets, "2026-08-02T23:00:00Z")
    monkeypatch.setattr(
        kalshi_odds, "events_for",
        lambda s, limit=200: [{"event_ticker": "STALE"}, {"event_ticker": "CURRENT"}],
    )
    monkeypatch.setattr(
        kalshi_odds, "markets_for", lambda t: stale if t == "STALE" else current
    )
    assert kalshi_odds.latest_event("KXPGATOUR")["event_ticker"] == "CURRENT"


def test_latest_event_says_so_when_more_than_one_is_tradeable(monkeypatch, no_sleep, kalshi_markets, capsys):
    monkeypatch.setattr(
        kalshi_odds, "events_for",
        lambda s, limit=200: [{"event_ticker": "A"}, {"event_ticker": "B"}],
    )
    monkeypatch.setattr(
        kalshi_odds, "markets_for",
        lambda t: _dated(kalshi_markets, "2026-08-01T00:00:00Z" if t == "A" else "2026-08-02T00:00:00Z"),
    )
    assert kalshi_odds.latest_event("KXPGATOUR")["event_ticker"] == "B"
    assert "2 probed" in capsys.readouterr().err


def test_latest_event_raises_when_nothing_is_tradeable(monkeypatch, no_sleep, kalshi_markets):
    settled_only = [m for m in kalshi_markets if m["status"] != "active"]
    monkeypatch.setattr(kalshi_odds, "events_for", lambda s, limit=200: [{"event_ticker": "OLD"}])
    monkeypatch.setattr(kalshi_odds, "markets_for", lambda t: settled_only)
    with pytest.raises(RuntimeError, match="no tradeable event"):
        kalshi_odds.latest_event("KXPGATOUR")


def test_latest_event_raises_on_an_empty_series(monkeypatch, no_sleep):
    monkeypatch.setattr(kalshi_odds, "events_for", lambda s, limit=200: [])
    with pytest.raises(RuntimeError, match="no events at all"):
        kalshi_odds.latest_event("KXPGATOUR")


# ---------------------------------------------------------------------------
# markets_for  --  cursor paging
# ---------------------------------------------------------------------------

def test_markets_for_follows_the_cursor(monkeypatch):
    pages = [
        {"markets": [{"ticker": "A"}], "cursor": "c1"},
        {"markets": [{"ticker": "B"}], "cursor": ""},
    ]
    monkeypatch.setattr(kalshi_odds, "get", lambda path, **kw: pages.pop(0))
    assert [m["ticker"] for m in kalshi_odds.markets_for("KXPGATOUR-WYC26")] == ["A", "B"]


def test_markets_for_raises_on_a_repeated_cursor(monkeypatch):
    """A server echoing its own cursor would otherwise loop forever and duplicate the field."""
    monkeypatch.setattr(
        kalshi_odds, "get",
        lambda path, **kw: {"markets": [{"ticker": "A"}], "cursor": "stuck"},
    )
    with pytest.raises(RuntimeError, match="repeated cursor"):
        kalshi_odds.markets_for("KXPGATOUR-WYC26")


def test_markets_for_refuses_to_return_a_partial_field(monkeypatch):
    counter = {"n": 0}

    def endless(path, **kw):
        counter["n"] += 1
        return {"markets": [{"ticker": f"M{counter['n']}"}], "cursor": f"c{counter['n']}"}

    monkeypatch.setattr(kalshi_odds, "get", endless)
    with pytest.raises(RuntimeError, match="did not finish paging"):
        kalshi_odds.markets_for("KXPGATOUR-WYC26", max_pages=5)
    assert counter["n"] == 5


# ---------------------------------------------------------------------------
# fetch_golfers
# ---------------------------------------------------------------------------

def test_fetch_golfers_sorts_by_probability(monkeypatch, kalshi_markets):
    monkeypatch.setattr(kalshi_odds, "markets_for", lambda t: kalshi_markets)
    _, golfers, report = kalshi_odds.fetch_golfers("KXPGATOUR-WYC26")
    assert [g["odds"] for g in golfers] == sorted((g["odds"] for g in golfers), reverse=True)
    assert report["golfers"] == len(golfers)


def test_fetch_golfers_raises_on_an_all_settled_event(monkeypatch, kalshi_markets):
    settled_only = [m for m in kalshi_markets if m["status"] != "active"]
    monkeypatch.setattr(kalshi_odds, "markets_for", lambda t: settled_only)
    with pytest.raises(RuntimeError, match="none are active"):
        kalshi_odds.fetch_golfers("KXPGATOUR-OLD26")


def test_fetch_golfers_raises_on_zero_markets(monkeypatch):
    monkeypatch.setattr(kalshi_odds, "markets_for", lambda t: [])
    with pytest.raises(RuntimeError, match="zero markets"):
        kalshi_odds.fetch_golfers("KXPGATOUR-NOPE")


# ---------------------------------------------------------------------------
# write_capture
# ---------------------------------------------------------------------------

def test_write_capture_writes_both_files(tmp_path, kalshi_markets):
    golfers = kalshi_odds.to_golfers(kalshi_markets)
    rep = kalshi_odds.liquidity_report(golfers)
    raw_path, odds_path = kalshi_odds.write_capture(
        "KXPGATOUR-WYC26", golfers, rep, "ask", str(tmp_path), markets=kalshi_markets
    )
    import json as _json
    assert _json.load(open(raw_path))["market_count"] == len(kalshi_markets)
    written = _json.load(open(odds_path))
    assert written["price_mode"] == "ask"
    assert len(written["golfers"]) == len(golfers)
