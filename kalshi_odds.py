#!/usr/bin/env python3
"""
kalshi_odds.py -- pull golf odds from the official Kalshi REST API.

This is the odds source for the grouper. It replaced DraftKings, which moved its
endpoint every season and required an annual sit-in-DevTools ritual to find again.
Kalshi publishes a documented, stable, unauthenticated read API instead:

    https://api.elections.kalshi.com/trade-api/v2

Endpoints used here (all public GETs, no API key needed for market data):
    /series?category=Sports          -> discover series; golf ones carry tags:["Golf"]
    /events?series_ticker=<TICKER>   -> tournaments within a series, newest first
    /markets?event_ticker=<TICKER>   -> one market per golfer, with live bid/ask

MARKET COVERAGE (verified 2026-08-02, re-verified live 2026-08-03):
    Full parity with the DK markets the grouper used. For a single tournament:
        KXPGATOUR-<EV>     outright Winner      <- DK "Winner" equivalent, the default
        KXPGATOP5-<EV>     Top 5 Finishers
        KXPGATOP10-<EV>    Top 10 Finishers
        KXPGAMAKECUT-<EV>  To Make the Cut
    All carry the complete field (143 golfers for the 2026 Wyndham Championship).
    Every one returns the identical market shape, so the market type is a series
    prefix on the event ticker and nothing more -- the direct analogue of the old
    DraftKings ODDS_TYPE knob.

WHY WINNER, AND WHY THE ASK:
    Winner ticks at $0.001 below 10c (price_level_structure: tapered_deci_cent).
    Top 5 / Top 10 / MakeCut are linear_cent, a flat $0.01 -- 10x coarser exactly
    where a golf field lives. Winner is the better signal, not the worse one.

    The ask is the price to use. Measured on the live 143-golfer KXPGATOUR-WYC26 field
    on 2026-08-03 (prices move, so treat the figures as a shape, not a constant; the
    earlier 2026-08-02 capture in kalshi-migration/START-HERE.md read 1.294 / 0.877):
        ask: 31 distinct levels, sums to 1.308, zero golfers at 0.0000
        bid: 30 distinct levels, sums to 0.906, 46 golfers at 0.0000
    A bid book that sums under 1.0 is not a distribution, and a third of the field
    would enter the grouper at zero weight. "Mid" inherits the same hole: with 46
    missing bids it is often just half the ask, a number this code invented.

RATE LIMITING -- READ THIS BEFORE EDITING discovery code:
    Kalshi 429s aggressive bursts. An 8-wide sweep across 112 series returned
    99 x HTTP 429. Crucially, a 429 is NOT an exception -- if you do
    `(resp.json().get("events") or [])` inside a bare try/except you will read
    "no events" and conclude a market does not exist. That exact bug produced a
    confidently wrong "Kalshi has no PGA winner market" conclusion in this repo's
    history. Every request below goes through get(), which raises on non-200 and
    backs off on 429. Keep it that way.

USAGE
-----
    python kalshi_odds.py --list-events                   # what golf is on right now
    python kalshi_odds.py --latest                        # newest live PGA Tour winner event
    python kalshi_odds.py --event KXPGATOUR-WYC26         # fetch one event
    python kalshi_odds.py --event KXPGATOUR-WYC26 --write-odds-file

Output shape matches what group.py consumes:
    [{"golfer_name": str, "odds": float_implied_probability, "golfer_id": str}, ...]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:  # only the network path needs it; the parsers do not
    requests = None

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Series that price a whole tournament field, one market per golfer.
# These are the only ones useful for building balanced groups.
FIELD_SERIES = {
    "winner": "KXPGATOUR",           # outright tournament winner -- the DK "Winner" analogue
    "top5": "KXPGATOP5",             # tournament Top 5
    "top10": "KXPGATOP10",           # tournament Top 10
    "makecut": "KXPGAMAKECUT",       # tournament Make Cut
    "r3lead": "KXPGAR3LEAD",         # single-round
    "r3top10": "KXPGAR3TOP10",       # single-round
    "r1top5": "KXPGAR1TOP5",         # single-round
    "dpw_winner": "KXDPWORLDTOUR",   # outright winner, DP World Tour
}

# The default market. Swapping this for KXPGATOP5 etc. costs one string, but read
# the tick-structure note in the module docstring before you do.
DEFAULT_SERIES = "KXPGATOUR"
DEFAULT_PRICE = "ask"
PRICE_MODES = ("ask", "mid", "last", "bid")

_SESSION = None


def session():
    """The shared HTTP session, built on first use so `requests` is only needed for network calls."""
    global _SESSION
    if requests is None:
        sys.exit("This script needs `requests` to reach the Kalshi API.  pip install requests")
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"Accept": "application/json"})
    return _SESSION


def get(path, _retries=4, **params):
    """
    GET with explicit non-200 handling and 429 backoff.

    Deliberately raises rather than returning an empty dict: a swallowed 429
    looks identical to "this market does not exist", which is how the earlier
    version of this exploration reached a wrong conclusion.
    """
    delay = 1.0
    for attempt in range(_retries):
        r = session().get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            if attempt == _retries - 1:
                raise RuntimeError(
                    f"rate limited (429) on {path} after {_retries} attempts -- "
                    "slow down; do NOT treat this as an empty result"
                )
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"unreachable: {path}")


def golf_series():
    """All series tagged Golf."""
    data = get("/series", category="Sports")
    seen, out = set(), []
    for s in data.get("series", []):
        if "Golf" in (s.get("tags") or []) and s["ticker"] not in seen:
            seen.add(s["ticker"])
            out.append({"ticker": s["ticker"], "title": s.get("title")})
    return out


def events_for(series_ticker, limit=200):
    """
    Events in a series, newest first.

    Do NOT pass status= to /events. Every value tried (open, closed, settled,
    unopened) returns zero results. The unfiltered call already includes settled
    events.
    """
    data = get("/events", series_ticker=series_ticker, limit=limit)
    return data.get("events", [])


def list_field_events():
    """
    Tournaments that currently have a full-field market on Kalshi.

    Paced deliberately, and failures are reported loudly. If a series errors we
    record it rather than dropping it, so a partial sweep can never masquerade
    as "this market doesn't exist".
    """
    rows, failures = [], []
    for label, ticker in FIELD_SERIES.items():
        try:
            for e in events_for(ticker):
                rows.append(
                    {
                        "market_type": label,
                        "series": ticker,
                        "event_ticker": e.get("event_ticker"),
                        "title": e.get("title"),
                        "tournament": e.get("sub_title"),
                    }
                )
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
        time.sleep(0.3)

    if failures:
        print("\n!! INCOMPLETE SWEEP -- these series did not return data:", file=sys.stderr)
        for f in failures:
            print(f"   {f}", file=sys.stderr)
        print("   Results below are PARTIAL. Do not conclude a market is absent.\n", file=sys.stderr)
    return rows


def markets_for(event_ticker, max_pages=20):
    """
    All golfer markets in an event, paging until exhausted.

    limit=500 covers any golf field in one page, so in practice this never loops. The
    guards are for the case where it does: a server that echoes the same cursor back
    would otherwise spin forever and duplicate the field, and a swallowed page limit
    would silently truncate it. Both are bounded here, and exhausting max_pages raises
    rather than returning a partial field.
    """
    out, cursor, seen = [], None, set()
    for _ in range(max_pages):
        params = {"event_ticker": event_ticker, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        data = get("/markets", **params)
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            return out
        if cursor in seen:
            raise RuntimeError(
                f"/markets returned a repeated cursor {cursor!r} for {event_ticker}. "
                "Paging is not advancing; the field would be duplicated."
            )
        seen.add(cursor)
    raise RuntimeError(
        f"/markets did not finish paging {event_ticker} within {max_pages} pages "
        f"({len(out)} markets so far). Refusing to return a partial field."
    )


def latest_event(series_ticker=DEFAULT_SERIES, max_probe=4):
    """
    The current tournament in a series: the one whose markets were created most recently
    and that still has at least one active market.

    /events appears to return newest first, but the event objects carry no date field to
    verify that with -- strike_date is null on every one. So rather than trusting list
    position, this probes the `max_probe` head events and picks by the newest market
    created_time among those that are tradeable. An old event with one stray active
    market can therefore no longer win on position alone.

    Probing is sequential and paced, because bursts get rate limited.

    Raises if nothing tradeable is found. It never returns None -- a silent None here
    would read exactly like "no tournament this week", which is the failure mode this
    module exists to avoid.
    """
    events = events_for(series_ticker)
    if not events:
        raise RuntimeError(
            f"no events at all for series {series_ticker}. That is unexpected -- "
            "check the response before assuming the series is empty."
        )

    candidates, tried = [], []
    for e in events[:max_probe]:
        ticker = e.get("event_ticker")
        if not ticker:
            continue
        markets = markets_for(ticker)
        active = [m for m in markets if m.get("status") == "active"]
        if active:
            candidates.append(
                {
                    "event_ticker": ticker,
                    "title": e.get("title"),
                    "tournament": e.get("sub_title"),
                    "markets": markets,
                    "active_count": len(active),
                    "newest_market": max((m.get("created_time") or "") for m in active),
                }
            )
        else:
            tried.append(f"{ticker} ({len(markets)} markets, 0 active)")
        time.sleep(0.3)

    if candidates:
        best = max(candidates, key=lambda c: c["newest_market"])
        if len(candidates) > 1:
            others = [c["event_ticker"] for c in candidates if c is not best]
            print(
                f"note: {len(candidates)} probed {series_ticker} events are tradeable "
                f"({', '.join(c['event_ticker'] for c in candidates)}); chose "
                f"{best['event_ticker']} on newest market created_time. Pass --event "
                f"explicitly to pick one of {others}.",
                file=sys.stderr,
            )
        return best

    raise RuntimeError(
        f"no tradeable event in the {max_probe} newest {series_ticker} events: "
        + "; ".join(tried)
        + ". Winner markets post Sunday ~23:00Z of tournament week -- if it is earlier "
        "than that, the field is simply not up yet."
    )


def _f(v):
    """
    Coerce a Kalshi money field to float.

    Money fields arrive as STRINGS -- "yes_bid_dollars": "0.0110". Summing them
    without coercion silently concatenates instead of adding.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_golfers(markets, price=DEFAULT_PRICE, active_only=True, strict=True):
    """
    Convert Kalshi markets to the {golfer_name, odds} shape group.py expects.

    Kalshi prices ARE implied probabilities already (a YES contract settles at
    $1.00), so no fractional->probability conversion is needed. That makes the old
    fractional_odds_to_implied_probability() unnecessary on this path.

    price: "ask"  -> the YES ask. THE DEFAULT, and the right one. It is a real
                     quoted price on the tick grid and it is always present, even
                     on a golfer who has never traded: on a binary contract a NO
                     bid at 0.9970 IS a YES ask at 0.0030, and Kalshi synthesizes
                     the YES ask off the NO book. Someone is always willing to
                     sell NO on a 1000/1 golfer, so the ask side never empties.
           "mid"  -> (bid + ask) / 2. Looks fairer than it is: roughly a third of
                     the field has no bid, so mid is frequently half the ask -- a
                     number this code made up rather than read.
           "bid"  -> disqualified for grouping. The bid book sums UNDER 1.0 and
                     leaves ~46 of 143 golfers at zero weight.
           "last" -> last traded price, stale or absent on thin markets.

    strict=True asserts ask > 0 on every emitted market and raises if it fails.
    There is deliberately no silent fallback: a quiet zero is exactly how this
    project previously produced a confident wrong answer.
    """
    if price not in PRICE_MODES:
        raise ValueError(f"unknown price mode: {price!r} (expected one of {PRICE_MODES})")

    golfers = []
    for m in markets:
        if active_only and m.get("status") != "active":
            continue
        bid = _f(m.get("yes_bid_dollars"))
        ask = _f(m.get("yes_ask_dollars"))
        last = _f(m.get("last_price_dollars"))

        name = m.get("yes_sub_title") or m.get("title")
        if not name:
            continue

        if strict and ask <= 0:
            raise ValueError(
                f"{m.get('ticker')} ({name}) quotes no ask (yes_ask_dollars="
                f"{m.get('yes_ask_dollars')!r}). Kalshi synthesizes the YES ask off "
                "the NO book, so an active market should never lack one. Investigate "
                "rather than defaulting it to zero -- pass strict=False only if you "
                "have decided a zero-weight golfer is acceptable."
            )

        if price == "ask":
            odds = ask
        elif price == "mid":
            odds = (bid + ask) / 2 if (bid or ask) else last
        elif price == "last":
            odds = last
        else:  # "bid"
            odds = bid

        golfers.append(
            {
                "golfer_name": name,
                "odds": round(odds, 6),
                # Stable per-golfer UUID. A far better join key to ESPN than a
                # display name -- it kills the "Matt / Matthew Fitzpatrick" bug class.
                "golfer_id": (m.get("custom_strike") or {}).get("golf_competitor"),
                "_bid": bid,
                "_ask": ask,
                "_spread": round(ask - bid, 4) if (bid and ask) else None,
                "_ticker": m.get("ticker"),
            }
        )
    return golfers


def clean_golfers(golfers):
    """Strip the underscore-prefixed diagnostic fields, keeping what the grouper uses."""
    return [
        {"golfer_name": g["golfer_name"], "odds": g["odds"], "golfer_id": g.get("golfer_id")}
        for g in golfers
    ]


def liquidity_report(golfers):
    n = len(golfers)
    two_sided = [g for g in golfers if g["_bid"] and g["_ask"]]
    wide = [g for g in two_sided if g["_spread"] and g["_spread"] > 0.05]
    total = sum(g["odds"] for g in golfers)
    return {
        "golfers": n,
        "two_sided_quotes": len(two_sided),
        "one_sided_or_empty": n - len(two_sided),
        "spreads_over_5c": len(wide),
        "probability_sum": round(total, 4),
        "note": (
            "probability_sum is the book, not a distribution: a Winner ask book runs "
            "~1.3 (the overround), a Top 5 book runs toward 5 and Top 10 toward 10. The "
            "grouper de-vigs by this observed sum, so never assume it is ~1.0. Far BELOW "
            "the market's nominal target means the field is stale or thin."
        ),
    }


def fetch_golfers(event_ticker=None, price=DEFAULT_PRICE, series=DEFAULT_SERIES, strict=True):
    """
    One call for callers that just want odds: fetch, convert, sort, return.

    Pass an explicit event_ticker, or leave it None to resolve the newest event in
    `series` that has active markets. Returns (event_ticker, golfers, report).
    """
    if event_ticker:
        markets = markets_for(event_ticker)
    else:
        resolved = latest_event(series)
        event_ticker, markets = resolved["event_ticker"], resolved["markets"]

    if not markets:
        raise RuntimeError(
            f"{event_ticker} returned zero markets. Confirm the request succeeded "
            "before concluding the tournament is not posted."
        )

    golfers = to_golfers(markets, price=price, strict=strict)
    if not golfers:
        raise RuntimeError(
            f"{event_ticker} returned {len(markets)} markets but none are active. "
            "Settled markets quote bid=0.0000/ask=1.0000 and are filtered out."
        )
    golfers.sort(key=lambda g: g["odds"], reverse=True)
    return event_ticker, golfers, liquidity_report(golfers)


def write_capture(event_ticker, golfers, report, price, dump_dir, markets=None):
    """Write the raw payload and the grouper-ready odds file. Returns both paths."""
    os.makedirs(dump_dir, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    raw_path = None
    if markets is not None:
        raw_path = os.path.join(dump_dir, f"kalshi_{event_ticker}_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fetched_at": fetched_at,
                    "endpoint": f"{BASE}/markets?event_ticker={event_ticker}&limit=500",
                    "market_count": len(markets),
                    "markets": markets,
                },
                f,
                indent=2,
            )

    # The price mode is in the filename on purpose. Without it a `--price bid` run
    # overwrites an `--price ask` capture of the same event, and the replacement is a
    # book summing under 1.0 with a third of the field at zero weight -- which a later
    # --data-file run would grade as real.
    odds_path = os.path.join(dump_dir, f"kalshi_{event_ticker}_{price}_golfers.json")
    with open(odds_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "kalshi",
                "event": event_ticker,
                "price_mode": price,
                "fetched_at": fetched_at,
                "report": report,
                "golfers": clean_golfers(golfers),
            },
            f,
            indent=2,
        )
    return raw_path, odds_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-events", action="store_true", help="show golf tournaments with full-field markets")
    ap.add_argument("--list-series", action="store_true", help="show every Golf-tagged series ticker")
    ap.add_argument("--event", help="event ticker, e.g. KXPGATOUR-WYC26")
    ap.add_argument(
        "--latest",
        action="store_true",
        help=f"resolve the newest {DEFAULT_SERIES} event that has active markets",
    )
    ap.add_argument(
        "--series",
        default=DEFAULT_SERIES,
        help=f"series to resolve --latest against (default {DEFAULT_SERIES})",
    )
    ap.add_argument("--price", default=DEFAULT_PRICE, choices=list(PRICE_MODES))
    ap.add_argument("--include-closed", action="store_true", help="keep settled/finalized markets")
    ap.add_argument(
        "--allow-missing-ask",
        action="store_true",
        help="do not raise when an active market quotes no ask (you almost certainly do not want this)",
    )
    ap.add_argument("--dump-dir", default="captures", help="where to write raw payloads")
    ap.add_argument(
        "--write-odds-file",
        nargs="?",
        const="kalshi_data.json",
        metavar="PATH",
        help="also write the grouper input file (default ./kalshi_data.json)",
    )
    args = ap.parse_args()

    if args.list_series:
        for s in golf_series():
            print(f"{s['ticker']:<24} {s['title']}")
        return

    if args.list_events:
        rows = list_field_events()
        rows.sort(key=lambda r: (r["market_type"], r["event_ticker"] or ""))
        print(f"\n{len(rows)} full-field golf events on Kalshi:\n")
        for r in rows:
            print(f"  [{r['market_type']:<10}] {r['event_ticker']:<28} {r['tournament'] or r['title']}")
        print(
            "\nWinner markets (KXPGATOUR-*) post Sunday ~23:00Z of tournament week; "
            "Top 5 / Top 10 follow ~21h later.\nMarkets keep being added through Wednesday "
            "as the field firms up, so pull Wednesday night."
        )
        return

    if not args.event and not args.latest:
        ap.error("give --event TICKER or --latest, or use --list-events / --list-series")

    strict = not args.allow_missing_ask

    if args.event:
        event_ticker = args.event
        print(f"Fetching {event_ticker} ...")
        markets = markets_for(event_ticker)
    else:
        print(f"Resolving newest live {args.series} event ...")
        resolved = latest_event(args.series)
        event_ticker, markets = resolved["event_ticker"], resolved["markets"]
        print(f"  {event_ticker}  {resolved['tournament'] or resolved['title']}")

    print(f"  {len(markets)} markets returned")

    golfers = to_golfers(markets, price=args.price, active_only=not args.include_closed, strict=strict)
    golfers.sort(key=lambda g: g["odds"], reverse=True)
    rep = liquidity_report(golfers)

    raw_path, odds_path = write_capture(
        event_ticker, golfers, rep, args.price, args.dump_dir, markets=markets
    )
    print(f"  raw dump -> {raw_path}")

    print("\nLiquidity:")
    for k, v in rep.items():
        if k != "note":
            print(f"  {k:<22} {v}")

    print(f"\nTop 12 by implied probability ({args.price}):")
    for g in golfers[:12]:
        sp = f"spread {g['_spread']}" if g["_spread"] is not None else "one-sided"
        print(f"  {g['golfer_name']:<28} {g['odds']:.4f}   ({sp})")

    print(f"\nGrouper-ready odds -> {odds_path}")

    if args.write_odds_file:
        with open(args.write_odds_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": "kalshi",
                    "event": event_ticker,
                    "price_mode": args.price,
                    # group.py prints this back, so a file left over from last week is
                    # visible rather than silently winning the odds-resolution order.
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "golfers": clean_golfers(golfers),
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.write_odds_file}  (group.py reads this directly)")
        if args.include_closed:
            print(
                "!! WARNING: written with --include-closed, so it holds settled markets "
                "quoting ask=1.0000. group.py will refuse this file.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
