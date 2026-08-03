#!/usr/bin/env python3
"""
kalshi_odds.py -- pull golf odds from the official Kalshi REST API.

Unlike DraftKings, Kalshi publishes a documented, stable, unauthenticated read API.
No HAR archaeology required. Base URL:

    https://api.elections.kalshi.com/trade-api/v2

Endpoints used here (all public GETs, no API key needed for market data):
    /series?category=Sports          -> discover series; golf ones carry tags:["Golf"]
    /events?series_ticker=<TICKER>   -> tournaments within a series
    /markets?event_ticker=<TICKER>   -> one market per golfer, with live bid/ask

MARKET COVERAGE (verified 2026-08-02):
    Full parity with the DK markets the grouper uses. For a single tournament:
        KXPGATOUR-<EV>     outright Winner      <- DK "Winner" equivalent
        KXPGATOP5-<EV>     Top 5 Finishers
        KXPGATOP10-<EV>    Top 10 Finishers
        KXPGAMAKECUT-<EV>  To Make the Cut
    All carry the complete field (151 golfers for the 2026 Rocket Classic).

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
    python kalshi_odds.py --list-events                  # what golf is on right now
    python kalshi_odds.py --event KXPGATOP10-ROC26       # fetch one event
    python kalshi_odds.py --event KXPGATOP10-ROC26 --write-dk-data

Output shape matches what group.py's list_dk_golf_odds() returns:
    [{"golfer_name": str, "odds": float_implied_probability}, ...]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("This script needs `requests`.  pip install requests")

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

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def get(path, _retries=4, **params):
    """
    GET with explicit non-200 handling and 429 backoff.

    Deliberately raises rather than returning an empty dict: a swallowed 429
    looks identical to "this market does not exist", which is how the earlier
    version of this exploration reached a wrong conclusion.
    """
    delay = 1.0
    for attempt in range(_retries):
        r = SESSION.get(BASE + path, params=params, timeout=30)
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


def markets_for(event_ticker):
    """All golfer markets in an event, paging until exhausted."""
    out, cursor = [], None
    while True:
        params = {"event_ticker": event_ticker, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        data = get("/markets", **params)
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            break
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_golfers(markets, price="mid", active_only=True):
    """
    Convert Kalshi markets to the {golfer_name, odds} shape group.py expects.

    Kalshi prices ARE implied probabilities already (a YES contract settles at
    $1.00), so no fractional->probability conversion is needed. This is strictly
    cleaner than DK: no vig to strip on the individual contract, though the book
    still shows a bid/ask spread.

    price: "mid"  -> (bid + ask) / 2, the fairest read
           "last" -> last traded price, stale on thin markets
           "bid" / "ask" -> the raw side
    """
    golfers = []
    for m in markets:
        if active_only and m.get("status") != "active":
            continue
        bid = _f(m.get("yes_bid_dollars"))
        ask = _f(m.get("yes_ask_dollars"))
        last = _f(m.get("last_price_dollars"))

        if price == "mid":
            odds = (bid + ask) / 2 if (bid or ask) else last
        elif price == "last":
            odds = last
        elif price == "bid":
            odds = bid
        elif price == "ask":
            odds = ask
        else:
            raise ValueError(f"unknown price mode: {price}")

        name = m.get("yes_sub_title") or m.get("title")
        if not name:
            continue
        golfers.append(
            {
                "golfer_name": name,
                "odds": round(odds, 6),
                "_bid": bid,
                "_ask": ask,
                "_spread": round(ask - bid, 4) if (bid and ask) else None,
                "_ticker": m.get("ticker"),
            }
        )
    return golfers


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
            "probability_sum well above the market's nominal target (5 for Top 5, "
            "10 for Top 10) means overround; far below means the field is stale or thin."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-events", action="store_true", help="show golf tournaments with full-field markets")
    ap.add_argument("--list-series", action="store_true", help="show every Golf-tagged series ticker")
    ap.add_argument("--event", help="event ticker, e.g. KXPGATOP10-ROC26")
    ap.add_argument("--price", default="mid", choices=["mid", "last", "bid", "ask"])
    ap.add_argument("--include-closed", action="store_true", help="keep settled/finalized markets")
    ap.add_argument("--dump-dir", default="captures", help="where to write raw payloads")
    ap.add_argument("--write-dk-data", action="store_true", help="also write ./dk_data.json in grouper format")
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
        print("\nNote: no outright-winner series exists for PGA Tour events.")
        return

    if not args.event:
        ap.error("give --event TICKER, or use --list-events / --list-series")

    os.makedirs(args.dump_dir, exist_ok=True)
    print(f"Fetching {args.event} ...")
    markets = markets_for(args.event)
    print(f"  {len(markets)} markets returned")

    raw_path = os.path.join(args.dump_dir, f"kalshi_{args.event}_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "endpoint": f"{BASE}/markets?event_ticker={args.event}&limit=500",
                "market_count": len(markets),
                "markets": markets,
            },
            f,
            indent=2,
        )
    print(f"  raw dump -> {raw_path}")

    golfers = to_golfers(markets, price=args.price, active_only=not args.include_closed)
    golfers.sort(key=lambda g: g["odds"], reverse=True)

    rep = liquidity_report(golfers)
    print("\nLiquidity:")
    for k, v in rep.items():
        if k != "note":
            print(f"  {k:<22} {v}")

    print(f"\nTop 12 by implied probability ({args.price}):")
    for g in golfers[:12]:
        sp = f"spread {g['_spread']}" if g["_spread"] is not None else "one-sided"
        print(f"  {g['golfer_name']:<28} {g['odds']:.4f}   ({sp})")

    clean = [{"golfer_name": g["golfer_name"], "odds": g["odds"]} for g in golfers]
    out_path = os.path.join(args.dump_dir, f"kalshi_{args.event}_golfers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"event": args.event, "price_mode": args.price, "report": rep, "golfers": clean}, f, indent=2)
    print(f"\nGrouper-ready odds -> {out_path}")

    if args.write_dk_data:
        with open("dk_data.json", "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
        print("Wrote ./dk_data.json  (NOTE: this is the flat golfers list, not the")
        print("DK markets/selections envelope -- group.py needs a small shim to read it.)")


if __name__ == "__main__":
    main()
