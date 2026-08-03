#!/usr/bin/env python3
"""
ESPN PGA leaderboard reader. Stdlib only.

Verified against a live Rocket Classic R2 payload (2026-07-31):

  * competitor["score"]["displayValue"] counts COMPLETED ROUNDS ONLY.
    Mid-round it is stale (46/147 players wrong). Compute the live total
    by summing linescore displayValues instead.
  * linescores[] contains STUB entries for future rounds with no
    "value"/"displayValue" key at all. Filter before summing.
  * Withdrawn/no-score rounds carry displayValue "-".
  * Current round is competitions[0].status.period (NOT event.status).
  * status.position.isTie is a bool -- no need to parse the "T".
"""

import argparse, json, sys, urllib.request, urllib.error

URL = "https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga"
UA = "Mozilla/5.0 (compatible; leaderboard/3.0)"


def to_par(x):
    """'E'->0, '-2'->-2, '+3'->3, '-' or junk -> None."""
    s = str(x).strip()
    if s == "E":
        return 0
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return None


def fmt_par(n):
    return "E" if n == 0 else format(n, "+d")


def load(url):
    if url.startswith("file://"):
        return json.load(open(url[7:]))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def parse(data):
    """-> (meta dict, list of player dicts) — the shape to feed a frontend."""
    events = data.get("events") or []
    if not events:
        return None, []
    ev = events[0]
    comp = (ev.get("competitions") or [{}])[0]
    meta = {
        "event": ev.get("name"),
        "course": (ev.get("courses") or [{}])[0].get("name"),
        "par": (ev.get("courses") or [{}])[0].get("shotsToPar"),
        "round": comp.get("status", {}).get("period"),
        "detail": comp.get("status", {}).get("type", {}).get("detail"),
        "state": ev.get("status", {}).get("type", {}).get("state"),
        "start": ev.get("date"),
        "end": ev.get("endDate"),
    }
    players = []
    for c in comp.get("competitors") or []:
        st = c.get("status") or {}
        # keep only real rounds; drop future-round stubs and "-" scores
        rounds = []
        for ls in c.get("linescores") or []:
            v = to_par(ls.get("displayValue")) if "displayValue" in ls else None
            if v is not None:
                rounds.append({"round": ls.get("period"), "toPar": v,
                               "strokes": ls.get("value")})
        live = sum(r["toPar"] for r in rounds) if rounds else None
        players.append({
            "name": (c.get("athlete") or {}).get("displayName"),
            "pos": (st.get("position") or {}).get("displayName"),
            "tied": bool((st.get("position") or {}).get("isTie")),
            "thru": st.get("displayThru") or st.get("thru"),
            "total": fmt_par(live) if live is not None else "-",
            "stale_total": (c.get("score") or {}).get("displayValue"),
            "rounds": rounds,
            "sort": c.get("sortOrder", 9999),
        })
    players.sort(key=lambda p: p["sort"])
    return meta, players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="emit parsed JSON")
    ap.add_argument("--audit", action="store_true", help="show stale-vs-live diffs")
    ap.add_argument("--save", metavar="FILE", help="write raw payload to FILE")
    ap.add_argument("--raw", action="store_true", help="dump verbatim structure")
    a = ap.parse_args()

    try:
        data = load(a.url)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        sys.exit(f"failed: {e}")

    if a.save:
        with open(a.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"saved raw payload -> {a.save}")

    if a.raw:
        ev = (data.get("events") or [{}])[0]
        comp = (ev.get("competitions") or [{}])[0]
        cs = comp.get("competitors") or []
        print(f"top-level keys: {sorted(data.keys())}")
        print(f"event keys: {sorted(ev.keys())}")
        print(f"competition keys: {sorted(comp.keys())}")
        print("\n--- competition.status verbatim ---")
        print(json.dumps(comp.get("status"), indent=2))
        if cs:
            print(f"\ncompetitor keys: {sorted(cs[0].keys())}")
            print("\n--- competitor.status verbatim ---")
            print(json.dumps(cs[0].get("status"), indent=2))
            print("\n--- linescores verbatim (first player) ---")
            print(json.dumps(cs[0].get("linescores"), indent=2))
        print()

    meta, players = parse(data)
    if not meta:
        sys.exit("No events -- between tournaments.")
    if a.json:
        print(json.dumps({"meta": meta, "players": players}, indent=2))
        return

    print(f"{meta['event']} @ {meta['course']} (par {meta['par']})")
    print(f"{meta['detail']}  [{meta['start']} -> {meta['end']}]\n")
    print(f"{'POS':<6} {'PLAYER':<24} {'TOT':>5} {'THRU':>5}   ROUNDS")
    print("-" * 72)
    for p in players[:a.top]:
        rs = ", ".join(f"{fmt_par(r['toPar'])} R{r['round']}" for r in p["rounds"])
        print(f"{str(p['pos']):<6} {str(p['name']):<24} {p['total']:>5} "
              f"{str(p['thru']):>5}   {rs or '--'}")

    if a.audit:
        bad = [p for p in players if p["total"] != p["stale_total"]]
        print(f"\n{len(bad)}/{len(players)} differ from score.displayValue:")
        for p in bad[:12]:
            print(f"  {p['name']:<24} live={p['total']:>4}  "
                  f"score field={str(p['stale_total']):>4}  thru {p['thru']}")


if __name__ == "__main__":
    main()
