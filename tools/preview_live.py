#!/usr/bin/env python3
"""
preview_live.py -- see the ranked scoreboard on a Wednesday.

A pool is drawn the night before, and the page that comes out of the build shows the
draw: every team, every roster, the prices they were drawn on, nothing ranked. It turns
into a scoreboard by itself at the first tee time, in whatever browser has it open
(README, "One build, and the page decides when to rank"). That is the right behaviour
and an awkward thing to show somebody on a Wednesday, because the only honest way to
see the ranked page is to wait for Thursday.

This builds a preview of it, out of real ESPN data and nothing else:

    this week's field   -- the real leaderboard payload for this competition's event,
                           every athlete id, name and headshot as ESPN publishes them
    a played tournament -- a real ESPN leaderboard from a tournament that has already
                           happened, with its positions, ties, thru-counts, linescores,
                           cut and statuses intact

and transplants the second onto the first. Every score on the preview page is a number
ESPN published; what is invented is only WHICH of this week's golfers is wearing it.

The payload is then baked into the page as a `data:` URI, so the page polls it exactly
as it polls ESPN and renders it with the same code. Nothing in `frontend/` knows the
difference and nothing in `frontend/` was changed to make this work -- what you are
looking at is the page, not a mock-up of it.

    python tools/preview_live.py --result build/result.json --stage round2
    python bundle_frontend.py --result build/preview-round2.json --out dist/

Two stages ship, because they are the two the page draws differently:

    round2   Friday afternoon, mid-round. 74 golfers in with a second round, 66 still
             out on the course at "thru 5", 6 yet to tee off. The status pill is Live
             and the page keeps polling.
    final    Sunday night. 73 made the cut, 74 are CUT and greyed out, the pill says
             Final and the poll loop stops.

WHO ENDS UP WHERE
-----------------
A played leaderboard has to be dealt out to this week's field somehow, and the honest
options are a coin toss or the market. The default is the market: one sample from the
same de-vigged prices the groups were drawn on, by Plackett-Luce -- draw the winner
with probability proportional to price, then the runner-up from whoever is left, and so
on down the field. Favourites turn up near the top about as often as the book says they
should, so the board reads like a tournament rather than a raffle, and every position is
still a draw rather than a forecast. `--draw uniform` shuffles instead, which is the
same machinery with the prices ignored.

Either way the draw is seeded, so a preview is reproducible: same result file, same
seed, same board.

NOT FOR HANDING ROUND
---------------------
The scores did not happen. Somebody sent a preview page believing it is this week's
scoreboard reads a leaderboard that never existed, which is worse than being sent
nothing, so the preview says what it is in three places: the file is named `preview-*`,
its result JSON carries a `preview` block, and the masthead on the page itself is
labelled. `--no-label` removes only the last of those, for a screenshot.
"""

import argparse
import base64
import copy
import datetime as dt
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import espn_leaderboard                                            # noqa: E402
import standings                                                   # noqa: E402

# The two donors, both real ESPN payloads already in the repository for the tests. They
# are the same tournament at two points in its life, which is deliberate: the pair is
# what proves the page draws a live board and a finished one differently.
STAGES = {
    "round2": {
        "path": os.path.join(ROOT, "espn-api", "lb.json"),
        "what": "Rocket Classic, round 2 in progress -- 66 golfers on the course",
    },
    "final": {
        "path": os.path.join(ROOT, "tests", "fixtures", "espn_final_with_cut.json"),
        "what": "Rocket Classic, final -- 73 made the cut, 74 did not",
    },
}

# ESPN writes its timestamps one way everywhere in a leaderboard payload, so the tee
# times can be moved onto this week's dates without parsing the whole document.
ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2}(?::\d{2})?Z)$")

PREVIEW_NOTE = (
    "Every score, position and thru-count on this page is real ESPN data from a "
    "tournament that has been played, dealt out to this week's field. Nothing here is a "
    "forecast and none of it happened. It exists to show what the page looks like once "
    "play starts."
)


# ---------------------------------------------------------------------------
# Dealing the leaderboard out
# ---------------------------------------------------------------------------

def market_order(weights, rng):
    """
    One finishing order sampled from the prices, best first (Plackett-Luce).

    Draw the winner with probability proportional to weight, remove them, draw again
    from what is left. It is the standard way to turn a set of win probabilities into a
    whole finishing order, and it is the right shape here for one reason: the odds this
    competition was drawn on already say who is likely to be near the top, so a preview
    that ignores them puts a 0.09% shot in the lead about as often as the favourite and
    reads as broken rather than as random.

    A golfer with no price still plays -- ESPN's field is not always Kalshi's -- so a
    zero weight is floored rather than dropped.
    """
    idx = list(range(len(weights)))
    w = [max(float(x), 0.0) for x in weights]
    floor = min([x for x in w if x > 0] or [1.0]) / 2
    w = [x or floor for x in w]

    order = []
    total = sum(w)
    while idx:
        draw = rng.random() * total
        acc = 0.0
        pick = len(idx) - 1
        for i, j in enumerate(idx):
            acc += w[j]
            if acc >= draw:
                pick = i
                break
        j = idx.pop(pick)
        total -= w[j]
        order.append(j)
    return order


def uniform_order(count, rng):
    """A coin toss over the whole field: every golfer as likely to win as any other."""
    order = list(range(count))
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# The transplant
# ---------------------------------------------------------------------------

def shift_dates(node, days):
    """
    Move every ESPN timestamp in a copied fragment by whole days, in place.

    Tee times are the visible half of this -- the page prints "tees off later" rather
    than a raw timestamp, but the payload is the thing somebody will open next to the
    page to check it, and last month's dates in it read as a bug rather than as a donor.
    Whole days only: a tournament's rounds keep their shape, and 07:45 stays 07:45.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = shift_dates(value, days)
        return node
    if isinstance(node, list):
        return [shift_dates(v, days) for v in node]
    if isinstance(node, str):
        m = ISO.match(node)
        if m:
            date = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (date + dt.timedelta(days=days)).isoformat() + "T" + m.group(4)
    return node


def day_shift(field_payload, donor_payload):
    """Whole days from the donor tournament's first round to this one's."""
    def start(payload):
        text = ((payload.get("events") or [{}])[0]).get("date") or ""
        m = ISO.match(text)
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    a, b = start(donor_payload), start(field_payload)
    return (b - a).days if a and b else 0


def transplant(field_payload, donor_payload, order):
    """
    -> a leaderboard payload: this week's field, wearing a played tournament's scores.

    `order` is the finishing order as indices into the field's competitor list, best
    first. The donor's rows are taken in ITS finishing order and handed out down that
    list, so position 1 gets position 1, every tie stays a tie and the cut line falls
    exactly where the donor's did. Nothing about the leaderboard is recomputed here --
    the whole point is that it is a real one.

    Copied per golfer: `status` (position, thru, tee time, and which of playing /
    finished / cut / not yet out they are), `linescores`, `score`, `sortOrder`,
    `movement`, `earnings`. Kept: everything that says who they are -- athlete id, name,
    headshot, flag, amateur flag.

    A field larger than the donor's is possible (they are different tournaments), and
    the golfers past the end of the donor are left not-yet-teed-off rather than invented
    for: that is a state the page already draws.
    """
    out = copy.deepcopy(field_payload)
    event = out["events"][0]
    comp = (event.get("competitions") or [{}])[0]
    field = comp.get("competitors") or []

    donor_event = donor_payload["events"][0]
    donor_comp = (donor_event.get("competitions") or [{}])[0]
    donor_rows = sorted(donor_comp.get("competitors") or [],
                        key=lambda c: c.get("sortOrder", 9999))

    days = day_shift(field_payload, donor_payload)

    # The envelope decides whether the page ranks at all: `state` in / post is half of
    # `has_started`, and `completed` is what stops the poll loop on a finished
    # tournament. Both come from the donor; the event's identity does not.
    event["status"] = copy.deepcopy(donor_event.get("status"))
    comp["status"] = copy.deepcopy(donor_comp.get("status"))

    for rank, target_index in enumerate(order):
        row = field[target_index]
        if rank >= len(donor_rows):
            row["sortOrder"] = 9000 + rank
            row["status"] = {"period": 0, "displayValue": "-",
                             "position": {"displayName": "-", "isTie": False},
                             "type": {"id": "1", "name": "STATUS_SCHEDULED", "state": "pre",
                                      "completed": False, "description": "Scheduled"}}
            row["linescores"] = []
            row.pop("score", None)
            continue
        donor = donor_rows[rank]
        row["sortOrder"] = donor.get("sortOrder", rank + 1)
        row["status"] = shift_dates(copy.deepcopy(donor.get("status") or {}), days)
        row["linescores"] = shift_dates(copy.deepcopy(donor.get("linescores") or []), days)
        for key in ("score", "movement", "earnings"):
            if key in donor:
                row[key] = copy.deepcopy(donor[key])
            else:
                row.pop(key, None)

    field.sort(key=lambda c: c.get("sortOrder", 9999))
    return out


# ---------------------------------------------------------------------------
# Baking it into the competition
# ---------------------------------------------------------------------------

def data_uri(payload):
    """
    The payload as a URL the page can fetch with no server and no network.

    `fetch()` reads a data: URI from a file:// page, so the poll loop, the event-id
    check, the parse and the ranking all run unmodified against it. Base64 rather than
    percent-encoding because the result of this lands inside a <script> tag in the
    exported HTML, and base64 cannot contain a "</script>".
    """
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return "data:application/json;base64," + base64.b64encode(body).decode("ascii")


def preview_result(result, payload, stage, donor, seed, draw, label=True):
    """The same competition, pointed at the simulated leaderboard instead of ESPN."""
    out = copy.deepcopy(result)
    out["live"]["espn_leaderboard_url"] = data_uri(payload)
    out["preview"] = {
        "simulated": True,
        "stage": stage,
        "donor": donor["what"],
        "donor_file": os.path.relpath(donor["path"], ROOT),
        "field": (f"ESPN {result['sources']['espn']['event_id']} "
                  f"{result['tournament']['name']}"),
        "draw": draw,
        "seed": seed,
        "tool": "tools/preview_live.py",
        "note": PREVIEW_NOTE,
    }
    # The masthead is where somebody looking at the page will see it, and the tagline is
    # the one line of it a competition owns. The build's own tagline is kept alongside
    # rather than replaced -- this is still that league's page.
    if label:
        tagline = out["league"].get("tagline")
        out["league"]["tagline"] = (f"{tagline} · PREVIEW · simulated scoring"
                                    if tagline else "PREVIEW · simulated scoring")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def league_table(result, players):
    """The standings the preview page will show, off the reference implementation."""
    by_team = {}
    for golfer in result["golfers"]:
        if golfer.get("team_id"):
            by_team.setdefault(golfer["team_id"], []).append(golfer)
    teams = [{"team_id": t["team_id"], "golfers": by_team.get(t["team_id"], [])}
             for t in result["teams"]]
    names = {t["team_id"]: t["team_name"] for t in result["teams"]}
    by_id = {p["athlete_id"]: p for p in players if p.get("athlete_id")}
    return [(row, names.get(row["team_id"], row["team_id"]))
            for row in standings.compute(teams, by_id)]


def read_field(result, path):
    """This week's real leaderboard payload, off disk or off ESPN."""
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    event_id = result["live"]["espn_event_id"]
    league = (result["sources"].get("espn") or {}).get("league") or espn_leaderboard.DEFAULT_LEAGUE
    return espn_leaderboard.fetch_leaderboard(event_id, league), "ESPN"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="result JSON from build_competition.py")
    ap.add_argument("--stage", default="round2", choices=sorted(STAGES),
                    help="which point in a tournament to preview (default round2)")
    ap.add_argument("--donor", help="a leaderboard payload to take the scores from, "
                                    "instead of one of the shipped stages")
    ap.add_argument("--field", help="this week's leaderboard payload (default: fetch it "
                                    "from ESPN)")
    ap.add_argument("--draw", default="market", choices=("market", "uniform"),
                    help="how the finishing order is dealt out (default market)")
    ap.add_argument("--seed", type=int, default=7, help="reproducible draw (default 7)")
    ap.add_argument("--no-label", action="store_true",
                    help="leave the masthead unlabelled -- for a screenshot, not for sending")
    ap.add_argument("--out", default="build", help="where to write (default build/)")
    args = ap.parse_args(argv)

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)
    if not result.get("live"):
        raise SystemExit("this result file has no `live` block, so it carries no ESPN "
                         "athlete ids and nothing on a page built from it could score.")

    donor = dict(STAGES[args.stage])
    if args.donor:
        donor = {"path": args.donor, "what": f"custom donor: {args.donor}"}
    with open(donor["path"], encoding="utf-8") as f:
        donor_payload = json.load(f)

    field_payload, field_from = read_field(result, args.field)

    # The page refuses a payload for the wrong event, so a preview built against one
    # would render nothing at all and blame ESPN for it. Catch it here, where the
    # message can say what actually happened.
    meta, players = espn_leaderboard.parse_leaderboard(field_payload)
    want = str(result["live"]["espn_event_id"])
    if not players:
        raise SystemExit(f"no competitors in the field payload from {field_from} -- "
                         "nothing to transplant a leaderboard onto.")
    if str((meta or {}).get("event_id")) != want:
        raise SystemExit(f"that field payload is ESPN event {(meta or {}).get('event_id')}, "
                         f"but this competition is event {want}. The page checks the same "
                         "thing and would refuse it.")

    weight = {g["espn"]["athlete_id"]: g["odds"]["grouping_weight"]
              for g in result["golfers"]
              if (g.get("espn") or {}).get("athlete_id") and g.get("odds")}
    competitors = field_payload["events"][0]["competitions"][0]["competitors"]
    weights = [weight.get((c.get("athlete") or {}).get("id"), 0.0) for c in competitors]

    rng = random.Random(args.seed)
    order = (market_order(weights, rng) if args.draw == "market"
             else uniform_order(len(competitors), rng))

    payload = transplant(field_payload, donor_payload, order)
    out = preview_result(result, payload, args.stage, donor, args.seed, args.draw,
                         label=not args.no_label)

    os.makedirs(args.out, exist_ok=True)
    leaderboard_path = os.path.join(args.out, f"preview-{args.stage}-leaderboard.json")
    result_path = os.path.join(args.out, f"preview-{args.stage}.json")
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    preview_meta, preview_players = espn_leaderboard.parse_leaderboard(payload)
    scored = sum(1 for p in preview_players if p["to_par"] is not None)
    print(f"Field:   {len(players)} competitors from {field_from} "
          f"[{result['tournament']['name']}, {meta.get('state')}]")
    print(f"Scores:  {donor['what']}")
    print(f"Draw:    {args.draw}, seed {args.seed} -- "
          f"{'sampled from the prices the groups were drawn on' if args.draw == 'market' else 'a straight shuffle'}")
    print(f"Preview: round {preview_meta.get('round')}, {preview_meta.get('detail')}, "
          f"started={espn_leaderboard.has_started(preview_meta.get('state'), preview_players)}, "
          f"{scored}/{len(preview_players)} with a score")

    print("\n------------- STANDINGS THE PAGE WILL SHOW -------------")
    for row, name in league_table(out, preview_players):
        best = row["best"]
        lead = "—"
        if best and best["espn"]:
            player = best["espn"]
            lead = (f"{best['name']} {player.get('position') or player.get('status_short') or '—'}"
                    f" ({player.get('to_par_display')})")
        decided = f"decided on golfer #{row['decided_at']}" if row["decided_at"] else ""
        print(f"  {row['position']:>3}  {name:<22} {lead:<34} "
              f"{row['counting']}/{row['roster']} in it  {decided}")

    print(f"\nLeaderboard -> {leaderboard_path}  ({os.path.getsize(leaderboard_path) // 1024} KB)")
    print(f"Preview     -> {result_path}  ({os.path.getsize(result_path) // 1024} KB)")
    print(f"\n    python bundle_frontend.py --result {result_path} --out dist/ "
          f"--name {os.path.basename(result_path)[:-5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
