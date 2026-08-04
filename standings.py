"""
standings.py -- rank the league off the live ESPN leaderboard.

This is the reference implementation of the rule the pool actually plays by, and it
exists in Python so the rule can be TESTED. The scoreboard the players look at runs
the same algorithm in the browser; `tests/fixtures/standings_golden.json` is the
contract between the two, so a frontend rewrite cannot quietly change who is winning.

THE RULE
--------
A team is ranked by the best position any of its golfers holds on the live
leaderboard. Teams tied on that are separated by their next-best golfer, then the
one after that, for as far as it takes. Groups are uneven, so a team can run out of
golfers partway down -- when it does, the team that still has one wins.

That is a lexicographic comparison of each team's golfer positions in ascending
order, with the shorter list padded by something worse than everything.

WHAT THIS RULE REQUIRES OF ITS INPUT
------------------------------------
A LEADERBOARD. Not merely a field -- ESPN posts a tournament's field about two days
before the first round and gives every player in it position "-", so a payload can
be complete, correct and entirely unrankable. Run this over one and all 147 golfers
land in tier 1 and come out ordered by ESPN's pre-tournament sortOrder: a full
league table, with a leader and tie-breaks, invented from nothing.

That is not a defect here and is not fixed here. This rule's contract is a
leaderboard, and the guard belongs in front of it -- `espn_leaderboard.has_started`,
mirrored by `hasStarted` in frontend/lib.js, which is what the pages ask before they
rank anything. tests/test_standings.py demonstrates the failure rather than
describing it.

RANKING A SINGLE GOLFER
-----------------------
ESPN gives a golfer a position only while they are playing and still in it, so the
rank key is a pair (tier, value) rather than a number:

    tier 0   still playing        value = position number   (T12 -> 12)
    tier 1   in the field, no position: cut / WD / DQ / not yet teed off
                                  value = sortOrder
    tier 2   not in the field     value = 0
    tier 3   the team has no golfer this deep  (padding)

Tier 0 uses the DISPLAYED position, which shares a number between tied golfers --
that is what makes the tie-break fire at all. Tier 1 falls back to `sortOrder`
because ESPN publishes no position for a cut player ("-" for all 74 of them in the
measured field) while `sortOrder` still ranks them sensibly among themselves and
below everyone who made the cut. Tier 2 is a golfer Kalshi priced who never teed
off; 4 of 151 in the measured field. They sit below every golfer who is in the
tournament but still ahead of holding nothing, because a team that drafted 12
golfers has drafted 12 golfers.

Two teams whose whole vectors are equal are genuinely tied, and are reported tied
rather than separated on something invented.
"""

# Worse than any real golfer. Used to pad a short team so the comparison can keep
# going past the end of its roster.
PADDING = (3, 0)


def golfer_rank(player):
    """
    (tier, value) for one golfer. `player` is a parsed ESPN player, or None.

    Never returns a number. A cut golfer is not "position 74"; they are in a
    different class from everyone still playing, and collapsing the two lets a team
    of cut golfers outrank a team holding someone in contention.
    """
    if player is None:
        return (2, 0)
    if player.get("position_number") is not None:
        return (0, player["position_number"])
    # Tier 1 also catches a golfer who simply has not teed off yet, which is why
    # nothing may call this before play starts. See the module docstring.
    return (1, player.get("sort_order") or 9999)


def team_vector(players):
    """A team's golfer ranks, best first. This IS the team's score."""
    return sorted(golfer_rank(p) for p in players)


def compare(vector_a, vector_b):
    """
    -1 / 0 / +1, and the depth that decided it (1-based), or None if fully tied.

    The depth is worth carrying: "won on their 3rd golfer" is the whole story of a
    close pool day, and it is free here and unrecoverable later.
    """
    depth = max(len(vector_a), len(vector_b))
    for i in range(depth):
        a = vector_a[i] if i < len(vector_a) else PADDING
        b = vector_b[i] if i < len(vector_b) else PADDING
        if a != b:
            return (-1 if a < b else 1), i + 1
    return 0, None


def _cmp_key(vector, depth):
    """The vector padded to a fixed length, so plain tuple ordering is the rule."""
    return tuple(vector[i] if i < len(vector) else PADDING for i in range(depth))


def compute(teams, players_by_key):
    """
    Rank the league.

    `teams` is a list of {team_id, golfers: [...]}, where each golfer carries enough
    to be looked up. `players_by_key` maps any of a golfer's keys -- ESPN athlete id,
    Kalshi golfer id, or name -- to a parsed ESPN player.

    Returns a list of team results in finishing order, each with its rank, its golfer
    detail in leaderboard order, and the depth at which it separated from the team
    directly above it.
    """
    rows = []
    for team in teams:
        detail = []
        for g in team.get("golfers") or []:
            player = _lookup(g, players_by_key)
            detail.append({
                "golfer_id": g.get("golfer_id"),
                "name": g.get("name"),
                "espn": player,
                "rank": golfer_rank(player),
                "in_field": player is not None,
                "made_cut": bool(player and player.get("position_number") is not None),
            })
        detail.sort(key=lambda d: d["rank"])
        vector = [d["rank"] for d in detail]
        scored = [d for d in detail if d["rank"][0] == 0]
        rows.append({
            "team_id": team.get("team_id"),
            "vector": vector,
            "golfers": detail,
            "best": detail[0] if detail else None,
            "counting": len(scored),
            "in_field": sum(1 for d in detail if d["in_field"]),
            "roster": len(detail),
            "to_par": _sum_to_par(detail),
        })

    depth = max((len(r["vector"]) for r in rows), default=0)
    # Sort on the padded vector, then team_id, so the order is total and stable. The
    # tie-break on team_id decides nothing about who is winning -- teams that reach
    # here equal are marked tied below and share a rank.
    rows.sort(key=lambda r: (_cmp_key(r["vector"], depth), str(r["team_id"])))

    rank = 0
    for i, row in enumerate(rows):
        if i == 0:
            rank = 1
            row["decided_at"] = None
            row["tied"] = False
        else:
            result, at = compare(rows[i - 1]["vector"], row["vector"])
            if result == 0:
                row["tied"] = True
                rows[i - 1]["tied"] = True
                row["decided_at"] = None
            else:
                rank = i + 1
                row["tied"] = False
                row["decided_at"] = at
        row["rank"] = rank

    # A team can be marked tied by the row after it, so the display position of the
    # first member of a tie group is only correct once the whole pass is done.
    for row in rows:
        row["position"] = f"T{row['rank']}" if row["tied"] else str(row["rank"])
    return rows


def _lookup(golfer, players_by_key):
    """
    The ESPN player for one golfer, or None.

    A golfer that carries an `espn` block came out of a result file, where the build
    already settled who they are. That block is then the ONLY answer: an athlete id
    resolves, and a null athlete id means the build looked and found nobody -- either a
    confirmed withdrawal or a name nobody has settled yet. Falling through to a name
    match there would score a golfer the build deliberately did not, and would put this
    reference implementation out of step with the page, which has no name match at all.

    Golfers with no `espn` block are handed in by hand -- the golden fixtures and
    tools/make_golden.py do this -- and keep the older, more forgiving lookup.
    """
    if "espn" in golfer:
        athlete_id = (golfer["espn"] or {}).get("athlete_id")
        return players_by_key.get(athlete_id) if athlete_id else None
    for key in ("espn_athlete_id", "golfer_id", "name"):
        value = golfer.get(key)
        if value and value in players_by_key:
            return players_by_key[value]
    return None


def _sum_to_par(detail):
    """Aggregate to-par over the golfers still in the tournament. Display only."""
    values = [d["espn"]["to_par"] for d in detail
              if d["espn"] and d["espn"].get("to_par") is not None]
    return sum(values) if values else None


def index_players(players, matches=None):
    """
    Build the lookup `compute` wants: every key a golfer might carry -> ESPN player.

    `matches` is the build-time Kalshi-name -> ESPN-player join from
    espn_leaderboard.match_field, folded in so a golfer the join settled by alias or
    by review -- whose Kalshi name is not the ESPN display name -- still resolves
    here. The scoreboard does not need this: the build writes an athlete id onto every
    golfer it resolved and the page joins on that. This is the reference
    implementation, and it is handed golfers straight out of a Kalshi field.
    """
    by_key = {}
    for p in players:
        for key in (p.get("athlete_id"), p.get("name")):
            if key:
                by_key.setdefault(key, p)
    for name, hit in (matches or {}).items():
        by_key.setdefault(name, hit["player"] if isinstance(hit, dict) and "player" in hit else hit)
    return by_key
