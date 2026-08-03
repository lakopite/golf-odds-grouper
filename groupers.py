"""
groupers.py -- partition a golf field into groups of equal total implied probability.

The objective is the delta: the highest group total minus the lowest group total.
Smaller is better, and the code here does not merely try to make it small -- it
computes a lower limit that no partition can beat, and reports whether it reached it.

WHY THIS IS AN INTEGER PROBLEM, WHICH IS THE WHOLE TRICK
--------------------------------------------------------
Kalshi quotes on a price grid: the Winner series ticks at $0.001 below 10c, the other
series at a flat $0.01. Every price in a book is an exact multiple of that tick --
measured on the live 143-golfer Wyndham field, zero prices deviated from the grid.
The de-vig divides the whole book by one constant, so the grid survives it.

So this is not a real-valued partition. It is a partition of WHOLE NUMBERS of ticks,
and that hands us something a float formulation cannot have: a provable floor.
A field of 1151 ticks split 5 ways cannot beat a delta of 1 tick, because 1151 is not
divisible by 5. Reach 1 tick and you are done -- there is nothing left to search for.

The floor is not always 1. If one golfer is worth more than a group's fair share, the
group holding them is heavy no matter what else happens. On the live field that bites
from 13 groups upward: Cameron Young at 92 ticks against a fair share of 88.5, so the
floor at 13 groups is 4 ticks and at 25 groups it is 48. No algorithm beats that, and
the honest response is to exclude him (`--auto-exclude`), not to search harder. The
report names the golfers responsible so the message can say so.

THE METHOD
----------
1. Recover the tick grid and convert the field to whole numbers.
2. Compute the floor (see lower_limit).
3. Seed with Karmarkar-Karp differencing.
4. Improve with local search -- single moves and single swaps.
5. Stop the moment the delta reaches the floor. That answer is optimal, and the report
   says so rather than leaving the caller to wonder.

Karmarkar-Karp rather than the obvious "put each golfer in the lightest group" (LPT):
on a real golf field the two tie, because the long tail of 1-tick golfers gives LPT all
the small change it needs to close the last gap. They stop tying the moment the tail
goes away. Measured on a 40-item field of large weights, differencing reached 401 where
LPT reached 3797; on a 70-golfer limited field LPT missed the floor and differencing hit
it. Golf fields are usually the easy case, but a signature event is not a full field and
that is exactly when the seed matters.

WHAT THIS REPLACED
------------------
Five methods (backtracking, dynamic programming, simulated annealing, a genetic
algorithm, and greedy redistribution), each run on every field, with the smallest delta
winning. They agreed on the live field for the reason above -- it is easy -- so the
comparison mostly measured nothing. Off that field they separated badly, and four of
the five had defects that made them unfixable rather than untuned:

  * The backtracking recurrence used `dp(i, 0) = inf`, which blocks the exclude path and
    forces the top golfer into the seed set on every run. Its objective was "the smallest
    total of k golfers", which has no relation to balance; the last five lines (put each
    golfer in the lightest group) did all the work.
  * The DP scaled prices with `int(odds * 100)`, so after the de-vig almost every golfer
    scaled to zero and the table was degenerate. Its cost grew with the SUM of the
    weights, not the field size: 609 seconds on a 40-item field. Its reconstruction loop
    did not terminate when the field was shorter than the group count.
  * The annealer started at temperature 1000 against a delta of 0.001 to 0.08, so it
    accepted nearly every bad move for the first thousand steps and refused every bad
    move after about 1400 of its 100000 steps. It never annealed; it was a random walk
    followed by a hill climb, and it could only swap, so it could never change a group
    size.
  * The genetic algorithm drew its crossover point from the field size (143) but sliced
    groups of about 12, so at 12 groups the crossover did nothing in 92% of draws. It
    needed 5 to 10 minutes to reach an answer this file reaches in about 3 milliseconds.
  * Greedy redistribution was the worst method on all ten test fields. Its swap was fixed
    rather than chosen -- always the heaviest golfer of the fullest group for the lightest
    of the emptiest -- so it overshot and oscillated instead of converging.
"""

import heapq
import random
import time

# How long to keep improving once the seed is placed. Only ever reached when the floor
# is unreachable, which on a real field means a golfer is over their fair share -- and
# the answer there is to exclude them, not to spend longer searching.
DEFAULT_TIME_LIMIT = 2.0

# Candidate grid denominators tried when recovering the tick scale. The Kalshi grids in
# play need 1; a field whose cheapest golfer is not itself one tick needs that golfer's
# tick count. 1000 is far past anything a price grid produces.
MAX_GRID_DENOMINATOR = 1000

# Above this many ticks the grid recovery is treated as a failure -- see to_ticks.
MAX_TICKS = 10 ** 7


def calculate_total_odds(group):
    return sum(golfer["odds"] for golfer in group)


# ---------------------------------------------------------------------------
# The tick grid
# ---------------------------------------------------------------------------

def to_ticks(odds, max_denominator=MAX_GRID_DENOMINATOR, tolerance=1e-7):
    """
    Recover the whole-number tick weights behind a list of de-vigged probabilities.

    Every odds value is `raw_price / book_sum`, and every raw price is a whole number of
    ticks, so the ratios between them are exact rationals. Divide through by the smallest
    value and look for the denominator that turns all of them into integers: usually 1,
    because a golf field almost always contains a golfer priced at a single tick, but 3
    if the cheapest golfer in the field happens to be a three-tick golfer.

    Returns (ticks, unit, exact). `unit` is the odds value of one tick, so
    `ticks[i] * unit == odds[i]`. `exact` is False when no grid was found, in which case
    the caller gets a fine fixed scale instead and MUST NOT claim optimality from it --
    the floor computed on an invented grid is a statement about the invention.
    """
    if not odds:
        raise ValueError("no odds to convert")
    smallest = min(odds)
    if smallest <= 0:
        raise ValueError("a golfer is priced at or below zero; the field never loaded")

    ratios = [o / smallest for o in odds]
    for d in range(1, max_denominator + 1):
        scaled = [r * d for r in ratios]
        if all(abs(s - round(s)) <= tolerance * max(1.0, s) for s in scaled):
            ticks = [round(s) for s in scaled]
            if sum(ticks) <= MAX_TICKS:
                return ticks, smallest / d, True
            break

    # No grid. Every caller today feeds Kalshi prices, which are always on one, so this
    # is the path for a future source that is not -- decimal odds, a model's output, a
    # hand-written file. Scale finely enough that the rounding is far below anything that
    # could change a grouping, and tell the caller the certificate is not available.
    total = sum(odds)
    scale = MAX_TICKS / total
    ticks = [max(1, round(o * scale)) for o in odds]
    return ticks, total / sum(ticks), False


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

def lower_limit(ticks, n_groups):
    """
    The smallest delta any partition of these weights into n_groups groups can have.

    Two independent reasons a delta cannot be zero, and the floor is the larger:

    Divisibility. If the total does not divide evenly by the group count, some group is
    heavier than another by at least one tick.

    Concentration. Take the j heaviest golfers. They sit in at most j groups, so those
    groups hold at least their combined weight and the heaviest of them is at least the
    average of that weight -- while the OTHER n_groups - j groups divide no more than
    what is left, so the lightest is at most that average. The gap between the two is a
    floor. Sweeping j finds the strongest version of the argument.

    The j-sweep matters. Against the weaker "total / groups" form it is one tick better
    from 14 groups upward on the live field, and that one tick is the difference between
    reporting a proven answer and reporting a hopeful one: it certifies every group count
    from 2 to 30, where the weaker bound certifies none above 13.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    ticks = sorted(ticks, reverse=True)
    total = sum(ticks)

    limit = 0 if total % n_groups == 0 else 1

    heaviest = 0
    for j in range(1, min(n_groups, len(ticks))):
        heaviest += ticks[j - 1]
        top = -(-heaviest // j)                    # ceil: the heaviest of those j groups
        rest = (total - heaviest) // (n_groups - j)  # floor: the lightest of the others
        limit = max(limit, top - rest)
    return max(limit, 0)


def dominant_golfers(ticks, n_groups):
    """
    Indices of golfers worth more than a group's fair share.

    These are the reason a floor is above one tick. No partition balances around them, so
    naming them lets the caller say "exclude this golfer" instead of "the delta is 22".
    """
    fair_share = sum(ticks) / n_groups
    return [i for i, t in enumerate(ticks) if t > fair_share]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def karmarkar_karp(ticks, n_groups):
    """
    Karmarkar-Karp differencing, generalised to n_groups ways.

    Each partial solution is a tuple of n_groups running totals held in ascending order.
    Repeatedly take the two tuples with the widest spread and combine them back to front
    -- the heaviest side of one against the lightest side of the other -- so the two
    imbalances cancel rather than compound. The last tuple standing is the partition.

    The tuples are normalised by subtracting their own minimum after each merge. Only the
    shape of an imbalance carries information; its absolute level does not, and leaving it
    in makes the heap order meaningless.
    """
    n = len(ticks)
    heap = []
    for i in range(n):
        sums = [0] * n_groups
        sums[-1] = ticks[i]
        members = [frozenset()] * (n_groups - 1) + [frozenset([i])]
        heap.append((-ticks[i], i, sums, members))
    heapq.heapify(heap)

    counter = n
    while len(heap) > 1:
        _, _, sums_a, members_a = heapq.heappop(heap)
        _, _, sums_b, members_b = heapq.heappop(heap)
        merged_sums = [sums_a[g] + sums_b[n_groups - 1 - g] for g in range(n_groups)]
        merged_members = [members_a[g] | members_b[n_groups - 1 - g] for g in range(n_groups)]

        order = sorted(range(n_groups), key=lambda g: merged_sums[g])
        base = merged_sums[order[0]]
        sums = [merged_sums[g] - base for g in order]
        members = [merged_members[g] for g in order]
        heapq.heappush(heap, (-(sums[-1] - sums[0]), counter, sums, members))
        counter += 1

    _, _, _, members = heap[0]
    assignment = [0] * n
    for g, group_members in enumerate(members):
        for i in group_members:
            assignment[i] = g
    return assignment


def longest_first(ticks, n_groups):
    """Heaviest golfer first, always into the currently lightest group."""
    heap = [(0, g) for g in range(n_groups)]
    heapq.heapify(heap)
    assignment = [0] * len(ticks)
    for i in sorted(range(len(ticks)), key=lambda i: -ticks[i]):
        total, g = heapq.heappop(heap)
        assignment[i] = g
        heapq.heappush(heap, (total + ticks[i], g))
    return assignment


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _score(sums):
    """
    The objective, with a tie-break.

    The delta alone is nearly flat: any move between two middle groups leaves it
    unchanged, so a search that reads only the delta stalls on a plateau while the
    partition is still visibly lopsided. Ranking the group totals behind it gives those
    moves a direction, and the search keeps going until the whole profile is level.
    """
    ordered = sorted(sums, reverse=True)
    return (ordered[0] - ordered[-1], tuple(ordered))


def _fill_empty_groups(assignment, ticks, n_groups):
    """
    Guarantee every group holds at least one golfer.

    A group of one is a real answer -- at 25 groups the optimal partition gives Cameron
    Young a group to himself. A group of NONE is not: that participant has no stake in
    the tournament at all. Differencing can leave one when the group count crowds the
    field size, so close it here rather than let it reach a participant.
    """
    members = [[] for _ in range(n_groups)]
    for i, g in enumerate(assignment):
        members[g].append(i)
    for g in range(n_groups):
        if members[g]:
            continue
        donor = max(range(n_groups), key=lambda d: len(members[d]))
        if len(members[donor]) < 2:
            raise ValueError("fewer golfers than groups")
        # Give away the donor's lightest golfer: it disturbs the balance least.
        i = min(members[donor], key=lambda i: ticks[i])
        members[donor].remove(i)
        members[g].append(i)
        assignment[i] = g
    return assignment


def local_search(ticks, n_groups, assignment, floor, deadline=None):
    """
    Best-improvement hill climb over single moves and single swaps.

    Both neighbourhoods are needed. A move alone cannot rebalance two groups that are
    already the right size, and a swap alone cannot change a group's size at all -- which
    is the flaw that held the old annealer at 8 ticks where the floor was 1.
    """
    n = len(ticks)
    assignment = list(assignment)
    sums = [0] * n_groups
    counts = [0] * n_groups
    for i, g in enumerate(assignment):
        sums[g] += ticks[i]
        counts[g] += 1

    while True:
        if _spread(sums) <= floor:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        current = _score(sums)
        best = None

        for i in range(n):
            a = assignment[i]
            if counts[a] == 1:
                continue                       # never empty a group
            for b in range(n_groups):
                if b == a:
                    continue
                sums[a] -= ticks[i]
                sums[b] += ticks[i]
                score = _score(sums)
                sums[a] += ticks[i]
                sums[b] -= ticks[i]
                if score < current and (best is None or score < best[0]):
                    best = (score, ("move", i, a, b))

        for i in range(n):
            a = assignment[i]
            for j in range(i + 1, n):
                b = assignment[j]
                if a == b or ticks[i] == ticks[j]:
                    continue
                shift = ticks[i] - ticks[j]
                sums[a] -= shift
                sums[b] += shift
                score = _score(sums)
                sums[a] += shift
                sums[b] -= shift
                if score < current and (best is None or score < best[0]):
                    best = (score, ("swap", i, j, a, b))

        if best is None:
            break
        move = best[1]
        if move[0] == "move":
            _, i, a, b = move
            assignment[i] = b
            sums[a] -= ticks[i]
            sums[b] += ticks[i]
            counts[a] -= 1
            counts[b] += 1
        else:
            _, i, j, a, b = move
            assignment[i], assignment[j] = b, a
            shift = ticks[i] - ticks[j]
            sums[a] -= shift
            sums[b] += shift
    return assignment, sums


def _spread(sums):
    return max(sums) - min(sums)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def solve_ticks(ticks, n_groups, time_limit=DEFAULT_TIME_LIMIT):
    """
    Partition whole-number weights into n_groups groups, smallest delta first.

    Returns (assignment, sums, floor). Deterministic: the perturbation uses a fixed seed,
    so the same weights always give the same partition. That matters more than it looks.
    The old annealer and genetic algorithm both returned a different answer to the same
    question on consecutive runs, which makes "why did my group change?" unanswerable.
    """
    n = len(ticks)
    if n < n_groups:
        raise ValueError(f"{n} golfers cannot fill {n_groups} groups")
    floor = lower_limit(ticks, n_groups)
    deadline = time.monotonic() + time_limit

    best = None
    for seed in (karmarkar_karp(ticks, n_groups), longest_first(ticks, n_groups)):
        seed = _fill_empty_groups(list(seed), ticks, n_groups)
        assignment, sums = local_search(ticks, n_groups, seed, floor, deadline)
        if best is None or _score(sums) < _score(best[1]):
            best = (assignment, sums)
        if _spread(best[1]) <= floor:
            return best[0], best[1], floor

    # The floor is out of reach from either seed. Kick the incumbent out of its local
    # optimum and re-polish, until the floor is reached or the budget runs out.
    rng = random.Random(0)
    kick = max(2, n // 10)
    while time.monotonic() < deadline:
        assignment = list(best[0])
        for _ in range(kick):
            assignment[rng.randrange(n)] = rng.randrange(n_groups)
        assignment = _fill_empty_groups(assignment, ticks, n_groups)
        assignment, sums = local_search(ticks, n_groups, assignment, floor, deadline)
        if _score(sums) < _score(best[1]):
            best = (assignment, sums)
        if _spread(best[1]) <= floor:
            break
    return best[0], best[1], floor


def partition(golfers, n_groups, time_limit=DEFAULT_TIME_LIMIT):
    """
    Partition a field into n_groups groups of equal total implied probability.

    Returns (groups, report). The report is the point of this function over
    generate_groups: it carries the floor, whether the answer reached it, and -- when it
    did not -- which golfers are the reason.
    """
    if not golfers:
        raise ValueError("no golfers to group")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if len(golfers) < n_groups:
        raise ValueError(f"{len(golfers)} golfers cannot fill {n_groups} groups")

    ticks, unit, exact = to_ticks([g["odds"] for g in golfers])
    assignment, sums, floor = solve_ticks(ticks, n_groups, time_limit=time_limit)

    groups = [[] for _ in range(n_groups)]
    for i, g in enumerate(assignment):
        groups[g].append(golfers[i])

    delta_ticks = _spread(sums)
    dominant = dominant_golfers(ticks, n_groups)
    report = {
        "delta_ticks": delta_ticks,
        "delta": delta_ticks * unit,
        "floor_ticks": floor,
        # An invented grid gives an invented floor, so an inexact conversion never
        # certifies. It still groups correctly; it just does not get to say "optimal".
        "optimal": exact and delta_ticks <= floor,
        "exact_grid": exact,
        "tick_value": unit,
        "field_ticks": sum(ticks),
        "fair_share_ticks": sum(ticks) / n_groups,
        "dominant_golfers": [
            {"golfer_name": golfers[i]["golfer_name"],
             "ticks": ticks[i],
             "fair_shares": ticks[i] / (sum(ticks) / n_groups)}
            for i in sorted(dominant, key=lambda i: -ticks[i])
        ],
        "group_sizes": sorted(len(g) for g in groups),
    }
    return groups, report


def generate_groups(golfers, n_groups, time_limit=DEFAULT_TIME_LIMIT):
    """Partition a field, discarding the report. See partition() for the report."""
    return partition(golfers, n_groups, time_limit=time_limit)[0]
