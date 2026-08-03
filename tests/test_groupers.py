"""
Tests for the partitioner.

The claim this file has to hold up is stronger than "the groups look balanced". The
solver reports a proven floor and says whether it reached it, so these tests check the
floor is genuinely a floor, that the answer never beats it, and that "optimal" is only
ever claimed when it is earned.
"""

import random

import pytest

import groupers


def _field(odds):
    return [{"golfer_name": f"G{i:03d}", "odds": o} for i, o in enumerate(odds)]


def _sums(groups):
    return [sum(g["odds"] for g in grp) for grp in groups]


def _delta(groups):
    s = _sums(groups)
    return max(s) - min(s)


# ---------------------------------------------------------------------------
# The tick grid -- the fact the whole approach rests on
# ---------------------------------------------------------------------------

def test_a_kalshi_book_recovers_its_own_tick_grid():
    """Kalshi Winner prices are whole $0.001 ticks, and the de-vig divides by one constant."""
    raw = [92, 40, 38, 13, 7, 4, 2, 1, 1, 1]
    total = sum(raw)
    ticks, unit, exact = groupers.to_ticks([r / total for r in raw])
    assert exact
    assert ticks == raw
    assert unit == pytest.approx(1 / total)


def test_a_grid_is_found_when_the_cheapest_golfer_is_not_one_tick():
    """Dividing by the smallest value is not enough on its own: 3, 4, 5 needs a denominator."""
    raw = [9, 5, 4, 3]
    ticks, _, exact = groupers.to_ticks([r / sum(raw) for r in raw])
    assert exact and ticks == raw


def test_odds_off_any_grid_fall_back_without_claiming_a_certificate():
    rng = random.Random(0)
    odds = [rng.random() for _ in range(30)]
    total = sum(odds)
    ticks, unit, exact = groupers.to_ticks([o / total for o in odds])
    assert not exact
    assert all(t >= 1 for t in ticks)
    # Still a faithful representation, just an invented one.
    assert sum(t * unit for t in ticks) == pytest.approx(1.0, rel=1e-4)


def test_a_zero_price_is_refused_rather_than_grouped():
    """A bid book leaves a third of the field at zero. That is not a field."""
    with pytest.raises(ValueError, match="at or below zero"):
        groupers.to_ticks([0.5, 0.3, 0.0])


def test_an_off_grid_field_never_claims_optimality():
    rng = random.Random(5)
    field = _field([rng.random() for _ in range(24)])
    _, report = groupers.partition(field, 4, time_limit=0.2)
    assert not report["exact_grid"]
    assert report["optimal"] is False


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

def test_an_indivisible_total_cannot_reach_zero():
    assert groupers.lower_limit([1] * 7, 2) == 1        # 7 is odd
    assert groupers.lower_limit([1] * 8, 2) == 0        # 8 is not


def test_a_golfer_over_their_fair_share_raises_the_floor():
    """One golfer worth 2 fair shares: their group is heavy whatever happens elsewhere."""
    ticks = [92] + [1] * 100
    assert groupers.lower_limit(ticks, 2) == 0          # 96 vs 96 is reachable
    assert groupers.lower_limit(ticks, 25) >= 46


def test_the_top_j_sweep_beats_the_plain_average_bound():
    """
    The concentration argument is what certifies the awkward group counts. On the live
    field shape at 16 groups the plain "total / groups" bound gives 21 and the answer is
    22, so only the sweep can prove the answer optimal.
    """
    ticks = [92, 40, 38, 32, 31, 30, 27, 25, 21] + [19] * 3 + [13] * 6 + [8] * 7 + \
            [5] * 10 + [4] * 18 + [3] * 17 + [2] * 19 + [1] * 15
    total = sum(ticks)
    plain = max(1, max(ticks) - total // 16)
    assert groupers.lower_limit(ticks, 16) > plain


def test_the_floor_is_never_negative_and_never_beaten():
    rng = random.Random(2)
    for _ in range(40):
        n = rng.randint(6, 40)
        k = rng.randint(2, min(8, n))
        ticks = [rng.randint(1, 60) for _ in range(n)]
        floor = groupers.lower_limit(ticks, k)
        assert floor >= 0
        _, sums, _ = groupers.solve_ticks(ticks, k, time_limit=0.15)
        assert max(sums) - min(sums) >= floor, "the floor must be a floor"


def test_lower_limit_rejects_zero_groups():
    with pytest.raises(ValueError, match="must be positive"):
        groupers.lower_limit([1, 2, 3], 0)


# ---------------------------------------------------------------------------
# The partition itself
# ---------------------------------------------------------------------------

def test_a_real_shaped_field_is_partitioned_optimally():
    """The live Wyndham shape: 1151 ticks, so 5 groups cannot beat 1 tick, and should not."""
    ticks = [92, 40, 38, 32, 31, 30, 27, 25, 21] + [19] * 3 + [17] * 2 + [16] * 4 + \
            [15] * 2 + [14] + [13] * 6 + [12] * 3 + [11] * 4 + [10] * 4 + [9] * 5 + \
            [8] * 7 + [7] * 7 + [6] * 7 + [5] * 10 + [4] * 18 + [3] * 17 + [2] * 19 + [1] * 15
    total = sum(ticks)
    field = _field([t / total for t in ticks])
    for k in (2, 5, 8, 12, 16, 25):
        groups, report = groupers.partition(field, k)
        assert report["optimal"], f"{k} groups: {report['delta_ticks']} vs floor {report['floor_ticks']}"
        assert report["delta_ticks"] == report["floor_ticks"]


def test_every_golfer_lands_in_exactly_one_group():
    ticks = [92, 40, 38, 13, 7, 4, 4, 3, 2, 2, 1, 1, 1, 1]
    field = _field([t / sum(ticks) for t in ticks])
    groups, _ = groupers.partition(field, 4)
    names = [g["golfer_name"] for grp in groups for g in grp]
    assert sorted(names) == sorted(g["golfer_name"] for g in field)
    assert len(names) == len(set(names))


def test_no_group_is_ever_empty():
    """
    A group of one is a real answer -- at 25 groups the optimum gives the favourite a
    group to himself. A group of none is not: that participant has no stake at all.
    """
    field = _field([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    for k in range(2, 7):
        groups, _ = groupers.partition(field, k)
        assert all(len(g) >= 1 for g in groups), f"{k} groups produced an empty group"


def test_the_group_totals_still_sum_to_the_field():
    ticks = [92, 40, 38, 13, 7, 4, 2, 1, 1, 1]
    field = _field([t / sum(ticks) for t in ticks])
    groups, _ = groupers.partition(field, 3)
    assert sum(_sums(groups)) == pytest.approx(sum(g["odds"] for g in field))


def test_the_same_field_always_gives_the_same_groups():
    """
    The old annealer and genetic algorithm each returned a different answer to the same
    question on consecutive runs, which makes "why did my group change?" unanswerable.
    """
    ticks = [92, 40, 38, 32, 13, 13, 8, 7, 5, 4, 4, 3, 2, 2, 1, 1, 1]
    field = _field([t / sum(ticks) for t in ticks])

    def signature():
        groups, _ = groupers.partition(list(field), 4)
        return sorted(sorted(g["golfer_name"] for g in grp) for grp in groups)

    first = signature()
    random.seed(99)
    for _ in range(4):
        assert signature() == first


def test_the_field_order_does_not_change_the_partition_quality():
    ticks = [92, 40, 38, 32, 31, 13, 13, 8, 7, 5, 4, 4, 3, 2, 2, 1, 1, 1]
    total = sum(ticks)
    rng = random.Random(8)
    shuffled = list(ticks)
    rng.shuffle(shuffled)
    a, ra = groupers.partition(_field([t / total for t in ticks]), 4)
    b, rb = groupers.partition(_field([t / total for t in shuffled]), 4)
    assert ra["delta_ticks"] == rb["delta_ticks"]


def test_a_field_shorter_than_the_group_count_is_refused_not_hung():
    with pytest.raises(ValueError, match="cannot fill"):
        groupers.partition(_field([0.5, 0.5]), 3)


def test_an_empty_field_is_refused():
    with pytest.raises(ValueError, match="no golfers"):
        groupers.partition([], 3)


def test_zero_groups_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        groupers.partition(_field([0.5, 0.5]), 0)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_names_the_golfers_who_put_a_floor_under_the_delta():
    ticks = [92] + [4] * 20
    field = _field([t / sum(ticks) for t in ticks])
    _, report = groupers.partition(field, 10)
    names = [g["golfer_name"] for g in report["dominant_golfers"]]
    assert names == ["G000"]
    assert report["dominant_golfers"][0]["fair_shares"] > 1
    assert report["floor_ticks"] > 1


def test_a_balanced_field_reports_nobody_as_dominant():
    field = _field([1 / 40] * 40)
    _, report = groupers.partition(field, 5)
    assert report["dominant_golfers"] == []


def test_the_reported_delta_matches_the_groups():
    ticks = [92, 40, 38, 13, 7, 4, 2, 1, 1, 1]
    field = _field([t / sum(ticks) for t in ticks])
    groups, report = groupers.partition(field, 3)
    assert report["delta"] == pytest.approx(_delta(groups))
    assert report["delta_ticks"] * report["tick_value"] == pytest.approx(report["delta"])


# ---------------------------------------------------------------------------
# Why differencing rather than the obvious greedy
# ---------------------------------------------------------------------------

def test_differencing_beats_lightest_group_first_when_the_tail_is_gone():
    """
    A golf field's long tail of one-tick golfers is what lets "put each golfer in the
    lightest group" close the last gap, and on a full field the two tie. Take the tail
    away and they stop tying -- which is exactly a limited-field or signature event.
    """
    rng = random.Random(3)
    ticks = [rng.randrange(10 ** 5, 10 ** 6) for _ in range(40)]

    greedy = groupers.longest_first(ticks, 5)
    greedy_sums = [0] * 5
    for i, g in enumerate(greedy):
        greedy_sums[g] += ticks[i]

    differencing = groupers.karmarkar_karp(ticks, 5)
    kk_sums = [0] * 5
    for i, g in enumerate(differencing):
        kk_sums[g] += ticks[i]

    assert max(kk_sums) - min(kk_sums) < max(greedy_sums) - min(greedy_sums)


def test_local_search_improves_on_both_seeds():
    rng = random.Random(12)
    ticks = [rng.randrange(10 ** 4, 10 ** 5) for _ in range(30)]
    floor = groupers.lower_limit(ticks, 4)

    seed = groupers.longest_first(ticks, 4)
    seed_sums = [0] * 4
    for i, g in enumerate(seed):
        seed_sums[g] += ticks[i]

    _, sums = groupers.local_search(ticks, 4, seed, floor)
    assert max(sums) - min(sums) <= max(seed_sums) - min(seed_sums)


def test_karmarkar_karp_returns_a_complete_assignment():
    rng = random.Random(21)
    for _ in range(20):
        n, k = rng.randint(5, 30), rng.randint(2, 5)
        ticks = [rng.randint(1, 100) for _ in range(n)]
        assignment = groupers.karmarkar_karp(ticks, k)
        assert len(assignment) == n
        assert all(0 <= g < k for g in assignment)
