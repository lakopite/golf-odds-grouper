"""Tests for the grouper's odds loading, de-vig, and exclusion logic."""

import json

import pytest

import group


# ---------------------------------------------------------------------------
# Shape dispatch -- both envelopes carry a top-level "markets" key
# ---------------------------------------------------------------------------

def test_loads_the_raw_kalshi_markets_envelope(kalshi_raw):
    golfers = group.load_golfers(kalshi_raw)
    assert len(golfers) == sum(1 for m in kalshi_raw["markets"] if m["status"] == "active")
    assert golfers[0]["golfer_name"]
    assert all(0 < g["odds"] <= 1 for g in golfers)


def test_loads_the_converted_kalshi_odds_file(kalshi_odds_file):
    golfers = group.load_golfers(kalshi_odds_file)
    assert len(golfers) == len(kalshi_odds_file["golfers"])
    assert golfers[0]["golfer_id"]


def test_loads_a_bare_golfer_list():
    golfers = group.load_golfers([{"golfer_name": "A", "odds": 0.1}])
    assert golfers == [{"golfer_name": "A", "odds": 0.1, "golfer_id": None}]


def test_loads_the_legacy_draftkings_envelope(dk_payload):
    golfers = group.load_golfers(dk_payload)
    names = [g["golfer_name"] for g in golfers]
    assert "Cameron Young" in names
    # Only the "Winner" market's selections; the Top 5 market must not leak in.
    assert len(golfers) == 8


def test_draftkings_and_kalshi_envelopes_are_told_apart(dk_payload, kalshi_raw):
    """Both have markets[]. Dispatch must read the contents, not just the key."""
    assert len(group.load_golfers(dk_payload)) == 8
    assert len(group.load_golfers(kalshi_raw)) == 18


def test_unrecognised_shape_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unrecognised odds file shape"):
        group.load_golfers({"something": "else"})


def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty list"):
        group.load_golfers([])


def test_malformed_list_raises():
    with pytest.raises(ValueError, match=r"\{golfer_name, odds\}"):
        group.load_golfers([{"name": "A", "price": 0.1}])


def test_markets_without_selections_or_kalshi_fields_raises():
    with pytest.raises(ValueError, match="truncated"):
        group.load_golfers({"markets": [{"id": "m1", "name": "Winner"}]})


def test_read_odds_file_round_trips(tmp_path, kalshi_odds_file):
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(kalshi_odds_file))
    assert len(group.read_odds_file(str(path))) == len(kalshi_odds_file["golfers"])


# ---------------------------------------------------------------------------
# check_book -- the guards to_golfers() applies do not reach a converted file
# ---------------------------------------------------------------------------

def test_a_settled_field_is_refused(kalshi_odds_file):
    """
    kalshi_odds.py --include-closed writes settled markets, which quote ask=1.0000.
    Read back, they normalize cleanly and yield a confident grouping of a tournament
    that finished months ago. That is exactly the failure this repo has already paid for.
    """
    settled = {**kalshi_odds_file,
               "golfers": [{**g, "odds": 1.0} for g in kalshi_odds_file["golfers"]]}
    with pytest.raises(ValueError, match="settled markets"):
        group.load_golfers(settled)


def test_one_settled_market_is_enough_to_refuse(kalshi_odds_file):
    poisoned = {**kalshi_odds_file, "golfers": list(kalshi_odds_file["golfers"])}
    poisoned["golfers"][3] = {**poisoned["golfers"][3], "odds": 1.0}
    with pytest.raises(ValueError, match="priced at or above 1.0"):
        group.load_golfers(poisoned)


def test_a_price_mode_mismatch_warns(kalshi_odds_file, capsys):
    """A converted file cannot be re-priced, so --price ask on a bid file must say so."""
    group.load_golfers({**kalshi_odds_file, "price_mode": "bid"}, price="ask")
    assert "written with --price bid" in capsys.readouterr().out


def test_a_matching_price_mode_is_silent(kalshi_odds_file, capsys):
    group.load_golfers(kalshi_odds_file, price="ask")
    assert "--price" not in capsys.readouterr().out


def test_a_top_n_book_warns_that_it_is_not_a_probability(capsys):
    top5 = {"golfers": [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(50)]}
    group.load_golfers(top5)
    assert "NOT mutually" in capsys.readouterr().out


def test_a_thin_book_warns(capsys):
    thin = {"golfers": [{"golfer_name": f"G{i}", "odds": 0.01} for i in range(50)]}
    group.load_golfers(thin)
    assert "below 1.0" in capsys.readouterr().out


def test_a_normal_winner_book_warns_about_nothing(capsys):
    winner = {"golfers": [{"golfer_name": f"G{i}", "odds": 1.308 / 100} for i in range(100)]}
    group.load_golfers(winner)
    assert "WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# DraftKings legacy parser
# ---------------------------------------------------------------------------

def test_fractional_odds_decode():
    # "9/1" is a 10% shot: 1 / (1 + 9).
    assert group.fractional_odds_to_implied_probability("9/1") == pytest.approx(0.1)
    assert group.fractional_odds_to_implied_probability("1/1") == pytest.approx(0.5)


def test_dk_odds_type_selects_the_right_market(dk_payload):
    top5 = group.list_dk_golf_odds(dk_payload, odds_type="Top 5 (Including Ties)")
    assert [g["golfer_name"] for g in top5] == ["Cameron Young", "Hideki Matsuyama"]


def test_missing_dk_market_raises_and_names_the_alternatives(dk_payload):
    with pytest.raises(ValueError, match="no DraftKings market"):
        group.list_dk_golf_odds(dk_payload, odds_type="Top 3")


# ---------------------------------------------------------------------------
# De-vig
# ---------------------------------------------------------------------------

def _book(sum_to, n=10):
    """A synthetic book summing to `sum_to`, top-heavy the way a real field is."""
    weights = [2 ** -i for i in range(n)]
    scale = sum_to / sum(weights)
    return [{"golfer_name": f"G{i}", "odds": w * scale} for i, w in enumerate(weights)]


def _flat_book(sum_to, n=40):
    """A book of indistinguishable longshots -- the bottom third of a golf field."""
    return [{"golfer_name": f"G{i}", "odds": sum_to / n} for i in range(n)]


def test_normalize_turns_any_book_into_a_distribution():
    # A Kalshi Winner ask book runs ~1.30, a Top 5 book toward 5, a Top 10 toward 10.
    for book_sum in (1.308, 5.0, 10.0, 0.906):
        fair = group.normalize_probabilities(_book(book_sum))
        assert sum(g["odds"] for g in fair) == pytest.approx(1.0)


def test_normalize_preserves_relative_weight():
    raw = _book(1.308)
    fair = group.normalize_probabilities(raw)
    assert fair[0]["odds"] / fair[1]["odds"] == pytest.approx(raw[0]["odds"] / raw[1]["odds"])


def test_normalize_keeps_other_fields():
    fair = group.normalize_probabilities([{"golfer_name": "A", "odds": 2.0, "golfer_id": "x"}])
    assert fair[0]["golfer_id"] == "x"


def test_normalize_rejects_an_empty_book():
    with pytest.raises(ValueError, match="sum to zero"):
        group.normalize_probabilities([{"golfer_name": "A", "odds": 0.0}])


def test_devig_is_idempotent():
    once = group.normalize_probabilities(_book(1.308))
    assert [g["odds"] for g in group.normalize_probabilities(once)] == pytest.approx(
        [g["odds"] for g in once]
    )


# ---------------------------------------------------------------------------
# Conditional odds
# ---------------------------------------------------------------------------

def test_conditional_odds_still_sum_to_one_off_a_1_3_book():
    """The scaling question: the de-vig must key off the OBSERVED sum, not 1.0."""
    conditional = group.odds_to_conditional(_book(1.308), ["G0", "G1"])
    assert sum(g["odds"] for g in conditional) == pytest.approx(1.0)
    assert {g["golfer_name"] for g in conditional}.isdisjoint({"G0", "G1"})


def test_conditional_odds_are_scale_invariant():
    """Same field, different overround -> identical conditional probabilities."""
    a = group.odds_to_conditional(_book(1.308), ["G0"])
    b = group.odds_to_conditional(_book(5.0), ["G0"])
    assert [g["odds"] for g in a] == pytest.approx([g["odds"] for g in b])


def test_conditional_odds_preserve_relative_weight_among_survivors():
    raw = _book(1.308)
    conditional = group.odds_to_conditional(raw, ["G0"])
    assert conditional[0]["odds"] / conditional[1]["odds"] == pytest.approx(
        raw[1]["odds"] / raw[2]["odds"]
    )


def test_excluding_nobody_is_just_a_de_vig():
    raw = _book(1.308)
    assert [g["odds"] for g in group.odds_to_conditional(raw, [])] == pytest.approx(
        [g["odds"] for g in group.normalize_probabilities(raw)]
    )


def test_excluding_a_name_not_in_the_field_warns(capsys):
    group.odds_to_conditional(_book(1.308), ["Scottie Scheffler"])
    assert "not in this field" in capsys.readouterr().out


def test_excluding_the_whole_field_raises():
    book = _book(1.308, n=3)
    with pytest.raises(ValueError, match="whole field"):
        group.odds_to_conditional(book, ["G0", "G1", "G2"])


@pytest.mark.parametrize(
    "odds",
    [
        [0.21, 0.11, 0.07, 0.03, 0.01],   # normalizes to 0.9999999999999999
        [0.09, 0.044, 0.041, 0.038],
        [1 / 3, 1 / 3, 1 / 3],
        [0.7, 0.2, 0.1],
    ],
)
def test_excluding_the_whole_field_raises_despite_float_error(odds):
    """
    The excluded probabilities are normalized floats, so they land just under 1.0 far
    more often than on it. A guard written as `sum(excluded) >= 1.0` misses those and
    returns an empty field, which then reaches the partitioners.
    """
    book = [{"golfer_name": f"G{i}", "odds": o} for i, o in enumerate(odds)]
    with pytest.raises(ValueError, match="whole field"):
        group.odds_to_conditional(book, [g["golfer_name"] for g in book])


def test_conditional_odds_never_return_an_empty_field():
    book = [{"golfer_name": f"G{i}", "odds": o} for i, o in enumerate([0.21, 0.11, 0.07, 0.03, 0.01])]
    survivors = group.odds_to_conditional(book, ["G0", "G1", "G2", "G3"])
    assert [g["golfer_name"] for g in survivors] == ["G4"]
    assert survivors[0]["odds"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The 1/participants threshold
# ---------------------------------------------------------------------------

def test_threshold_is_measured_after_de_vig():
    """
    Against the raw 1.308-sum ask book a golfer at 0.21 looks over a 1/5 threshold.
    De-vigged they are 0.16, and they are not. Compare on the same scale or the rule
    fires ~30% too eagerly.
    """
    book = [{"golfer_name": "Fav", "odds": 0.21}] + [
        {"golfer_name": f"G{i}", "odds": (1.308 - 0.21) / 20} for i in range(20)
    ]
    assert group.golfers_over_threshold(book, 5) == []
    assert [g["golfer_name"] for g in group.golfers_over_threshold(book, 7)] == ["Fav"]


def test_threshold_flags_a_genuine_outlier():
    book = [{"golfer_name": "Fav", "odds": 0.60}] + [
        {"golfer_name": f"G{i}", "odds": 0.05} for i in range(10)
    ]
    assert [g["golfer_name"] for g in group.golfers_over_threshold(book, 5)] == ["Fav"]


def test_threshold_rejects_zero_participants():
    with pytest.raises(ValueError, match="must be positive"):
        group.golfers_over_threshold(_book(1.0), 0)


def test_auto_exclusion_iterates_to_a_fixed_point():
    """
    Removing a golfer redistributes their weight, which can push the next one over.
    With 4 groups the threshold is 0.25: only A is over to begin with, but once A
    goes B climbs from 0.20 to 0.286 and goes too. One pass would miss B.
    """
    book = (
        [{"golfer_name": "A", "odds": 0.30}, {"golfer_name": "B", "odds": 0.20}]
        + [{"golfer_name": f"G{i}", "odds": 0.0625} for i in range(8)]
    )
    assert [g["golfer_name"] for g in group.golfers_over_threshold(book, 4)] == ["A"]
    assert group.auto_exclusions(book, 4) == ["A", "B"]


def test_auto_exclusion_excludes_nobody_on_a_flat_field():
    assert group.auto_exclusions(_flat_book(1.308, n=40), 5) == []


@pytest.mark.parametrize(
    "odds,n_participants",
    [
        ([0.40, 0.35, 0.15, 0.10], 3),                    # both A and B clear 1/3
        ([0.25, 0.24, 0.23, 0.22, 0.03, 0.03], 5),        # four clear 1/5 in sequence
        ([0.5, 0.3, 0.2], 2),
        ([0.9, 0.05, 0.03, 0.02], 3),
    ],
)
def test_auto_exclusion_leaves_at_least_one_golfer_per_group(odds, n_participants):
    """
    The cascade must stop rather than eat the field. A field shorter than the group
    count is not groupable, and dp_generate_groups does not fail gracefully on one --
    its reconstruction loop never terminates.
    """
    book = [{"golfer_name": f"G{i}", "odds": o} for i, o in enumerate(odds)]
    excluded = group.auto_exclusions(book, n_participants)
    assert len(book) - len(excluded) >= n_participants


def test_auto_exclusion_takes_the_heaviest_first_when_it_must_stop_short():
    book = [
        {"golfer_name": "A", "odds": 0.40}, {"golfer_name": "B", "odds": 0.35},
        {"golfer_name": "C", "odds": 0.15}, {"golfer_name": "D", "odds": 0.10},
    ]
    assert group.auto_exclusions(book, 3) == ["A"]


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------

def test_validate_groups_accepts_a_clean_partition():
    golfers = [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(4)]
    groups = {"Group 0": golfers[:2], "Group 1": golfers[2:]}
    assert group.validate_groups(groups, golfers)


def test_validate_groups_rejects_a_duplicated_golfer():
    golfers = [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(4)]
    groups = {"Group 0": [golfers[0], golfers[0]], "Group 1": golfers[2:]}
    assert not group.validate_groups(groups, golfers)


def test_validate_groups_rejects_a_dropped_golfer():
    golfers = [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(4)]
    groups = {"Group 0": golfers[:2], "Group 1": golfers[2:3]}
    assert not group.validate_groups(groups, golfers)


def test_percentage_difference_handles_two_zeroes():
    assert group.percentage_difference(0, 0) == float('inf')


def test_confirm_group_writes_its_json(tmp_path):
    golfers = [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(4)]
    groups = {"Group 0": golfers[:2], "Group 1": golfers[2:]}
    totals = {"Group 0": 0.2, "Group 1": 0.2}
    info = group.confirm_group("Greedy Algorithm", groups, totals, golfers, output_dir=str(tmp_path))
    assert info["valid"] and info["delta"] == 0
    assert json.loads((tmp_path / "Greedy Algorithm.json").read_text())["method"] == "Greedy Algorithm"


def test_confirm_group_returns_none_for_an_invalid_partition(tmp_path):
    golfers = [{"golfer_name": f"G{i}", "odds": 0.1} for i in range(4)]
    groups = {"Group 0": golfers[:2], "Group 1": golfers[:2]}
    totals = {"Group 0": 0.2, "Group 1": 0.2}
    assert group.confirm_group("Backtracking", groups, totals, golfers, output_dir=str(tmp_path)) is None
