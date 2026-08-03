"""
The migration's actual blocker, tested end to end.

Before the migration, kalshi_odds.py emitted a flat golfers list and group.py could
only read the DraftKings markets/selections envelope, so the file one wrote was a
file the other could not read. These tests hold that seam shut.
"""

import json
import os
import random

import pytest

import group
import kalshi_odds


# The solver returns in milliseconds on any field it can certify, and only spends the
# time limit when the floor is out of reach. Keep it short so a pathological fixture
# cannot stall the suite.
FAST = ["--seed", "1", "--time-limit", "0.5"]


@pytest.fixture
def workspace(tmp_path, participants, monkeypatch):
    (tmp_path / "participants.json").write_text(json.dumps(participants))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(workspace, *args):
    rc = group.main([
        "--participants", "participants.json",
        "--output-dir", str(workspace / "output"),
        *FAST, *args,
    ])
    assert rc == 0
    return json.loads((workspace / "output" / "BESTGROUPS.json").read_text())


def test_kalshi_odds_output_feeds_the_grouper(workspace, kalshi_markets, participants):
    """Write with kalshi_odds.py, read with group.py. This is the seam that was broken."""
    golfers = kalshi_odds.to_golfers(kalshi_markets)
    golfers.sort(key=lambda g: g["odds"], reverse=True)
    rep = kalshi_odds.liquidity_report(golfers)
    _, odds_path = kalshi_odds.write_capture(
        "KXPGATOUR-WYC26", golfers, rep, "ask", str(workspace / "captures")
    )

    best = _run(workspace, "--data-file", odds_path, "--no-exclude")
    assert set(best["groups"]) == set(participants)
    assert sum(len(g) for g in best["groups"].values()) == len(golfers)


def test_groups_partition_the_field_exactly(workspace, kalshi_odds_file, tmp_path):
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(kalshi_odds_file))
    best = _run(workspace, "--data-file", str(path), "--no-exclude")

    names = [g["golfer_name"] for grp in best["groups"].values() for g in grp]
    assert sorted(names) == sorted(g["golfer_name"] for g in kalshi_odds_file["golfers"])
    assert len(names) == len(set(names)), "no golfer may appear twice"


def test_group_totals_sum_to_one_after_the_de_vig(workspace, kalshi_odds_file, tmp_path):
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(kalshi_odds_file))
    best = _run(workspace, "--data-file", str(path), "--no-exclude")
    assert sum(best["totals"].values()) == pytest.approx(1.0)


def test_kalshi_data_json_is_picked_up_with_no_arguments(workspace, kalshi_odds_file):
    (workspace / "kalshi_data.json").write_text(json.dumps(kalshi_odds_file))
    best = _run(workspace, "--no-exclude")
    assert sum(len(g) for g in best["groups"].values()) == len(kalshi_odds_file["golfers"])


def test_an_archived_dk_payload_still_runs(workspace, dk_payload):
    (workspace / "dk_data.json").write_text(json.dumps(dk_payload))
    best = _run(workspace, "--no-exclude")
    names = [g["golfer_name"] for grp in best["groups"].values() for g in grp]
    assert "Scottie Scheffler" in names


def test_kalshi_data_wins_over_a_stale_dk_file(workspace, kalshi_odds_file, dk_payload):
    (workspace / "dk_data.json").write_text(json.dumps(dk_payload))
    (workspace / "kalshi_data.json").write_text(json.dumps(kalshi_odds_file))
    best = _run(workspace, "--no-exclude")
    names = {g["golfer_name"] for grp in best["groups"].values() for g in grp}
    assert "Scottie Scheffler" not in names
    assert "Cameron Young" in names


def test_exclusions_drop_the_named_golfers(workspace, dk_payload):
    (workspace / "dk_data.json").write_text(json.dumps(dk_payload))
    best = _run(workspace, "--exclude", "Scottie Scheffler", "--exclude", "Rory McIlroy")
    names = {g["golfer_name"] for grp in best["groups"].values() for g in grp}
    assert names.isdisjoint({"Scottie Scheffler", "Rory McIlroy"})
    assert sum(best["totals"].values()) == pytest.approx(1.0)


def test_auto_exclude_removes_the_dominant_favourite(workspace, dk_payload):
    """Scheffler at 4/1 is 20% of a de-vigged 8-golfer book, over the 1/5 fair share."""
    (workspace / "dk_data.json").write_text(json.dumps(dk_payload))
    best = _run(workspace, "--no-exclude", "--auto-exclude")
    names = {g["golfer_name"] for grp in best["groups"].values() for g in grp}
    assert "Scottie Scheffler" not in names


def test_the_run_records_its_delta_and_whether_it_is_optimal(workspace, kalshi_odds_file, tmp_path):
    """
    The old output could say only "this method won". Winning says nothing about whether
    the answer is good, so the run now records the proven floor alongside the delta.
    """
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(kalshi_odds_file))
    _run(workspace, "--data-file", str(path), "--no-exclude")
    written = set(os.listdir(workspace / "output"))
    assert written == {"BESTGROUPS.json", "GROUPING.json"}

    info = json.loads((workspace / "output" / "GROUPING.json").read_text())
    assert info["valid"]
    assert info["delta_ticks"] >= info["floor_ticks"]
    assert info["optimal"] is (info["delta_ticks"] <= info["floor_ticks"])


def test_a_book_with_an_overround_still_produces_totals_of_one(workspace, kalshi_odds_file, tmp_path):
    """
    The trimmed fixture's ask book sums to 0.327, so on its own it never exercises an
    overround. A live Winner book sums to ~1.31; scale the fixture up to match.
    """
    scaled = {**kalshi_odds_file,
              "golfers": [{**g, "odds": g["odds"] * 4} for g in kalshi_odds_file["golfers"]]}
    assert 1.2 < sum(g["odds"] for g in scaled["golfers"]) < 1.4
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(scaled))
    best = _run(workspace, "--data-file", str(path), "--no-exclude")
    assert sum(best["totals"].values()) == pytest.approx(1.0)


def _signature(best):
    return sorted(sorted(g["golfer_name"] for g in grp) for grp in best["groups"].values())


def test_the_load_order_does_not_change_the_partition(workspace, kalshi_raw, kalshi_odds_file, tmp_path):
    """
    The partitioner is order-sensitive -- differencing breaks ties by input position --
    so loading must sort. The same odds then give the same groups whichever envelope
    they arrive in.

    Both fixtures are shuffled first. Written in file order they are already sorted by
    probability, which would let a missing sort pass unnoticed.
    """
    shuffled_raw = {**kalshi_raw, "markets": list(kalshi_raw["markets"])}
    random.Random(4).shuffle(shuffled_raw["markets"])
    shuffled_file = {**kalshi_odds_file, "golfers": list(kalshi_odds_file["golfers"])}
    random.Random(9).shuffle(shuffled_file["golfers"])
    assert [g["golfer_name"] for g in shuffled_file["golfers"]] != [
        g["golfer_name"] for g in kalshi_odds_file["golfers"]
    ], "fixture must actually be shuffled for this test to mean anything"

    raw_path, file_path = tmp_path / "raw.json", tmp_path / "converted.json"
    raw_path.write_text(json.dumps(shuffled_raw))
    file_path.write_text(json.dumps(shuffled_file))

    from_raw = _run(workspace, "--data-file", str(raw_path), "--no-exclude")
    from_file = _run(workspace, "--data-file", str(file_path), "--no-exclude")

    assert _signature(from_raw) == _signature(from_file)


def test_a_shuffled_file_groups_identically_to_a_sorted_one(workspace, kalshi_odds_file, tmp_path):
    """Same odds, different file order, same groups -- the sort on load is what guarantees it."""
    shuffled = {**kalshi_odds_file, "golfers": list(kalshi_odds_file["golfers"])}
    random.Random(17).shuffle(shuffled["golfers"])

    sorted_path, shuffled_path = tmp_path / "sorted.json", tmp_path / "shuffled.json"
    sorted_path.write_text(json.dumps(kalshi_odds_file))
    shuffled_path.write_text(json.dumps(shuffled))

    a = _run(workspace, "--data-file", str(sorted_path), "--no-exclude")
    b = _run(workspace, "--data-file", str(shuffled_path), "--no-exclude")
    assert _signature(a) == _signature(b)


def test_exclusions_that_shorten_the_field_are_refused_not_hung(workspace, participants, tmp_path):
    """
    A field shorter than the group count is not groupable at all. The check that catches
    it runs AFTER exclusions, because the exclusion list -- named or automatic -- is what
    shortens the field.
    """
    (workspace / "kalshi_data.json").write_text(json.dumps({
        "golfers": [{"golfer_name": n, "odds": o} for n, o in
                    [("A", .3), ("B", .25), ("C", .2), ("D", .15), ("E", .1), ("F", .05)]]
    }))
    with pytest.raises(SystemExit, match="cannot fill 5 groups"):
        group.main(["--participants", "participants.json",
                    "--output-dir", str(workspace / "output"),
                    "--exclude", "A", "--exclude", "B", *FAST])


def test_auto_exclude_never_shortens_the_field_into_the_hang(workspace, participants):
    (workspace / "kalshi_data.json").write_text(json.dumps({
        "golfers": [{"golfer_name": n, "odds": o} for n, o in
                    [("A", .25), ("B", .24), ("C", .23), ("D", .22), ("E", .03), ("F", .03)]]
    }))
    best = _run(workspace, "--no-exclude", "--auto-exclude")
    assert len(best["groups"]) == 5
    assert sum(len(g) for g in best["groups"].values()) >= 5


def test_a_stale_capture_announces_itself(workspace, kalshi_odds_file, capsys):
    """A local file outranks a live pull, so its provenance has to be visible."""
    (workspace / "kalshi_data.json").write_text(json.dumps(
        {**kalshi_odds_file, "fetched_at": "2026-06-01T00:00:00+00:00"}
    ))
    _run(workspace, "--no-exclude")
    out = capsys.readouterr().out
    assert "KXPGATOUR-WYC26" in out and "2026-06-01" in out


def test_more_groups_than_golfers_is_refused(workspace, participants):
    (workspace / "kalshi_data.json").write_text(
        json.dumps({"golfers": [{"golfer_name": "A", "odds": 0.5}]})
    )
    with pytest.raises(SystemExit, match="only 1 golfers"):
        group.main(["--participants", "participants.json",
                    "--output-dir", str(workspace / "output"), *FAST])


def test_the_run_never_touches_the_network_when_a_file_is_present(
    workspace, kalshi_odds_file, monkeypatch
):
    def boom(*a, **kw):
        raise AssertionError("group.py hit the network although a local odds file exists")

    monkeypatch.setattr(kalshi_odds, "get", boom)
    (workspace / "kalshi_data.json").write_text(json.dumps(kalshi_odds_file))
    _run(workspace, "--no-exclude")
