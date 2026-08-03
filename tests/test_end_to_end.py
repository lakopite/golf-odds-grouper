"""
The migration's actual blocker, tested end to end.

Before the migration, kalshi_odds.py emitted a flat golfers list and group.py could
only read the DraftKings markets/selections envelope, so the file one wrote was a
file the other could not read. These tests hold that seam shut.
"""

import json
import os

import pytest

import group
import kalshi_odds


FAST = ["--methods", "backtracking,dp,sa,ga,greedy",
        "--sa-iter", "200", "--ga-pop", "8", "--ga-generations", "3", "--seed", "1"]


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


def test_every_method_writes_its_own_result_file(workspace, kalshi_odds_file, tmp_path):
    path = tmp_path / "kalshi_data.json"
    path.write_text(json.dumps(kalshi_odds_file))
    _run(workspace, "--data-file", str(path), "--no-exclude")
    written = set(os.listdir(workspace / "output"))
    for label in group.METHOD_LABELS.values():
        assert f"{label}.json" in written


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


def test_the_load_order_does_not_change_the_partition(workspace, kalshi_raw, kalshi_odds_file, tmp_path):
    """
    The partitioners are order-sensitive: feeding the DP the API's own market order
    rather than a sorted one measurably worsens the partition. Loading must sort, so
    the same odds give the same groups whichever envelope they arrive in.
    """
    raw_path, file_path = tmp_path / "raw.json", tmp_path / "converted.json"
    raw_path.write_text(json.dumps(kalshi_raw))
    file_path.write_text(json.dumps(kalshi_odds_file))

    from_raw = _run(workspace, "--data-file", str(raw_path), "--no-exclude", "--methods", "dp")
    from_file = _run(workspace, "--data-file", str(file_path), "--no-exclude", "--methods", "dp")

    def signature(best):
        return sorted(sorted(g["golfer_name"] for g in grp) for grp in best["groups"].values())

    assert signature(from_raw) == signature(from_file)


def test_exclusions_that_shorten_the_field_are_refused_not_hung(workspace, participants, tmp_path):
    """
    dp_generate_groups does not terminate when given fewer golfers than groups, so a
    field shortened by exclusions must be caught before it reaches the partitioners.
    Without the post-exclusion check this test hangs rather than fails.
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


def test_an_unknown_method_is_refused(workspace, kalshi_odds_file):
    (workspace / "kalshi_data.json").write_text(json.dumps(kalshi_odds_file))
    with pytest.raises(SystemExit, match="unknown method"):
        group.main(["--participants", "participants.json", "--methods", "annealing"])


def test_the_run_never_touches_the_network_when_a_file_is_present(
    workspace, kalshi_odds_file, monkeypatch
):
    def boom(*a, **kw):
        raise AssertionError("group.py hit the network although a local odds file exists")

    monkeypatch.setattr(kalshi_odds, "get", boom)
    (workspace / "kalshi_data.json").write_text(json.dumps(kalshi_odds_file))
    _run(workspace, "--no-exclude")
