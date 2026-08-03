"""
Rebuilding a competition from its own result file, offline.

A result file is a complete description of a competition, which makes it the input to
the next build of it: `--from-result` reads the league, the tournament on both APIs,
the market, the price mode, the hand-picked exclusions and the seed straight back out.

What these tests are really holding shut is the thing a rebuild must never do. People
have already been told which golfers they own. A rebuild refreshes what the world did
-- the ESPN join above all, which a Wednesday build cannot finish -- and leaves the
draw exactly where it was. Re-dealing is `--regroup`, it is opt-in, and it says so.
"""

import json

import pytest

import build_competition as bc
import espn_leaderboard as espn
from test_build_competition import make_result


# ---------------------------------------------------------------------------
# Fixtures: a built competition, and an ESPN that answers without a network
# ---------------------------------------------------------------------------

# make_result() labels its synthetic field "Golfer 00".. "Golfer 39", which normalises
# to one token for all of them -- every name identical, every tier-2 key absent. That is
# fine for arithmetic and useless for a name join, so the field is relabelled with names
# shaped like the real thing: two words, distinct first initials within a surname.
FIRST = ["Adam", "Ben", "Cal", "Dan", "Eli", "Fin", "Gus", "Hal", "Ian", "Jon", "Kit", "Lee",
         "Max", "Ned", "Otto", "Pat", "Quin", "Rob", "Sam", "Tom", "Uri", "Vic", "Walt", "Xan",
         "Yves", "Zed"]
LAST = ["Ash", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Gum", "Hazel", "Ivy", "Juniper"]


@pytest.fixture
def result_file(tmp_path):
    """A four-team competition, written where a rebuild can read it."""
    result = make_result(n_teams=4, n_golfers=40)

    renamed = {}
    for i, golfer in enumerate(result["golfers"]):
        renamed[golfer["name"]] = f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST)]}"
        golfer["name"] = renamed[golfer["name"]]
    for team in result["teams"]:
        team["golfer_names"] = [renamed[n] for n in team["golfer_names"]]
    for row in result["odds_snapshot"]["excluded"]:
        row["golfer_name"] = renamed[row["golfer_name"]]

    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    return str(path), result


def leaderboard(names, event_id="401811961", state="post"):
    return {"events": [{
        "id": event_id, "name": "Wyndham Championship", "date": "2026-08-06T07:00Z",
        "status": {"type": {"state": state, "completed": state == "post"}},
        "courses": [{"name": "Sedgefield", "shotsToPar": 70}],
        "competitions": [{
            "status": {"period": 4, "type": {"detail": "Final"}},
            "competitors": [{
                "athlete": {"id": f"a{i:03d}", "displayName": name, "shortName": name,
                            "headshot": {"href": f"https://x/{i}.png"},
                            "flag": {"alt": "USA"}},
                "sortOrder": i + 1,
                "status": {"position": {"displayName": str(i + 1), "isTie": False}},
                "linescores": [{"period": 1, "displayValue": "-2", "value": 68}],
            } for i, name in enumerate(names)],
        }],
    }]}


@pytest.fixture
def espn_field(monkeypatch, result_file):
    """
    ESPN with this week's field posted: every golfer in the draw is playing.

    Returns the list of event ids fetched, so a test can assert that history was not
    walked when the field could answer for everyone.
    """
    _, result = result_file
    names = [g["name"] for g in result["golfers"]]
    fetched = []

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        fetched.append(str(event_id))
        return leaderboard(names)

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
    monkeypatch.setattr(espn, "season_calendar", lambda *a, **k: [
        {"event_id": "401811960", "name": "Rocket Classic",
         "start": "2026-07-30T07:00Z", "end": "2026-08-02T07:00Z"}])
    return fetched


@pytest.fixture
def espn_pre(monkeypatch, result_file):
    """
    ESPN as it is the night the groups are drawn: this week's event returns nothing,
    and last week's tournament has the whole field in it.
    """
    _, result = result_file
    names = [g["name"] for g in result["golfers"]]
    fetched = []

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        fetched.append(str(event_id))
        if str(event_id) == "401811961":
            return leaderboard([], state="pre")
        return leaderboard(names, event_id="401811960")

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
    monkeypatch.setattr(espn, "season_calendar", lambda *a, **k: [
        {"event_id": "401811960", "name": "Rocket Classic",
         "start": "2026-07-30T07:00Z", "end": "2026-08-02T07:00Z"}])
    return fetched


def run(argv):
    assert bc.main(argv) == 0


def rebuilt(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reading a result file back
# ---------------------------------------------------------------------------

def test_a_result_file_round_trips_into_a_league(result_file):
    """
    The teams come out of the result file rather than out of the league file, ids and
    all. Re-reading the league file instead would mint fresh ids the moment somebody
    had renamed a team -- ids are derived from names -- and hand every player a
    different group.
    """
    _, result = result_file
    league = bc.league_from_result(result)
    assert league["league_id"] == result["league"]["league_id"]
    assert [t["team_id"] for t in league["teams"]] == [t["team_id"] for t in result["teams"]]
    assert all("golfer_names" not in t for t in league["teams"])


def test_a_file_that_is_not_a_result_is_refused(tmp_path):
    path = tmp_path / "league.json"
    path.write_text(json.dumps({"league_name": "Test", "teams": []}))
    with pytest.raises(SystemExit) as exc:
        bc.load_result(str(path))
    assert "not a result file" in str(exc.value)


def test_a_missing_file_says_what_it_wanted(tmp_path):
    with pytest.raises(SystemExit) as exc:
        bc.load_result(str(tmp_path / "nope.json"))
    assert "--from-result" in str(exc.value)


def hydrate(argv, result):
    """Parse a rebuild command line exactly as main() does, and fill it from the file."""
    args = bc.build_parser().parse_args(argv)
    bc.apply_result_defaults(args, result, bc.typed_options(argv, args))
    return args


def test_every_input_is_read_back_out_of_the_file(result_file):
    args = hydrate(["--from-result", result_file[0]], result_file[1])

    assert args.kalshi_event == "KXPGATOUR-WYC26"
    assert args.tournament == "Wyndham Championship"
    assert args.espn_event == "401811961"
    assert args.odds == "winner"
    assert args.price == "ask"
    assert args.season == 2026
    assert args.seed == 7


def test_what_is_typed_beats_what_is_recorded(result_file):
    """A rebuild with one change is one flag, not a re-typed command line."""
    args = hydrate(["--from-result", result_file[0], "--seed", "99",
                    "--price", "mid", "--poll-interval", "15"], result_file[1])
    assert (args.seed, args.price, args.poll_interval) == (99, "mid", 15)
    assert args.kalshi_event == "KXPGATOUR-WYC26"


def test_typing_a_value_that_happens_to_be_the_default_still_beats_the_file(result_file):
    """
    The trap argparse sets: `--odds winner` and an `--odds` nobody passed look identical
    once parsed. If the recorded value won, somebody moving a Top 5 competition back to
    the Winner market would silently get Top 5, with their instruction dropped.
    """
    _, result = result_file
    result["sources"]["kalshi"].update(odds_type="top5", series_ticker="KXPGATOP5",
                                       price_mode="mid")
    assert hydrate(["--from-result", result_file[0]], result).odds == "top5"

    args = hydrate(["--from-result", result_file[0], "--odds", "winner", "--price", "ask"], result)
    assert args.odds == "winner"
    assert args.price == "ask"


def test_a_raw_series_ticker_survives_the_round_trip(result_file):
    """`--odds KXPGAR3LEAD` is recorded as odds_type "custom"; the ticker is the series."""
    _, result = result_file
    result["sources"]["kalshi"].update(odds_type="custom", series_ticker="KXPGAR3LEAD")
    assert hydrate(["--from-result", result_file[0]], result).odds == "KXPGAR3LEAD"


def test_named_exclusions_are_carried_but_the_fair_share_rule_is_not(tmp_path):
    """
    One is a decision and one is a rule. Who the pool chose to leave out is theirs and
    survives; who was over 1/n of the field depends on the field and is re-derived.
    """
    result = make_result(n_teams=4, n_golfers=40, excluded_names=["Golfer 00"])
    result["odds_snapshot"]["excluded"].append(
        {"golfer_name": "Golfer 01", "reason": "over_fair_share", "raw_odds": 0.1,
         "devigged_odds": 0.1})
    path = tmp_path / "r.json"
    path.write_text(json.dumps(result))
    assert hydrate(["--from-result", str(path)], result).exclude == ["Golfer 00"]


# ---------------------------------------------------------------------------
# What a rebuild preserves
# ---------------------------------------------------------------------------

def test_a_rebuild_deals_nobody_a_different_group(result_file, espn_field, tmp_path):
    """
    The one thing a rebuild must never do. People have been told which golfers they
    own; a run that quietly re-partitions takes them away.
    """
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert [t["golfer_names"] for t in after["teams"]] == [t["golfer_names"] for t in before["teams"]]
    assert [t["total_odds"] for t in after["teams"]] == [t["total_odds"] for t in before["teams"]]
    assert [t["group_index"] for t in after["teams"]] == [t["group_index"] for t in before["teams"]]
    assert after["competition_id"] == before["competition_id"]
    assert after["generator"]["seed"] == before["generator"]["seed"]


def test_a_rebuild_keeps_the_odds_the_groups_were_drawn_on(result_file, espn_field, tmp_path):
    """
    Wednesday's prices are not re-derivable on Thursday, and they are the evidence that
    the draw was even. They are carried, capture time and all.
    """
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert after["odds_snapshot"]["captured_at"] == before["odds_snapshot"]["captured_at"]
    assert after["odds_snapshot"]["captured_at"] != after["generated_at"]
    assert after["odds_snapshot"]["raw_book_sum"] == before["odds_snapshot"]["raw_book_sum"]
    assert ([(g["name"], g["odds"]["raw"], g["odds"]["grouping_weight"]) for g in after["golfers"]]
            == [(g["name"], g["odds"]["raw"], g["odds"]["grouping_weight"]) for g in before["golfers"]])
    assert after["grouping"] == before["grouping"]


def test_a_rebuild_says_what_it_was_rebuilt_from(result_file, espn_field, tmp_path):
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    first = rebuilt(out)

    assert first["rebuilt_from"]["source_generated_at"] == before["generated_at"]
    assert first["rebuilt_from"]["rebuild_count"] == 1
    assert first["rebuilt_from"]["mode"] == "refresh"
    assert first["rebuilt_from"]["first_built_at"] == before["generated_at"]

    # And a rebuild of a rebuild still points at the original build.
    again = str(tmp_path / "again.json")
    run(["--from-result", out, "--output", again])
    assert rebuilt(again)["rebuilt_from"]["rebuild_count"] == 2
    assert rebuilt(again)["rebuilt_from"]["first_built_at"] == before["generated_at"]


def test_a_rebuild_is_itself_rebuildable(result_file, espn_field, tmp_path):
    """The output is the same shape as the input, or the second rebuild is a rewrite."""
    path, _ = result_file
    out = str(tmp_path / "a.json")
    run(["--from-result", path, "--output", out])
    run(["--from-result", out, "--output", str(tmp_path / "b.json")])
    a, b = rebuilt(out), rebuilt(str(tmp_path / "b.json"))
    assert sorted(a) == sorted(b)
    assert [t["golfer_names"] for t in a["teams"]] == [t["golfer_names"] for t in b["teams"]]


def test_a_league_file_for_a_different_league_is_refused(result_file, espn_field, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"league_name": "Someone Else",
                                 "teams": [{"team_name": "X", "player_name": "Y"}]}))
    with pytest.raises(SystemExit) as exc:
        bc.main(["--from-result", result_file[0], "--league", str(other),
                 "--output", str(tmp_path / "out.json")])
    assert "team ids do not line up" in str(exc.value).replace("\n", " ")


# ---------------------------------------------------------------------------
# What a rebuild refreshes
# ---------------------------------------------------------------------------

def test_a_rebuild_finishes_the_espn_join_the_first_build_could_not(result_file, espn_field,
                                                                    tmp_path):
    """
    The reason to rebuild at all. A build run before the first tee time has no field to
    join against and every golfer comes out `deferred`; the same competition rebuilt on
    Thursday has athlete ids for all of them.
    """
    path, before = result_file
    assert all(g["espn"]["match"] == "deferred" for g in before["golfers"])

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert all(g["espn"]["athlete_id"] for g in after["golfers"])
    assert all(g["espn"]["match"] == "exact" for g in after["golfers"])
    assert all(g["espn"]["source"] == "field" for g in after["golfers"])
    assert all(g["espn"]["in_field"] is True for g in after["golfers"])
    assert after["sources"]["espn"]["field_available_at_build"] is True


def test_a_rebuild_before_the_field_exists_still_identifies_the_golfers(result_file, espn_pre,
                                                                       tmp_path):
    """
    The Wednesday case, which is most of the life of a result file. ESPN answers with no
    competitors, so the join falls back to the tournaments that are over -- and takes
    identity from them and nothing else.
    """
    path, _ = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert after["sources"]["espn"]["field_available_at_build"] is False
    assert after["sources"]["espn"]["identities_from_history"] == len(after["golfers"])
    for g in after["golfers"]:
        assert g["espn"]["athlete_id"] and g["espn"]["headshot"]
        assert g["espn"]["source"] == "history"
        assert g["espn"]["from_event"]["name"] == "Rocket Classic"
        # Unknown, not false: this week's field has not been published.
        assert g["espn"]["in_field"] is None


def test_history_is_not_walked_when_this_weeks_field_answers(result_file, espn_field, tmp_path):
    path, _ = result_file
    run(["--from-result", path, "--output", str(tmp_path / "after.json")])
    assert espn_field == ["401811961"], "no reason to read last week's leaderboard"


def test_history_can_be_switched_off(result_file, espn_pre, tmp_path):
    path, _ = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--no-espn-history", "--output", out])
    assert espn_pre == ["401811961"]
    assert all(g["espn"]["match"] == "deferred" for g in rebuilt(out)["golfers"])


def test_a_golfer_who_withdrew_is_identified_but_marked_out_of_the_field(result_file, espn_pre,
                                                                        monkeypatch, tmp_path):
    """
    History can hand back a withdrawn golfer's headshot; it cannot make them a starter.
    Both facts land in the file, because the pool will ask about both.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    absent = names[0]

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        if str(event_id) == "401811961":
            return leaderboard(names[1:], state="in")
        return leaderboard(names, event_id="401811960")

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])

    golfer = next(g for g in rebuilt(out)["golfers"] if g["name"] == absent)
    assert golfer["espn"]["athlete_id"], "we know who they are"
    assert golfer["espn"]["source"] == "history"
    assert golfer["espn"]["in_field"] is False, "and we know they are not playing"
    assert rebuilt(out)["sources"]["espn"]["match_report"]["not_in_field"] == [absent]


# ---------------------------------------------------------------------------
# --refresh-odds
# ---------------------------------------------------------------------------

def kalshi_market(golfer, ask, ticker=None):
    return {"ticker": ticker or f"KX-{golfer['golfer_id']}", "status": "active",
            "yes_sub_title": golfer["name"], "yes_bid_dollars": f"{max(ask - 0.001, 0):.4f}",
            "yes_ask_dollars": f"{ask:.4f}", "last_price_dollars": f"{ask:.4f}",
            "custom_strike": {"golf_competitor": golfer["golfer_id"]},
            "price_level_structure": "tapered_deci_cent"}


@pytest.fixture
def moving_market(monkeypatch, result_file):
    """The same field, one price up, one golfer gone, one golfer new."""
    _, result = result_file
    golfers = result["golfers"]
    markets = [kalshi_market(g, g["odds"]["raw"]) for g in golfers[1:]]
    markets[0] = kalshi_market(golfers[1], golfers[1]["odds"]["raw"] + 0.01)
    markets.append({"ticker": "KX-NEW", "status": "active", "yes_sub_title": "Monday Qualifier",
                    "yes_bid_dollars": "0.0010", "yes_ask_dollars": "0.0020",
                    "last_price_dollars": "0.0020",
                    "custom_strike": {"golf_competitor": "new-id"}})
    monkeypatch.setattr(bc.kalshi_odds, "markets_for", lambda *a, **kw: markets)
    return result


def test_refreshed_odds_sit_beside_the_drawn_ones_never_on_top_of_them(result_file, espn_field,
                                                                      moving_market, tmp_path):
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--refresh-odds", "--output", out])
    after = rebuilt(out)

    moved = next(g for g in after["golfers"] if g["name"] == before["golfers"][1]["name"])
    assert moved["odds"]["raw"] == before["golfers"][1]["odds"]["raw"], "the draw price is fixed"
    assert moved["odds"]["current"] == pytest.approx(moved["odds"]["raw"] + 0.01)
    assert after["odds_snapshot"]["captured_at"] == before["odds_snapshot"]["captured_at"]
    assert after["odds_snapshot"]["refreshed"]["at"] == after["generated_at"]
    assert after["rebuilt_from"]["mode"] == "refresh-odds"


def test_a_refresh_names_the_golfers_who_left_and_arrived(result_file, espn_field, moving_market,
                                                          tmp_path):
    """
    Two facts the pool will argue about: a drawn golfer with no market has been pulled
    off the board, and a golfer priced after the draw is in nobody's group.
    """
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--refresh-odds", "--output", out])
    refreshed = rebuilt(out)["odds_snapshot"]["refreshed"]

    assert refreshed["no_longer_priced"] == [before["golfers"][0]["name"]]
    assert refreshed["priced_since_the_draw"] == ["Monday Qualifier"]
    assert refreshed["matched"] == len(before["golfers"]) - 1


def test_a_settled_market_carries_the_snapshot_rather_than_failing(result_file, espn_field,
                                                                   monkeypatch, tmp_path, capsys):
    """
    Rebuilding after the tournament is a normal thing to want -- the final leaderboard
    is the interesting one. Every market has settled by then, so there is nothing live
    to read, and that is not a reason to refuse to write the file.
    """
    path, before = result_file
    monkeypatch.setattr(bc.kalshi_odds, "markets_for", lambda *a, **kw: [
        {"ticker": "KX-1", "status": "settled", "yes_sub_title": "Golfer 00",
         "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}])
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--refresh-odds", "--output", out])

    after = rebuilt(out)
    assert after["odds_snapshot"]["refreshed"] is None
    assert after["odds_snapshot"]["raw_book_sum"] == before["odds_snapshot"]["raw_book_sum"]
    assert all(g["odds"]["current"] is None for g in after["golfers"])
    assert "settled" in capsys.readouterr().out


def test_a_kalshi_outage_does_not_take_the_rebuild_with_it(result_file, espn_field, monkeypatch,
                                                           tmp_path):
    def boom(*a, **kw):
        raise RuntimeError("rate limited (429)")

    monkeypatch.setattr(bc.kalshi_odds, "markets_for", boom)
    out = str(tmp_path / "after.json")
    run(["--from-result", result_file[0], "--refresh-odds", "--output", out])
    assert rebuilt(out)["odds_snapshot"]["refreshed"] is None


# ---------------------------------------------------------------------------
# --regroup
# ---------------------------------------------------------------------------

def test_regroup_refuses_to_overwrite_the_draw_it_read(result_file, tmp_path):
    """
    That file is the record of the groups people already hold. Replacing it with a
    different draw is a thing to ask for out loud.
    """
    path, _ = result_file
    with pytest.raises(SystemExit):
        bc.main(["--from-result", path, "--regroup", "--output", path])


def test_regroup_and_refresh_odds_are_not_a_combination(result_file, espn_field, monkeypatch,
                                                        tmp_path, capsys):
    """--regroup already pulls fresh odds; there is nothing left for --refresh-odds."""
    path, result = result_file
    monkeypatch.setattr(bc.kalshi_odds, "markets_for",
                        lambda *a, **kw: [kalshi_market(g, g["odds"]["raw"])
                                          for g in result["golfers"]])
    run(["--from-result", path, "--regroup", "--refresh-odds", "--time-limit", "0.5",
         "--output", str(tmp_path / "out.json")])
    assert "ignored" in capsys.readouterr().out


def test_regroup_partitions_the_new_field_and_says_it_was_a_regroup(result_file, espn_field,
                                                                    monkeypatch, tmp_path):
    path, before = result_file
    monkeypatch.setattr(bc.kalshi_odds, "markets_for",
                        lambda *a, **kw: [kalshi_market(g, g["odds"]["raw"])
                                          for g in before["golfers"]])
    out = str(tmp_path / "out.json")
    run(["--from-result", path, "--regroup", "--time-limit", "0.5", "--output", out])

    after = rebuilt(out)
    assert after["rebuilt_from"]["mode"] == "regroup"
    assert after["odds_snapshot"]["captured_at"] == after["generated_at"], "a new snapshot"
    assert sum(t["golfer_count"] for t in after["teams"]) == len(after["golfers"])
    assert [t["team_id"] for t in after["teams"]] == [t["team_id"] for t in before["teams"]]


# ---------------------------------------------------------------------------
# The flags that only mean something together
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--league", "x.json", "--tournament", "wyndham", "--refresh-odds"],
    ["--league", "x.json", "--tournament", "wyndham", "--regroup"],
])
def test_the_rebuild_flags_need_a_rebuild(argv):
    with pytest.raises(SystemExit):
        bc.main(argv)


def test_a_build_needs_either_a_league_or_a_result_file():
    with pytest.raises(SystemExit):
        bc.main(["--tournament", "wyndham"])


def test_a_truncated_result_file_is_refused_rather_than_rebuilt_wrong(result_file, tmp_path):
    """
    A team roster naming a golfer who carries no weight would rebuild silently at zero,
    and the file would then report team totals nobody was dealt. There is no evidence
    left to reconstruct the real weight from, so the honest answer is to stop.
    """
    path, result = result_file
    result["golfers"] = result["golfers"][:-1]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(result))
    with pytest.raises(SystemExit) as exc:
        bc.main(["--from-result", str(broken), "--output", str(tmp_path / "out.json")])
    assert "no grouping weight" in str(exc.value)


def test_an_exported_zip_can_be_rebuilt_from_directly(result_file, espn_field, tmp_path):
    """
    The zip is usually the copy a user still has -- the page is for reading and the
    JSON inside it is for re-running. Making them unzip it first is a step with no
    thought in it.
    """
    import bundle_frontend

    path, before = result_file
    paths, _ = bundle_frontend.bundle(before, bundle_frontend.DEFAULT_TEMPLATE, str(tmp_path / "dist"))
    zip_path = next(p for p in paths if p.endswith(".zip"))

    out = str(tmp_path / "after.json")
    run(["--from-result", zip_path, "--output", out])
    assert [t["golfer_names"] for t in rebuilt(out)["teams"]] == \
        [t["golfer_names"] for t in before["teams"]]


def test_a_zip_without_a_result_in_it_says_so(tmp_path):
    import zipfile

    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("index.html", "<html></html>")
    with pytest.raises(SystemExit) as exc:
        bc.load_result(str(path))
    assert "no result.json" in str(exc.value)
