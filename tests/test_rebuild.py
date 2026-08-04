"""
Rebuilding a competition from its own result file, offline.

A result file is a complete description of a competition, which makes it the input to
the next build of it: `--from-result` reads the league, the tournament on both APIs,
the market, the price mode, the hand-picked exclusions and the seed straight back out.

What these tests are really holding shut is the thing a rebuild must never do. People
have already been told which golfers they own. A rebuild refreshes what the world did
-- the ESPN join above all, which is the entire difference between the groups sheet
drawn on Wednesday and the scoreboard built on Thursday -- and leaves the draw exactly
where it was. Re-dealing is `--regroup`, it is opt-in, and it says so.
"""

import json

import pytest

import build_competition as bc
import espn_leaderboard as espn
import match_review
from test_build_competition import make_result


# ---------------------------------------------------------------------------
# Fixtures: a built competition, and an ESPN that answers without a network
# ---------------------------------------------------------------------------

# make_result() labels its synthetic field "Golfer Zero".."Golfer Thirty Nine", which is
# distinct enough to match but shares a first token with all of it. A real field does
# not, so the field is relabelled with names shaped like the real thing: two words, with
# distinct first initials inside any one surname.
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

    Returns the list of event ids fetched. Every test that uses this fixture asserts the
    list is non-empty, because a stub that silently stops being reached does not turn the
    suite red -- it turns it slow, and starts hammering the real ESPN API from CI.
    """
    _, result = result_file
    names = [g["name"] for g in result["golfers"]]
    fetched = []

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        fetched.append(str(event_id))
        return leaderboard(names)

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
    return fetched


@pytest.fixture
def espn_no_field(monkeypatch):
    """
    ESPN as it is the night the groups are drawn: the event exists and lists nobody.

    There is nothing else to stub. A build that finds no field does not go looking
    anywhere else -- not at last week's leaderboard, not at the season calendar. It
    writes the groups and says so.
    """
    fetched = []

    def fake_fetch(event_id, league=espn.DEFAULT_LEAGUE):
        fetched.append(str(event_id))
        return leaderboard([], state="pre")

    monkeypatch.setattr(espn, "fetch_leaderboard", fake_fetch)
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
    The reason to rebuild at all, and the whole shape of a normal week in one test.

    The first build ran before the first tee time. ESPN listed nobody, so it made no
    claim about anybody: `build_mode` is groups, `live` is null and every golfer's `espn`
    block is null. The same competition rebuilt on Thursday has an athlete id for every
    one of them and a `live` block telling the page where to poll.
    """
    path, before = result_file
    assert before["build_mode"] == "groups"
    assert before["live"] is None
    assert all(g["espn"] is None for g in before["golfers"])

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert espn_field == ["401811961"], "the stub must be what answered"
    assert after["build_mode"] == "live"
    assert set(after["live"]) == {"espn_leaderboard_url", "espn_event_id",
                                  "poll_interval_seconds"}
    assert all(g["espn"]["athlete_id"] for g in after["golfers"])
    assert all(g["espn"]["match"] == "exact" for g in after["golfers"])
    assert all(g["espn"]["in_field"] is True for g in after["golfers"])
    assert after["sources"]["espn"]["field_size_at_build"] == len(after["golfers"])


def test_a_rebuild_before_the_field_exists_stays_a_groups_sheet(result_file, espn_no_field,
                                                                tmp_path):
    """
    The Wednesday case, which is most of the life of a result file. ESPN answers with no
    competitors, so the rebuild claims nothing about anybody and asks nowhere else.

    It used to go looking through the season's finished tournaments for an athlete id.
    That recovered an identity the page could not score with, at the cost of a join whose
    correctness nothing could check -- so it does not, and the file says so plainly
    instead.
    """
    path, _ = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert espn_no_field == ["401811961"], "one request, and no reason for a second"
    assert after["build_mode"] == "groups"
    assert after["live"] is None
    assert after["sources"]["espn"]["field_size_at_build"] == 0
    assert after["sources"]["espn"]["match_report"] is None
    assert all(g["espn"] is None for g in after["golfers"])


def test_a_golfer_missing_from_a_posted_field_is_unresolved_until_somebody_looks(
        result_file, monkeypatch, tmp_path):
    """
    Half of the fact, honestly stated. This golfer is not in the leaderboard the build
    read -- but "not in the field" and "the join could not spell their name" are the same
    silence, and only one of them is a withdrawal.

    So the build says what it actually knows: no athlete id, `unresolved`, and `in_field`
    None rather than False. It also writes them into the review file, because the
    difference between the two is exactly what a review settles.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    absent = names[0]
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(names[1:], state="in"))

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    golfer = next(g for g in after["golfers"] if g["name"] == absent)
    assert golfer["espn"]["athlete_id"] is None
    assert golfer["espn"]["match"] == "unresolved"
    assert golfer["espn"]["in_field"] is None, "nobody has looked yet"
    assert after["sources"]["espn"]["match_report"]["unresolved"] == [absent]
    assert after["sources"]["espn"]["match_report"]["absent"] == []

    review = json.loads((tmp_path / "match-review.json").read_text())
    assert [row["kalshi_name"] for row in review["pending"]] == [absent]


def test_a_reviewed_withdrawal_stops_being_a_question(result_file, monkeypatch, tmp_path):
    """
    The other half. Once somebody has looked at the leaderboard and confirmed this golfer
    is not in it, the file stops calling it unresolved -- and the next rebuild does not
    re-ask, because the decision travels in the result file.

    The golfer still scores nothing. That was already true. What changes is that the
    build is now finished rather than merely stuck.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    absent = names[0]
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(names[1:], state="in"))

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])

    review_path = tmp_path / "match-review.json"
    review = json.loads(review_path.read_text())
    review["decisions"] = {absent: {"absent": True, "note": "withdrew before the first round"}}
    review_path.write_text(json.dumps(review))

    again = str(tmp_path / "again.json")
    run(["--from-result", out, "--output", again])
    settled = rebuilt(again)

    golfer = next(g for g in settled["golfers"] if g["name"] == absent)
    assert golfer["espn"]["match"] == "absent"
    assert golfer["espn"]["in_field"] is False, "checked, and not playing"
    assert settled["sources"]["espn"]["match_report"]["unresolved"] == []
    assert settled["sources"]["espn"]["match_report"]["absent"] == [absent]
    assert settled["sources"]["espn"]["match_decisions"] == {
        absent: {"absent": True, "note": "withdrew before the first round"}}


def test_a_reviewed_binding_is_applied_and_then_carried_without_re_asking(
        result_file, monkeypatch, tmp_path):
    """
    The round trip that replaced the fuzzy tier. ESPN spells one golfer differently, the
    build refuses to guess, a decision binds them by athlete id, and the decision comes
    back out of the result file on the run after that.

    The carry is the part worth holding shut. Re-asking the same question every time
    somebody rebuilds on Friday, Saturday and Sunday is how a review step stops being
    done at all.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    misspelled = names[0]
    espn_names = ["Adam Ashe"] + names[1:]          # ESPN writes "Ashe", Kalshi wrote "Ash"
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(espn_names, state="in"))

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    review_path = tmp_path / "match-review.json"
    review = json.loads(review_path.read_text())
    pending = next(row for row in review["pending"] if row["kalshi_name"] == misspelled)
    assert pending["suggestions"][0]["espn_name"] == "Adam Ashe", "the answer is offered"

    review["decisions"] = {misspelled: {"athlete_id": pending["suggestions"][0]["athlete_id"]}}
    review_path.write_text(json.dumps(review))

    again = str(tmp_path / "again.json")
    run(["--from-result", out, "--output", again])
    settled = rebuilt(again)
    golfer = next(g for g in settled["golfers"] if g["name"] == misspelled)
    assert golfer["espn"]["match"] == "decision"
    assert golfer["espn"]["display_name"] == "Adam Ashe"
    assert golfer["espn"]["in_field"] is True

    # And again, with the review file deleted: the decision is in the result file now.
    review_path.unlink()
    third = str(tmp_path / "third.json")
    run(["--from-result", again, "--output", third])
    golfer = next(g for g in rebuilt(third)["golfers"] if g["name"] == misspelled)
    assert golfer["espn"]["match"] == "decision"
    assert golfer["espn"]["display_name"] == "Adam Ashe"


def test_a_review_file_for_another_tournament_binds_nobody(result_file, espn_field, tmp_path):
    """
    Its athlete ids are ids in somebody else's field. Applying them would attach real
    people to the wrong tournament, which is the exact failure the whole review path
    exists to prevent -- so the file is refused whole rather than filtered.
    """
    path, result = result_file
    stale = tmp_path / "match-review.json"
    stale.write_text(json.dumps({
        "schema": match_review.SCHEMA,
        "espn": {"event_id": "999999", "league": "pga"},
        "decisions": {result["golfers"][0]["name"]: {"absent": True}},
    }))

    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)
    assert after["sources"]["espn"]["match_decisions"] == {}
    assert after["sources"]["espn"]["match_report"]["absent"] == []


def test_an_alias_that_fired_is_recorded_so_a_rebuild_elsewhere_still_resolves_it(
        result_file, monkeypatch, tmp_path):
    """
    A result file gets exported, mailed and rebuilt on a machine with no alias file. The
    aliases that actually fired travel with it, so the same golfers resolve the same way.

    Only the ones that fired. The whole alias file is repo state, not a fact about this
    competition, and copying it in would make every result file a snapshot of somebody
    else's data.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    espn_names = ["Adam Ashe"] + names[1:]
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(espn_names, state="in"))

    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"aliases": {names[0]: "Adam Ashe", "Nobody At All": "X"}}))
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--alias-file", str(aliases), "--output", out])

    after = rebuilt(out)
    assert after["sources"]["espn"]["aliases_applied"] == {names[0]: "Adam Ashe"}
    golfer = next(g for g in after["golfers"] if g["name"] == names[0])
    assert golfer["espn"]["match"] == "alias"

    # Now rebuild with no alias file at all. The recorded one still answers.
    again = str(tmp_path / "again.json")
    run(["--from-result", out, "--alias-file", str(tmp_path / "nothing.json"),
         "--output", again])
    golfer = next(g for g in rebuilt(again)["golfers"] if g["name"] == names[0])
    assert golfer["espn"]["match"] == "alias" and golfer["espn"]["display_name"] == "Adam Ashe"


def test_a_rebuild_will_not_turn_a_scoreboard_back_into_a_groups_sheet(result_file, monkeypatch,
                                                                       tmp_path, capsys):
    """
    ESPN does not withdraw a field once it has posted one, so a rebuild that finds none
    where the last one found forty has failed to ask rather than learned something.

    Writing the file anyway would null `live` and every athlete id, and the page built
    from it would announce that a tournament halfway through its final round has not
    started. Nothing is written and the run says why; the file that was passed in is
    still correct, and rebuilding is cheap.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(names, state="in"))
    live_path = str(tmp_path / "live.json")
    run(["--from-result", path, "--output", live_path])
    assert rebuilt(live_path)["build_mode"] == "live"

    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard([], state="in"))
    out = tmp_path / "downgraded.json"
    with pytest.raises(SystemExit) as exc:
        bc.main(["--from-result", live_path, "--output", str(out)])
    assert "found none" in str(exc.value)
    assert not out.exists(), "a refused rebuild must not leave a half-written file"


def test_an_espn_outage_during_a_live_tournament_is_refused_too(result_file, monkeypatch,
                                                                tmp_path):
    """
    Same rule, other cause. An exception and an empty field are the same empty list to
    everything downstream, and neither is a reason to throw away a working scoreboard.
    """
    path, result = result_file
    names = [g["name"] for g in result["golfers"]]
    monkeypatch.setattr(espn, "fetch_leaderboard",
                        lambda event_id, league=espn.DEFAULT_LEAGUE:
                        leaderboard(names, state="in"))
    live_path = str(tmp_path / "live.json")
    run(["--from-result", path, "--output", live_path])

    def boom(event_id, league=espn.DEFAULT_LEAGUE):
        raise RuntimeError("ESPN returned HTTP 503")

    monkeypatch.setattr(espn, "fetch_leaderboard", boom)
    with pytest.raises(SystemExit) as exc:
        bc.main(["--from-result", live_path, "--output", str(tmp_path / "nope.json")])
    assert "503" in str(exc.value)


def test_the_mode_a_file_was_built_in_is_readable_without_the_key(result_file):
    """
    Files written before 2.0 carry no `build_mode`, and a rebuild has to know the mode to
    refuse a downgrade. Inferring it is exact rather than a guess: those files recorded
    the ESPN field size, and having a field is the whole of what the mode means.
    """
    _, result = result_file
    assert bc.prior_build_mode(result) == "groups"
    assert bc.prior_build_mode({"build_mode": "live"}) == "live"

    legacy = {"sources": {"espn": {"field_size_at_build": 156}}}
    assert bc.prior_build_mode(legacy) == "live"
    assert bc.prior_build_mode({"sources": {"espn": {"field_size_at_build": 0}}}) == "groups"
    assert bc.prior_build_mode({}) == "groups"


def kalshi_market(golfer, ask, ticker=None):
    return {"ticker": ticker or f"KX-{golfer['golfer_id']}", "status": "active",
            "yes_sub_title": golfer["name"], "yes_bid_dollars": f"{max(ask - 0.001, 0):.4f}",
            "yes_ask_dollars": f"{ask:.4f}", "last_price_dollars": f"{ask:.4f}",
            "custom_strike": {"golf_competitor": golfer["golfer_id"]},
            "price_level_structure": "tapered_deci_cent"}


# ---------------------------------------------------------------------------
# A rebuild does not go back to the market
#
# There was once a --refresh-odds flag that re-read Kalshi during a rebuild and baked
# a second price in beside the drawn one. It is gone. The odds are read once, when the
# groups are drawn, and a league mate looking at two prices for the same golfer -- one
# he was dealt on, one he was not -- reasonably concluded the draw was being fiddled
# with after the fact. These are the tests that keep the second price from coming back.
# ---------------------------------------------------------------------------

def test_a_rebuild_writes_no_price_that_was_read_after_the_draw(result_file, espn_field,
                                                                tmp_path):
    """
    Not `is None` -- `not in`. A field that is present and null is a slot waiting to be
    filled, and the whole point is that there is no slot.
    """
    path, _ = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--output", out])
    after = rebuilt(out)

    assert "refreshed" not in after["odds_snapshot"]
    assert all("current" not in g["odds"] for g in after["golfers"])
    assert all(set(g["odds"]) == {"raw", "devigged", "grouping_weight"}
               for g in after["golfers"])


def test_a_rebuild_never_calls_kalshi_at_all(result_file, espn_field, monkeypatch, tmp_path):
    """
    The strongest form of the claim, and the one that cannot rot: not "it ignores what
    it read" but "it does not read". Any call at all detonates.
    """
    def boom(*a, **kw):
        raise AssertionError("a rebuild asked Kalshi for prices")

    monkeypatch.setattr(bc.kalshi_odds, "markets_for", boom)
    out = str(tmp_path / "after.json")
    run(["--from-result", result_file[0], "--output", out])
    assert rebuilt(out)["odds_snapshot"]["captured_at"] == result_file[1][
        "odds_snapshot"]["captured_at"]


def test_the_flag_that_used_to_re_read_the_market_is_gone(result_file, capsys):
    """
    argparse exits 2 on an unknown flag, which is what we want -- but SystemExit alone
    would also be raised by a dozen unrelated failures on this command line (no ESPN
    stub, no --output), so it would keep passing if the flag came back. The message is
    what says argparse refused the flag rather than the run failing later.
    """
    with pytest.raises(SystemExit):
        bc.main(["--from-result", result_file[0], "--refresh-odds"])
    assert "unrecognized arguments: --refresh-odds" in capsys.readouterr().err


def test_a_rebuild_says_it_is_a_refresh_and_never_a_refresh_of_odds(result_file, espn_field,
                                                                    tmp_path):
    out = str(tmp_path / "after.json")
    run(["--from-result", result_file[0], "--output", out])
    assert rebuilt(out)["rebuilt_from"]["mode"] == "refresh"


def test_a_rebuild_at_a_different_price_does_not_relabel_the_drawn_snapshot(result_file,
                                                                            espn_field,
                                                                            tmp_path):
    """
    `--price mid` on a rebuild cannot make Wednesday's ask prices retroactively mids.
    The snapshot keeps the mode it was actually read at.
    """
    path, before = result_file
    out = str(tmp_path / "after.json")
    run(["--from-result", path, "--price", "mid", "--output", out])
    after = rebuilt(out)
    assert after["odds_snapshot"]["price_mode"] == before["odds_snapshot"]["price_mode"]
    assert after["sources"]["kalshi"]["price_mode"] == before["sources"]["kalshi"]["price_mode"]


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


def test_regroup_is_the_only_rebuild_that_reads_kalshi_at_all(result_file, espn_field,
                                                              monkeypatch, tmp_path):
    """
    A regroup re-partitions, so it needs today's book -- and the prices it reads become
    the odds at creation of a NEW draw, which is the only honest way a competition ever
    gets a second set of prices.
    """
    path, result = result_file
    reads = []

    def markets(*a, **kw):
        reads.append(a)
        return [kalshi_market(g, g["odds"]["raw"]) for g in result["golfers"]]

    monkeypatch.setattr(bc.kalshi_odds, "markets_for", markets)
    out = str(tmp_path / "out.json")
    run(["--from-result", path, "--regroup", "--time-limit", "0.5", "--output", out])
    assert reads, "a regroup has to price the field it is dealing"
    assert "refreshed" not in rebuilt(out)["odds_snapshot"]


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

def test_regroup_needs_a_rebuild(capsys):
    """
    Checked on the message, not just on SystemExit: argparse exits 2 for any of a dozen
    reasons, so `pytest.raises(SystemExit)` alone would keep passing if the flag were
    quietly removed.
    """
    with pytest.raises(SystemExit):
        bc.main(["--league", "x.json", "--tournament", "wyndham", "--regroup"])
    assert "--regroup only means something with --from-result" in capsys.readouterr().err


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


def test_excluding_a_golfer_on_a_rebuild_does_not_crash_the_parser(result_file, espn_field,
                                                                   tmp_path):
    """
    --exclude is an `append` option, so argparse appends to whatever its default is.
    Probing for typed options by replacing defaults with a sentinel made the parser
    itself raise AttributeError the moment anybody used it.
    """
    path, result = result_file
    args = hydrate(["--from-result", path, "--exclude", result["golfers"][0]["name"]], result)
    assert args.exclude == [result["golfers"][0]["name"]]

    # And end to end, through main().
    run(["--from-result", path, "--exclude", result["golfers"][0]["name"],
         "--output", str(tmp_path / "out.json")])


def test_the_rebuild_inputs_and_the_file_they_are_read_from_stay_in_step(result_file):
    """
    REBUILD_INPUTS is the list of what a rebuild reads back, and both the probe and the
    fill iterate it. A key added to one and not the other should raise here rather than
    silently stop being carried.
    """
    args = hydrate(["--from-result", result_file[0]], result_file[1])
    for dest in bc.REBUILD_INPUTS:
        assert hasattr(args, dest), dest


# ---------------------------------------------------------------------------
# What a rebuild must not quietly lose
# ---------------------------------------------------------------------------

def test_the_tournaments_dates_and_course_survive_an_espn_outage(result_file, monkeypatch,
                                                                 tmp_path):
    """
    Start, end and course are static facts about the tournament -- ESPN being down does
    not un-schedule it. Nulling them costs the page its "Not started -- first round
    <date>" line, which is exactly the state a rebuild is usually run to serve.
    """
    path, before = result_file
    before["tournament"].update(start="2026-08-06T07:00Z", end="2026-08-09T07:00Z",
                                course={"name": "Sedgefield CC", "par": 70})
    (tmp_path / "with-dates.json").write_text(json.dumps(before))

    def down(*a, **kw):
        raise RuntimeError("ESPN returned HTTP 503")

    monkeypatch.setattr(espn, "fetch_leaderboard", down)
    monkeypatch.setattr(espn, "season_calendar", lambda *a, **k: [])

    out = str(tmp_path / "after.json")
    run(["--from-result", str(tmp_path / "with-dates.json"), "--output", out])
    after = rebuilt(out)["tournament"]

    assert after["start"] == "2026-08-06T07:00Z"
    assert after["end"] == "2026-08-09T07:00Z"
    assert after["course"] == {"name": "Sedgefield CC", "par": 70}
    # But not the state: this run could not read it, so it does not claim to know it.
    assert after["state_at_build"] is None


def test_the_fair_share_rule_being_switched_off_survives_a_regroup(tmp_path, espn_field,
                                                                   monkeypatch):
    """
    A file with no over_fair_share exclusions either had nobody over the line or had the
    rule switched off, and those two rebuild differently. Without recording which, a
    --regroup of a --no-auto-exclude competition drops the favourite out of the pool
    altogether -- not to another team, out.
    """
    result = make_result(n_teams=4, n_golfers=12)
    # One golfer worth more than a quarter of the field, kept deliberately.
    result["golfers"][0]["odds"].update(raw=0.4, devigged=0.4, grouping_weight=0.4)
    result["odds_snapshot"]["auto_exclude"] = False
    path = tmp_path / "kept.json"
    path.write_text(json.dumps(result))

    monkeypatch.setattr(bc.kalshi_odds, "markets_for", lambda *a, **kw: [
        kalshi_market(g, g["odds"]["raw"]) for g in result["golfers"]])
    out = str(tmp_path / "out.json")
    run(["--from-result", str(path), "--regroup", "--time-limit", "0.3", "--output", out])

    after = rebuilt(out)
    assert after["odds_snapshot"]["excluded"] == []
    assert all(g["team_id"] for g in after["golfers"]), "nobody was dropped from the pool"
    assert after["odds_snapshot"]["auto_exclude"] is False, "and the decision is still recorded"


def test_a_fresh_build_records_whether_the_fair_share_rule_ran():
    from test_build_competition import make_result as fresh

    assert fresh()["odds_snapshot"]["auto_exclude"] is True


def test_replacing_the_recorded_exclusions_is_said_out_loud(result_file, capsys):
    """Silently re-admitting a golfer the pool excluded by hand is not on."""
    _, result = result_file
    result["odds_snapshot"]["excluded"] = [
        {"golfer_name": "Adam Ash", "reason": "named", "raw_odds": 0.1, "devigged_odds": 0.1}]
    hydrate(["--from-result", "x.json", "--exclude", "Ben Ash"], result)
    assert "replaces the 1 exclusion(s) recorded" in capsys.readouterr().out


@pytest.mark.parametrize("dropped", ["odds_snapshot", "tournament", "generator", "generated_at"])
def test_a_result_file_missing_a_block_the_rebuild_reads_is_refused(result_file, tmp_path,
                                                                    dropped):
    """
    Checking six blocks and then dying on a KeyError in the seventh reports the same
    damaged file twice, once badly.
    """
    _, result = result_file
    result.pop(dropped)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(result))
    with pytest.raises(SystemExit) as exc:
        bc.load_result(str(path))
    assert dropped in str(exc.value)
