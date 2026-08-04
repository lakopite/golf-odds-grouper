"""
Tests for build_competition.py, offline.

The build's one network-free job is assembling the result file, and the result file is
the source of truth for a whole competition -- so what is tested here is mostly that
every number in it means what its key says it means: that the weights sum to 1.0 over
the golfers that were grouped, that an excluded golfer has no weight, that every golfer
belongs to exactly one team, and that nothing silently loses a golfer along the way.

The live pull is covered by tests/test_live.py, which is skipped by default.
"""

import json
import os
import types

import pytest

import build_competition as bc
import espn_leaderboard
import group as grouper_cli
import groupers
import league as league_mod
import match_review


# ---------------------------------------------------------------------------
# Tournament resolution
# ---------------------------------------------------------------------------

KALSHI_EVENTS = [
    {"event_ticker": "KXPGATOUR-WYC26", "title": "Wyndham Championship Winner",
     "sub_title": "2026 Wyndham Championship"},
    {"event_ticker": "KXPGATOUR-ROC26", "title": "Rocket Classic Winner",
     "sub_title": "2026 Rocket Classic"},
    {"event_ticker": "KXPGATOUR-THOC26", "title": "The Open Championship Winner",
     "sub_title": "2026 The Open Championship"},
]


@pytest.fixture
def kalshi_events(monkeypatch):
    monkeypatch.setattr(bc.kalshi_odds, "events_for", lambda series, **kw: KALSHI_EVENTS)


@pytest.mark.parametrize("query,expected", [
    ("Wyndham Championship", "KXPGATOUR-WYC26"),
    ("wyndham", "KXPGATOUR-WYC26"),
    ("2026 Wyndham Championship", "KXPGATOUR-WYC26"),
    ("Rocket Classic", "KXPGATOUR-ROC26"),
    ("KXPGATOUR-WYC26", "KXPGATOUR-WYC26"),
    ("WYC26", "KXPGATOUR-WYC26"),
    ("wyc26", "KXPGATOUR-WYC26"),
])
def test_resolve_kalshi_event(kalshi_events, query, expected):
    best, _ = bc.resolve_kalshi_event(query, "KXPGATOUR")
    assert best["event_ticker"] == expected


def test_a_ticker_beats_a_name_lookup(kalshi_events):
    """Event codes are not derivable, so an explicit one must never be re-guessed."""
    best, ranked = bc.resolve_kalshi_event("ROC26", "KXPGATOUR")
    assert best["event_ticker"] == "KXPGATOUR-ROC26" and ranked[0]["score"] == 1.0


def test_an_unknown_tournament_resolves_to_nothing(kalshi_events):
    best, _ = bc.resolve_kalshi_event("Ryder Cup", "KXPGATOUR")
    assert best is None


def test_the_kalshi_and_espn_resolvers_use_the_same_scorer():
    """
    Both sides have to agree on what "wyndham" means, or the build pairs a Kalshi
    event with someone else's leaderboard and scores the league against the wrong
    tournament -- which looks entirely normal until Sunday.
    """
    assert espn_leaderboard.score_name("wyndham", "2026 Wyndham Championship") == 1.0
    assert espn_leaderboard.score_name("wyndham", "Wyndham Championship") == 1.0
    assert espn_leaderboard.score_name("wyndham", "Rocket Classic") == 0.0


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------

def field(*pairs):
    return [{"golfer_name": n, "odds": o, "golfer_id": n.lower()} for n, o in pairs]


def test_named_exclusions_carry_their_reason():
    golfers = field(("A", 0.3), ("B", 0.2), ("C", 0.1))
    assert bc.resolve_exclusions(golfers, 3, ["A"], auto=False) == [
        {"golfer_name": "A", "reason": "named"}]


def test_a_named_exclusion_that_is_not_in_the_field_is_dropped_with_a_warning(capsys):
    golfers = field(("A", 0.3), ("B", 0.2))
    assert bc.resolve_exclusions(golfers, 2, ["Tiger Woods"], auto=False) == []
    assert "excluded nothing" in capsys.readouterr().out


def test_auto_exclusion_drops_anyone_over_a_group_s_fair_share():
    """One golfer worth more than 1/n of the field cannot be balanced around."""
    golfers = field(("Big", 0.6), ("B", 0.1), ("C", 0.1), ("D", 0.1), ("E", 0.1))
    out = bc.resolve_exclusions(golfers, 2, [], auto=True)
    assert [e["golfer_name"] for e in out] == ["Big"]
    assert out[0]["reason"] == "over_fair_share"


def test_auto_exclusion_is_measured_after_the_named_ones(capsys):
    """
    Removing a golfer redistributes their weight, so the threshold has to be re-read
    against the field that is left. B is under 1/2 of the whole field and over 1/2 of
    the field without A.
    """
    golfers = field(("A", 0.40), ("B", 0.35), ("C", 0.15), ("D", 0.10))
    named_only = bc.resolve_exclusions(golfers, 2, [], auto=True)
    assert [e["golfer_name"] for e in named_only] == []

    with_a_gone = bc.resolve_exclusions(golfers, 2, ["A"], auto=True)
    assert [e["golfer_name"] for e in with_a_gone] == ["A", "B"]


def test_auto_exclusion_off_leaves_the_field_alone():
    golfers = field(("Big", 0.9), ("B", 0.05), ("C", 0.05))
    assert bc.resolve_exclusions(golfers, 2, [], auto=False) == []


# ---------------------------------------------------------------------------
# assemble()
# ---------------------------------------------------------------------------

# Synthetic golfer names that survive normalisation. "Golfer 00" does not: normalising
# drops every non-letter, so all forty of them fold to "golfer", collide, and are refused
# by the ambiguity guard -- which is the guard working correctly and a fixture that
# cannot exercise a single match. Spelled-out numbers keep them distinct as NAMES.
_ONES = ("Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen "
         "Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen").split()
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def golfer_name(i):
    """
    `Golfer Seven`, `Golfer Thirty Four`, `Golfer One Hundred Six`.

    Distinct to the name matcher, unlike a number, which normalisation deletes. Goes
    past a hundred because the render suite deals out a real 147-player field.
    """
    return f"Golfer {_spell(i)}"


def _spell(i):
    if i >= 100:
        rest = i % 100
        return f"{_ONES[i // 100]} Hundred" + (f" {_spell(rest)}" if rest else "")
    if i < 20:
        return _ONES[i]
    return _TENS[i // 10] + ("" if i % 10 == 0 else f" {_ONES[i % 10]}")


def espn_stage(names, players=None, matches=None, report=None, decisions=None,
               started=False):
    """
    The ESPN half of a build: what build_competition.espn_stage returns.

    `matches` is keyed by Kalshi name and shaped as espn_leaderboard.match_field returns
    it. Anything not in it and not named in `report["absent"]` comes out `unresolved`.

    `started` defaults to False because that is the ordinary case: a pool is drawn the
    night before, when ESPN has posted the field and nobody has teed off. It changes
    nothing about the file except the one fact that records the clock -- the join, the
    athlete ids and the `live` block are identical either way, which is the whole point
    of there being one build.
    """
    players = players if players is not None else [
        {"athlete_id": f"a{i:02d}", "name": name, "headshot": None, "country": None,
         "position": str(i + 1), "sort_order": i + 1}
        for i, name in enumerate(names)]
    if matches is None:
        by_name = {p["name"]: p for p in players}
        matches = {n: {"player": by_name[n], "match": "exact"} for n in names if n in by_name}
    report = report if report is not None else {
        "espn_field_size": len(players), "requested": len(names), "matched": len(matches),
        "matched_decision": 0, "matched_alias": 0, "matched_exact": len(matches),
        "absent": [], "unresolved": [n for n in names if n not in matches],
        "ambiguous_names": [], "problems": [],
    }
    return {"meta": {"state": "in" if started else "pre", "course": "Sedgefield", "par": 70,
                     "started": started},
            "players": players, "matches": matches, "report": report,
            "decisions": decisions or {}, "review": None}


def make_result(n_teams=4, n_golfers=40, excluded_names=(), tmp_path=None, espn=None):
    """
    Run the whole assembly with a synthetic field and no network.

    By default the ESPN join resolves the whole field and play has not started, which is
    what a build made the night before looks like. Pass `espn=espn_stage([...],
    started=True)` for one made once play is under way.
    """
    teams = [{"team_id": f"t{i}", "team_name": f"Team {i}", "player_name": f"P{i}",
              "team_logo": None} for i in range(n_teams)]

    raw = []
    for i in range(n_golfers):
        odds = round(0.002 * (n_golfers - i), 4)
        raw.append({"golfer_name": golfer_name(i), "odds": odds, "golfer_id": f"g{i:02d}",
                    "_bid": odds - 0.001, "_ask": odds, "_spread": 0.001,
                    "_ticker": f"KX-T-{i:02d}"})
    raw = grouper_cli.sort_field(raw)

    excluded = [{"golfer_name": n, "reason": "named"} for n in excluded_names]
    devigged = {g["golfer_name"]: g["odds"] for g in grouper_cli.normalize_probabilities(raw)}
    weighted = (grouper_cli.odds_to_conditional(raw, set(excluded_names))
                if excluded_names else grouper_cli.normalize_probabilities(raw))
    groups, report = groupers.partition(weighted, n_teams)
    order = list(range(n_teams))
    team_groups = {teams[i]["team_id"]: groups[order[i]] for i in range(n_teams)}

    args = types.SimpleNamespace(price="ask", espn_league="pga", poll_interval=60)
    return bc.assemble(
        now="2026-08-03T00:00:00+00:00", args=args,
        league={"league_id": "L", "league_name": "Test", "league_slug": "test",
                "source_file": "x.json"},
        teams=teams, team_groups=team_groups, odds_type="winner", series="KXPGATOUR",
        market_label="Outright Winner", exclusive=True, event_ticker="KXPGATOUR-WYC26",
        tournament_name="Wyndham Championship", season=2026,
        espn_event={"event_id": "401811961", "name": "Wyndham Championship", "state": "pre"},
        espn=espn if espn is not None else espn_stage([g["golfer_name"] for g in raw]),
        field=raw, devigged=devigged, weighted=weighted, excluded=excluded,
        liquidity={"golfers": n_golfers}, raw_sum=sum(g["odds"] for g in raw),
        auto_exclude=True,
        tick_structures=["tapered_deci_cent"], report=report, groups=groups,
        order=order, seed=7,
    )


def test_every_golfer_belongs_to_exactly_one_team():
    result = make_result()
    assigned = [g for g in result["golfers"] if g["team_id"]]
    assert len(assigned) == len(result["golfers"])
    assert sum(t["golfer_count"] for t in result["teams"]) == len(assigned)
    seen = [name for t in result["teams"] for name in t["golfer_names"]]
    assert len(seen) == len(set(seen)) == len(assigned)


def test_grouping_weights_sum_to_one_over_the_grouped_field():
    result = make_result()
    total = sum(g["odds"]["grouping_weight"] for g in result["golfers"]
                if g["odds"]["grouping_weight"] is not None)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_team_totals_are_the_sum_of_their_golfers_weights():
    result = make_result()
    by_name = {g["name"]: g for g in result["golfers"]}
    for team in result["teams"]:
        expected = sum(by_name[n]["odds"]["grouping_weight"] for n in team["golfer_names"])
        assert team["total_odds"] == pytest.approx(expected, abs=1e-9)


def test_team_totals_add_up_to_one():
    """
    Approximately, and deliberately so: every weight in the file is rounded for
    readability, so the totals add to 1.0 within the accumulated rounding rather than
    exactly. What IS exact is the relation above -- a team's total is the sum of its
    own golfers -- because that is the one a reader can check by hand.
    """
    result = make_result()
    assert sum(t["total_odds"] for t in result["teams"]) == pytest.approx(1.0, abs=1e-6)


def test_an_excluded_golfer_has_no_weight_and_no_team():
    result = make_result(excluded_names=[golfer_name(0)])
    excluded = next(g for g in result["golfers"] if g["name"] == golfer_name(0))
    assert excluded["excluded"] is True
    assert excluded["odds"]["grouping_weight"] is None
    assert excluded["team_id"] is None
    assert excluded["odds"]["devigged"] > 0            # it still records what they were worth
    assert result["odds_snapshot"]["excluded"][0]["golfer_name"] == golfer_name(0)


def test_exclusion_rescales_the_survivors_back_to_one():
    result = make_result(excluded_names=[golfer_name(0), golfer_name(1)])
    total = sum(g["odds"]["grouping_weight"] for g in result["golfers"]
                if g["odds"]["grouping_weight"] is not None)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert sum(t["total_odds"] for t in result["teams"]) == pytest.approx(1.0, abs=1e-6)


def test_devigged_is_over_the_whole_field_and_sums_to_one_even_with_exclusions():
    """
    The distinction the two keys exist to draw: `devigged` is the de-vig of the whole
    field, `grouping_weight` is what the partitioner actually saw.
    """
    result = make_result(excluded_names=[golfer_name(0)])
    assert sum(g["odds"]["devigged"] for g in result["golfers"]) == pytest.approx(1.0, abs=1e-6)


def test_golfers_are_listed_strongest_first():
    result = make_result()
    raw = [g["odds"]["raw"] for g in result["golfers"]]
    assert raw == sorted(raw, reverse=True)


def test_the_result_records_no_price_read_after_the_groups_were_drawn():
    """
    `not in`, never `is None`. A field present and null is a slot waiting to be filled,
    and the whole claim is that there is no slot: the odds were read once, when the
    groups were drawn, and that reading is the competition.

    Without this, a half-finished removal -- flag gone, function gone, frontend gone,
    assemble() still writing `"current": null` onto all 150 golfers forever -- leaves
    the suite completely green.
    """
    for result in (make_result(), make_result(espn=espn_stage(
            [golfer_name(0), golfer_name(1)], started=True))):
        assert "refreshed" not in result["odds_snapshot"]
        assert all("current" not in g["odds"] for g in result["golfers"])
        assert all(set(g["odds"]) == {"raw", "devigged", "grouping_weight"}
                   for g in result["golfers"])
        # The version is how a reader of an old file finds out the shape changed. A
        # silent drop would leave two incompatible documents both calling themselves 2.0.
        # 4.0 removed `build_mode`; 3.0 removed the inlined crest and banner in favour of
        # an art slug. The one price per golfer this test is about is unchanged through
        # both and is still 2.1's doing.
        assert result["schema_version"] == "4.0"


def test_the_result_records_where_every_number_came_from():
    result = make_result()
    k = result["sources"]["kalshi"]
    assert k["event_ticker"] == "KXPGATOUR-WYC26"
    assert k["markets_endpoint"].startswith("https://api.elections.kalshi.com")
    assert "event_ticker=KXPGATOUR-WYC26" in k["markets_endpoint"]
    assert k["price_mode"] == "ask"
    assert result["sources"]["espn"]["leaderboard_endpoint"].endswith("event=401811961")
    assert result["odds_snapshot"]["captured_at"] == result["generated_at"]


def test_the_result_says_kalshi_is_unreachable_from_a_browser():
    """
    Measured, not assumed: Kalshi 403s every origin but its own. A frontend that
    believes otherwise ships a permanently broken panel.
    """
    result = make_result()
    assert result["sources"]["kalshi"]["browser_reachable"] is False
    assert result["sources"]["espn"]["browser_reachable"] is True


def test_the_build_offers_the_page_espn_and_nothing_else_to_fetch():
    """
    `live` is what the page does while it is open, and that is ESPN and nothing else. A
    Kalshi URL or a relay slot in here is an invitation to write a fetch that can only
    403, which is the panel this simplification removed.

    There is no name-matching block either. The build has already written an athlete id
    onto every golfer it resolved, so the page joins on the id and needs nothing here to
    do it with.
    """
    live = make_result(espn=espn_stage([golfer_name(0), golfer_name(1)]))["live"]
    assert set(live) == {"espn_leaderboard_url", "espn_event_id", "poll_interval_seconds"}
    assert "espn.com" in live["espn_leaderboard_url"]


def test_a_build_made_before_the_first_tee_time_still_tells_the_page_where_to_poll():
    """
    The heart of the simplification, and the thing most easily lost by accident.

    A competition drawn the night before is a FINISHED competition. ESPN has posted the
    field, so every golfer already carries an athlete id, and `live` is present and
    populated -- because the page has to be polling from the moment it opens. That poll
    is how it discovers the first tee time and starts ranking, with nothing re-run and
    no second link. `live: null` here would freeze the page as a groups sheet forever,
    which is exactly what this used to do and no longer does.
    """
    result = make_result()
    assert result["sources"]["espn"]["started_at_build"] is False
    assert set(result["live"]) == {"espn_leaderboard_url", "espn_event_id",
                                   "poll_interval_seconds"}
    assert all(g["espn"]["athlete_id"] for g in result["golfers"])


def test_the_file_has_no_build_mode_and_no_half_that_can_be_null():
    """
    4.0 removed `build_mode` and the nulled-out half it selected. A reader no longer has
    to work out which document they are holding before reading it, and nothing should
    reintroduce a key that says they do.

    Checked over a build made before play and one made during it, because the claim is
    that those two files differ only in `started_at_build` and whatever ESPN said.
    """
    for started in (False, True):
        result = make_result(espn=espn_stage([golfer_name(i) for i in range(40)],
                                             started=started))
        assert "build_mode" not in result
        assert result["live"] is not None
        assert result["sources"]["espn"]["match_report"] is not None
        assert all(g["espn"] is not None for g in result["golfers"])


def test_the_result_carries_the_grouping_certificate():
    result = make_result()
    g = result["grouping"]
    assert g["optimal"] is True and g["exact_grid"] is True
    assert g["delta_ticks"] <= g["floor_ticks"]
    assert "PROVEN OPTIMAL" in g["summary"]
    assert sum(g["group_sizes"]) == len(result["golfers"])


def test_the_clock_at_build_time_is_recorded_and_is_the_only_thing_it_changes():
    """
    `started_at_build` is a record of when this ran, not an instruction to the page --
    the page asks the leaderboard the same question on every poll and believes that.

    So it is worth stating that it changes nothing else. Two builds of the same
    competition, one before play and one during it, must agree on every golfer's athlete
    id and on the whole `live` block. If they ever diverge, the second build is doing
    something the first could have done, and the two-step week is back.
    """
    names = [golfer_name(i) for i in range(40)]
    early = make_result(espn=espn_stage(names))
    late = make_result(espn=espn_stage(names, started=True))

    assert early["sources"]["espn"]["started_at_build"] is False
    assert late["sources"]["espn"]["started_at_build"] is True
    assert early["live"] == late["live"]
    assert ([g["espn"] for g in early["golfers"]] == [g["espn"] for g in late["golfers"]])
    # The event is recorded either way; it is what every poll is aimed at.
    assert early["sources"]["espn"]["event_id"] == "401811961"


def test_the_build_bakes_an_athlete_id_onto_every_golfer_it_resolved():
    """
    The page joins on this id and on nothing else, so a golfer without one scores
    nothing for the life of that page. That makes this the single most load-bearing
    field in a result file -- and, since the join now happens the night before, the
    single best reason to read the build's output before handing the page over.
    """
    names = [golfer_name(i) for i in range(12)]
    result = make_result(n_teams=3, n_golfers=12, espn=espn_stage(names))
    assert result["sources"]["espn"]["field_size_at_build"] == 12
    for golfer in result["golfers"]:
        assert golfer["espn"]["athlete_id"], golfer["name"]
        assert golfer["espn"]["match"] == "exact"
        assert golfer["espn"]["in_field"] is True


def test_a_golfer_nobody_has_looked_at_is_not_reported_as_a_withdrawal():
    """
    Three states, because there are three. A golfer in the field is in it; a golfer a
    review confirmed absent is not; a golfer the join could not settle is UNKNOWN, and
    saying False there claims a withdrawal on evidence nobody gathered.

    Both of the last two score nothing, so the standings cannot tell them apart and do
    not need to. A person deciding whether the build is finished very much does.
    """
    names = [golfer_name(i) for i in range(12)]
    # Golfer Zero was reviewed and is not playing; Golfer One nobody has looked at.
    players = [{"athlete_id": f"a{i:02d}", "name": n, "position": str(i + 1),
                "headshot": None, "country": None, "sort_order": i + 1}
               for i, n in enumerate(names) if i > 1]
    matches, report = espn_leaderboard.match_field(
        names, players, decisions={golfer_name(0): {"absent": True}})
    result = make_result(n_teams=3, n_golfers=12,
                         espn=espn_stage(names, players=players, matches=matches, report=report))

    by_name = {g["name"]: g["espn"] for g in result["golfers"]}
    assert by_name[golfer_name(0)] == {"athlete_id": None, "display_name": None, "headshot": None,
                                    "country": None, "match": "absent", "in_field": False}
    assert by_name[golfer_name(1)]["match"] == "unresolved"
    assert by_name[golfer_name(1)]["in_field"] is None
    assert by_name[golfer_name(2)]["in_field"] is True


def test_the_result_carries_the_standings_rule_it_expects_to_be_scored_by():
    rules = make_result()["standings_rules"]
    assert set(rules["golfer_rank_tiers"]) == {"0", "1", "2", "3"}
    assert "lexicographic" in rules["comparison"]


def test_the_competition_id_is_stable_and_specific():
    a = make_result()["competition_id"]
    assert make_result()["competition_id"] == a
    assert league_mod.uuid.UUID(a).version == 5


def test_the_result_is_json_serialisable():
    json.dumps(make_result())


@pytest.mark.parametrize("n_teams", [2, 5, 13])
def test_assembly_holds_at_several_league_sizes(n_teams):
    result = make_result(n_teams=n_teams, n_golfers=60)
    assert len(result["teams"]) == n_teams
    assert all(t["golfer_count"] >= 1 for t in result["teams"])
    assert sum(t["total_odds"] for t in result["teams"]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Team logos
#
# The one kind of image this half of the pipeline still opens. The league's own art is
# a slug that stays a slug until export -- see the section below.
# ---------------------------------------------------------------------------

def test_a_local_logo_is_inlined(tmp_path):
    logo = tmp_path / "a.svg"
    logo.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    out = bc.inline_logo("a.svg", str(tmp_path))
    assert out.startswith("data:image/svg+xml;base64,")


def test_a_remote_logo_is_left_alone(tmp_path):
    for url in ("https://example.com/a.png", "data:image/png;base64,AAAA"):
        assert bc.inline_logo(url, str(tmp_path)) == url


def test_a_missing_logo_becomes_none_rather_than_a_broken_path(tmp_path, capsys):
    assert bc.inline_logo("nope.png", str(tmp_path)) is None
    assert "logo not found" in capsys.readouterr().out


def test_no_logo_stays_no_logo(tmp_path):
    assert bc.inline_logo(None, str(tmp_path)) is None


def test_an_oversized_logo_is_refused_rather_than_inlined(tmp_path, capsys):
    """It would land in every copy of the result JSON and every page built from it."""
    big = tmp_path / "big.png"
    big.write_bytes(b"\x00" * (bc.MAX_INLINE_LOGO_BYTES + 1))
    assert bc.inline_logo("big.png", str(tmp_path)) == "big.png"
    assert "inline limit" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The league's own identity
# ---------------------------------------------------------------------------

def test_the_result_carries_the_leagues_logo_slug_and_tagline():
    result = make_result(n_teams=3, n_golfers=10)
    assert set(("logo", "tagline")) <= set(result["league"])
    assert result["league"]["logo"] is None        # make_result's league has no art


def test_the_result_carries_no_images_of_its_own(tmp_path):
    """
    The point of the slug. A result file is the document that describes a competition,
    and it used to be mostly an envelope for two base64 PNGs that nothing in it read.
    Whatever else changes, no image belongs in the league block.
    """
    result = make_result(n_teams=2, n_golfers=6)
    result["league"]["logo"] = "example"
    args = art_args(league="leagues/example-league.json",
                    output=str(tmp_path / "out" / "result.json"),
                    update_aliases=False, alias_file=str(tmp_path / "a.json"))
    bc.finish(result, args, {"review": None, "decisions": {}, "matches": {}, "report": None},
              {}, 0.0)

    assert result["league"]["logo"] == "example"
    assert "data:" not in json.dumps(result["league"])


def test_a_rebuild_carries_the_branding_forward():
    """
    A rebuild has no league file to read the slug out of again, and dropping it would
    quietly un-brand a page somebody has already seen.
    """
    result = make_result(n_teams=2, n_golfers=6)
    result["league"].update(logo="wcw", tagline="10th Anniversary")
    league = bc.league_from_result(result)
    assert league["logo"] == "wcw"
    assert league["tagline"] == "10th Anniversary"


def test_a_result_file_written_before_branding_existed_still_rebuilds():
    result = make_result(n_teams=2, n_golfers=6)
    for field in ("logo", "tagline"):
        result["league"].pop(field)
    assert bc.league_from_result(result)["logo"] is None


def test_a_2x_file_loses_its_inlined_art_and_is_told_so(capsys):
    """
    3.0 has nowhere to put two data: URIs and would not want one. Dropping them is the
    change working as intended -- doing it silently would un-brand a page somebody has
    already sent round, which is the kind of quiet difference this tool does not make.
    """
    result = make_result(n_teams=2, n_golfers=6)
    result["league"].pop("logo")
    result["league"].update(crest="data:image/png;base64,AA", banner=None)
    league = bc.league_from_result(result)
    assert league["logo"] is None and "crest" not in league
    out = capsys.readouterr().out
    assert "predates the art slug" in out and "--logo" in out


# ---------------------------------------------------------------------------
# Which art slug a competition carries
#
# Two sources -- the command line and the league file -- and the whole of the rule is
# which one wins. Nothing here opens an image: the slug is a name until export, which
# is what keeps the result file a document rather than an envelope.
# ---------------------------------------------------------------------------

def art_args(**kw):
    """A parsed command line with the fields resolve_league_logo and finish read."""
    base = dict(league="leagues/wcw.json", from_result=None, logo=None, no_logo=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def art(said=None, leagues_dir=None, **cli):
    """
    Settle one competition's art slug.

    `said` is what the league file (or, on a rebuild, the result file) carries; the
    keyword arguments are the command line. Kept apart because `--logo` and a file's
    `logo` are exactly the two things these tests are about telling apart.
    """
    result = {"league": dict({"logo": None}, **(said or {}))}
    return bc.resolve_league_logo(result, art_args(**cli), leagues_dir)["league"]


def test_a_league_that_supplies_no_art_gets_none():
    """There is no shipped default any more. Filling one in meant a page could open
    wearing another league's crest, and a masthead with just a name in it is a shape
    the design draws."""
    assert art()["logo"] is None


def test_the_league_file_is_where_the_slug_usually_comes_from(tmp_path):
    assert art({"logo": "ours"}, leagues_dir=str(tmp_path))["logo"] == "ours"


def test_the_command_line_beats_the_league_file(tmp_path):
    assert art({"logo": "ours"}, leagues_dir=str(tmp_path), logo="mine")["logo"] == "mine"


def test_no_logo_wins_over_everything(tmp_path):
    assert art({"logo": "ours"}, leagues_dir=str(tmp_path), no_logo=True)["logo"] is None


def test_a_rebuild_keeps_the_slug_the_first_build_settled(tmp_path):
    league = art({"logo": "wcw"}, leagues_dir=str(tmp_path),
                 league=None, from_result="build/result.json")
    assert league["logo"] == "wcw"


def test_a_slug_that_names_nothing_is_said_out_loud(tmp_path, capsys):
    """A build that recorded art the export cannot find is a blank masthead somebody
    discovers after sending the page round. It costs one line to say it here."""
    assert art({"logo": "ghost"}, leagues_dir=str(tmp_path))["logo"] == "ghost"
    assert "names no art" in capsys.readouterr().out


def test_half_the_art_is_a_note_rather_than_a_warning(tmp_path, capsys):
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "logo.png").write_bytes(b"\x89PNG")
    art({"logo": "half"}, leagues_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "no banner image" in out and "names no art" not in out


def test_a_path_typed_where_a_slug_belongs_is_an_error_not_a_warning(capsys):
    """
    A slug somebody typed is a thing they meant, unlike a slug a league file merely
    carries. And the check has to come before the build: everything between the command
    line and the end is network, so a shrug here costs the Kalshi fetch and the ESPN
    join twice.
    """
    with pytest.raises(SystemExit):
        bc.main(["--league", "leagues/example-league.json", "--tournament", "Wyndham",
                 "--logo", "leagues/wcw/logo.png"])
    assert "is a slug" in capsys.readouterr().err


def test_a_typed_slug_that_names_no_art_is_an_error(capsys):
    with pytest.raises(SystemExit):
        bc.main(["--league", "leagues/example-league.json", "--tournament", "Wyndham",
                 "--logo", "nosuchleague"])
    assert "names no art" in capsys.readouterr().err


def test_asking_for_art_and_no_art_at_once_is_refused(capsys):
    with pytest.raises(SystemExit):
        bc.main(["--league", "leagues/example-league.json", "--tournament", "Wyndham",
                 "--logo", "example", "--no-logo"])
    assert "--logo and --no-logo" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

def test_a_reviewed_name_binding_is_worth_learning_and_an_exact_match_is_not():
    """
    The alias file only grows through review now, which is the whole reason the review
    step is worth doing twice: settle "Nicolas Echavarria is Nico Echavarria" once and
    every later tournament resolves it with nobody looking.

    An exact match teaches nothing -- the two names were already the same string.
    """
    decisions = {"Zachary Bauchou": {"athlete_id": "3"}}
    matches = {
        "Cameron Young": {"match": "exact", "player": {"name": "Cameron Young"}},
        "Zachary Bauchou": {"match": "decision", "player": {"name": "Zach Bauchou"}},
    }
    assert match_review.learned_aliases(decisions, matches, {}) == {
        "Zachary Bauchou": "Zach Bauchou"}


def test_a_confirmed_withdrawal_is_not_learned_as_an_alias():
    """
    "Jason Day is not in the field" is true of one tournament and false of the next one
    he enters. It belongs to this competition and nowhere else, so it stays in the
    result file and never reaches the alias file.
    """
    decisions = {"Jason Day": {"absent": True, "note": "withdrew"}}
    assert match_review.learned_aliases(decisions, {}, {}) == {}


def test_an_alias_already_known_is_not_relearned():
    decisions = {"Zachary Bauchou": {"athlete_id": "3"}}
    matches = {"Zachary Bauchou": {"match": "decision", "player": {"name": "Zach Bauchou"}}}
    assert match_review.learned_aliases(
        decisions, matches, {"Zachary Bauchou": "Zach Bauchou"}) == {}


def test_aliases_round_trip(tmp_path):
    path = str(tmp_path / "nested" / "aliases.json")
    bc.save_aliases(path, {"A": "B"})
    assert bc.load_aliases(path) == {"A": "B"}


def test_a_missing_alias_file_is_an_empty_map_not_an_error(tmp_path):
    assert bc.load_aliases(str(tmp_path / "nope.json")) == {}
    assert bc.load_aliases(None) == {}


# ---------------------------------------------------------------------------
# Odds types
# ---------------------------------------------------------------------------

def test_every_odds_type_maps_to_a_real_kalshi_series():
    for key, spec in bc.ODDS_TYPES.items():
        assert spec["series"] in bc.kalshi_odds.FIELD_SERIES.values()
        assert isinstance(spec["exclusive"], bool)


def test_only_the_winner_market_has_mutually_exclusive_outcomes():
    """
    Five golfers finish top 5, so a Top 5 book sums toward 5 and the de-vig gives
    share-of-five-slots rather than a probability. The result file has to say which.
    """
    assert bc.ODDS_TYPES["winner"]["exclusive"] is True
    assert not any(bc.ODDS_TYPES[k]["exclusive"] for k in ("top5", "top10", "makecut"))
