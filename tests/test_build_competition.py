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


def groups_stage():
    """
    The ESPN half of a build made before the field existed.

    This is what `espn_stage` returns in groups mode, and it is mostly nulls on purpose:
    there was no join, so there is no join to report on. See build_competition's module
    docstring.
    """
    return {"mode": "groups", "meta": None, "players": [], "matches": {}, "report": None,
            "decisions": {}, "error": None, "review": None}


def live_stage(names, players=None, matches=None, report=None, decisions=None):
    """
    The ESPN half of a build made once the field was published.

    `matches` is keyed by Kalshi name and shaped as espn_leaderboard.match_field returns
    it. Anything not in it and not named in `report["absent"]` comes out `unresolved`.
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
    return {"mode": "live", "meta": {"state": "in", "course": "Sedgefield", "par": 70},
            "players": players, "matches": matches, "report": report,
            "decisions": decisions or {}, "error": None, "review": None}


def make_result(n_teams=4, n_golfers=40, excluded_names=(), tmp_path=None, espn=None):
    """
    Run the whole assembly with a synthetic field and no network.

    Groups mode by default, because that is the first build of any competition. Pass
    `espn=live_stage([...])` for the Thursday one.
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
        espn=espn if espn is not None else groups_stage(),
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
    both_halves = (make_result(),
                   make_result(espn=live_stage([golfer_name(0), golfer_name(1)])))
    for result in both_halves:
        assert "refreshed" not in result["odds_snapshot"]
        assert all("current" not in g["odds"] for g in result["golfers"])
        assert all(set(g["odds"]) == {"raw", "devigged", "grouping_weight"}
                   for g in result["golfers"])


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


def test_a_live_build_offers_the_page_espn_and_nothing_else_to_fetch():
    """
    `live` is what the page does while it is open, and that is ESPN and nothing else. A
    Kalshi URL or a relay slot in here is an invitation to write a fetch that can only
    403, which is the panel this simplification removed.

    There is no name-matching block either. A live build has already written an athlete
    id onto every golfer it resolved, so the page joins on the id and needs nothing
    here to do it with.
    """
    live = make_result(espn=live_stage([golfer_name(0), golfer_name(1)]))["live"]
    assert set(live) == {"espn_leaderboard_url", "espn_event_id", "poll_interval_seconds"}
    assert "espn.com" in live["espn_leaderboard_url"]


def test_a_groups_build_gives_the_page_nothing_to_fetch_at_all():
    """
    Null, not an empty object, and the difference is the whole instruction. An empty
    object is a page that polls an endpoint it was not given; null is a page that does
    not poll. Before the first tee time there is no field to score against, so a fetch
    would be asking a question whose answer it could not use.
    """
    assert make_result()["live"] is None


def test_the_result_carries_the_grouping_certificate():
    result = make_result()
    g = result["grouping"]
    assert g["optimal"] is True and g["exact_grid"] is True
    assert g["delta_ticks"] <= g["floor_ticks"]
    assert "PROVEN OPTIMAL" in g["summary"]
    assert sum(g["group_sizes"]) == len(result["golfers"])


def test_a_groups_build_says_nothing_at_all_about_espn_athletes():
    """
    Before the first tee time nobody knows who this week's competitors are, so the file
    says nothing rather than something empty. `espn: null` on a golfer is the absence of
    a claim; a block of null fields would read as a join that ran and failed.

    `match_report` is null for the same reason. There was no join, so there is no report
    on one -- which is a different thing from a join that matched nobody, and a file
    that cannot tell those apart cannot say whether the build is finished.
    """
    result = make_result()
    assert result["build_mode"] == "groups"
    assert result["sources"]["espn"]["field_size_at_build"] == 0
    assert result["sources"]["espn"]["match_report"] is None
    assert all(g["espn"] is None for g in result["golfers"])
    # The event is still recorded. The tournament was resolved on ESPN even though its
    # field was not published, and the next build needs the id to ask again.
    assert result["sources"]["espn"]["event_id"] == "401811961"


def test_a_live_build_bakes_an_athlete_id_onto_every_golfer_it_resolved():
    """
    The page joins on this id and on nothing else, so a golfer without one scores
    nothing for the life of that page. That makes this the single most load-bearing
    field in a live result file.
    """
    names = [golfer_name(i) for i in range(12)]
    result = make_result(n_teams=3, n_golfers=12, espn=live_stage(names))
    assert result["build_mode"] == "live"
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
                         espn=live_stage(names, players=players, matches=matches, report=report))

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
# Logos
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


def test_a_warning_names_what_is_missing_rather_than_calling_it_a_logo(tmp_path, capsys):
    """The same inliner runs over crests and banners, and a build that says "logo not
    found" while looking for a banner sends somebody hunting through the wrong list."""
    assert bc.inline_logo("nope.png", str(tmp_path), "banner") is None
    assert "banner not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The league's own identity
# ---------------------------------------------------------------------------

def test_the_result_carries_the_leagues_crest_banner_and_tagline():
    result = make_result(n_teams=3, n_golfers=10)
    assert set(("crest", "banner", "tagline")) <= set(result["league"])
    assert result["league"]["crest"] is None       # make_result's league has no art


def test_finish_inlines_the_crest_and_the_banner_against_the_league_file(tmp_path):
    """
    Relative to the league file, not to wherever the build was run from -- exactly as
    the team logos are, and for the same reason: that is where somebody put them.
    """
    (tmp_path / "logos").mkdir()
    for name in ("crest.svg", "banner.svg"):
        (tmp_path / "logos" / name).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    league_path = tmp_path / "wcw.json"
    league_path.write_text("{}")

    result = make_result(n_teams=2, n_golfers=6)
    result["league"]["crest"] = "logos/crest.svg"
    result["league"]["banner"] = "logos/banner.svg"

    args = types.SimpleNamespace(league=str(league_path), from_result=None,
                                 crest=None, banner=None, no_crest=False, no_banner=False,
                                 output=str(tmp_path / "out" / "result.json"),
                                 update_aliases=False, alias_file=str(tmp_path / "a.json"))
    bc.finish(result, args, {"review": None, "decisions": {}, "matches": {}, "report": None},
              {}, 0.0)

    assert result["league"]["crest"].startswith("data:image/svg+xml;base64,")
    assert result["league"]["banner"].startswith("data:image/svg+xml;base64,")


def test_a_rebuild_carries_the_branding_forward(tmp_path):
    """
    By the time a result file exists these are data: URIs, and a rebuild has no league
    file to read them out of again. Dropping them would quietly un-brand a page
    somebody has already seen.
    """
    result = make_result(n_teams=2, n_golfers=6)
    result["league"].update(crest="data:image/png;base64,AA", banner=None,
                            tagline="10th Anniversary")
    league = bc.league_from_result(result)
    assert league["crest"] == "data:image/png;base64,AA"
    assert league["banner"] is None
    assert league["tagline"] == "10th Anniversary"


def test_a_result_file_written_before_branding_existed_still_rebuilds():
    result = make_result(n_teams=2, n_golfers=6)
    for field in ("crest", "banner", "tagline"):
        result["league"].pop(field)
    assert bc.league_from_result(result)["crest"] is None


# ---------------------------------------------------------------------------
# Where the masthead art comes from
#
# Three sources -- the command line, the league file, and the art the tool ships --
# and the whole of the rule is which one wins. The chrome around the art is the
# template's and is the same for every league; these two images are not.
# ---------------------------------------------------------------------------

def art_args(**kw):
    """A parsed command line with only the fields resolve_league_art reads."""
    base = dict(league="leagues/wcw.json", from_result=None,
                crest=None, banner=None, no_crest=False, no_banner=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def art(said=None, **cli):
    """
    Resolve one competition's masthead art.

    `said` is what the league file (or, on a rebuild, the result file) carries, as
    {"crest": ..., "banner": ...}; the keyword arguments are the command line. Kept
    apart because `--crest` and a file's `crest` are exactly the two things these
    tests are about telling apart.
    """
    result = {"league": dict({"crest": None, "banner": None}, **(said or {}))}
    return bc.resolve_league_art(result, art_args(**cli))["league"]


def test_a_league_that_supplies_no_art_gets_the_default():
    """The point of shipping a default: a competition created with nothing but a roster
    still opens looking like the design, not like a page whose images 404'd."""
    league = art()
    assert league["crest"] == bc.DEFAULT_CREST
    assert league["banner"] == bc.DEFAULT_BANNER
    assert os.path.isfile(league["crest"]) and os.path.isfile(league["banner"])


def test_the_league_file_beats_the_default():
    assert art({"crest": "logos/ours.png"})["crest"] == "logos/ours.png"


def test_the_command_line_beats_the_league_file(tmp_path):
    """`--crest` is how the image arrives beside the league JSON when a competition is
    created, so it has to win over art the file happens to carry."""
    mine = tmp_path / "mine.png"
    mine.write_bytes(b"\x89PNG")
    assert art({"crest": "logos/ours.png"}, crest=str(mine))["crest"] == str(mine)


def test_a_typed_path_is_resolved_against_the_working_directory(tmp_path, monkeypatch):
    """
    Not against the league file, which is where a path *inside* that file resolves.
    Making it absolute here is what keeps the inliner from later joining a relative
    `--crest` onto the league directory and looking in a place nobody meant.
    """
    (tmp_path / "art.png").write_bytes(b"\x89PNG")
    monkeypatch.chdir(tmp_path)
    resolved = art(crest="art.png")["crest"]
    assert os.path.isabs(resolved) and os.path.isfile(resolved)


@pytest.mark.parametrize("field", ["crest", "banner"])
def test_no_crest_and_no_banner_win_over_everything(field):
    assert art({field: "logos/ours.png"}, **{f"no_{field}": True})[field] is None


@pytest.mark.parametrize("field", ["crest", "banner"])
def test_false_in_the_league_file_means_none_rather_than_the_default(field):
    """A league that wants a bare masthead has to be able to say so once, in the file,
    rather than remembering a flag on every build."""
    assert art({field: False})[field] is None


def test_a_rebuild_does_not_fill_in_art_the_first_build_left_empty():
    """
    The first build settled this. A rebuild that re-answered it would put a crest on a
    page that has already gone round without one -- and the competition would change
    its appearance on a run that was supposed to be about the leaderboard.
    """
    league = art(league=None, from_result="build/result.json")
    assert league["crest"] is None and league["banner"] is None


def test_a_rebuild_still_takes_art_it_is_handed(tmp_path):
    """Not filling in a default is not the same as refusing to be told."""
    mine = tmp_path / "new.png"
    mine.write_bytes(b"\x89PNG")
    league = art({"crest": "data:image/png;base64,AA"},
                 league=None, from_result="build/result.json", crest=str(mine))
    assert league["crest"] == str(mine)


def test_a_rebuild_carries_an_already_inlined_crest_through_untouched():
    league = art({"crest": "data:image/png;base64,AA"},
                 league=None, from_result="b.json")
    assert league["crest"] == "data:image/png;base64,AA"


def test_the_default_art_survives_the_inliner(tmp_path):
    """
    The default is only a default if it actually lands in the page. It is a real file
    of a real size, and the inliner refuses anything over the limit -- so this is the
    test that fails if somebody drops a 3 MB banner in as the default.
    """
    for what, path in (("crest", bc.DEFAULT_CREST), ("banner", bc.DEFAULT_BANNER)):
        assert os.path.getsize(path) <= bc.MAX_INLINE_LOGO_BYTES, what
        assert bc.inline_logo(path, str(tmp_path), what).startswith("data:image/png;base64,")


@pytest.mark.parametrize("field", ["crest", "banner"])
def test_a_typo_in_a_typed_path_is_an_error_not_a_warning(field, capsys):
    """
    A path somebody typed is a thing they meant, unlike art a league file merely
    mentions. And the check has to come before the build: everything between the
    command line and the inliner is network, so a shrug here costs the Kalshi fetch
    and the ESPN join twice.
    """
    with pytest.raises(SystemExit):
        bc.main(["--league", "leagues/example-league.json", "--tournament", "Wyndham",
                 f"--{field}", "nope.png"])
    assert f"--{field} nope.png is not a file" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["crest", "banner"])
def test_asking_for_art_and_no_art_at_once_is_refused(field, capsys):
    with pytest.raises(SystemExit):
        bc.main(["--league", "leagues/example-league.json", "--tournament", "Wyndham",
                 f"--{field}", "leagues/logos/wcw-crest.png", f"--no-{field}"])
    assert f"--{field} and --no-{field}" in capsys.readouterr().err


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
